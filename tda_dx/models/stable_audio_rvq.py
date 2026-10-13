import json
import sys
import types
from pathlib import Path

import packaging
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from vector_quantize_pytorch import ResidualVQ

from tda_dx.utils import beartype_jit, exists


def _ensure_pkg_resources_packaging():
    if 'pkg_resources' in sys.modules:
        return
    sys.modules['pkg_resources'] = types.SimpleNamespace(packaging=packaging)


def _load_local_stable_audio_autoencoder(model_config_path: str, ckpt_path: str):
    _ensure_pkg_resources_packaging()

    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(ckpt_path))
    return model, model_config


def load_stable_audio_autoencoder(
    *,
    pretrained_name: str | None = None,
    model_config_path: str | None = None,
    ckpt_path: str | None = None,
):
    _ensure_pkg_resources_packaging()

    if exists(model_config_path) or exists(ckpt_path):
        if not (exists(model_config_path) and exists(ckpt_path)):
            raise ValueError('Both model_config_path and ckpt_path must be provided for local Stable Audio loading.')
        return _load_local_stable_audio_autoencoder(model_config_path, ckpt_path)

    if not exists(pretrained_name):
        raise ValueError(
            'Stable Audio autoencoder requires either pretrained_name or both model_config_path and ckpt_path.'
        )

    from stable_audio_tools.models.pretrained import get_pretrained_model

    return get_pretrained_model(pretrained_name)


@beartype_jit
class StableAudioRVQCodec(nn.Module):
    def __init__(
        self,
        *,
        autoencoder: nn.Module,
        rq_num_quantizers: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        rq_ema_decay: float = 0.95,
        threshold_ema_dead_code: float = 0.0,
        use_cosine_sim: bool = False,
        output_hz: int | None = None,
        rvq_checkpoint_path: str | None = None,
        learn_rvq: bool = False,
    ):
        super().__init__()

        self.autoencoder = autoencoder.eval()
        self.learn_rvq = learn_rvq

        for param in self.autoencoder.parameters():
            param.requires_grad = False

        latent_dim = int(getattr(self.autoencoder, 'latent_dim'))
        sample_rate = int(getattr(self.autoencoder, 'sample_rate'))
        downsampling_ratio = int(getattr(self.autoencoder, 'downsampling_ratio'))

        self.sample_rate = sample_rate
        self.downsampling_ratio = downsampling_ratio
        self.audio_channels = int(getattr(self.autoencoder, 'io_channels', getattr(self.autoencoder, 'audio_channels', 1)))
        self.output_hz = int(output_hz) if exists(output_hz) else int(round(sample_rate / downsampling_ratio))
        self.num_quantizers = int(rq_num_quantizers)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim) if exists(codebook_dim) else latent_dim
        self.use_cosine_sim = bool(use_cosine_sim)

        self.rq = ResidualVQ(
            dim=latent_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            commitment_weight=0.0,
            decay=rq_ema_decay,
            kmeans_init=True,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=self.use_cosine_sim,
        )

        if exists(rvq_checkpoint_path):
            rvq_state = torch.load(rvq_checkpoint_path, map_location='cpu')
            self.rq.load_state_dict(rvq_state)

    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = rearrange(x, 'b t -> b 1 t')

        # Stable Audio's VAE is stereo. Our training waveforms are often mono,
        # so promote mono to stereo instead of failing on the first conv layer.
        if x.shape[1] == 1 and self.audio_channels == 2:
            x = x.repeat(1, 2, 1)
        elif x.shape[1] != self.audio_channels:
            raise ValueError(
                f'Expected {self.audio_channels} audio channels for Stable Audio VAE, got {x.shape[1]}'
            )

        with torch.no_grad():
            self.autoencoder.eval()
            # Use the VAE *mean* latent instead of a stochastic posterior sample.
            # VAEBottleneck.encode() always samples (z = mean + std*eps) even in
            # eval mode, which bakes irreducible noise into the stored latents and
            # sets a hard floor on RVQ reconstruction quality.  Calling the
            # encoder directly and taking only the mean half avoids this.
            enc_out = self.autoencoder.encoder(x)
            if hasattr(self.autoencoder, 'bottleneck') and self.autoencoder.bottleneck is not None:
                # VAE bottleneck: encoder outputs [B, 2*C, T] (mean + log-scale).
                # Grab just the mean (first half) and pass it through the decoder
                # path as-is, bypassing the stochastic sampling step.
                mean, _scale = enc_out.chunk(2, dim=1)
                latents = mean
            else:
                # Non-VAE bottleneck or no bottleneck — behave as before.
                latents = self.autoencoder.encode(x)

        return latents

    def quantize_latents(self, latents: torch.Tensor, return_rvq_loss: bool = False):
        with torch.set_grad_enabled(self.learn_rvq):
            self.rq.train(self.learn_rvq)
            q, indices, _ = self.rq(rearrange(latents, 'b c t -> b t c'))

        if return_rvq_loss:
            return F.mse_loss(q, rearrange(latents, 'b c t -> b t c')).item()

        return rearrange(indices, 'b t q -> b t q')

    def analyze_quantization(self, latents: torch.Tensor):
        with torch.set_grad_enabled(self.learn_rvq):
            self.rq.train(self.learn_rvq)
            q, indices, per_quantizer_losses = self.rq(rearrange(latents, 'b c t -> b t c'))

        target = rearrange(latents, 'b c t -> b t c')
        mse = F.mse_loss(q, target).item()
        cosine = F.cosine_similarity(q.reshape(-1, q.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1).mean().item()

        perplexities = []
        dead_code_fractions = []
        usage_fractions = []

        for quantizer_index in range(indices.shape[-1]):
            quantizer_indices = indices[..., quantizer_index].reshape(-1)
            counts = torch.bincount(quantizer_indices, minlength=self.codebook_size).to(torch.float32)
            probs = counts / counts.sum().clamp(min=1.0)
            entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
            perplexity = entropy.exp().item()
            used_codes = (counts > 0).sum().item()
            perplexities.append(perplexity)
            dead_code_fractions.append(1.0 - (used_codes / self.codebook_size))
            usage_fractions.append(used_codes / self.codebook_size)

        return {
            'quantized_latents': rearrange(q, 'b t c -> b c t'),
            'indices': rearrange(indices, 'b t q -> b t q'),
            'mse': mse,
            'cosine': cosine,
            'per_quantizer_losses': per_quantizer_losses.mean(dim=0).detach().cpu().tolist(),
            'perplexities': perplexities,
            'dead_code_fractions': dead_code_fractions,
            'usage_fractions': usage_fractions,
        }

    def collect_quantization_statistics(self, latents: torch.Tensor):
        with torch.set_grad_enabled(self.learn_rvq):
            self.rq.train(self.learn_rvq)
            q, indices, _ = self.rq(rearrange(latents, 'b c t -> b t c'))

        target = rearrange(latents, 'b c t -> b t c')
        sq_error = (q - target) ** 2
        cosine = F.cosine_similarity(q, target, dim=-1)

        sample_mse = sq_error.mean(dim=(1, 2))
        sample_cosine = cosine.mean(dim=1)

        code_counts = []
        for quantizer_index in range(indices.shape[-1]):
            quantizer_indices = indices[..., quantizer_index].reshape(-1)
            counts = torch.bincount(quantizer_indices, minlength=self.codebook_size).to(torch.float32)
            code_counts.append(counts)

        return {
            'quantized_latents': rearrange(q, 'b t c -> b c t'),
            'indices': indices,
            'sample_mse': sample_mse.detach(),
            'sample_cosine': sample_cosine.detach(),
            'mse_sum': sq_error.sum().detach(),
            'mse_count': torch.tensor(sq_error.numel(), device=sq_error.device, dtype=torch.float32),
            'cosine_sum': cosine.sum().detach(),
            'cosine_count': torch.tensor(cosine.numel(), device=cosine.device, dtype=torch.float32),
            'code_counts': torch.stack(code_counts, dim=0).detach(),
        }

    def training_loss(self, latents: torch.Tensor):
        self.rq.train(True)
        target = rearrange(latents, 'b c t -> b t c')
        q, _, _ = self.rq(target)
        return F.mse_loss(q, target)

    def forward(self, x: torch.Tensor, return_encoded: bool = True, **kwargs):
        assert return_encoded is True
        latents = self.encode_latents(x)
        indices = self.quantize_latents(latents)
        return None, indices, None

    def decode_from_codebook_indices(self, quantized_indices: torch.Tensor):
        if quantized_indices.dim() != 3:
            raise ValueError(
                f'Expected quantized_indices to have shape [B, T, n_q], got {tuple(quantized_indices.shape)}'
            )

        if quantized_indices.shape[-1] > self.num_quantizers:
            raise ValueError(
                f'Expected at most {self.num_quantizers} quantizers for decode, got {quantized_indices.shape[-1]}'
            )

        if quantized_indices.shape[-1] < self.num_quantizers:
            pad_shape = (*quantized_indices.shape[:-1], self.num_quantizers - quantized_indices.shape[-1])
            zero_pad = torch.zeros(pad_shape, dtype=quantized_indices.dtype, device=quantized_indices.device)
            quantized_indices = torch.cat((quantized_indices, zero_pad), dim=-1)

        latents = self.rq.get_outputs_from_indices(quantized_indices)
        latents = rearrange(latents, 'b t c -> b c t')

        with torch.no_grad():
            self.autoencoder.eval()
            wave = self.autoencoder.decode(latents)

        return wave

    def save(self, path: str | Path):
        torch.save(self.rq.state_dict(), str(path))
