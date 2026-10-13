import contextlib
import csv
import itertools
import math
import os
import time
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import (Accelerator, DistributedDataParallelKwargs,
                        DistributedType)
from beartype.door import is_bearable
from beartype.typing import Dict, List, Literal, Optional, Union
from beartype.vale import Is
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from typing_extensions import Annotated

from tda_dx.models.clap_quantized import ClapQuantized
from tda_dx.data.audio_dataset import (ChunkedSoundDataset, NpySoundDataset, SoundDataset, get_dataloader)
from tda_dx.data.memmap import MemmapPreprocessedDataset, get_fast_dataloader
from tda_dx.models.hubert_kmeans import HfHubertWithKmeans, learn_kmeans
from tda_dx.models.types import NeuralCodec, Wav2Vec
from tda_dx.models.musiclm import (CoarseStage, FineStage, SemanticStage,
                           TokenConditionedTransformer)
from tda_dx.training.optimizer import get_linear_scheduler, get_optimizer, get_warmup_cosine_scheduler
from tda_dx.utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, beartype_jit, ceil_div,
                    copy_file_to_folder, default, eval_decorator, exists,
                    generate_mask_with_prob, get_embeds, gumbel_sample,
                    mask_out_after_eos_id, round_down_nearest_multiple, top_k)

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    input_audio=Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def sanitize_hparams(hps):
    for key, value in hps.items():
        if not (
            isinstance(value, int) or
            isinstance(value, float) or
            isinstance(value, str) or
            isinstance(value, bool) or
            isinstance(value, torch.Tensor)
        ):
            hps[key] = str(value)
    return hps


def _normalize_split_name(split_name: str):
    v = str(split_name).strip().lower()
    if v in {'train', 'training'}:
        return 'training'
    if v in {'val', 'valid', 'validation'}:
        return 'validation'
    if v in {'test', 'testing'}:
        return 'test'
    return v


def _normalize_subset_name(subset_name: str):
    return str(subset_name).strip().lower()


def _parse_subset_filter_values(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if value in {'', 'all', '*', 'none'}:
        return None

    values = {
        _normalize_subset_name(part)
        for token in value.split(',')
        for part in token.strip().split()
        if part.strip()
    }
    return values if len(values) > 0 else None

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)


def noop(*args, **kwargs):
    pass


@beartype_jit
class SingleStageTrainer(nn.Module):
    """
    General trainer for any stage of MusicLM.
        semantic: requires audio_conditioner and wav2vec
        coarse: requires audio_conditioner, wav2vec, and neural_codec
        fine: requires audio_conditioner and neural_codec
    """

    def __init__(
        self,
        transformer: TokenConditionedTransformer,
        stage: Literal['semantic', 'coarse', 'fine'],
        *,
        num_train_steps,
        batch_size,
        model_config,
        training_config,
        dataset: Optional[Dataset] = None,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        data_max_length_seconds = 1,
        ignore_files: Optional[List[str]]=None,
        cross_entropy_loss_weights: Optional[List[float]]=None,
        ignore_load_errors=True,
        folder=None,
        use_preprocessed_data=False,
        use_mmap_data=True,
        lr=3e-4,
        lr_warmup=0,
        lr_schedule='constant',
        lr_min_factor=0.01,
        grad_accum_every=1,
        wd=0.,
        max_grad_norm=0.5,
        valid_frac=0.05,
        valid_freq=1.0,
        random_split_seed=42,
        save_results_every=100,
        val_every_n_epochs=0,
        val_every_n_steps=0,
        save_predicted_tokens=True,
        save_reconstructed_wave=True,
        save_model_every=1000,
        save_top_k_checkpoints=0,
        alert_every=0,
        val_early_stop_patience_validations=0,
        val_early_stop_patience_epochs=0,
        val_early_stop_metric='loss',
        val_top_k: Optional[List[int]] = None,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        use_fma_official_split=True,
        fma_train_split='training',
        fma_valid_split='validation',
        fma_test_split='test',
        fma_tracks_csv_path=None,
        fma_valid_subset=None,
        fma_test_subset=None,
        exhaustive_stride=1,
        jitter_seconds=0.0,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        accelerate_kwargs = dict(accelerate_kwargs)
        accelerate_kwargs.setdefault('step_scheduler_with_optimizer', False)
        kwargs_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[kwargs_handler])

        self.log_with = accelerate_kwargs['log_with'] if 'log_with' in accelerate_kwargs else None

        self.use_preprocessed_data = use_preprocessed_data
        self.use_mmap_data = use_mmap_data
        self.model_config = model_config
        self.training_config = training_config
        self.use_fma_official_split = bool(use_fma_official_split)
        self.fma_train_split = _normalize_split_name(fma_train_split)
        self.fma_valid_split = _normalize_split_name(fma_valid_split)
        self.fma_test_split = _normalize_split_name(fma_test_split)
        self.fma_tracks_csv_path = fma_tracks_csv_path
        self.fma_valid_subset_filter = _parse_subset_filter_values(fma_valid_subset)
        self.fma_test_subset_filter = _parse_subset_filter_values(fma_test_subset)

        self.transformer = transformer

        self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner
        self.neural_codec = neural_codec

        self.stage = stage

        if stage == 'semantic':
            assert self.use_preprocessed_data or (exists(audio_conditioner) and exists(wav2vec))
            self.train_wrapper = SemanticStage(
                semantic_transformer=transformer,
                wav2vec=wav2vec,
                clap=audio_conditioner,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 1.])
            )
            if self.use_preprocessed_data:
                self.ds_fields = ('clap_token_ids', 'semantic_token_ids')
            else:
                self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic')
                target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz)
                normalize = (False, True)
                seq_len_multiple_of = wav2vec.seq_len_multiple_of
        elif stage == 'coarse':
            assert self.use_preprocessed_data or (exists(wav2vec) and exists(audio_conditioner) and exists(neural_codec))
            self.train_wrapper = CoarseStage(
                coarse_transformer=transformer,
                neural_codec=neural_codec,
                wav2vec=wav2vec,
                clap=audio_conditioner,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 0., 1.])
            )
            if self.use_preprocessed_data:
                self.ds_fields = ('clap_token_ids', 'semantic_token_ids', 'coarse_token_ids')
            else:
                self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic', 'raw_wave_for_acoustic')
                target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz, neural_codec.sample_rate)
                normalize = (False, True, False)
                seq_len_multiple_of = wav2vec.seq_len_multiple_of
        elif stage == 'fine':
            assert self.use_preprocessed_data or (exists(audio_conditioner) and exists(neural_codec))
            self.train_wrapper = FineStage(
                fine_transformer=transformer,
                clap=audio_conditioner,
                neural_codec=neural_codec,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 0., 1.])
            )
            if self.use_preprocessed_data:
                self.ds_fields = ('clap_token_ids', 'coarse_token_ids', 'fine_token_ids')
            else:
                self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_acoustic')
                target_sample_hz = (audio_conditioner.sample_rate, neural_codec.sample_rate)
                normalize = (False, False)
                seq_len_multiple_of = None
        else:
            raise ValueError(f'invalid stage: {stage}')

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        dataloader_num_workers = max(0, int(num_workers))
        dataloader_prefetch_factor = max(1, int(prefetch_factor))
        dataloader_persistent_workers = bool(persistent_workers) and dataloader_num_workers > 0
        dataloader_pin_memory = bool(pin_memory)

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr=lr, wd=wd)

        if lr_schedule == 'cosine':
            self.scheduler = get_warmup_cosine_scheduler(
                self.optim,
                warmup_iters=lr_warmup,
                total_iters=num_train_steps,
                min_lr_factor=lr_min_factor,
            )
        elif lr_warmup > 0:
            self.scheduler = get_linear_scheduler(
                self.optim,
                total_iters=lr_warmup,
            )
        else:
            self.scheduler = None

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        if self.use_preprocessed_data:
            if not self.use_mmap_data:
                raise ValueError(
                    'use_preprocessed_data=True requires use_mmap_data=True. '
                    'DB-backed preprocessed dataset is no longer supported.'
                )
            print('🚀 Using memory-mapped dataset for ultra-fast loading')
            self.ds = MemmapPreprocessedDataset(
                folder,
                stage=self.stage,
                semantic_window_seconds=int(self.model_config.global_cfg.semantic_audio_length_seconds),
                coarse_window_seconds=int(self.model_config.global_cfg.coarse_audio_length_seconds),
                fine_window_seconds=int(self.model_config.global_cfg.fine_audio_length_seconds),
                semantic_steps_per_second=self.model_config.hubert_kmeans_cfg.output_hz,
                acoustic_steps_per_second=self.model_config.acoustic_output_hz,
                exhaustive_stride=int(exhaustive_stride),
                jitter_seconds=float(jitter_seconds),
            )
        else:
            self.ds = dataset
            if not exists(self.ds):
                assert exists(
                    folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

                self.ds = SoundDataset(
                    folder,
                    max_length_seconds=data_max_length_seconds,
                    normalize=normalize,
                    target_sample_hz=target_sample_hz,
                    seq_len_multiple_of=seq_len_multiple_of,
                    ignore_files=default(ignore_files, []),
                    ignore_load_errors=ignore_load_errors
                )

        # split for validation

        self.test_ds = None

        if self.use_preprocessed_data and self.use_fma_official_split:
            self.ds, self.valid_ds, self.test_ds = self._split_preprocessed_dataset_by_fma(
                dataset=self.ds,
                folder=folder,
                train_split=self.fma_train_split,
                valid_split=self.fma_valid_split,
                test_split=self.fma_test_split,
                tracks_csv_path=self.fma_tracks_csv_path,
                valid_subset_filter=self.fma_valid_subset_filter,
                test_subset_filter=self.fma_test_subset_filter,
            )
            test_count = 0 if self.test_ds is None else len(self.test_ds)
            self.print(
                f'training with dataset of {len(self.ds)} samples, '
                f'validating with official FMA split {len(self.valid_ds)} samples, '
                f'testing with official FMA split {test_count} samples'
            )
            if self.test_ds is None:
                self.print('warning: no test split samples found; final test evaluation will be skipped')
        elif valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader
        train_dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
        if dataloader_num_workers > 0:
            train_dl_kwargs['prefetch_factor'] = dataloader_prefetch_factor
            train_dl_kwargs['persistent_workers'] = dataloader_persistent_workers

        valid_dl_kwargs = dict(train_dl_kwargs)
        valid_dl_kwargs['shuffle'] = False

        if self.use_preprocessed_data:
            self.dl = get_fast_dataloader(self.ds, drop_last=True, **train_dl_kwargs)
            self.valid_dl = get_fast_dataloader(self.valid_ds, drop_last=False, **valid_dl_kwargs)
            self.test_dl = (
                get_fast_dataloader(self.test_ds, drop_last=False, **valid_dl_kwargs)
                if exists(self.test_ds) else None
            )
        else:
            self.dl = get_dataloader(self.ds, **train_dl_kwargs)
            self.valid_dl = get_dataloader(self.valid_ds, **valid_dl_kwargs)
            self.test_dl = (
                get_dataloader(self.test_ds, **valid_dl_kwargs)
                if exists(self.test_ds) else None
            )

        # prepare with accelerator

        (
            self.train_wrapper,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.dl,
            self.valid_dl
        )

        if exists(self.scheduler):
            self.scheduler = self.accelerator.prepare(self.scheduler)

        if exists(self.test_dl):
            self.test_dl = self.accelerator.prepare(self.test_dl)

        # dataloader iterators

        self.dl_iter = cycle(self.dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.val_every_n_epochs = max(0, int(val_every_n_epochs))
        configured_val_every_n_steps = max(0, int(val_every_n_steps))
        self.valid_freq = float(valid_freq)
        if self.valid_freq < 0.0 or self.valid_freq > 1.0:
            raise ValueError(f'valid_freq must be in [0, 1], got: {valid_freq}')
        self.save_top_k_checkpoints = max(0, int(save_top_k_checkpoints))
        self.alert_every = max(0, int(alert_every))
        self.num_workers = dataloader_num_workers
        self.prefetch_factor = dataloader_prefetch_factor
        self.persistent_workers = dataloader_persistent_workers
        self.pin_memory = dataloader_pin_memory
        parsed_val_top_k = [] if val_top_k is None else [int(k) for k in val_top_k]
        parsed_val_top_k = sorted(set(k for k in parsed_val_top_k if k > 0))
        self.val_top_k = parsed_val_top_k if len(parsed_val_top_k) > 0 else [1]
        self.save_predicted_tokens = save_predicted_tokens
        self.save_reconstructed_wave = save_reconstructed_wave
        self.best_checkpoints = []
        self.should_stop_early = False
        self.early_stop_best_metric = None
        self.early_stop_bad_validations = 0
        self.early_stop_best_step = None

        steps_per_epoch = ceil_div(len(self.dl), self.grad_accum_every)
        self.steps_per_epoch = max(1, int(steps_per_epoch))

        # valid_freq is the fraction of one epoch after which validation runs.
        if self.valid_freq > 0:
            self.val_every_n_steps = max(1, int(math.ceil(self.steps_per_epoch * self.valid_freq)))
            self.print(
                f'validation schedule: every {self.val_every_n_steps} train steps '
                f'(~{self.valid_freq * 100:.1f}% of one epoch)'
            )
        else:
            epoch_based_val_every_n_steps = (
                self.steps_per_epoch * self.val_every_n_epochs
                if self.val_every_n_epochs > 0 else 0
            )
            self.val_every_n_steps = configured_val_every_n_steps or epoch_based_val_every_n_steps

            if configured_val_every_n_steps > 0:
                self.print(
                    f'validation schedule: every {self.val_every_n_steps} train steps '
                    f'(val_every_n_steps)'
                )
            elif self.val_every_n_epochs > 0:
                self.print(
                    f'validation schedule: every {self.val_every_n_epochs} epoch(s) '
                    f'(~{self.val_every_n_steps} train steps)'
                )
        if self.save_top_k_checkpoints > 0:
            self.print(
                f'best-checkpoint mode: keep top {self.save_top_k_checkpoints} '
                f'by configured validation metric'
            )
        self.print(
            f'dataloader config: num_workers={self.num_workers}, '
            f'prefetch_factor={self.prefetch_factor if self.num_workers > 0 else "n/a"}, '
            f'persistent_workers={self.persistent_workers}, pin_memory={self.pin_memory}'
        )

        metric = str(val_early_stop_metric).lower().strip()
        if metric not in ('loss', 'accuracy'):
            raise ValueError(f'val_early_stop_metric must be one of (loss, accuracy), got: {val_early_stop_metric}')

        self.val_early_stop_metric = metric
        self.val_early_stop_patience_validations = max(0, int(val_early_stop_patience_validations))
        self.val_early_stop_patience_epochs = max(0, int(val_early_stop_patience_epochs))

        if self.val_early_stop_patience_validations > 0:
            self.early_stop_patience_validations = self.val_early_stop_patience_validations
            self.print(
                f'early-stop enabled: metric={self.val_early_stop_metric}, '
                f'patience={self.early_stop_patience_validations} validation checks'
            )
        elif self.val_early_stop_patience_epochs > 0 and self.val_every_n_epochs <= 0:
            self.print('early-stop disabled: val_early_stop_patience_epochs > 0 but val_every_n_epochs <= 0')
            self.early_stop_patience_validations = 0
        elif self.val_early_stop_patience_epochs > 0:
            self.early_stop_patience_validations = ceil_div(
                self.val_early_stop_patience_epochs,
                max(1, self.val_every_n_epochs),
            )
            self.print(
                f'early-stop enabled: metric={self.val_early_stop_metric}, '
                f'patience={self.val_early_stop_patience_epochs} epoch(s) '
                f'(~{self.early_stop_patience_validations} validation checks)'
            )
        else:
            self.early_stop_patience_validations = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        if self.is_main and len([*self.results_folder.glob('**/*')]) > 0:
            self.print('results folder is not empty; keeping existing files (no interactive prompt)')

        self.accelerator.wait_for_everyone()

        if exists(save_reconstructed_wave):
            self.waves_folder = self.results_folder / 'reconstructed_waves'
            if self.is_main:
                self.waves_folder.mkdir(parents=True, exist_ok=True)
        if exists(save_predicted_tokens):
            self.tokens_folder = self.results_folder / 'tokens'
            if self.is_main:
                self.tokens_folder.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()

        hps = {}
        model_sections = {
            'model.global': asdict(self.model_config.global_cfg),
            'model.clap_rvq': asdict(self.model_config.clap_rvq_cfg),
            'model.hubert_kmeans': asdict(self.model_config.hubert_kmeans_cfg),
            'model.semantic': asdict(self.model_config.semantic_cfg),
            'model.coarse': asdict(self.model_config.coarse_cfg),
            'model.fine': asdict(self.model_config.fine_cfg),
        }
        if exists(self.model_config.encodec_cfg):
            model_sections['model.encodec'] = asdict(self.model_config.encodec_cfg)
        if exists(self.model_config.dac_cfg):
            model_sections['model.dac'] = asdict(self.model_config.dac_cfg)
        if exists(getattr(self.model_config, 'stable_audio_rvq_cfg', None)):
            model_sections['model.stable_audio_rvq'] = asdict(self.model_config.stable_audio_rvq_cfg)

        for prefix, section in model_sections.items():
            hps.update({f'{prefix}.{k}': v for k, v in section.items()})

        if stage == 'semantic':
            hps.update({f'training.semantic.{k}': v for k, v in asdict(self.training_config.semantic_trainer_cfg).items()})
        elif stage == 'coarse':
            hps.update({f'training.coarse.{k}': v for k, v in asdict(self.training_config.coarse_trainer_cfg).items()})
        elif stage == 'fine':
            hps.update({f'training.fine.{k}': v for k, v in asdict(self.training_config.fine_trainer_cfg).items()})

        hps = sanitize_hparams(hps)

        tracker_name = f"{stage}_stage_{int(time.time() * 1000)}" if 'tensorboard' in self.log_with else f"{stage}_stage"
        if 'wandb' in self.log_with and os.environ.get('WANDB_PROJECT'):
            # accelerate's WandB tracker passes tracker_name as wandb.init(project=...)
            tracker_name = os.environ['WANDB_PROJECT']

        init_kwargs = {}
        if 'wandb' in self.log_with:
            wandb_kwargs = {}
            if os.environ.get('WANDB_NAME'):
                wandb_kwargs['name'] = os.environ['WANDB_NAME']
            if os.environ.get('WANDB_TAGS'):
                tags = [t.strip() for t in os.environ['WANDB_TAGS'].split(',') if t.strip()]
                if tags:
                    wandb_kwargs['tags'] = tags
            if len(wandb_kwargs) > 0:
                init_kwargs['wandb'] = wandb_kwargs

        self.accelerator.init_trackers(
            tracker_name,
            config=hps,
            init_kwargs=init_kwargs,
        )

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

    def save(self, model_path, optim_path, scheduler_path=None):
        model_state_dict = self.accelerator.get_state_dict(self.transformer)
        torch.save(model_state_dict, model_path)

        optim_state_dict = self.optim.state_dict()
        torch.save(optim_state_dict, optim_path)

        if exists(self.scheduler):
            assert exists(scheduler_path)
            scheduler_state_dict = self.scheduler.state_dict()
            torch.save(scheduler_state_dict, scheduler_path)

    def load(self, model_path, optim_path, scheduler_path=None, steps=0):
        model_path = Path(model_path)
        optim_path = Path(optim_path)
        assert model_path.exists() and optim_path.exists()

        model_state_dict = torch.load(model_path, map_location=self.device)
        optim_state_dict = torch.load(optim_path, map_location=self.device)
        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(model_state_dict)
        self.optim.load_state_dict(optim_state_dict)

        if exists(self.scheduler):
            assert exists(scheduler_path), 'the config specifies a lr schedule that requires a scheduler checkpoint, but none was given. try setting lr_warmup to 0 and lr_schedule to constant.'
            scheduler_path = Path(scheduler_path)
            assert scheduler_path.exists()
            scheduler_state_dict = torch.load(scheduler_path, map_location=self.device)
            self.scheduler.load_state_dict(scheduler_state_dict)

        if steps > 0:
            assert int(self.steps.item()) == 0, 'steps should be 0 when loading a checkpoint for the first time'
            self.steps += steps

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def _load_memmap_metadata(self, folder: Path):
        metadata_path = folder / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f'metadata.json not found at {metadata_path}')
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _load_idx_to_split_for_memmap(self, folder: Path):
        metadata = self._load_memmap_metadata(folder)
        split_file = metadata.get('idx_to_split_file', 'idx_to_split.json')
        split_path = folder / split_file
        if split_path.exists():
            with open(split_path, 'r') as f:
                raw = json.load(f)
            return {int(k): _normalize_split_name(v) for k, v in raw.items()}
        raise FileNotFoundError(
            f'idx->split map not found ({split_path}). '
            f'Regenerate preprocessing to include split metadata.'
        )

    def _load_idx_to_track_id_for_memmap(self, folder: Path):
        metadata = self._load_memmap_metadata(folder)
        track_file = metadata.get('idx_to_track_id_file', 'idx_to_track_id.json')
        track_path = folder / track_file
        if track_path.exists():
            with open(track_path, 'r') as f:
                raw = json.load(f)
            return {int(k): str(v) for k, v in raw.items()}
        raise FileNotFoundError(
            f'idx->track_id map not found ({track_path}). '
            f'Regenerate preprocessing to include track-id metadata.'
        )

    def _load_track_id_to_subset_map(self, tracks_csv_path: str):
        tracks_path = Path(str(tracks_csv_path).strip())
        if (not tracks_path.exists()) or (not tracks_path.is_file()):
            raise FileNotFoundError(f'tracks.csv not found: {tracks_path}')

        mapping = {}
        with open(tracks_path, newline='') as f:
            reader = csv.reader(f)
            header1 = next(reader)
            header2 = next(reader)

            subset_idx = None
            for i, (a, b) in enumerate(zip(header1, header2)):
                if a.strip().lower() == 'set' and b.strip().lower() == 'subset':
                    subset_idx = i
                    break
            if subset_idx is None:
                raise ValueError(f'set.subset column not found in {tracks_path}')

            for row in reader:
                if len(row) <= subset_idx:
                    continue
                try:
                    track_id = str(int(row[0]))
                except ValueError:
                    continue
                mapping[track_id] = _normalize_subset_name(row[subset_idx])

        return mapping

    def _split_preprocessed_dataset_by_fma(
        self,
        dataset,
        folder,
        train_split: str,
        valid_split: str,
        test_split: Optional[str] = None,
        tracks_csv_path: Optional[str] = None,
        valid_subset_filter=None,
        test_subset_filter=None,
    ):
        folder_path = Path(folder)
        idx_to_split = self._load_idx_to_split_for_memmap(folder_path)

        apply_subset_filter = (valid_subset_filter is not None) or (test_subset_filter is not None)
        idx_to_track_id = {}
        track_id_to_subset = {}
        if apply_subset_filter:
            if (not exists(tracks_csv_path)) or (len(str(tracks_csv_path).strip()) == 0):
                raise ValueError(
                    'subset filtering for validation/test requires --fma_tracks_csv_path (path to tracks.csv).'
                )
            idx_to_track_id = self._load_idx_to_track_id_for_memmap(folder_path)
            track_id_to_subset = self._load_track_id_to_subset_map(tracks_csv_path)

        train_positions = []
        valid_positions = []
        test_positions = []
        for pos in range(len(dataset)):
            if hasattr(dataset, 'get_sqlite_idx_for_sample'):
                idx = int(dataset.get_sqlite_idx_for_sample(pos))
            else:
                idx = int(dataset.indices[pos])
            split_name = _normalize_split_name(idx_to_split.get(int(idx), ''))

            if split_name == train_split:
                train_positions.append(pos)
                continue

            track_subset = None
            if apply_subset_filter and (split_name == valid_split or (exists(test_split) and split_name == test_split)):
                track_id = str(idx_to_track_id.get(int(idx), '')).strip()
                if len(track_id) == 0:
                    raise RuntimeError(
                        f'missing track_id for idx={idx} while subset filtering is enabled. '
                        f'Check idx_to_track_id metadata in {folder_path}.'
                    )
                if track_id not in track_id_to_subset:
                    raise RuntimeError(
                        f'track_id={track_id} (idx={idx}) not found in tracks.csv subset metadata: {tracks_csv_path}'
                    )
                track_subset = track_id_to_subset[track_id]

            if split_name == valid_split:
                if valid_subset_filter is not None and track_subset not in valid_subset_filter:
                    continue
                valid_positions.append(pos)
            elif exists(test_split) and split_name == test_split:
                if test_subset_filter is not None and track_subset not in test_subset_filter:
                    continue
                test_positions.append(pos)

        if len(train_positions) == 0 or len(valid_positions) == 0:
            raise RuntimeError(
                f'official split produced empty subsets: train={len(train_positions)}, valid={len(valid_positions)}. '
                f'filters(valid={valid_subset_filter}, test={test_subset_filter}). '
                f'Check tracks.csv and preprocessed metadata alignment.'
            )

        test_subset = Subset(dataset, test_positions) if len(test_positions) > 0 else None
        return Subset(dataset, train_positions), Subset(dataset, valid_positions), test_subset

    def _evaluate_split_metrics(self, eval_dl, split_name: str):
        self.train_wrapper.eval()

        loss_weighted_sum = 0.0
        batch_count = 0
        loss_token_count_total = 0
        token_count_total = 0
        token_correct_total = 0.0
        topk_correct_totals = {}
        topk_count_totals = {}
        codebook_loss_weighted_sums = {}
        codebook_token_totals = {}

        try:
            eval_total_batches = len(eval_dl)
        except TypeError:
            eval_total_batches = None

        show_progress = bool(self.is_main)
        eval_iterator = tqdm(
            eval_dl,
            desc=f'evaluating {split_name}',
            total=eval_total_batches,
            disable=not show_progress,
            mininterval=5,
            leave=False,
        )

        with torch.no_grad():
            for eval_batch in eval_iterator:
                data_kwargs = dict(zip(self.ds_fields, eval_batch))
                if len(data_kwargs) == 0:
                    continue

                batch_loss, all_logits, all_labels, loss_breakdown = self.accelerator.unwrap_model(self.train_wrapper)(
                    **data_kwargs,
                    return_loss=True,
                    return_loss_breakdown=True,
                )
                batch_loss = self.accelerator.reduce(batch_loss, 'mean').item()
                batch_count += 1

                final_logits = self.accelerator.gather_for_metrics(all_logits[-1].transpose(1, 2).contiguous())
                gt_tokens = self.accelerator.gather_for_metrics(all_labels[-1].contiguous())

                final_logits = final_logits.detach().cpu().float()
                gt_tokens = gt_tokens.detach().cpu().long()
                pred_tokens = final_logits.argmax(-1)

                batch_token_count = int(gt_tokens.numel())
                loss_weighted_sum += float(batch_loss) * batch_token_count
                loss_token_count_total += batch_token_count

                for key, value in loss_breakdown.items():
                    if key.startswith('pred_q') and key.endswith('_ce'):
                        q_index = key[len('pred_q'):-3]
                        token_key = f'pred_q{q_index}_tokens'
                        q_token_count = int(loss_breakdown.get(token_key, batch_token_count))
                        q_name = f'{self.stage}_q{q_index}'
                        value_scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                        codebook_loss_weighted_sums[q_name] = (
                            codebook_loss_weighted_sums.get(q_name, 0.0)
                            + float(value_scalar) * q_token_count
                        )
                        codebook_token_totals[q_name] = (
                            codebook_token_totals.get(q_name, 0)
                            + q_token_count
                        )

                token_correct_total += float((pred_tokens == gt_tokens).sum().item())
                token_count_total += int(gt_tokens.numel())

                num_classes = int(final_logits.shape[-1])
                for k in self.val_top_k:
                    k_eff = min(int(k), num_classes)
                    topk_indices = final_logits.topk(k_eff, dim=-1).indices
                    topk_match = topk_indices.eq(gt_tokens.unsqueeze(-1)).any(dim=-1)
                    topk_correct_totals[k_eff] = topk_correct_totals.get(k_eff, 0.0) + float(topk_match.sum().item())
                    topk_count_totals[k_eff] = topk_count_totals.get(k_eff, 0) + int(topk_match.numel())

                if show_progress and (batch_count % 10 == 0):
                    eval_iterator.set_postfix({
                        'batches': batch_count,
                        'tokens': token_count_total,
                    }, refresh=False)

        if batch_count == 0 or token_count_total == 0:
            self.print(f'warning: {split_name} evaluation produced no non-empty batches')
            return None

        split_loss = (loss_weighted_sum / loss_token_count_total) if loss_token_count_total > 0 else None
        split_topk = {
            k: (topk_correct_totals[k] / max(1, topk_count_totals[k]))
            for k in sorted(topk_correct_totals.keys())
        }
        split_accuracy = split_topk.get(1, token_correct_total / token_count_total)
        split_codebook_logs = {
            f'{split_name}_codebook_ce/{q_name}': (codebook_loss_weighted_sums[q_name] / codebook_token_totals[q_name])
            for q_name in sorted(codebook_loss_weighted_sums.keys())
            if codebook_token_totals.get(q_name, 0) > 0
        }

        topk_summary = ', '.join(
            f'acc@{k} {split_topk[k]:.6f}' for k in sorted(split_topk.keys())
        )
        self.print(
            f'{split_name}: loss {split_loss}, {topk_summary}, batches={batch_count}'
        )

        return {
            'loss': split_loss,
            'accuracy': split_accuracy,
            'topk': split_topk,
            'codebook_logs': split_codebook_logs,
            'batches': batch_count,
            'tokens': token_count_total,
        }

    def evaluate_validation(self, save_json: bool = False, json_path: Optional[str] = None):
        if not exists(self.valid_dl):
            self.print('validation split is unavailable; skipping validation evaluation')
            return None

        metrics = self._evaluate_split_metrics(self.valid_dl, split_name='validation')
        if metrics is None:
            return None

        step = int(self.steps.item())
        log_payload = {
            'validation_eval_loss': metrics['loss'],
            'validation_eval_accuracy': metrics['accuracy'],
            'validation_eval_batches': metrics['batches'],
            'validation_eval_tokens': metrics['tokens'],
        }
        for k, value in metrics['topk'].items():
            log_payload[f'validation_eval_accuracy_top{k}'] = value
        for key, value in metrics['codebook_logs'].items():
            log_payload[key] = value

        self.accelerator.log(log_payload, step=step)

        payload = {
            'stage': self.stage,
            'step': step,
            'metrics': {
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'topk': {str(k): v for k, v in metrics['topk'].items()},
                'codebook': metrics['codebook_logs'],
                'batches': metrics['batches'],
                'tokens': metrics['tokens'],
            },
            'results_folder': str(self.results_folder),
            'timestamp_unix': int(time.time()),
        }

        if self.is_main and save_json:
            out_path = Path(json_path) if exists(json_path) else (self.results_folder / f'{self.stage}.validation_metrics.final.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            self.print(f'validation metrics json saved to {str(out_path)}')

        return payload

    def evaluate_test(self, save_json: bool = True, json_path: Optional[str] = None):
        if not exists(self.test_dl):
            self.print('test split is unavailable; skipping test evaluation')
            return None

        metrics = self._evaluate_split_metrics(self.test_dl, split_name='test')
        if metrics is None:
            return None

        step = int(self.steps.item())
        log_payload = {
            'test_loss': metrics['loss'],
            'test_accuracy': metrics['accuracy'],
            'test_batches': metrics['batches'],
            'test_tokens': metrics['tokens'],
        }
        for k, value in metrics['topk'].items():
            log_payload[f'test_accuracy_top{k}'] = value
        for key, value in metrics['codebook_logs'].items():
            log_payload[key] = value

        self.accelerator.log(log_payload, step=step)

        payload = {
            'stage': self.stage,
            'step': step,
            'metrics': {
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'topk': {str(k): v for k, v in metrics['topk'].items()},
                'codebook': metrics['codebook_logs'],
                'batches': metrics['batches'],
                'tokens': metrics['tokens'],
            },
            'results_folder': str(self.results_folder),
            'timestamp_unix': int(time.time()),
        }

        if self.is_main and save_json:
            out_path = Path(json_path) if exists(json_path) else (self.results_folder / f'{self.stage}.test_metrics.final.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            self.print(f'test metrics json saved to {str(out_path)}')

        return payload

    def train_step(self):
        device = self.device
        step_start_time = time.perf_counter()

        steps = int(self.steps.item())

        # Restore full training mode after validation toggles `train_wrapper.eval()`.
        # Wrapper-level `self.training` gates logic inside stage wrappers (e.g. mask_prob path).
        self.train_wrapper.train()

        # logs

        logs = {}

        # update vae (generator)

        for accum_i in range(self.grad_accum_every):
            is_last_accum = (accum_i == self.grad_accum_every - 1)
            no_sync_ctx = (
                contextlib.nullcontext()
                if (not self.is_distributed or is_last_accum)
                else self.accelerator.no_sync(self.train_wrapper)
            )
            with no_sync_ctx:
                non_empty_batch = False
                while non_empty_batch is False:
                    data_kwargs = dict(zip(self.ds_fields, next(self.dl_iter)))
                    if len(data_kwargs) == 0:
                        continue

                    non_empty_batch = True

                    loss, _, _, loss_breakdown = self.train_wrapper(
                        **data_kwargs,
                        return_loss=True,
                        return_loss_breakdown=True,
                    )

                    self.accelerator.backward(loss / self.grad_accum_every)

                    accum_log(logs, {'loss': loss.item() / self.grad_accum_every})
                    train_codebook_logs = {
                        f'train_codebook_ce/{self.stage}_q{key[len("pred_q"):-3]}': value / self.grad_accum_every
                        for key, value in loss_breakdown.items()
                        if key.startswith('pred_q') and key.endswith('_ce')
                    }
                    if len(train_codebook_logs) > 0:
                        accum_log(logs, train_codebook_logs)

        grad_norm = None
        if exists(self.max_grad_norm):
            grad_norm = self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)
            if torch.is_tensor(grad_norm):
                grad_norm = float(grad_norm.detach().item())
            elif grad_norm is not None:
                grad_norm = float(grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        if exists(self.scheduler):
            self.scheduler.step()

        lr = float(self.optim.param_groups[0]['lr'])
        step_time_sec = float(time.perf_counter() - step_start_time)

        # sample results every so often

        valid_loss = None
        valid_accuracy = None
        valid_topk = {}
        valid_codebook_logs = {}
        early_stop_triggered = False
        should_validate = False
        if self.val_every_n_steps > 0:
            should_validate = (
                ((steps + 1) % self.val_every_n_steps == 0)
                or (steps + 1 == self.num_train_steps)
            )
        elif self.save_results_every > 0:
            should_validate = not (steps % self.save_results_every)

        if should_validate:
            self.train_wrapper.eval()

            valid_loss_weighted_sum = 0.0
            valid_batch_count = 0
            valid_loss_token_count_total = 0
            token_count_total = 0
            token_correct_total = 0.0
            topk_correct_totals = {}
            topk_count_totals = {}
            valid_codebook_loss_weighted_sums = {}
            valid_codebook_token_totals = {}
            first_pred_tokens = None
            first_gt_tokens = None
            first_all_logits = None
            first_all_labels = None

            try:
                valid_total_batches = len(self.valid_dl)
            except TypeError:
                valid_total_batches = None

            show_valid_progress = bool(self.is_main)
            valid_iterator = tqdm(
                self.valid_dl,
                desc='validating',
                total=valid_total_batches,
                disable=not show_valid_progress,
                mininterval=5,
                leave=False,
            )

            with torch.no_grad():
                for valid_batch in valid_iterator:
                    data_kwargs = dict(zip(self.ds_fields, valid_batch))
                    if len(data_kwargs) == 0:
                        continue

                    batch_valid_loss, all_logits, all_labels, loss_breakdown = self.accelerator.unwrap_model(self.train_wrapper)(
                        **data_kwargs,
                        return_loss=True,
                        return_loss_breakdown=True,
                    )
                    batch_valid_loss = self.accelerator.reduce(batch_valid_loss, 'mean').item()
                    valid_batch_count += 1

                    final_logits = self.accelerator.gather_for_metrics(all_logits[-1].transpose(1, 2).contiguous())
                    gt_tokens = self.accelerator.gather_for_metrics(all_labels[-1].contiguous())

                    final_logits = final_logits.detach().cpu().float()
                    gt_tokens = gt_tokens.detach().cpu().long()
                    pred_tokens = final_logits.argmax(-1)

                    batch_token_count = int(gt_tokens.numel())
                    valid_loss_weighted_sum += float(batch_valid_loss) * batch_token_count
                    valid_loss_token_count_total += batch_token_count

                    for key, value in loss_breakdown.items():
                        if key.startswith('pred_q') and key.endswith('_ce'):
                            q_index = key[len('pred_q'):-3]
                            token_key = f'pred_q{q_index}_tokens'
                            q_token_count = int(loss_breakdown.get(token_key, batch_token_count))
                            q_name = f'{self.stage}_q{q_index}'
                            valid_codebook_loss_weighted_sums[q_name] = (
                                valid_codebook_loss_weighted_sums.get(q_name, 0.0)
                                + float(value) * q_token_count
                            )
                            valid_codebook_token_totals[q_name] = (
                                valid_codebook_token_totals.get(q_name, 0)
                                + q_token_count
                            )

                    token_correct_total += float((pred_tokens == gt_tokens).sum().item())
                    token_count_total += int(gt_tokens.numel())

                    num_classes = int(final_logits.shape[-1])
                    for k in self.val_top_k:
                        k_eff = min(int(k), num_classes)
                        topk_indices = final_logits.topk(k_eff, dim=-1).indices
                        topk_match = topk_indices.eq(gt_tokens.unsqueeze(-1)).any(dim=-1)
                        topk_correct_totals[k_eff] = topk_correct_totals.get(k_eff, 0.0) + float(topk_match.sum().item())
                        topk_count_totals[k_eff] = topk_count_totals.get(k_eff, 0) + int(topk_match.numel())

                    if show_valid_progress and (valid_batch_count % 10 == 0):
                        valid_iterator.set_postfix({
                            'batches': valid_batch_count,
                            'tokens': token_count_total,
                        }, refresh=False)

                    if first_pred_tokens is None:
                        first_pred_tokens = pred_tokens
                        first_gt_tokens = gt_tokens
                        first_all_logits = all_logits
                        first_all_labels = all_labels

            if valid_batch_count > 0 and token_count_total > 0:
                if valid_loss_token_count_total > 0:
                    valid_loss = valid_loss_weighted_sum / valid_loss_token_count_total
                else:
                    valid_loss = None
                valid_topk = {
                    k: (topk_correct_totals[k] / max(1, topk_count_totals[k]))
                    for k in sorted(topk_correct_totals.keys())
                }
                valid_codebook_logs = {
                    f'valid_codebook_ce/{q_name}': (valid_codebook_loss_weighted_sums[q_name] / valid_codebook_token_totals[q_name])
                    for q_name in sorted(valid_codebook_loss_weighted_sums.keys())
                    if valid_codebook_token_totals.get(q_name, 0) > 0
                }
                valid_accuracy = valid_topk.get(1, token_correct_total / token_count_total)

                topk_summary = ', '.join(
                    f'acc@{k} {valid_topk[k]:.6f}' for k in sorted(valid_topk.keys())
                )
                self.print(
                    f'{steps}: valid loss {valid_loss}, {topk_summary}, '
                    f'batches={valid_batch_count}'
                )
            else:
                self.print(f'{steps}: warning: validation produced no non-empty batches')

            if (
                self.is_main and
                self.save_predicted_tokens and
                first_pred_tokens is not None and
                first_gt_tokens is not None
            ):
                # save one representative validation batch for qualitative inspection
                assert exists(self.tokens_folder)

                interleave = torch.empty(
                    (first_pred_tokens.shape[0] + first_gt_tokens.shape[0], first_pred_tokens.shape[1]),
                    dtype=first_pred_tokens.dtype
                )
                interleave[0::2] = first_pred_tokens
                interleave[1::2] = first_gt_tokens

                np.savetxt(
                    str(self.tokens_folder / f'{self.stage}.tokens.{steps}.txt'),
                    interleave,
                    fmt='%-6s',
                    header='predicted and ground truth tokens from the validation set. row 0%2 is predicted, 1%2 is ground truth\n '
                )

            if (
                self.is_main and
                self.save_reconstructed_wave and
                (self.stage == 'coarse' or self.stage == 'fine') and
                first_all_logits is not None and
                first_all_labels is not None
            ):
                # For coarse/fine stages, reconstruct teacher-forced wave from one representative validation batch.
                assert exists(self.neural_codec)
                assert exists(self.waves_folder)

                pred_tokens = first_all_logits[-1].detach().argmax(1)[:, :-1]
                pred_tokens[pred_tokens == self.transformer.eos_ids[-1]] = 0

                num_quantizers = self.transformer.token_sequences[-1].num_quantizers
                pred_tokens = rearrange(pred_tokens, 'b (n q) -> b n q', q=num_quantizers)

                if self.stage == 'fine':
                    coarse_tokens = first_all_labels[-2][:, :-1]
                    coarse_quantizers = self.transformer.token_sequences[-2].num_quantizers
                    coarse_tokens = rearrange(coarse_tokens, 'b (n q) -> b n q', q=coarse_quantizers)
                    pred_tokens = torch.cat((coarse_tokens, pred_tokens), dim=-1)

                waves = self.neural_codec.decode_from_codebook_indices(pred_tokens)
                waves = waves.cpu()

                file_paths = []

                max_files_to_save = 4
                for i, wave in enumerate(waves):
                    if i < max_files_to_save:
                        file_path = str(self.waves_folder / f'{self.stage}.reconstructed_wave_{i}.{steps}.wav')
                        torchaudio.save(file_path, wave, self.neural_codec.sample_rate)
                        file_paths.append(file_path)
                    else:
                        break

                if 'wandb' in self.log_with and exists(wandb):
                    audios = [wandb.Audio(file_path, caption=f'reconstructed wave at {steps} steps') for file_path in file_paths]
                    wandb.log({'reconstructed_wave': audios})

            if self.is_main and self.save_top_k_checkpoints > 0 and valid_loss is not None:
                if self.val_early_stop_metric == 'accuracy':
                    checkpoint_metric = float(valid_accuracy if valid_accuracy is not None else float('-inf'))
                else:
                    checkpoint_metric = float(valid_loss)

                model_path = str(self.results_folder / f'{self.stage}.best.transformer.{steps}.pt')
                optim_path = str(self.results_folder / f'{self.stage}.best.optimizer.{steps}.pt')
                scheduler_path = str(self.results_folder / f'{self.stage}.best.scheduler.{steps}.pt')

                self.save(model_path, optim_path, scheduler_path)

                self.best_checkpoints.append({
                    'step': steps,
                    'metric': checkpoint_metric,
                    'metric_name': self.val_early_stop_metric,
                    'model_path': model_path,
                    'optim_path': optim_path,
                    'scheduler_path': scheduler_path if exists(self.scheduler) else None,
                })
                self.best_checkpoints.sort(
                    key=lambda x: x['metric'],
                    reverse=(self.val_early_stop_metric == 'accuracy')
                )

                while len(self.best_checkpoints) > self.save_top_k_checkpoints:
                    to_remove = self.best_checkpoints.pop(-1)
                    for path_key in ('model_path', 'optim_path', 'scheduler_path'):
                        path = to_remove.get(path_key)
                        if exists(path) and Path(path).exists():
                            Path(path).unlink()
                    self.print(
                        f"{steps}: pruned checkpoint (step={to_remove['step']}, "
                        f"{to_remove.get('metric_name', 'metric')}={to_remove['metric']:.6f})"
                    )
                self.print(
                    f"{steps}: saved best checkpoint candidate "
                    f"({self.val_early_stop_metric}={checkpoint_metric:.6f}, kept={len(self.best_checkpoints)})"
                )

            if self.early_stop_patience_validations > 0 and valid_loss is not None:
                if self.val_early_stop_metric == 'accuracy':
                    current_metric = float(valid_accuracy if valid_accuracy is not None else float('-inf'))
                    improved = (
                        self.early_stop_best_metric is None
                        or current_metric > self.early_stop_best_metric
                    )
                else:
                    current_metric = float(valid_loss)
                    improved = (
                        self.early_stop_best_metric is None
                        or current_metric < self.early_stop_best_metric
                    )

                if improved:
                    self.early_stop_best_metric = current_metric
                    self.early_stop_bad_validations = 0
                    self.early_stop_best_step = steps
                else:
                    self.early_stop_bad_validations += 1

                if self.early_stop_bad_validations >= self.early_stop_patience_validations:
                    early_stop_triggered = True
                    self.print(
                        f'{steps}: early-stop triggered '
                        f'(metric={self.val_early_stop_metric}, '
                        f'best_step={self.early_stop_best_step}, '
                        f'bad_validations={self.early_stop_bad_validations})'
                    )

            stop_tensor = torch.tensor(
                [1 if early_stop_triggered else 0],
                device=self.device,
                dtype=torch.int64,
            )
            global_stop = int(self.accelerator.reduce(stop_tensor, 'max').item()) > 0
            if global_stop:
                self.should_stop_early = True

        progress_pct = 100.0 * (steps + 1) / max(1, self.num_train_steps)
        epoch_float = (steps + 1) / max(1, self.steps_per_epoch)

        log_payload = {
            "train_loss": logs['loss'],
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "lr": lr,
            "step_time_sec": step_time_sec,
            "progress_pct": progress_pct,
            "epoch": epoch_float,
        }
        for k, value in valid_topk.items():
            log_payload[f"valid_accuracy_top{k}"] = value
        for key, value in logs.items():
            if key.startswith('train_codebook_ce/'):
                log_payload[key] = value
        for key, value in valid_codebook_logs.items():
            log_payload[key] = value
        if grad_norm is not None:
            log_payload["grad_norm"] = grad_norm

        self.accelerator.log(log_payload, step=steps)

        if (
            self.is_main and
            self.alert_every > 0 and
            self.log_with is not None and 'wandb' in self.log_with and
            exists(wandb) and
            valid_loss is not None
        ):
            wandb.alert(
                title=f"{self.stage} step {steps}/{self.num_train_steps}",
                text=(
                    f"train_loss={logs['loss']:.4f} "
                    f"valid_loss={valid_loss:.4f}"
                ),
                level=wandb.AlertLevel.INFO,
            )

        # save model every so often

        if self.is_main and self.save_model_every > 0 and not (steps % self.save_model_every):

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

            model_path = str(self.results_folder / f'{self.stage}.transformer.{steps}.pt')
            optim_path = str(self.results_folder / f'{self.stage}.optimizer.{steps}.pt')
            scheduler_path = str(self.results_folder / f'{self.stage}.scheduler.{steps}.pt')

            self.save(model_path, optim_path, scheduler_path)

            # save audio conditioner (clap) rvq checkpoint
            if exists(self.audio_conditioner) and self.audio_conditioner.learn_rvq:
                rvq_state_dict = self.audio_conditioner.rq.state_dict()
                torch.save(rvq_state_dict, str(self.results_folder / f'{self.stage}.conditioner_rvq.{steps}.pt'))

        self.steps += 1
        logs['lr'] = lr
        logs['valid_loss'] = valid_loss
        logs['valid_accuracy'] = valid_accuracy
        for k, value in valid_topk.items():
            logs[f'valid_accuracy_top{k}'] = value
        logs['step_time_sec'] = step_time_sec
        logs['progress_pct'] = progress_pct
        logs['epoch'] = epoch_float
        logs['early_stop'] = self.should_stop_early
        if grad_norm is not None:
            logs['grad_norm'] = grad_norm
        return logs

    def train(self, log_fn=noop):
        start_step = int(self.steps.item())
        disable_pbar = not self.is_local_main
        pbar = tqdm(
            total=self.num_train_steps,
            initial=start_step,
            desc=f'{self.stage} train',
            dynamic_ncols=True,
            disable=disable_pbar,
        )

        while self.steps < self.num_train_steps and not self.should_stop_early:
            logs = self.train_step()
            if not disable_pbar:
                pbar.update(1)
                postfix = {
                    'loss': f"{logs.get('loss', float('nan')):.4f}",
                    'lr': f"{logs.get('lr', float('nan')):.2e}",
                    't': f"{logs.get('step_time_sec', float('nan')):.2f}s",
                }
                valid_loss = logs.get('valid_loss', None)
                if valid_loss is not None:
                    postfix['v_loss'] = f"{valid_loss:.4f}"
                pbar.set_postfix(postfix)

            log_fn(logs)

        if not disable_pbar:
            pbar.close()

        # Always persist a final "last" checkpoint so interrupted/finished runs can be resumed.
        if self.is_main:
            final_step = int(self.steps.item())
            model_path = str(self.results_folder / f'{self.stage}.transformer.{final_step}.pt')
            optim_path = str(self.results_folder / f'{self.stage}.optimizer.{final_step}.pt')
            scheduler_path = str(self.results_folder / f'{self.stage}.scheduler.{final_step}.pt')
            self.save(model_path, optim_path, scheduler_path)
            self.print(f'{final_step}: saved last checkpoint')

        if self.is_main and self.alert_every > 0 and self.log_with is not None and 'wandb' in self.log_with and exists(wandb):
            status_text = (
                f"Early-stopped at {int(self.steps.item())} steps."
                if self.should_stop_early else
                f"Completed {int(self.steps.item())} steps."
            )
            wandb.alert(
                title=f"{self.stage} training complete",
                text=status_text,
                level=wandb.AlertLevel.INFO,
            )
        if self.should_stop_early:
            self.print('training complete (early stopped)')
        else:
            self.print('training complete')


@beartype_jit
class ClapRVQTrainer(nn.Module):
    """
    Learn the residual vector quantizer to turn CLAP embeddings into discrete tokens.
    """

    def __init__(
        self,
        *,
        num_train_steps,
        batch_size,
        accumulate_batches: Optional[int] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        wd=0.,
        max_grad_norm=0.5,
        data_max_length_seconds: Union[float, int] = 10,
        input_audio_format: Literal['audio', 'npy', 'chunked'] = 'audio',
        npy_sample_hz: int = 48000,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        valid_frac=0.05,
        random_split_seed=42,
        save_results_every=100,
        save_model_every=1000,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.log_with = accelerate_kwargs['log_with'] if 'log_with' in accelerate_kwargs else None

        self.audio_conditioner = audio_conditioner
        self.ds = dataset
        self.num_train_steps = num_train_steps
        self.accumulate_batches = accumulate_batches
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            if input_audio_format == 'npy':
                dataset_cls = NpySoundDataset
            elif input_audio_format == 'chunked':
                dataset_cls = ChunkedSoundDataset
            else:
                dataset_cls = SoundDataset

            if input_audio_format == 'chunked':
                dataset_kwargs = dict(
                    chunk_seconds=data_max_length_seconds,
                    target_sample_hz=audio_conditioner.sample_rate,
                    seq_len_multiple_of=None,
                    ignore_files=default(ignore_files, []),
                    ignore_load_errors=ignore_load_errors,
                )
            else:
                dataset_kwargs = dict(
                    max_length_seconds=data_max_length_seconds,
                    target_sample_hz=audio_conditioner.sample_rate,
                    seq_len_multiple_of=None,
                    ignore_files=default(ignore_files, []),
                    ignore_load_errors=ignore_load_errors,
                )
                if input_audio_format == 'npy':
                    dataset_kwargs['source_sample_hz'] = npy_sample_hz

            self.ds = dataset_cls(
                folder,
                **dataset_kwargs,
            )

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader
        dataloader_num_workers = max(0, int(num_workers))
        dataloader_prefetch_factor = max(1, int(prefetch_factor))
        dataloader_persistent_workers = bool(persistent_workers) and dataloader_num_workers > 0
        dataloader_pin_memory = bool(pin_memory)

        train_dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
        valid_dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
        if dataloader_num_workers > 0:
            train_dl_kwargs['prefetch_factor'] = dataloader_prefetch_factor
            train_dl_kwargs['persistent_workers'] = dataloader_persistent_workers
            valid_dl_kwargs['prefetch_factor'] = dataloader_prefetch_factor
            valid_dl_kwargs['persistent_workers'] = dataloader_persistent_workers

        self.dl = get_dataloader(self.ds, **train_dl_kwargs)

        self.valid_dl = get_dataloader(self.valid_ds, **valid_dl_kwargs)

        (
            self.audio_conditioner,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.audio_conditioner,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.num_workers = dataloader_num_workers
        self.prefetch_factor = dataloader_prefetch_factor
        self.persistent_workers = dataloader_persistent_workers
        self.pin_memory = dataloader_pin_memory

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        if len([*self.results_folder.glob('**/*')]) > 0:
            self.print('results folder is not empty; keeping existing files (no interactive prompt)')
        self.print(
            f'dataloader config: num_workers={self.num_workers}, '
            f'prefetch_factor={self.prefetch_factor if self.num_workers > 0 else "n/a"}, '
            f'persistent_workers={self.persistent_workers}, pin_memory={self.pin_memory}'
        )

        hps = {"num_train_steps": num_train_steps, "batch_size": batch_size, "accumulate_batches": accumulate_batches}

        if 'tensorboard' in self.log_with:
            self.accelerator.init_trackers(f"clap_rvq_{int(time.time() * 1000)}", config=hps)
        else:
            self.accelerator.init_trackers(f"clap_rvq", config=hps)

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        steps = int(self.steps.item())

        self.audio_conditioner.learn_rvq = True

        iters = default(self.accumulate_batches, 1)
        iters = math.ceil(iters / self.accelerator.num_processes)

        embeds = []
        for _ in tqdm(range(iters), desc='accumulating batches'):
            raw_wave_for_clap = next(self.dl_iter)[0]
            embed = self.audio_conditioner.forward(audio_input=raw_wave_for_clap.to(self.device), return_embedding=True)
            embeds.append(embed)

        embeds = torch.cat(embeds, dim=0)
        embeds = self.accelerator.gather_for_metrics(embeds)

        if self.is_main:
            loss = self.audio_conditioner.quantize(embeds, return_rvq_loss=True)

            self.print(f'loss: {loss}')

            # sample results every so often
            valid_loss = None
            if not (steps % self.save_results_every):
                raw_wave_for_clap = next(self.valid_dl_iter)[0]

                with torch.no_grad():
                    self.audio_conditioner.learn_rvq = False
                    valid_loss = self.audio_conditioner.forward(audio_input=raw_wave_for_clap.to(self.device), return_rvq_loss=True)

                self.print(f'{steps}: valid loss {valid_loss}')

            self.accelerator.log({
                "train_loss": loss,
                "valid_loss": valid_loss
            }, step=steps)

            # save model every so often

            if not (steps % self.save_model_every):
                # save audio conditioner (clap) rvq checkpoint
                rvq_state_dict = self.accelerator.unwrap_model(self.audio_conditioner).rq.state_dict()
                torch.save(rvq_state_dict, str(self.results_folder / f'clap.rvq.{steps}.pt'))

                self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1

    def train(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')


@beartype_jit
class HfHubertKmeansTrainer(nn.Module):
    """
    Trainer for kmeans part of HfHubertWithKmeans. Consists of two parts: 1) extracting Hubert features and 2) training kmeans model on these features.
    """

    def __init__(
        self,
        *,
        feature_extraction_num_steps: int,
        feature_extraction_batch_size: int,
        hubert_kmeans: HfHubertWithKmeans,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        data_max_length_seconds: Union[float, int] = 1,
        input_audio_format: Literal['audio', 'npy'] = 'audio',
        npy_sample_hz: int = 48000,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.ds = dataset
        self.feature_extraction_num_steps = feature_extraction_num_steps
        self.feature_extraction_batch_size = feature_extraction_batch_size
        self.hubert_kmeans = hubert_kmeans
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            dataset_cls = NpySoundDataset if input_audio_format == 'npy' else SoundDataset
            dataset_kwargs = dict(
                max_length_seconds=data_max_length_seconds,
                normalize=True,
                target_sample_hz=hubert_kmeans.target_sample_hz,
                seq_len_multiple_of=hubert_kmeans.seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )
            if input_audio_format == 'npy':
                dataset_kwargs['source_sample_hz'] = npy_sample_hz

            self.ds = dataset_cls(
                folder,
                **dataset_kwargs
            )
        self.print(
            f'training on {feature_extraction_num_steps * feature_extraction_batch_size} out of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=feature_extraction_batch_size, shuffle=True)

        (
            self.hubert_kmeans,
            self.dl
        ) = self.accelerator.prepare(
            self.hubert_kmeans,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        if len([*self.results_folder.glob('**/*')]) > 0:
            self.print('results folder is not empty; keeping existing files (no interactive prompt)')

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def extract_hubert_features(self):

        raw_wave = next(self.dl_iter)[0]

        embed = self.hubert_kmeans.forward(wav_input=raw_wave.to(self.device), return_embed=True)

        # get features
        embed = rearrange(embed, 'b t f -> (b t) f')
        embed = self.accelerator.gather_for_metrics(embed)
        embed = embed.detach().cpu().numpy()

        return embed

    def train(self, log_fn=noop, seed=0):

        self.print('step 1: extracting features. must wait for this to complete before training kmeans.')
        features = []
        num_steps = math.ceil(self.feature_extraction_num_steps / self.accelerator.num_processes)
        while self.steps < num_steps:
            self.print(f'{int(self.steps.item())} / {num_steps} steps')
            features.append(self.extract_hubert_features())
            self.steps += 1

        features = np.concatenate(features, axis=0)

        features = features[~np.any(np.isnan(features), axis=-1)]

        self.print('step 2: training kmeans')
        if self.is_main:
            learn_kmeans(
                features,
                seed,
                str(self.results_folder / 'kmeans.joblib'),
                n_clusters=self.accelerator.unwrap_model(self.hubert_kmeans).codebook_size)

        self.print('training complete')
