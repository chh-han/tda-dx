"""
Shared custom-backward projected-gradient helpers.
"""

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_projected_grads_custom_backward(
    *,
    stage_wrapper: nn.Module,
    model_parameters: Iterable[nn.Parameter],
    loader: DataLoader,
    ds_fields: Tuple[str, ...],
    device: str,
    proj_dim: int,
    seed: int,
    desc: str,
    normalize_grads: bool,
    ensemble_id: int = 0,
) -> torch.Tensor:
    """Compute projected per-sample gradients exactly via standard backward()."""
    from dattri.func.projection import random_project

    stage_wrapper.eval()
    params: List[nn.Parameter] = list(model_parameters)

    def _zero_grads() -> None:
        for p in params:
            p.grad = None

    def _flatten_grads() -> torch.Tensor:
        grads = []
        for p in params:
            if p.requires_grad:
                if p.grad is not None:
                    grads.append(
                        p.grad.detach().to(device=device, dtype=torch.float32).flatten()
                    )
                else:
                    grads.append(
                        torch.zeros(p.numel(), dtype=torch.float32, device=device)
                    )
        g = torch.cat(grads).unsqueeze(0)
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    proj_fn = None
    proj_list: List[torch.Tensor] = []
    num_samples = sum(batch[0].shape[0] for batch in loader)
    pbar = tqdm(total=num_samples, desc=desc, unit="sample", leave=True)

    for batch in loader:
        batch_size = batch[0].shape[0]
        batch_g_list: List[torch.Tensor] = []
        for s in range(batch_size):
            sample = tuple(t[s : s + 1] for t in batch)
            _zero_grads()
            data_kwargs = {k: sample[i].to(device) for i, k in enumerate(ds_fields)}
            loss, _, _ = stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()
            del loss

            batch_g_list.append(_flatten_grads())
            _zero_grads()

        g_batch = torch.cat(batch_g_list, dim=0)  # [B, D]
        if proj_fn is None:
            proj_fn = random_project(
                feature=g_batch,
                feature_batch_size=batch_size,
                proj_dim=proj_dim,
                proj_max_batch_size=16,
                proj_seed=seed,
                proj_type="sjlt" if "cuda" in str(device) else "normal",
                device=device,
            )

        g_projected = proj_fn(g_batch, ensemble_id=ensemble_id)  # [B, proj_dim]
        if normalize_grads:
            g_projected = g_projected / (g_projected.norm(dim=1, keepdim=True) + 1e-8)

        for g in g_projected:
            proj_list.append(g.detach().cpu())
        pbar.update(batch_size)

        del g_batch, g_projected, batch_g_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    return torch.stack(proj_list)
