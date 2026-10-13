"""
attributor/methods/grad_cos.py

Memory-efficient grad_cos attribution using standard loss.backward()
instead of functorch vmap(grad(fn)).

Mathematically identical to:
    TracInAttributor(normalized_grad=True, proj_dim=k).attribute(train, test)

Algorithm
---------
score(train_i, test_j) = cosine_similarity(proj(∇L_i), proj(∇L_j))

Where proj is a fixed random projection R^d → R^k provided by dattri:
  - CUDA: CudaProjector with SJLT (sparse JL transform, c=1 nonzero/col)
  - CPU:  BasicProjector with Gaussian projection

Why this is memory-efficient
----------------------------
- loss.backward() uses PyTorch's standard autograd engine which benefits
  from memory-sharing, in-place ops, and incremental graph freeing.
- functorch vmap(grad(fn)) materialises all activations simultaneously
  and has additional overhead, causing OOM for large models with long
  sequences (e.g. fine stage: 1200+ tokens, 8 codebooks).

Projection consistency
----------------------
The projector is initialised once with the same seed, so train and test
gradients are always projected by the same matrix R — cosine similarities
are valid.
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tda_dx.attribution.methods.base import AttributionMethod


# ─────────────────────────────────────────────────────────────────────────────
# Main attribution function
# ─────────────────────────────────────────────────────────────────────────────

def _compute_proj_grads(
    stage_wrapper: nn.Module,
    model_parameters: Iterable[nn.Parameter],
    loader: torch.utils.data.DataLoader,
    ds_fields: Tuple[str, ...],
    device: str,
    proj_dim: int,
    seed: int,
    chunk_size: int,
    desc: str,
) -> torch.Tensor:
    """Compute L2-normalized projected gradients for all samples in loader.

    Uses dattri's random_project() factory:
      - CUDA: CudaProjector with SJLT
      - CPU:  BasicProjector with Gaussian

    Returns
    -------
    proj_mat : torch.Tensor, shape (N, proj_dim), float32, on CPU
    """
    from dattri.func.projection import random_project

    stage_wrapper.eval()
    params: List[nn.Parameter] = list(model_parameters)

    def _zero_grads() -> None:
        for p in params:
            p.grad = None

    def _flatten_grads() -> torch.Tensor:
        """Flatten current .grad tensors into (1, feature_dim) on device."""
        grads = []
        for p in params:
            if p.requires_grad:
                if p.grad is not None:
                    grads.append(p.grad.detach().to(device=device, dtype=torch.float32).flatten())
                else:
                    grads.append(torch.zeros(p.numel(), dtype=torch.float32, device=device))
        g = torch.cat(grads).unsqueeze(0)  # (1, feature_dim)
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    proj_fn = None
    proj_list: List[torch.Tensor] = []
    num_samples = sum(batch[0].shape[0] for batch in loader)
    pbar = tqdm(total=num_samples, desc=desc, unit='sample', leave=True)

    for batch in loader:
        batch_size = batch[0].shape[0]
        batch_g_list: List[torch.Tensor] = []
        for s in range(batch_size):
            sample = tuple(t[s:s+1] for t in batch)
            _zero_grads()
            data_kwargs = {k: sample[i].to(device) for i, k in enumerate(ds_fields)}
            loss, _, _ = stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()
            del loss
            batch_g_list.append(_flatten_grads())
            _zero_grads()

        g_batch = torch.cat(batch_g_list, dim=0)  # (B, feature_dim)
        if proj_fn is None:
            proj_fn = random_project(
                feature=g_batch,
                feature_batch_size=batch_size,
                proj_dim=proj_dim,
                proj_max_batch_size=16,
                proj_seed=seed,
                proj_type='sjlt' if 'cuda' in str(device) else 'normal',
                device=device,
            )

        g_projected = proj_fn(g_batch, ensemble_id=0)  # (B, proj_dim)
        g_norm = g_projected / (g_projected.norm(dim=1, keepdim=True) + 1e-8)
        for g in g_norm:
            proj_list.append(g.cpu())
        pbar.update(batch_size)

        del g_batch, g_projected, g_norm, batch_g_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    return torch.stack(proj_list)  # (N, proj_dim)


def compute_train_proj_grads(
    stage_wrapper: nn.Module,
    model_parameters: Iterable[nn.Parameter],
    train_loader: torch.utils.data.DataLoader,
    ds_fields: Tuple[str, ...],
    device: str,
    proj_dim: int = 512,
    seed: int = 42,
    chunk_size: int = 50_000,
    print_fn: Callable = print,
) -> torch.Tensor:
    """Compute and return L2-normalized projected train gradients.

    Returns
    -------
    train_proj_mat : torch.Tensor, shape (num_train, proj_dim), float32, on CPU
    """
    print_fn('grad_cos: caching train projected gradients...')
    t0 = time.time()
    result = _compute_proj_grads(
        stage_wrapper, model_parameters, train_loader, ds_fields,
        device, proj_dim, seed, chunk_size, desc='train proj grads',
    )
    print_fn(f'grad_cos: train proj done: {list(result.shape)} in {time.time() - t0:.0f}s')
    return result


def compute_test_proj_grads(
    stage_wrapper: nn.Module,
    model_parameters: Iterable[nn.Parameter],
    test_loader: torch.utils.data.DataLoader,
    ds_fields: Tuple[str, ...],
    device: str,
    proj_dim: int = 512,
    seed: int = 42,
    chunk_size: int = 50_000,
    print_fn: Callable = print,
) -> torch.Tensor:
    """Compute and return L2-normalized projected test gradients.

    Returns
    -------
    test_proj_mat : torch.Tensor, shape (num_test, proj_dim), float32, on CPU
    """
    print_fn('grad_cos: caching test projected gradients...')
    t0 = time.time()
    result = _compute_proj_grads(
        stage_wrapper, model_parameters, test_loader, ds_fields,
        device, proj_dim, seed, chunk_size, desc='test proj grads',
    )
    print_fn(f'grad_cos: test proj done: {list(result.shape)} in {time.time() - t0:.0f}s')
    return result


def compute_grad_cos_from_projs(
    train_proj_mat: torch.Tensor,
    test_proj_mat: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity scores from pre-normalized projected gradients.

    Parameters
    ----------
    train_proj_mat : (num_train, proj_dim) L2-normalized
    test_proj_mat  : (num_test,  proj_dim) L2-normalized

    Returns
    -------
    scores : (num_train, num_test)
    """
    return train_proj_mat @ test_proj_mat.T


def compute_grad_cos_scores(
    stage_wrapper: nn.Module,
    model_parameters: Iterable[nn.Parameter],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    ds_fields: Tuple[str, ...],
    device: str,
    proj_dim: int = 512,
    seed: int = 42,
    chunk_size: int = 50_000,
    print_fn: Callable = print,
) -> torch.Tensor:
    """Compute grad_cos attribution scores via standard loss.backward().

    Equivalent to:
        TracInAttributor(normalized_grad=True).attribute(train_loader, test_loader)

    score[i, j] = cosine_similarity(proj(∇L_train_i), proj(∇L_test_j))

    Parameters
    ----------
    stage_wrapper : nn.Module
        SemanticStage / CoarseStage / FineStage.
        Called as: loss, _, _ = stage_wrapper(**data_kwargs, return_loss=True)
    model_parameters : iterable of nn.Parameter
        E.g. inner_transformer.parameters(). Materialised once; order must
        be consistent with the ChunkedGradProjector calls.
    train_loader : DataLoader
        Training samples. batch_size=1 recommended.
    test_loader : DataLoader
        Test (generated) samples.
    ds_fields : tuple of str
        Field names matching the stage, e.g.
        ('clap_token_ids', 'semantic_token_ids').
    device : str
    proj_dim : int
        Random projection dimension k.
    seed : int
        Determines the random projection matrix. Same seed → same matrix
        for train and test.
    chunk_size : int
        Parameter elements per chunk in projection. Trade speed vs memory.
    print_fn : callable

    Returns
    -------
    scores : torch.Tensor, shape (num_train, num_test), float32
        score[i, j] = cosine similarity of projected gradients.
    """
    from dattri.func.projection import random_project

    stage_wrapper.eval()

    # Materialise parameter list once so order is fixed.
    params: List[nn.Parameter] = list(model_parameters)

    proj_fn = None  # lazy init on first backward pass

    def _zero_grads() -> None:
        for p in params:
            p.grad = None  # faster than optimizer.zero_grad()

    def _flatten_grads() -> torch.Tensor:
        """Flatten current .grad tensors into (1, feature_dim) on device."""
        grads = []
        for p in params:
            if p.requires_grad:
                if p.grad is not None:
                    grads.append(p.grad.detach().to(device=device, dtype=torch.float32).flatten())
                else:
                    grads.append(torch.zeros(p.numel(), dtype=torch.float32, device=device))
        g = torch.cat(grads).unsqueeze(0)  # (1, feature_dim)
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    def _batch_project_normalize(batch_g: torch.Tensor) -> torch.Tensor:
        """Project a batch of gradients and L2-normalize → (B, proj_dim)."""
        nonlocal proj_fn
        if proj_fn is None:
            proj_fn = random_project(
                feature=batch_g,
                feature_batch_size=batch_g.shape[0],
                proj_dim=proj_dim,
                proj_max_batch_size=16,
                proj_seed=seed,
                proj_type='sjlt' if 'cuda' in str(device) else 'normal',
                device=device,
            )
        g = proj_fn(batch_g, ensemble_id=0)  # (B, proj_dim)
        return g / (g.norm(dim=1, keepdim=True) + 1e-8)

    # ── Phase 1: cache projected & normalised test gradients ─────────────────
    print_fn('grad_cos (custom backward): caching test gradients...')
    t0 = time.time()
    test_proj: List[torch.Tensor] = []

    num_test_samples = sum(batch[0].shape[0] for batch in test_loader)
    pbar_test = tqdm(total=num_test_samples, desc='caching test grads', unit='sample', leave=True)

    for batch in test_loader:
        batch_size = batch[0].shape[0]
        batch_g_list: List[torch.Tensor] = []
        for s in range(batch_size):
            sample = tuple(t[s:s+1] for t in batch)
            _zero_grads()
            data_kwargs = {k: sample[i].to(device) for i, k in enumerate(ds_fields)}
            loss, _, _ = stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()
            del loss
            batch_g_list.append(_flatten_grads())
            _zero_grads()

        g_batch = torch.cat(batch_g_list, dim=0)  # (B, feature_dim)
        g_norm = _batch_project_normalize(g_batch)  # (B, proj_dim)
        for g in g_norm:
            test_proj.append(g.cpu())
        pbar_test.update(batch_size)

        del g_batch, g_norm, batch_g_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar_test.close()

    # Stack and move to device once (400 × 512 × 4 bytes ≈ 0.8 MB — negligible)
    test_proj_mat = torch.stack(test_proj).to(device)  # (num_test, proj_dim)
    num_test = test_proj_mat.shape[0]
    print_fn(
        f'grad_cos (custom backward): cached {num_test} test gradients '
        f'in {time.time()-t0:.0f}s'
    )

    # ── Phase 2: train gradients → cosine similarities ───────────────────────
    print_fn('grad_cos (custom backward): computing train gradients...')
    t1 = time.time()
    score_rows: List[torch.Tensor] = []

    num_train_samples = sum(batch[0].shape[0] for batch in train_loader)
    pbar_train = tqdm(total=num_train_samples, desc='train grads', unit='sample', leave=True)

    for batch in train_loader:
        batch_size = batch[0].shape[0]
        batch_g_list = []
        for s in range(batch_size):
            sample = tuple(t[s:s+1] for t in batch)
            _zero_grads()
            data_kwargs = {k: sample[i].to(device) for i, k in enumerate(ds_fields)}
            loss, _, _ = stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()
            del loss
            batch_g_list.append(_flatten_grads())
            _zero_grads()

        g_batch = torch.cat(batch_g_list, dim=0)  # (B, feature_dim)
        g_norm = _batch_project_normalize(g_batch)  # (B, proj_dim)

        # (B, proj_dim) @ (proj_dim, num_test) = (B, num_test)
        rows = (g_norm @ test_proj_mat.t()).cpu()
        for row in rows:
            score_rows.append(row)
        pbar_train.update(batch_size)

        del g_batch, g_norm, batch_g_list, rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar_train.close()

    scores = torch.stack(score_rows)  # (num_train, num_test)
    print_fn(
        f'grad_cos (custom backward): done. '
        f'scores={list(scores.shape)} in {time.time()-t1:.0f}s'
    )
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# GradCosMethod: AttributionMethod wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GradCosMethod(AttributionMethod):
    """Attribution method using gradient cosine similarity (grad_cos).

    Wraps the custom-backward grad_cos implementation from grad_cos.py.
    Supports cache_shard / merge_attribute / full modes.
    """

    def __init__(self):
        self.stage_wrapper = None
        self.params = None
        self.ds_fields = None
        self.device = None
        self.args = None
        self._cached_self_sanity = None

    def setup(
        self,
        stage_wrapper: torch.nn.Module,
        params: List[torch.nn.Parameter],
        ds_fields: Tuple[str, ...],
        device: str,
        args,
    ) -> None:
        """Initialize method-specific state. Called once after model is loaded."""
        self.stage_wrapper = stage_wrapper
        self.params = params
        self.ds_fields = ds_fields
        self.device = device
        self.args = args
        self._cached_self_sanity = None

    def cache_shard(
        self,
        shard_loader: DataLoader,
        shard_positions: List[int],
        results_folder: Path,
        shard_id: int,
        num_shards: int,
    ) -> None:
        """Compute and save normalized projected train gradients for this shard."""
        args = self.args
        print(f'[SHARD {shard_id}/{num_shards}] '
              f'Samples: {len(shard_positions)} (of ??? total)')

        cache_start = time.time()
        train_proj = compute_train_proj_grads(
            stage_wrapper=self.stage_wrapper,
            model_parameters=iter(self.params),
            train_loader=shard_loader,
            ds_fields=self.ds_fields,
            device=self.device,
            proj_dim=args.proj_dim,
            seed=args.seed,
            chunk_size=args.grad_cos_chunk_size,
            print_fn=print,
        )
        cache_time = time.time() - cache_start

        shard_dir = results_folder / 'shards'
        shard_dir.mkdir(parents=True, exist_ok=True)
        proj_path = shard_dir / f'train_proj_{shard_id:03d}_of_{num_shards:03d}.npy'
        idx_path  = shard_dir / f'positions_{shard_id:03d}_of_{num_shards:03d}.npy'
        meta_path = shard_dir / f'meta_{shard_id:03d}_of_{num_shards:03d}.json'

        np.save(str(proj_path), train_proj.numpy())
        np.save(str(idx_path), np.array(shard_positions, dtype=np.int64))
        shard_meta = {
            'shard_id': shard_id,
            'num_shards': num_shards,
            'num_samples': len(shard_positions),
            'proj_dim': args.proj_dim,
            'cache_time_seconds': cache_time,
            'proj_shape': list(train_proj.shape),
        }
        with open(str(meta_path), 'w') as f:
            json.dump(shard_meta, f, indent=2)

        print(f'[SHARD {shard_id}] Saved: {proj_path.name}, {idx_path.name}')
        print(f'[SHARD {shard_id}] DONE in {cache_time:.0f}s ({cache_time / 60:.1f}min)')

        if args.wandb:
            import wandb
            wandb.log({
                'shard/id': shard_id,
                'shard/num_shards': num_shards,
                'shard/num_samples': len(shard_positions),
                'shard/cache_time_seconds': cache_time,
            })
            wandb.finish()

    def merge_attribute(
        self,
        num_train: int,
        results_folder: Path,
        num_shards: int,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        """Load cached train proj shards, compute test proj grads, return scores."""
        args = self.args
        print(f'Loading {num_shards} cached train proj shards '
              f'from {results_folder / "shards"}...')
        shard_dir = results_folder / 'shards'

        train_proj_full = torch.zeros(num_train, args.proj_dim)
        total_shard_time = 0.0
        for i in range(num_shards):
            proj_path = shard_dir / f'train_proj_{i:03d}_of_{num_shards:03d}.npy'
            idx_path  = shard_dir / f'positions_{i:03d}_of_{num_shards:03d}.npy'
            if not proj_path.exists():
                raise FileNotFoundError(f'Missing shard file: {proj_path}')
            proj_i      = torch.from_numpy(np.load(str(proj_path)))
            positions_i = np.load(str(idx_path))
            train_proj_full[positions_i] = proj_i
            meta_path = shard_dir / f'meta_{i:03d}_of_{num_shards:03d}.json'
            if meta_path.exists():
                with open(str(meta_path)) as f:
                    total_shard_time += json.load(f).get('cache_time_seconds', 0.0)
            print(f'  Loaded shard {i}: proj={list(proj_i.shape)}')
            del proj_i

        print(f'Merged train_proj: {list(train_proj_full.shape)}')
        print(f'Total shard compute time: {total_shard_time:.0f}s '
              f'({total_shard_time / 60:.1f}min)')

        # Compute test projected gradients (backward on test set)
        print('Computing test projected gradients...')
        test_proj = compute_test_proj_grads(
            stage_wrapper=self.stage_wrapper,
            model_parameters=iter(self.params),
            test_loader=test_loader,
            ds_fields=self.ds_fields,
            device=self.device,
            proj_dim=args.proj_dim,
            seed=args.seed,
            chunk_size=args.grad_cos_chunk_size,
            print_fn=print,
        )  # (num_test, proj_dim) on CPU

        # Attribution scores
        print('Computing attribution scores...')
        train_proj_dev = train_proj_full.to(self.device)
        test_proj_dev  = test_proj.to(self.device)
        scores = compute_grad_cos_from_projs(train_proj_dev, test_proj_dev)
        # scores: (num_train, num_test) on device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute self-sanity from cached train projections (chunked)
        print('Computing self-sanity metrics from cached train projections (chunked)...')
        self._cached_self_sanity = self._compute_chunked_self_sanity(
            train_proj_full=train_proj_full,
            train_proj_dev=train_proj_dev,
            num_train=num_train,
            device=self.device,
            chunk=512,
        )

        del train_proj_full, train_proj_dev, test_proj_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.wandb:
            import wandb
            wandb.log({
                'timing/total_shard_cache_seconds': total_shard_time,
            })

        return scores

    def _compute_chunked_self_sanity(
        self,
        train_proj_full: torch.Tensor,
        train_proj_dev: torch.Tensor,
        num_train: int,
        device: str,
        chunk: int = 512,
    ) -> Tuple[Dict, Dict]:
        """Chunked row-wise self-sanity to avoid N×N GPU allocation.

        Returns (metrics_dict, arrays_dict) in the same format as
        compute_self_sanity_metrics.
        """
        _N = num_train
        _diag_scores      = np.empty(_N, dtype=np.float32)
        _diag_ranks       = np.empty(_N, dtype=np.int64)
        _diag_percentiles = np.empty(_N, dtype=np.float32)
        _top1_hits_strict = np.empty(_N, dtype=bool)
        _top1_hits_tie    = np.empty(_N, dtype=bool)
        _top1_indices     = np.empty(_N, dtype=np.int64)

        with torch.no_grad():
            for _s in range(0, _N, chunk):
                _e = min(_s + chunk, _N)
                _rows = train_proj_full[_s:_e].to(device)             # (c, proj_dim)
                _rs = (_rows @ train_proj_dev.T).cpu().numpy()        # (c, N)
                del _rows
                _c_idx = np.arange(_s, _e)
                _loc   = np.arange(_e - _s)
                _d     = _rs[_loc, _c_idx]                            # diag values
                _mx_v  = _rs.max(axis=1)
                _mx_i  = _rs.argmax(axis=1)
                _diag_scores[_s:_e]      = _d
                _diag_ranks[_s:_e]       = (_rs > _d[:, None]).sum(axis=1) + 1
                _diag_percentiles[_s:_e] = (
                    100.0 * (_rs < _d[:, None]).sum(axis=1) / max(_N - 1, 1)
                )
                _top1_hits_strict[_s:_e] = (_mx_i == _c_idx)
                _top1_hits_tie[_s:_e]    = (_d >= _mx_v - 1e-6)
                _top1_indices[_s:_e]     = _mx_i

        self_arrays = {
            'diag_scores':      _diag_scores,
            'diag_ranks':       _diag_ranks,
            'diag_percentiles': _diag_percentiles,
            'top1_hits_strict': _top1_hits_strict,
            'top1_hits_tie':    _top1_hits_tie,
            'top1_indices':     _top1_indices,
        }
        self_metrics = {
            'top1_hit_rate_strict':  float(_top1_hits_strict.mean()),
            'top1_hit_rate_tie':     float(_top1_hits_tie.mean()),
            'mean_percentile':       float(_diag_percentiles.mean()),
            'median_percentile':     float(np.median(_diag_percentiles)),
            'p10_percentile':        float(np.percentile(_diag_percentiles, 10)),
            'p90_percentile':        float(np.percentile(_diag_percentiles, 90)),
            'mean_rank':             float(_diag_ranks.mean()),
            'median_rank':           float(np.median(_diag_ranks)),
            'num_train':             int(_N),
        }
        return self_metrics, self_arrays

    def full(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        """End-to-end attribution without sharding."""
        args = self.args
        print('Computing attribution scores (custom backward)...')
        scores = compute_grad_cos_scores(
            stage_wrapper=self.stage_wrapper,
            model_parameters=iter(self.params),
            train_loader=train_loader,
            test_loader=test_loader,
            ds_fields=self.ds_fields,
            device=self.device,
            proj_dim=args.proj_dim,
            seed=args.seed,
            chunk_size=args.grad_cos_chunk_size,
            print_fn=print,
        )
        return scores

    def self_sanity(
        self,
        train_loader: DataLoader,
        num_train: int,
    ) -> Tuple[Dict, Dict]:
        """Compute self-sanity metrics.

        In merge mode: returns cached result from merge_attribute (no extra backward).
        In full mode: recomputes train projections and uses chunked approach.
        """
        if self._cached_self_sanity is not None:
            return self._cached_self_sanity

        # full mode: recompute train projected gradients
        args = self.args
        print('Computing train projected gradients for self-sanity...')
        train_proj = compute_train_proj_grads(
            stage_wrapper=self.stage_wrapper,
            model_parameters=iter(self.params),
            train_loader=train_loader,
            ds_fields=self.ds_fields,
            device=self.device,
            proj_dim=args.proj_dim,
            seed=args.seed,
            chunk_size=args.grad_cos_chunk_size,
            print_fn=print,
        )  # (num_train, proj_dim) on CPU
        train_proj_dev = train_proj.to(self.device)
        result = self._compute_chunked_self_sanity(
            train_proj_full=train_proj,
            train_proj_dev=train_proj_dev,
            num_train=num_train,
            device=self.device,
            chunk=512,
        )
        del train_proj, train_proj_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


__all__ = [
    '_compute_proj_grads',
    'compute_train_proj_grads',
    'compute_test_proj_grads',
    'compute_grad_cos_from_projs',
    'compute_grad_cos_scores',
    'GradCosMethod',
]
