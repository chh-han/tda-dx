"""
attributor/methods/trak.py

TRAK attribution method wrapper.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tda_dx.attribution.methods.base import AttributionMethod
from tda_dx.attribution.data.loading import DS_FIELDS


class TRAKMethod(AttributionMethod):
    """Attribution method wrapping TRAK (dattri TRAKAttributor).

    Supports cache_shard / merge_attribute / full modes.
    """

    def __init__(self):
        self.stage_wrapper = None
        self.params = None
        self.ds_fields = None
        self.device = None
        self.args = None
        self.attributor = None
        self.task = None
        self.inner_transformer = None
        self.loss_fn = None

    @staticmethod
    def _parse_csv_list(raw: str | None) -> List[str]:
        if raw is None:
            return []
        return [item.strip() for item in raw.split(',') if item.strip()]

    def _get_num_checkpoints(self) -> int:
        return len(self.attributor.task.get_checkpoints())

    def setup(
        self,
        stage_wrapper: torch.nn.Module,
        params: List[torch.nn.Parameter],
        ds_fields: Tuple[str, ...],
        device: str,
        args,
    ) -> None:
        """Initialize TRAK state. Called once after model is loaded."""
        self.stage_wrapper = stage_wrapper
        self.params = params
        self.ds_fields = ds_fields
        self.device = device
        self.args = args

    def set_task_and_transformer(self, task, inner_transformer, loss_fn):
        """Set the dattri AttributionTask and inner transformer. Called by runner."""
        self.task = task
        self.inner_transformer = inner_transformer
        self.loss_fn = loss_fn

        checkpoint_paths = self._parse_csv_list(getattr(self.args, 'trak_checkpoint_paths', None))
        if checkpoint_paths:
            task.checkpoints = checkpoint_paths
            task.current_checkpoint_idx = None

        from dattri.algorithm.trak import TRAKAttributor
        args = self.args
        layer_names_for_attr = getattr(args, '_layer_names_for_attr', None)
        proj_type = "sjlt" if self.device == "cuda" else "normal"
        self.attributor = TRAKAttributor(
            task=task,
            correct_probability_func=lambda params, data_pair: torch.exp(-loss_fn(params, data_pair)),
            projector_kwargs={"proj_dim": args.proj_dim, "device": self.device, "proj_type": proj_type},
            layer_name=layer_names_for_attr,
            device=self.device,
            regularization=args.regularization,
        )

    def cache_shard(
        self,
        shard_loader: DataLoader,
        shard_positions: List[int],
        results_folder: Path,
        shard_id: int,
        num_shards: int,
    ) -> None:
        """Compute projected gradients + Q for a subset of training data."""
        from dattri.func.projection import random_project

        args = self.args
        attributor = self.attributor

        print(f'[SHARD {shard_id}/{num_shards}] '
              f'Samples: {len(shard_positions)} (of ??? total)')

        # Save shard data
        shard_dir = results_folder / 'shards'
        shard_dir.mkdir(parents=True, exist_ok=True)
        idx_path = shard_dir / f'positions_{shard_id:03d}_of_{num_shards:03d}.npy'
        np.save(str(idx_path), np.array(shard_positions))

        # Pre-allocate a float32 gradient buffer to avoid repeated large allocations
        # inside the sample loop.  Shared across checkpoints (same parameter count).
        _trainable_params = [p for p in self.inner_transformer.parameters() if p.requires_grad]
        _num_trainable = sum(p.numel() for p in _trainable_params)
        _grad_buffer = torch.zeros(_num_trainable, dtype=torch.float32, device=self.device)

        total_cache_time = 0.0
        num_ckpts = self._get_num_checkpoints()
        for ckpt_idx in range(num_ckpts):
            # Load checkpoint weights into the model (side effect of get_param).
            # full_parameters is no longer needed: Q is derived directly from the
            # loss value computed during the backward pass, avoiding a second forward.
            attributor.task.get_param(ckpt_idx=ckpt_idx, layer_name=attributor.layer_name)

            projected_grads = []
            Q_values = []
            cache_start = time.time()

            pbar = tqdm(
                shard_loader,
                total=len(shard_positions),
                desc=f'[Shard {shard_id}/{num_shards}][ckpt {ckpt_idx}] caching',
                dynamic_ncols=True,
            )
            for train_data in pbar:
                # train_data is a batch of (possibly) multiple samples.
                # TRAK requires per-sample gradients, so we loop over each sample
                # individually.  Peak GPU memory stays the same as batch_size=1;
                # the only gain from a larger DataLoader batch_size is reduced
                # data-loading overhead.
                batch_size_actual = train_data[0].shape[0]
                for sample_idx in range(batch_size_actual):
                    single_data = tuple(t[sample_idx:sample_idx + 1].to(self.device) for t in train_data)

                    # --- Single forward+backward: gradient AND loss value in one pass ---
                    for p in self.inner_transformer.parameters():
                        p.grad = None
                    _data_kwargs = {
                        k: single_data[i] for i, k in enumerate(self.ds_fields)
                    }
                    _loss, _, _ = self.stage_wrapper(**_data_kwargs, return_loss=True)
                    _loss_val = _loss.item()   # scalar loss for Q computation
                    _loss.backward()
                    del _loss

                    # Fill pre-allocated buffer in-place (avoids repeated large allocations)
                    _offset = 0
                    for p in _trainable_params:
                        _n = p.numel()
                        if p.grad is not None:
                            _grad_buffer[_offset:_offset + _n].copy_(p.grad.view(-1).float())
                        else:
                            _grad_buffer[_offset:_offset + _n].zero_()
                        p.grad = None   # free immediately
                        _offset += _n
                    torch.nan_to_num_(_grad_buffer)           # in-place, no new allocation
                    _grad_buffer /= attributor.norm_scaler    # in-place
                    grad_t = _grad_buffer.unsqueeze(0)        # view, no copy

                    grad_p = (
                        random_project(
                            grad_t, 1, **attributor.projector_kwargs,
                        )(grad_t, ensemble_id=ckpt_idx)
                        .clone().detach()
                    )
                    # Offload projected gradient to CPU immediately to free GPU memory
                    projected_grads.append(grad_p.cpu())

                    # Q = 1 - exp(-loss).  Mathematically identical to the original
                    # correct_probability_func but avoids a second forward pass.
                    q = torch.tensor([1.0 - math.exp(-_loss_val)], dtype=torch.float32)
                    Q_values.append(q)

                elapsed = time.time() - cache_start
                done = len(projected_grads)
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(shard_positions) - done) / speed if speed > 0 else 0
                pbar.set_postfix(loss=f'{_loss_val:.3f}', sps=f'{speed:.2f}', eta=f'{eta/60:.0f}m')

            Phi = torch.cat(projected_grads, dim=0)
            Q = torch.cat(Q_values, dim=0)
            cache_time = time.time() - cache_start
            total_cache_time += cache_time

            phi_path = shard_dir / f'phi_ckpt{ckpt_idx:03d}_{shard_id:03d}_of_{num_shards:03d}.pt'
            q_path = shard_dir / f'q_ckpt{ckpt_idx:03d}_{shard_id:03d}_of_{num_shards:03d}.pt'
            torch.save(Phi.cpu(), str(phi_path))
            torch.save(Q.cpu(), str(q_path))

            print(f'[SHARD {shard_id}][CKPT {ckpt_idx}] Gradient projection done: '
                  f'Phi={list(Phi.shape)}, Q={list(Q.shape)} ({cache_time:.0f}s)')

        shard_meta = {
            'shard_id': shard_id,
            'num_shards': num_shards,
            'num_samples': len(shard_positions),
            'proj_dim': args.proj_dim,
            'num_checkpoints': num_ckpts,
            'cache_time_seconds_total': total_cache_time,
        }
        meta_path = shard_dir / f'meta_{shard_id:03d}_of_{num_shards:03d}.json'
        with open(str(meta_path), 'w') as f:
            json.dump(shard_meta, f, indent=2)

        print(f'[SHARD {shard_id}] Saved shard cache for {num_ckpts} checkpoint(s), positions_{shard_id:03d}_of_{num_shards:03d}.npy')
        print(f'[SHARD {shard_id}] DONE in {total_cache_time:.0f}s ({total_cache_time / 60:.1f}min)')

        if args.wandb:
            import wandb
            wandb.log({
                "shard/id": shard_id,
                "shard/num_shards": num_shards,
                "shard/num_samples": len(shard_positions),
                "shard/cache_time_seconds": total_cache_time,
            })
            wandb.finish()

    def merge_attribute(
        self,
        num_train: int,
        results_folder: Path,
        num_shards: int,
        test_loader: DataLoader,
        shard_source_folder: Path = None,
    ) -> torch.Tensor:
        """Load cached shards, build kernel matrix, compute scores.

        Args:
            shard_source_folder: If provided, read Phi/Q shard files from this
                directory instead of results_folder.  Useful for re-running the
                merge step (e.g. with a different regularization) without
                re-caching gradients.
        """
        from dattri.func.projection import random_project

        args = self.args
        attributor = self.attributor
        inner_transformer = self.inner_transformer
        stage_wrapper = self.stage_wrapper
        ds_fields = self.ds_fields
        device = self.device

        shard_dir = (shard_source_folder if shard_source_folder is not None else results_folder) / 'shards'
        print(f'Loading {num_shards} cached shards from {shard_dir}...')

        num_ckpts = self._get_num_checkpoints()
        total_shard_time = 0
        inv_XTX_XT_list = []
        running_Q = 0
        running_count = 0

        for ckpt_idx in range(num_ckpts):
            Phi_full = torch.zeros(num_train, args.proj_dim)
            Q_full = torch.zeros(num_train)

            for i in range(num_shards):
                phi_path = shard_dir / f'phi_ckpt{ckpt_idx:03d}_{i:03d}_of_{num_shards:03d}.pt'
                q_path = shard_dir / f'q_ckpt{ckpt_idx:03d}_{i:03d}_of_{num_shards:03d}.pt'
                idx_path = shard_dir / f'positions_{i:03d}_of_{num_shards:03d}.npy'

                if not phi_path.exists():
                    raise FileNotFoundError(f'Missing shard file: {phi_path}')

                phi_i = torch.load(str(phi_path), map_location='cpu')
                q_i = torch.load(str(q_path), map_location='cpu')
                positions_i = np.load(str(idx_path))
                Phi_full[positions_i] = phi_i
                Q_full[positions_i] = q_i
                print(f'  Loaded shard {i} for ckpt {ckpt_idx}: Phi={list(phi_i.shape)}')
                del phi_i, q_i

                meta_path = shard_dir / f'meta_{i:03d}_of_{num_shards:03d}.json'
                if meta_path.exists() and ckpt_idx == 0:
                    with open(str(meta_path)) as f:
                        shard_meta = json.load(f)
                    total_shard_time += shard_meta.get('cache_time_seconds_total', 0)

            Phi_full = Phi_full.to(device)
            Q_full = Q_full.to(device)
            print(f'Merged Phi ckpt {ckpt_idx}: {list(Phi_full.shape)}, Q: {list(Q_full.shape)}')

            print(f'Building kernel matrix for ckpt {ckpt_idx}...')
            kernel_start = time.time()
            kernel_matrix = Phi_full.T @ Phi_full
            kernel_matrix.diagonal().add_(attributor.regularization)
            inv_XTX_XT = torch.linalg.solve(kernel_matrix, Phi_full.T)
            kernel_time = time.time() - kernel_start
            print(f'Kernel computation for ckpt {ckpt_idx} done in {kernel_time:.0f}s')
            inv_XTX_XT_list.append(inv_XTX_XT)
            running_Q = running_Q * running_count + Q_full
            running_count += 1
            running_Q /= running_count

            del Phi_full, Q_full, kernel_matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f'Total shard compute time: {total_shard_time:.0f}s ({total_shard_time / 60:.1f}min)')

        # Set attributor state so .attribute(test_loader) works
        attributor.inv_XTX_XT_list = inv_XTX_XT_list
        attributor.Q = running_Q
        attributor.full_train_dataloader = True  # sentinel (non-None to pass validation)

        # Store shard info so self_sanity can reuse cached Phi instead of recomputing gradients
        self._shard_dir = shard_dir
        self._shard_num_shards = num_shards
        self._shard_num_train = num_train
        self._shard_num_ckpts = num_ckpts

        if args.wandb:
            import wandb
            wandb.log({
                "timing/total_shard_cache_seconds": total_shard_time,
                "timing/kernel_seconds": kernel_time,
            })

        print('Computing attribution scores...')
        # Custom backward for test gradients (memory-efficient, avoids vmap OOM).
        from tqdm import tqdm
        stage_wrapper.eval()
        running_xinv_XTX_XT = 0
        running_count = 0
        for ckpt_idx in range(num_ckpts):
            attributor.task._load_checkpoints(ckpt_idx)
            _test_proj_grads = []
            for _test_data in tqdm(test_loader, desc=f'calculating gradient of test set (ckpt {ckpt_idx})...'):
                for p in inner_transformer.parameters():
                    p.grad = None
                _data_kwargs = {k: _test_data[i].to(device) for i, k in enumerate(ds_fields)}
                _loss, _, _ = stage_wrapper(**_data_kwargs, return_loss=True)
                _loss.backward()
                del _loss
                _grad_t = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in inner_transformer.parameters() if p.requires_grad
                ]).unsqueeze(0)
                _grad_t = torch.nan_to_num(_grad_t)
                _grad_t /= attributor.norm_scaler
                _grad_p = (
                    random_project(_grad_t, _grad_t.shape[0], **attributor.projector_kwargs)(
                        _grad_t, ensemble_id=ckpt_idx
                    ).clone().detach()
                )
                _test_proj_grads.append(_grad_p)
                for p in inner_transformer.parameters():
                    p.grad = None
            _test_proj_grads = torch.cat(_test_proj_grads, dim=0)
            running_xinv_XTX_XT = (
                running_xinv_XTX_XT * running_count
                + _test_proj_grads.float() @ attributor.inv_XTX_XT_list[ckpt_idx]
            )
            running_count += 1
            running_xinv_XTX_XT /= running_count

        scores = (running_xinv_XTX_XT * attributor.Q.to(device).unsqueeze(0)).T
        return scores

    def full(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        """End-to-end TRAK attribution without sharding."""
        attributor = self.attributor
        args = self.args

        print('Caching training representations...')
        cache_start = time.time()
        attributor.cache(train_loader)
        cache_time = time.time() - cache_start
        print(f'Caching done in {cache_time:.0f}s')

        if args.wandb:
            import wandb
            wandb.log({"timing/cache_seconds": cache_time})

        print('Computing attribution scores...')
        scores = attributor.attribute(test_loader)
        return scores

    def self_sanity(
        self,
        train_loader: DataLoader,
        num_train: int,
    ) -> Tuple[Dict, Dict]:
        """Compute TRAK self-sanity using chunked approach.

        If merge_attribute was called beforehand, reuses cached shard Phi matrices
        instead of recomputing training gradients from scratch.
        """
        from attributor.sanity.metrics import (
            compute_trak_self_sanity_chunked,
            _aggregate_self_sanity_arrays,
        )

        args = self.args
        attributor = self.attributor
        device = self.device

        if hasattr(self, '_shard_dir'):
            # Reuse shard Phi: no gradient recomputation needed.
            # scores[:, j] = phi_j @ avg(inv_XTX_XT) * Q  where phi_j = Phi_full[j]
            print('Self-sanity: reusing cached shard Phi (no gradient recomputation)...')
            shard_dir = self._shard_dir
            num_shards = self._shard_num_shards
            num_ckpts = self._shard_num_ckpts

            # Average inv_XTX_XT over checkpoints
            avg_inv = sum(attributor.inv_XTX_XT_list) / len(attributor.inv_XTX_XT_list)
            # avg_inv: (proj_dim, num_train)
            Q = attributor.Q.to(device)  # (num_train,)

            chunk_size = args.self_sanity_query_chunk_size
            n = num_train

            diag_scores = np.zeros(n, dtype=np.float64)
            top1_indices = np.zeros(n, dtype=np.int64)
            top1_hits_strict = np.zeros(n, dtype=np.bool_)
            top1_hits_tie = np.zeros(n, dtype=np.bool_)
            diag_ranks = np.zeros(n, dtype=np.float64)
            diag_percentiles = np.zeros(n, dtype=np.float64)

            for j_start in tqdm(range(0, n, chunk_size), desc='self-sanity (shard reuse)'):
                j_end = min(j_start + chunk_size, n)
                query_indices = np.arange(j_start, j_end, dtype=np.int64)

                # Load Phi rows for this query chunk from shards
                Phi_chunk = torch.zeros(j_end - j_start, args.proj_dim)
                for shard_idx in range(num_shards):
                    # Average over checkpoints (use ckpt 0 Phi; all shards same positions)
                    phi_path = shard_dir / f'phi_ckpt000_{shard_idx:03d}_of_{num_shards:03d}.pt'
                    idx_path = shard_dir / f'positions_{shard_idx:03d}_of_{num_shards:03d}.npy'
                    positions = np.load(str(idx_path))
                    mask = (positions >= j_start) & (positions < j_end)
                    if not mask.any():
                        continue
                    phi_i = torch.load(str(phi_path), map_location='cpu')
                    local_idx = positions[mask] - j_start
                    Phi_chunk[local_idx] = phi_i[mask]
                    del phi_i

                # For multi-checkpoint: average Phi over checkpoints
                if num_ckpts > 1:
                    Phi_chunk_sum = Phi_chunk.clone()
                    for ckpt_idx in range(1, num_ckpts):
                        for shard_idx in range(num_shards):
                            phi_path = shard_dir / f'phi_ckpt{ckpt_idx:03d}_{shard_idx:03d}_of_{num_shards:03d}.pt'
                            idx_path = shard_dir / f'positions_{shard_idx:03d}_of_{num_shards:03d}.npy'
                            positions = np.load(str(idx_path))
                            mask = (positions >= j_start) & (positions < j_end)
                            if not mask.any():
                                continue
                            phi_i = torch.load(str(phi_path), map_location='cpu')
                            local_idx = positions[mask] - j_start
                            Phi_chunk_sum[local_idx] += phi_i[mask]
                            del phi_i
                    Phi_chunk = Phi_chunk_sum / num_ckpts

                Phi_chunk = Phi_chunk.to(device).float()
                # score_chunk[q, i] = phi_q @ avg_inv[:, i] * Q[i]
                # = (Phi_chunk @ avg_inv) * Q.unsqueeze(0)
                # shape: (chunk, num_train)
                score_chunk = (Phi_chunk @ avg_inv).float() * Q.float().unsqueeze(0)
                # transpose to (num_train, chunk) to match compute_trak_self_sanity_chunked convention
                score_chunk = score_chunk.T.cpu().numpy()  # (num_train, chunk)

                local_row_idx = np.arange(j_end - j_start, dtype=np.int64)
                diag_vals = score_chunk[query_indices, local_row_idx].astype(np.float64)
                max_vals = score_chunk.max(axis=0).astype(np.float64)
                argmax_idx = score_chunk.argmax(axis=0).astype(np.int64)

                gt_counts = (score_chunk > diag_vals[None, :]).sum(axis=0).astype(np.float64)
                eq_counts = (score_chunk == diag_vals[None, :]).sum(axis=0).astype(np.float64)
                best_rank = gt_counts + 1.0
                worst_rank = gt_counts + eq_counts
                avg_rank = 0.5 * (best_rank + worst_rank)
                percentiles = 100.0 * (n - avg_rank) / (n - 1) if n > 1 else np.full_like(avg_rank, 100.0)

                diag_scores[query_indices] = diag_vals
                top1_indices[query_indices] = argmax_idx
                top1_hits_strict[query_indices] = (argmax_idx == query_indices)
                top1_hits_tie[query_indices] = (diag_vals >= max_vals)
                diag_ranks[query_indices] = avg_rank
                diag_percentiles[query_indices] = percentiles

                del Phi_chunk, score_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return _aggregate_self_sanity_arrays(
                diag_scores=diag_scores,
                top1_indices=top1_indices,
                top1_hits_strict=top1_hits_strict,
                top1_hits_tie=top1_hits_tie,
                diag_ranks=diag_ranks,
                diag_percentiles=diag_percentiles,
            )

        # Fallback: recompute training gradients via dattri cache
        print('Caching training representations...')
        cache_start = time.time()
        attributor.cache(train_loader)
        print(f'Caching done in {time.time() - cache_start:.0f}s')

        if args.wandb:
            import wandb
            wandb.log({"timing/cache_seconds": time.time() - cache_start})

        print(
            'Computing self-sanity metrics with chunked TRAK attribution '
            f'(query_chunk_size={args.self_sanity_query_chunk_size})...'
        )
        return compute_trak_self_sanity_chunked(
            attributor=attributor,
            test_dataset=train_loader.dataset,
            num_train=num_train,
            grad_batch_size=args.batch_size,
            query_chunk_size=args.self_sanity_query_chunk_size,
        )


__all__ = ['TRAKMethod']
