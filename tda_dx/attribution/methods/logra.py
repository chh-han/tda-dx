"""
attributor/methods/logra.py

LoGra attribution method wrapper.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from tda_dx.attribution.methods.base import AttributionMethod
from tda_dx.attribution.methods.block_projected_utils import (
    configure_block_projected_params,
    validate_single_batch_contract,
)
from tda_dx.attribution.sanity.metrics import (
    _aggregate_self_sanity_arrays,
    compute_self_sanity_metrics,
)


class LoGraMethod(AttributionMethod):
    """Attribution method wrapping dattri LoGraAttributor.

    Supports `full` mode and cache-backed `merge_attribute` mode.
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
        self.layer_names = None

    def setup(
        self,
        stage_wrapper: torch.nn.Module,
        params: List[torch.nn.Parameter],
        ds_fields: Tuple[str, ...],
        device: str,
        args,
    ) -> None:
        self.stage_wrapper = stage_wrapper
        self.params = params
        self.ds_fields = ds_fields
        self.device = device
        self.args = args

    def set_task_and_transformer(self, task, inner_transformer, loss_fn):
        """Set the dattri AttributionTask and construct the LoGra attributor."""
        del loss_fn  # LoGra uses the task loss directly.

        self.task = task
        self.inner_transformer = inner_transformer

        from dattri.algorithm.logra import LoGraAttributor

        layer_names = configure_block_projected_params(inner_transformer, "LoGra")
        self.layer_names = layer_names

        cache_dir = self.args.logra_cache_dir
        if cache_dir is None and self.args.logra_offload == 'disk':
            cache_dir = str(Path(self.args.results_folder) / 'logra_cache')

        self.attributor = LoGraAttributor(
            task=task,
            layer_names=layer_names,
            hessian=self.args.logra_hessian,
            damping=self.args.logra_damping,
            device=self.device,
            proj_dim=self.args.proj_dim,
            offload=self.args.logra_offload,
            cache_dir=cache_dir,
            chunk_size=self.args.logra_chunk_size,
        )
        self.attributor.release_train_gradients_after_ifvp = (
            self.args.logra_offload != 'disk'
        )

    def _validate_single_batch_contract(self, train_loader: DataLoader) -> None:
        validate_single_batch_contract(
            method_label="LoGra",
            attributor=self.attributor,
            inner_transformer=self.inner_transformer,
            layer_names=self.layer_names,
            train_loader=train_loader,
            device=self.device,
        )

    def cache_shard(
        self,
        shard_loader: DataLoader,
        shard_positions: List[int],
        results_folder: Path,
        shard_id: int,
        num_shards: int,
    ) -> None:
        del shard_loader, shard_positions, results_folder, shard_id, num_shards
        raise NotImplementedError('LoGra currently supports only --mode full in attributor.')

    def merge_attribute(
        self,
        num_train: int,
        results_folder: Path,
        num_shards: int,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        del results_folder, num_shards

        if self.args.logra_cache_dir is None:
            raise RuntimeError(
                "LoGra merge_attribute requires --logra_cache_dir pointing to an "
                "existing cache directory created by self.attributor.cache(...)."
            )

        cache_dir = Path(self.args.logra_cache_dir)
        metadata_path = cache_dir / "batch_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(
                f"LoGra merge_attribute could not find cache metadata at {metadata_path}"
            )
        if not self.attributor.offload_manager.has_ifvp():
            raise RuntimeError(
                "LoGra merge_attribute requires cached IFVP data. "
                "Run the cache/full stage to completion first."
            )

        print(f"Loading cached LoGra state from {cache_dir}")
        if not self.attributor.metadata.batch_info:
            self.attributor.metadata._load_metadata_if_exists()

        total_train_samples = self.attributor.metadata.get_total_samples()
        if total_train_samples != num_train:
            raise RuntimeError(
                "LoGra cache metadata does not match the current training split. "
                f"Cache has total_samples={total_train_samples}, "
                f"but current run expects num_train={num_train}. "
                "Use a fresh stage-specific cache directory created from the same "
                "dataset split/checkpoint configuration."
            )
        if self.attributor.metadata.layer_names != self.layer_names:
            raise RuntimeError(
                "LoGra cache layer_names do not match the current model hook set. "
                "The cache was likely produced by a different model/configuration."
            )
        expected_total_proj_dim = len(self.layer_names) * int(self.args.proj_dim)
        if self.attributor.metadata.total_proj_dim != expected_total_proj_dim:
            raise RuntimeError(
                "LoGra cache projection dimension mismatch. "
                f"Cache total_proj_dim={self.attributor.metadata.total_proj_dim}, "
                f"expected {expected_total_proj_dim} "
                f"(layers={len(self.layer_names)}, proj_dim={self.args.proj_dim})."
            )

        self.attributor._sync_layer_dims()
        if self.attributor.compressors is None:
            self.attributor._setup_compressors(test_loader)

        print("Computing test gradients from cached LoGra state...")
        test_grads_tensor, _ = self.attributor._compute_test_gradients(test_loader)
        if test_grads_tensor is None or test_grads_tensor.numel() == 0:
            num_test = (
                len(test_loader.sampler)
                if hasattr(test_loader, "sampler")
                else len(test_loader.dataset)
            )
            return torch.zeros(num_train, num_test)

        batch_to_sample_mapping = self.attributor.metadata.get_batch_to_sample_mapping()
        test_sample_count = test_grads_tensor.shape[0]

        if_score = torch.zeros(
            total_train_samples,
            test_sample_count,
            device=self.device,
        )

        train_ifvp_dataloader = self.attributor.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=32,
            pin_memory=True,
        )
        test_batch_size = min(256, test_sample_count)

        # Pin test grads on device once if it fits.
        try:
            test_grads_device = self.attributor.offload_manager.move_to_device(
                test_grads_tensor,
            )
        except RuntimeError:
            test_grads_device = None

        for chunk_tensor, batch_mapping in tqdm(
            train_ifvp_dataloader,
            desc="Computing LoGra attribution from cache",
        ):
            chunk_tensor_device = self.attributor.offload_manager.move_to_device(
                chunk_tensor,
            ).to(dtype=test_grads_tensor.dtype)

            for test_start in range(0, test_sample_count, test_batch_size):
                test_end = min(test_start + test_batch_size, test_sample_count)
                if test_grads_device is not None:
                    test_batch_device = test_grads_device[test_start:test_end]
                else:
                    test_batch = test_grads_tensor[test_start:test_end]
                    test_batch_device = self.attributor.offload_manager.move_to_device(
                        test_batch,
                    )

                chunk_scores = torch.matmul(chunk_tensor_device, test_batch_device.t())

                for batch_idx, (start_row, end_row) in batch_mapping.items():
                    batch_idx = int(batch_idx)
                    if batch_idx not in batch_to_sample_mapping:
                        continue
                    train_start, train_end = batch_to_sample_mapping[batch_idx]
                    if_score[train_start:train_end, test_start:test_end] = chunk_scores[
                        start_row:end_row
                    ].to(if_score.device)

            del chunk_tensor_device

        return if_score

    def full(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        print('Validating LoGra projected-gradient contract on first batch...')
        self._validate_single_batch_contract(train_loader)
        print('Caching LoGra training representations...')
        self.attributor.cache(train_loader)
        print('Computing LoGra attribution scores...')
        return self.attributor.attribute(train_loader, test_loader)

    def resume_after_grad(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        # Recovery path: gradients already on disk under logra_cache_dir.
        # Skip the cache() gradient loop, run only preconditioner + IFVP + attribute.
        # Forces release_train_gradients_after_ifvp=True to keep peak disk bounded.
        attributor = self.attributor
        attributor.release_train_gradients_after_ifvp = True
        attributor.full_train_dataloader = train_loader

        print('[resume] Loading cached metadata...')
        attributor.metadata._load_metadata_if_exists()
        attributor._sync_layer_dims()
        if attributor.layer_dims is None:
            raise RuntimeError(
                'resume_after_grad: layer_dims not found in batch_metadata.json. '
                'Patch it from a chunk_*_metadata.json before running.'
            )

        if attributor.compressors is None:
            print('[resume] Setting up compressors deterministically...')
            attributor._setup_compressors(train_loader)

        print('[resume] Computing preconditioners (reading existing grad chunks)...')
        attributor.compute_preconditioners()
        print('[resume] Computing IFVP (releases grad chunks as it goes)...')
        attributor.compute_ifvp()
        print('[resume] Computing attribution scores...')
        return attributor.attribute(train_loader, test_loader)

    def _compute_cached_self_sanity(self, num_train: int) -> Tuple[Dict, Dict]:
        if self.args.logra_cache_dir is None:
            raise RuntimeError(
                "LoGra cached self-sanity requires --logra_cache_dir pointing to an "
                "existing stage-specific cache directory."
            )

        cache_dir = Path(self.args.logra_cache_dir)
        metadata_path = cache_dir / "batch_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(
                f"LoGra cached self-sanity could not find cache metadata at {metadata_path}"
            )
        if not self.attributor.offload_manager.has_ifvp():
            raise RuntimeError(
                "LoGra cached self-sanity requires cached IFVP data. "
                "Run the cache/full stage to completion first."
            )

        if not self.attributor.metadata.batch_info:
            self.attributor.metadata._load_metadata_if_exists()

        total_train_samples = self.attributor.metadata.get_total_samples()
        if total_train_samples != num_train:
            raise RuntimeError(
                "LoGra cache metadata does not match the current training split. "
                f"Cache has total_samples={total_train_samples}, "
                f"but current run expects num_train={num_train}."
            )
        if self.attributor.metadata.layer_names != self.layer_names:
            raise RuntimeError(
                "LoGra cache layer_names do not match the current model hook set."
            )
        expected_total_proj_dim = len(self.layer_names) * int(self.args.proj_dim)
        if self.attributor.metadata.total_proj_dim != expected_total_proj_dim:
            raise RuntimeError(
                "LoGra cache projection dimension mismatch. "
                f"Cache total_proj_dim={self.attributor.metadata.total_proj_dim}, "
                f"expected {expected_total_proj_dim}."
            )

        batch_to_sample_mapping = self.attributor.metadata.get_batch_to_sample_mapping()

        def _row_sample_indices(batch_mapping: Dict[int, Tuple[int, int]]) -> np.ndarray:
            parts = []
            for batch_idx, (row_start, row_end) in batch_mapping.items():
                batch_idx = int(batch_idx)
                if batch_idx not in batch_to_sample_mapping:
                    raise RuntimeError(
                        f"LoGra cache batch index {batch_idx} missing from metadata mapping."
                    )
                sample_start, sample_end = batch_to_sample_mapping[batch_idx]
                expected_rows = int(row_end) - int(row_start)
                expected_samples = int(sample_end) - int(sample_start)
                if expected_rows != expected_samples:
                    raise RuntimeError(
                        "LoGra cache row/sample count mismatch for batch "
                        f"{batch_idx}: rows={expected_rows}, samples={expected_samples}"
                    )
                parts.append(np.arange(sample_start, sample_end, dtype=np.int64))
            if not parts:
                return np.empty((0,), dtype=np.int64)
            return np.concatenate(parts, axis=0)

        # Single-pass O(N^2)-IO self-sanity:
        #   S = G_grad @ G_ifvp^T   (N x N influence matrix)
        # Stream candidate (ifvp) chunks once; for each, multiply against the
        # full set of query (gradient) blocks held on device when possible.
        # This replaces the prior O(N^2 / chunk^2) disk passes with O(N) reads
        # per stream and identical math.
        device = self.device
        total_proj_dim = int(self.attributor.metadata.total_proj_dim)

        grad_loader = self.attributor.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=32,
            pin_memory=True,
        )
        if grad_loader is None:
            raise RuntimeError("LoGra cached self-sanity requires cached gradients.")

        # Load all gradients into a single (num_train, D) tensor. For LoGra the
        # cached representation is `(num_train) x (sum_l proj_dim_l)` which is
        # the same memory the disk cache already holds; loading it once avoids
        # quadratic re-reads in the score loop.
        full_grads = torch.empty(
            (num_train, total_proj_dim),
            dtype=torch.float32,
        )
        full_grads_indices_seen = np.zeros(num_train, dtype=np.bool_)
        print("Loading cached LoGra gradients into memory...")
        for grad_tensor, grad_mapping in tqdm(
            grad_loader, desc="LoGra load gradients", total=len(grad_loader),
        ):
            indices = _row_sample_indices(grad_mapping)
            if len(indices) != grad_tensor.shape[0]:
                raise RuntimeError("LoGra cached query row mapping mismatch.")
            full_grads[indices] = grad_tensor.to(dtype=torch.float32)
            full_grads_indices_seen[indices] = True
        if not full_grads_indices_seen.all():
            missing = int((~full_grads_indices_seen).sum())
            raise RuntimeError(
                f"LoGra cached self-sanity missing {missing} sample gradients."
            )

        # Decide whether to keep gradients on GPU (large speedup) or stream from
        # CPU per chunk. ~num_train * D * 4 bytes; bail to CPU if it doesn't fit.
        full_grads_device = None
        if torch.cuda.is_available():
            try:
                full_grads_device = full_grads.to(device, non_blocking=True)
            except RuntimeError:
                full_grads_device = None

        diag_scores = np.zeros(num_train, dtype=np.float64)
        local_max = np.full(num_train, -np.inf, dtype=np.float64)
        local_argmax = np.full(num_train, -1, dtype=np.int64)
        local_gt = np.zeros(num_train, dtype=np.float64)
        local_eq = np.zeros(num_train, dtype=np.float64)

        # Pass 1: stream IFVP to fill the diagonal of S = G_ifvp @ G_grad^T.
        # Each sample i contributes diag_scores[i] = <ifvp_i, grad_i>.
        ifvp_loader = self.attributor.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=32,
            pin_memory=True,
        )
        if ifvp_loader is None:
            raise RuntimeError("LoGra cached self-sanity requires cached IFVP.")

        print("Computing cached LoGra self-sanity diagonal (pass 1)...")
        for cand_tensor, cand_mapping in tqdm(
            ifvp_loader, desc="LoGra self-sanity diag", total=len(ifvp_loader),
        ):
            cand_indices = _row_sample_indices(cand_mapping)
            if len(cand_indices) != cand_tensor.shape[0]:
                raise RuntimeError("LoGra cached candidate row mapping mismatch.")
            if full_grads_device is not None:
                cand_dev = self.attributor.offload_manager.move_to_device(
                    cand_tensor,
                ).to(dtype=full_grads_device.dtype)
                idx_dev = torch.as_tensor(cand_indices, device=full_grads_device.device)
                grads_rows = full_grads_device.index_select(0, idx_dev)
                diag_block = (cand_dev * grads_rows).sum(dim=1).detach().cpu().numpy()
            else:
                cand_t = cand_tensor.to(dtype=torch.float32)
                grads_rows = full_grads[cand_indices]
                diag_block = (cand_t * grads_rows).sum(dim=1).numpy()
            diag_scores[cand_indices] = diag_block.astype(np.float64)

        # Pass 2: stream IFVP again and compute score blocks against ALL queries
        # (rows of full_grads). diag_scores is fully populated now, so
        # rank statistics are exact.
        ifvp_loader_2 = self.attributor.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=32,
            pin_memory=True,
        )
        print("Computing cached LoGra self-sanity ranks (pass 2)...")
        for cand_tensor, cand_mapping in tqdm(
            ifvp_loader_2, desc="LoGra self-sanity ranks", total=len(ifvp_loader_2),
        ):
            cand_indices = _row_sample_indices(cand_mapping)
            if len(cand_indices) != cand_tensor.shape[0]:
                raise RuntimeError("LoGra cached candidate row mapping mismatch.")

            if full_grads_device is not None:
                cand_dev = self.attributor.offload_manager.move_to_device(
                    cand_tensor,
                ).to(dtype=full_grads_device.dtype)
                # score_block[c, q] = <ifvp_c, grad_q>  =>  shape (|cand|, num_train)
                score_block_dev = cand_dev @ full_grads_device.t()
                score_block = score_block_dev.detach().cpu().numpy().astype(np.float64)
            else:
                cand_np = cand_tensor.to(dtype=torch.float32).numpy()
                score_block = (cand_np @ full_grads.numpy().T).astype(np.float64)

            block_max = score_block.max(axis=0)
            block_argmax_local = score_block.argmax(axis=0)
            improve = block_max > local_max
            local_max[improve] = block_max[improve]
            local_argmax[improve] = cand_indices[block_argmax_local[improve]]

            local_gt += (score_block > diag_scores[None, :]).sum(axis=0)
            local_eq += (score_block == diag_scores[None, :]).sum(axis=0)

        best_rank = local_gt + 1.0
        worst_rank = local_gt + local_eq
        avg_rank = 0.5 * (best_rank + worst_rank)
        if num_train == 1:
            diag_percentiles = np.full_like(avg_rank, 100.0, dtype=np.float64)
        else:
            diag_percentiles = 100.0 * (num_train - avg_rank) / (num_train - 1)

        top1_indices = local_argmax
        top1_hits_strict = local_argmax == np.arange(num_train, dtype=np.int64)
        top1_hits_tie = diag_scores >= local_max
        diag_ranks = avg_rank

        return _aggregate_self_sanity_arrays(
            diag_scores=diag_scores,
            top1_indices=top1_indices,
            top1_hits_strict=top1_hits_strict,
            top1_hits_tie=top1_hits_tie,
            diag_ranks=diag_ranks,
            diag_percentiles=diag_percentiles,
        )

    def self_sanity(
        self,
        train_loader: DataLoader,
        num_train: int,
    ) -> Tuple[Dict, Dict]:
        if self.args.mode == "merge_attribute":
            return self._compute_cached_self_sanity(num_train)

        print('Validating LoGra projected-gradient contract on first batch...')
        self._validate_single_batch_contract(train_loader)
        print('Caching LoGra training representations...')
        self.attributor.cache(train_loader)
        print('Computing LoGra self-sanity scores...')
        scores = self.attributor.attribute(train_loader, train_loader)
        scores_np = scores.detach().cpu().numpy()
        if scores_np.shape != (num_train, num_train):
            raise RuntimeError(
                'LoGra self-sanity requires raw score shape (num_train, num_train); '
                f'got {scores_np.shape}, expected {(num_train, num_train)}'
            )
        return compute_self_sanity_metrics(scores_np)


__all__ = ['LoGraMethod']
