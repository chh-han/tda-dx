"""
attributor/sanity/metrics.py

Self-influence sanity check metrics.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tda_dx.attribution.data.loading import collate_token_tuples


def compute_self_sanity_metrics(scores_np):
    """
    Compute self-influence sanity metrics from a square (train x test) score matrix
    where test samples are exactly the training samples in the same order.
    """
    if scores_np.ndim != 2 or scores_np.shape[0] != scores_np.shape[1]:
        raise ValueError(
            f'self-sanity requires square score matrix, got {scores_np.shape}'
        )

    n = scores_np.shape[0]
    if n == 0:
        raise ValueError('self-sanity requires at least one sample')

    diag_scores = np.diag(scores_np).astype(np.float64)
    top1_hits_strict = np.zeros(n, dtype=np.bool_)
    top1_hits_tie = np.zeros(n, dtype=np.bool_)
    top1_indices = np.zeros(n, dtype=np.int64)
    diag_ranks = np.zeros(n, dtype=np.float64)
    diag_percentiles = np.zeros(n, dtype=np.float64)

    for j in range(n):
        col = scores_np[:, j]
        d = diag_scores[j]
        top1_idx = int(np.argmax(col))
        top1_indices[j] = top1_idx
        top1_hits_strict[j] = (top1_idx == j)
        top1_hits_tie[j] = (d >= np.max(col))

        greater = int(np.sum(col > d))
        equal = int(np.sum(col == d))
        best_rank = greater + 1
        worst_rank = greater + equal
        avg_rank = 0.5 * (best_rank + worst_rank)
        diag_ranks[j] = avg_rank

        if n == 1:
            diag_percentiles[j] = 100.0
        else:
            diag_percentiles[j] = 100.0 * (n - avg_rank) / (n - 1)

    metrics = {
        'num_samples': int(n),
        'top1_hit_rate_strict': float(np.mean(top1_hits_strict)),
        'top1_hit_rate_tie': float(np.mean(top1_hits_tie)),
        'mean_percentile': float(np.mean(diag_percentiles)),
        'median_percentile': float(np.median(diag_percentiles)),
        'p10_percentile': float(np.percentile(diag_percentiles, 10)),
        'p90_percentile': float(np.percentile(diag_percentiles, 90)),
        'mean_diag_rank': float(np.mean(diag_ranks)),
        'median_diag_rank': float(np.median(diag_ranks)),
    }
    arrays = {
        'diag_scores': diag_scores,
        'top1_hits_strict': top1_hits_strict,
        'top1_hits_tie': top1_hits_tie,
        'top1_indices': top1_indices,
        'diag_ranks': diag_ranks,
        'diag_percentiles': diag_percentiles,
    }
    return metrics, arrays


def _aggregate_self_sanity_arrays(diag_scores, top1_indices, top1_hits_strict, top1_hits_tie, diag_ranks, diag_percentiles):
    metrics = {
        'num_samples': int(diag_scores.shape[0]),
        'top1_hit_rate_strict': float(np.mean(top1_hits_strict)),
        'top1_hit_rate_tie': float(np.mean(top1_hits_tie)),
        'mean_percentile': float(np.mean(diag_percentiles)),
        'median_percentile': float(np.median(diag_percentiles)),
        'p10_percentile': float(np.percentile(diag_percentiles, 10)),
        'p90_percentile': float(np.percentile(diag_percentiles, 90)),
        'mean_diag_rank': float(np.mean(diag_ranks)),
        'median_diag_rank': float(np.median(diag_ranks)),
    }
    arrays = {
        'diag_scores': diag_scores,
        'top1_hits_strict': top1_hits_strict,
        'top1_hits_tie': top1_hits_tie,
        'top1_indices': top1_indices,
        'diag_ranks': diag_ranks,
        'diag_percentiles': diag_percentiles,
    }
    return metrics, arrays


def compute_trak_self_sanity_chunked(
    attributor,
    test_dataset,
    num_train: int,
    grad_batch_size: int,
    query_chunk_size: int,
):
    """
    Exact self-sanity metrics for TRAK without materializing the full NxN score matrix.

    For each query chunk, we compute score_chunk = S[:, q_chunk] via
    attributor.attribute(query_loader), then update exact rank/percentile stats.
    """
    if query_chunk_size <= 0:
        raise ValueError(f'query_chunk_size must be > 0, got {query_chunk_size}')

    n = int(num_train)
    diag_scores = np.zeros(n, dtype=np.float64)
    top1_indices = np.zeros(n, dtype=np.int64)
    top1_hits_strict = np.zeros(n, dtype=np.bool_)
    top1_hits_tie = np.zeros(n, dtype=np.bool_)
    diag_ranks = np.zeros(n, dtype=np.float64)
    diag_percentiles = np.zeros(n, dtype=np.float64)

    for start in range(0, n, query_chunk_size):
        end = min(start + query_chunk_size, n)
        query_indices = np.arange(start, end, dtype=np.int64)
        query_subset = Subset(test_dataset, query_indices.tolist())
        query_loader = DataLoader(
            query_subset,
            batch_size=grad_batch_size,
            shuffle=False,
            collate_fn=collate_token_tuples,
        )

        score_chunk_t = attributor.attribute(query_loader)  # (num_train, chunk)
        score_chunk = score_chunk_t.detach().cpu().numpy()
        del score_chunk_t

        if score_chunk.ndim != 2 or score_chunk.shape != (n, end - start):
            raise RuntimeError(
                'Unexpected TRAK chunk score shape in self-sanity: '
                f'got {score_chunk.shape}, expected {(n, end - start)}'
            )

        local_row_idx = np.arange(end - start, dtype=np.int64)
        diag_vals = score_chunk[query_indices, local_row_idx].astype(np.float64)
        max_vals = score_chunk.max(axis=0).astype(np.float64)
        argmax_idx = score_chunk.argmax(axis=0).astype(np.int64)

        gt_counts = (score_chunk > diag_vals[None, :]).sum(axis=0).astype(np.float64)
        eq_counts = (score_chunk == diag_vals[None, :]).sum(axis=0).astype(np.float64)

        best_rank = gt_counts + 1.0
        worst_rank = gt_counts + eq_counts
        avg_rank = 0.5 * (best_rank + worst_rank)

        if n == 1:
            percentiles = np.full_like(avg_rank, 100.0, dtype=np.float64)
        else:
            percentiles = 100.0 * (n - avg_rank) / (n - 1)

        diag_scores[query_indices] = diag_vals
        top1_indices[query_indices] = argmax_idx
        top1_hits_strict[query_indices] = (argmax_idx == query_indices)
        top1_hits_tie[query_indices] = (diag_vals >= max_vals)
        diag_ranks[query_indices] = avg_rank
        diag_percentiles[query_indices] = percentiles

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


__all__ = [
    'compute_self_sanity_metrics',
    '_aggregate_self_sanity_arrays',
    'compute_trak_self_sanity_chunked',
]
