"""
attributor/pipeline/saving.py

Score normalization, post-processing, result saving, and shard merging.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np


def normalize_score_shape(scores_np, expected_rows, expected_cols, label):
    """Accept either (rows, cols) or (cols, rows); transpose when needed."""
    if scores_np.ndim != 2:
        raise ValueError(f'Expected 2D {label} matrix, got shape={scores_np.shape}')
    if expected_rows == expected_cols and scores_np.shape == (expected_rows, expected_cols):
        print(
            f'WARNING: {label} matrix is square ({scores_np.shape}); '
            f'transpose ambiguity cannot be detected automatically.'
        )
    if scores_np.shape == (expected_rows, expected_cols):
        return scores_np
    if scores_np.shape == (expected_cols, expected_rows):
        print(
            f'WARNING: {label} matrix shape {scores_np.shape} appears transposed; '
            f'transposing to ({expected_rows}, {expected_cols}).'
        )
        return scores_np.T
    raise ValueError(
        f'Unexpected {label} matrix shape {scores_np.shape}; '
        f'expected ({expected_rows}, {expected_cols}) or ({expected_cols}, {expected_rows}).'
    )


def save_attribution_results(
    scores_np,
    args,
    results_folder: Path,
    num_train: int,
    num_test: int,
    train_indices,
    train_track_ids: List[str],
    train_sqlite_indices,
    test_track_ids: List[str],
    test_losses,
    token_scores_np,
    token_axis_path,
    attr_time: float,
    run_id: str,
    _shard_tag: str,
    _resolve_genre: Callable,
    generated_tokens_dataset=None,
    print_all_scores: bool = False,
):
    """
    Save attribution scores, rankings, metadata, and optionally log to wandb.

    This function encapsulates the post-processing block from run_attribution()
    (lines 1613-1840 in the original attribution.py).
    """
    import time as _time

    run_start_ref = _time.time()  # used only for wandb total_seconds

    np.save(str(results_folder / f'{args.stage}_{args.method}_scores{_shard_tag}.npy'), scores_np)
    np.save(str(results_folder / 'test_losses.npy'), test_losses)
    # Save scores + track IDs together so they are always aligned.
    # For sharded runs each shard writes its own .npz; a merge step later
    # concatenates them in shard order to produce the final result.
    np.savez(
        str(results_folder / f'{args.stage}_{args.method}_scores_with_ids{_shard_tag}.npz'),
        scores=scores_np,
        train_track_ids=np.array(train_track_ids, dtype=object),
        train_indices=train_indices,
    )

    # average across test samples to get global per-train influence summary
    # NOTE: this is a global aggregate across all test samples.
    mean_influence = scores_np.mean(axis=1)  # shape: (num_train,)
    top_k = min(10, len(mean_influence))
    top_indices = np.argsort(-mean_influence)[:top_k]
    bottom_indices = np.argsort(mean_influence)[:top_k]

    train_genres = [_resolve_genre(tid) for tid in train_track_ids]
    test_genres = [_resolve_genre(tid) for tid in test_track_ids]

    top_ds_indices = train_indices[top_indices].tolist()
    bottom_ds_indices = train_indices[bottom_indices].tolist()
    top_sqlite_indices = [train_sqlite_indices[i] for i in top_indices.tolist()]
    bottom_sqlite_indices = [train_sqlite_indices[i] for i in bottom_indices.tolist()]
    top_scores = mean_influence[top_indices].tolist()
    bottom_scores = mean_influence[bottom_indices].tolist()

    # per-test rankings: for each test sample, rank train samples by attribution score
    per_test_rank_k = min(10, num_train)
    per_test_rankings = {}
    per_test_rank_rows = []
    for test_idx in range(num_test):
        col_scores = scores_np[:, test_idx]
        top_pool_indices = np.argsort(-col_scores)[:per_test_rank_k]
        bottom_pool_indices = np.argsort(col_scores)[:per_test_rank_k]

        top_items = []
        for rank, train_pool_idx in enumerate(top_pool_indices, start=1):
            train_ds_idx = int(train_indices[train_pool_idx])
            train_sqlite_idx = int(train_sqlite_indices[train_pool_idx])
            item = {
                'rank': rank,
                'train_pool_index': int(train_pool_idx),
                'train_ds_index': train_ds_idx,
                'train_sqlite_idx': train_sqlite_idx,
                'train_track_id': train_track_ids[train_pool_idx],
                'train_genre': train_genres[train_pool_idx],
                'score': float(col_scores[train_pool_idx]),
            }
            top_items.append(item)
            per_test_rank_rows.append([
                test_idx,
                test_track_ids[test_idx],
                test_genres[test_idx],
                'top',
                rank,
                int(train_pool_idx),
                train_ds_idx,
                train_sqlite_idx,
                train_track_ids[train_pool_idx],
                train_genres[train_pool_idx],
                float(col_scores[train_pool_idx]),
            ])

        bottom_items = []
        for rank, train_pool_idx in enumerate(bottom_pool_indices, start=1):
            train_ds_idx = int(train_indices[train_pool_idx])
            train_sqlite_idx = int(train_sqlite_indices[train_pool_idx])
            item = {
                'rank': rank,
                'train_pool_index': int(train_pool_idx),
                'train_ds_index': train_ds_idx,
                'train_sqlite_idx': train_sqlite_idx,
                'train_track_id': train_track_ids[train_pool_idx],
                'train_genre': train_genres[train_pool_idx],
                'score': float(col_scores[train_pool_idx]),
            }
            bottom_items.append(item)
            per_test_rank_rows.append([
                test_idx,
                test_track_ids[test_idx],
                test_genres[test_idx],
                'bottom',
                rank,
                int(train_pool_idx),
                train_ds_idx,
                train_sqlite_idx,
                train_track_ids[train_pool_idx],
                train_genres[train_pool_idx],
                float(col_scores[train_pool_idx]),
            ])

        per_test_rankings[str(test_idx)] = {
            'test_index': test_idx,
            'test_track_id': test_track_ids[test_idx],
            'test_genre': test_genres[test_idx],
            'topk': top_items,
            'bottomk': bottom_items,
        }

    per_test_rank_json_path = results_folder / f'{args.stage}_{args.method}_per_test_rankings{_shard_tag}.json'
    with open(str(per_test_rank_json_path), 'w') as f:
        json.dump(per_test_rankings, f, indent=2)

    per_test_rank_tsv_path = results_folder / f'{args.stage}_{args.method}_per_test_rankings{_shard_tag}.tsv'
    with open(str(per_test_rank_tsv_path), 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'test_index',
            'test_track_id',
            'test_genre',
            'rank_type',
            'rank',
            'train_pool_index',
            'train_ds_index',
            'train_sqlite_idx',
            'train_track_id',
            'train_genre',
            'score',
        ])
        writer.writerows(per_test_rank_rows)

    baseline_test_loss_mean = float(test_losses.mean()) if test_losses.size > 0 else None
    baseline_test_loss_std = float(test_losses.std()) if test_losses.size > 0 else None

    # save metadata
    metadata = {
        'run_id': run_id,
        'condition': args.condition,
        'stage': args.stage,
        'method': args.method,
        'num_train_samples': num_train,
        'num_test_samples': num_test,
        'checkpoint_path': args.checkpoint_path,
        'scores_shape': list(scores_np.shape),
        'attribution_time_seconds': attr_time,
        'baseline_test_loss_mean': baseline_test_loss_mean,
        'baseline_test_loss_std': baseline_test_loss_std,
        'top_influential_indices': top_ds_indices,
        'top_influential_sqlite_indices': top_sqlite_indices,
        'top_influential_track_ids': [train_track_ids[i] for i in top_indices.tolist()],
        'top_influential_scores': top_scores,
        'bottom_influential_indices': bottom_ds_indices,
        'bottom_influential_sqlite_indices': bottom_sqlite_indices,
        'bottom_influential_track_ids': [train_track_ids[i] for i in bottom_indices.tolist()],
        'bottom_influential_scores': bottom_scores,
        'train_sqlite_indices': train_sqlite_indices,
        'train_track_ids': train_track_ids,
        'train_genres': train_genres,
        'test_track_ids': test_track_ids,
        'test_genres': test_genres,
        'tracks_csv': args.tracks_csv,
        'genre_column': args.genre_column,
        'per_test_rank_k': per_test_rank_k,
        'per_test_rankings_json_path': str(per_test_rank_json_path),
        'per_test_rankings_tsv_path': str(per_test_rank_tsv_path),
    }
    if token_scores_np is not None:
        metadata['token_scores_shape'] = list(token_scores_np.shape)
        metadata['token_axis_path'] = str(token_axis_path)
    with open(str(results_folder / f'{args.stage}_{args.method}_metadata{_shard_tag}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nScore matrix shape: {scores_np.shape} (train x test)')
    print(f'Top {top_k} most influential training samples (ds_indices): {top_ds_indices}')
    print(f'Top {top_k} most influential training samples (sqlite_idx): {top_sqlite_indices}')
    print(f'Top {top_k} most influential training samples (track_ids): {[train_track_ids[i] for i in top_indices.tolist()]}')
    print(f'Top {top_k} most influential scores (mean over test): {[float(s) for s in top_scores]}')
    print(f'Top {top_k} least influential training samples (ds_indices): {bottom_ds_indices}')
    print(f'Top {top_k} least influential training samples (sqlite_idx): {bottom_sqlite_indices}')
    print(f'Top {top_k} least influential training samples (track_ids): {[train_track_ids[i] for i in bottom_indices.tolist()]}')
    print(f'Top {top_k} least influential scores (mean over test): {[float(s) for s in bottom_scores]}')
    print(f'Per-test top/bottom-{per_test_rank_k} rankings saved to {per_test_rank_json_path}')
    print(f'Per-test ranking table saved to {per_test_rank_tsv_path}')
    if num_test > 0 and per_test_rank_k > 0:
        first_top = per_test_rankings['0']['topk'][0]
        first_bottom = per_test_rankings['0']['bottomk'][0]
        print(
            f'Example test[0]={test_track_ids[0]} top1 train={first_top["train_track_id"]} '
            f'(score={first_top["score"]:.6f}), bottom1 train={first_bottom["train_track_id"]} '
            f'(score={first_bottom["score"]:.6f})'
        )
    if token_scores_np is not None:
        print(f'Token-level scores saved to {results_folder / f"{args.stage}_{args.method}_token_scores.npy"}')
        print(f'Token-level axis mapping saved to {token_axis_path}')

    if print_all_scores:
        print('\n=== Full attribution score matrix (train x test) ===')
        print(np.array2string(scores_np, threshold=np.inf, max_line_width=240))
        print('\nTrain axis mapping (row -> ds_index, sqlite_idx, track_id):')
        for i, (ds_idx, sqlite_idx, track_id) in enumerate(zip(train_indices.tolist(), train_sqlite_indices, train_track_ids)):
            print(f'  row {i}: ds_index={ds_idx}, sqlite_idx={sqlite_idx}, track_id={track_id}')
        print('Test axis mapping (col -> test_index, track_id):')
        for j, track_id in enumerate(test_track_ids):
            print(f'  col {j}: test_index={j}, track_id={track_id}')
        if token_scores_np is not None:
            print('\n=== Full token-level attribution score matrix (train x test_token) ===')
            print(np.array2string(token_scores_np, threshold=np.inf, max_line_width=240))
    print(f'Results saved to {results_folder}')

    if args.wandb:
        import wandb
        import time as _t
        wandb.log({
            "attribution/time_seconds": attr_time,
            "attribution/time_minutes": attr_time / 60,
            "attribution/scores_mean": float(scores_np.mean()),
            "attribution/scores_std": float(scores_np.std()),
            "attribution/num_train": num_train,
            "attribution/num_test": num_test,
            "timing/total_seconds": _t.time() - run_start_ref,
        })
        if args.alert_every > 0:
            wandb.alert(
                title=f"Attribution complete: {args.method} on {args.stage}",
                text=(f"scores shape={scores_np.shape} time={attr_time/60:.0f}min\n"
                      f"baseline test loss={baseline_test_loss_mean if baseline_test_loss_mean is not None else 'n/a'}"),
                level=wandb.AlertLevel.INFO,
            )
        wandb.finish()


def merge_sharded_results(results_folder: Path, stage: str, method: str, num_shards: int,
                           top_k_rankings: int = 50) -> None:
    """
    Merge per-shard .npz files into a single authoritative per_test_rankings.json
    and metadata.json after all shards have completed.

    Reads  : {stage}_{method}_scores_with_ids_shard_NNN_of_TTT.npz
    Writes : {stage}_{method}_per_test_rankings.json   (full top-K over all shards)
             {stage}_{method}_metadata.json            (updated train_track_ids / scores_shape)
    """
    shard_tag_fmt = f'_shard_{{:03d}}_of_{num_shards:03d}'

    # ── load and concatenate all shards in order ─────────────────────────────
    all_scores: list[np.ndarray] = []
    all_track_ids: list[str] = []

    for shard_id in range(num_shards):
        tag = shard_tag_fmt.format(shard_id)
        npz_path = results_folder / f'{stage}_{method}_scores_with_ids{tag}.npz'
        if not npz_path.exists():
            raise FileNotFoundError(f'Missing shard file: {npz_path}')
        data = np.load(str(npz_path), allow_pickle=True)
        all_scores.append(data['scores'])
        all_track_ids.extend(data['train_track_ids'].tolist())
        print(f'  Loaded shard {shard_id}: scores={data["scores"].shape}, '
              f'tracks={len(data["train_track_ids"])}')

    scores_full = np.concatenate(all_scores, axis=0)  # (total_train, n_test)
    print(f'  Concatenated score matrix: {scores_full.shape}')

    # ── aggregate clips → tracks via max-pool ────────────────────────────────
    track_to_clips: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(all_track_ids):
        track_to_clips[tid].append(i)
    unique_tracks = list(track_to_clips.keys())
    n_test = scores_full.shape[1]
    track_scores = np.zeros((len(unique_tracks), n_test), dtype=scores_full.dtype)
    for ti, tid in enumerate(unique_tracks):
        idxs = track_to_clips[tid]
        track_scores[ti] = scores_full[idxs, :].max(axis=0)
    print(f'  Track-level score matrix: {track_scores.shape}')

    # ── load test track IDs from any shard's metadata ────────────────────────
    shard0_meta_path = results_folder / f'{stage}_{method}_metadata{shard_tag_fmt.format(0)}.json'
    test_track_ids: list[str] = []
    test_genres: list[str] = []
    base_meta: dict = {}
    if shard0_meta_path.exists():
        base_meta = json.loads(shard0_meta_path.read_text())
        test_track_ids = base_meta.get('test_track_ids', [str(i) for i in range(n_test)])
        test_genres = base_meta.get('test_genres', [''] * n_test)
    else:
        test_track_ids = [str(i) for i in range(n_test)]
        test_genres = [''] * n_test

    # ── compute global per-test top-K rankings ────────────────────────────────
    eff_k = min(top_k_rankings, len(unique_tracks))
    per_test_rankings: dict[str, dict] = {}
    for test_idx in range(n_test):
        col = track_scores[:, test_idx]
        top_idx = np.argsort(-col)[:eff_k]
        per_test_rankings[str(test_idx)] = {
            'test_index': test_idx,
            'test_track_id': test_track_ids[test_idx] if test_idx < len(test_track_ids) else '',
            'test_genre': test_genres[test_idx] if test_idx < len(test_genres) else '',
            'topk': [
                {
                    'rank': r + 1,
                    'train_pool_index': int(top_idx[r]),
                    'train_track_id': unique_tracks[top_idx[r]],
                    'score': float(col[top_idx[r]]),
                }
                for r in range(len(top_idx))
            ],
        }

    rankings_path = results_folder / f'{stage}_{method}_per_test_rankings.json'
    rankings_path.write_text(json.dumps(per_test_rankings, indent=2))
    print(f'  Written: {rankings_path.name}  ({len(per_test_rankings)} queries, top-{eff_k})')

    # ── write merged metadata ─────────────────────────────────────────────────
    meta_out = dict(base_meta)
    meta_out['train_track_ids'] = all_track_ids
    meta_out['scores_shape'] = list(scores_full.shape)
    meta_out['num_train_samples'] = len(all_track_ids)
    meta_out['per_test_rank_k'] = eff_k
    meta_out['per_test_rankings_json_path'] = str(rankings_path)
    meta_out['num_shards_merged'] = num_shards

    meta_path = results_folder / f'{stage}_{method}_metadata.json'
    meta_path.write_text(json.dumps(meta_out, indent=2))
    print(f'  Written: {meta_path.name}')


__all__ = [
    'normalize_score_shape',
    'save_attribution_results',
    'merge_sharded_results',
]
