"""
attributor/pipeline/merge_shard_scores.py

Concatenate per-shard score files produced by `--mode full --num_shards N` into
the canonical un-tagged result files. Used by methods (e.g. LoGra) that don't
have a built-in cross-shard merge step.

For one (stage, method) pair, expects files in `results_folder`:
    <stage>_<method>_scores_shard_<sid>_of_<num>.npy
    <stage>_<method>_scores_with_ids_shard_<sid>_of_<num>.npz
    <stage>_<method>_metadata_shard_<sid>_of_<num>.json
    train_indices_shard_<sid>_of_<num>.npy

Produces:
    <stage>_<method>_scores.npy                   shape (sum_N_shard, num_test)
    <stage>_<method>_scores_with_ids.npz
    <stage>_<method>_metadata.json                aggregated
    <stage>_<method>_per_test_rankings.{json,tsv} top/bottom-k from final scores
    train_indices.npy

Usage (CLI):
    python -m attributor.pipeline.merge_shard_scores \\
        --results_root /path/.../logra-proj4096-damping0.1 \\
        --stage fine \\
        --method logra
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


_SHARD_RE = re.compile(r"_shard_(\d+)_of_(\d+)\.npy$")


def _find_shard_score_files(results_folder: Path, stage: str, method: str):
    """Return sorted list of per-shard score npy files, with parsed shard_ids."""
    pattern = f"{stage}_{method}_scores_shard_*_of_*.npy"
    files = list(results_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No shard score files matching {pattern} under {results_folder}"
        )
    parsed = []
    num_shards_set = set()
    for fp in files:
        m = _SHARD_RE.search(fp.name)
        if m is None:
            continue
        sid, total = int(m.group(1)), int(m.group(2))
        parsed.append((sid, total, fp))
        num_shards_set.add(total)
    if len(num_shards_set) != 1:
        raise RuntimeError(
            f"Inconsistent shard totals across files: {num_shards_set}"
        )
    num_shards = num_shards_set.pop()
    if len(parsed) != num_shards:
        present = sorted(s for s, _, _ in parsed)
        raise FileNotFoundError(
            f"Expected {num_shards} shards, found {len(parsed)}: shard_ids={present}"
        )
    parsed.sort(key=lambda x: x[0])
    return parsed, num_shards


def _shard_path(results_folder: Path, basename: str, sid: int, num: int):
    tag = f"_shard_{sid:03d}_of_{num:03d}"
    stem, _, ext = basename.rpartition(".")
    return results_folder / f"{stem}{tag}.{ext}"


def merge(results_folder: Path, stage: str, method: str, top_k: int = 10) -> None:
    score_shards, num_shards = _find_shard_score_files(results_folder, stage, method)
    print(f"[merge] {stage}/{method}: found {num_shards} shards")

    # --- 1. Concatenate scores ---
    score_arrays = []
    for sid, total, fp in score_shards:
        arr = np.load(str(fp))
        print(f"  shard {sid}: {fp.name}  shape={arr.shape}")
        score_arrays.append(arr)
    scores_np = np.concatenate(score_arrays, axis=0)
    num_train, num_test = scores_np.shape
    final_scores_path = results_folder / f"{stage}_{method}_scores.npy"
    np.save(str(final_scores_path), scores_np)
    print(f"[merge] wrote {final_scores_path}  shape={scores_np.shape}")

    # --- 2. Concatenate train_indices + train_track_ids (from .npz) ---
    train_indices_parts = []
    train_track_ids_parts = []
    for sid, total, _ in score_shards:
        ti_path = _shard_path(results_folder, "train_indices.npy", sid, total)
        train_indices_parts.append(np.load(str(ti_path)))
        npz_path = _shard_path(
            results_folder, f"{stage}_{method}_scores_with_ids.npz", sid, total
        )
        with np.load(str(npz_path), allow_pickle=True) as npz:
            train_track_ids_parts.append(npz["train_track_ids"])
    train_indices = np.concatenate(train_indices_parts, axis=0)
    train_track_ids = np.concatenate(train_track_ids_parts, axis=0)
    np.save(str(results_folder / "train_indices.npy"), train_indices)
    np.savez(
        str(results_folder / f"{stage}_{method}_scores_with_ids.npz"),
        scores=scores_np,
        train_track_ids=train_track_ids,
        train_indices=train_indices,
    )
    print(f"[merge] wrote train_indices.npy + scores_with_ids.npz")

    # --- 3. Aggregate metadata ---
    base_meta_path = _shard_path(
        results_folder, f"{stage}_{method}_metadata.json",
        score_shards[0][0], num_shards,
    )
    with open(str(base_meta_path)) as f:
        base_meta = json.load(f)

    # Concatenate train_sqlite_indices + train_genres across shards
    train_sqlite_indices_all = []
    train_genres_all = []
    attr_time_total = 0.0
    for sid, total, _ in score_shards:
        meta_p = _shard_path(
            results_folder, f"{stage}_{method}_metadata.json", sid, total
        )
        with open(str(meta_p)) as f:
            sm = json.load(f)
        train_sqlite_indices_all.extend(sm.get("train_sqlite_indices", []))
        train_genres_all.extend(sm.get("train_genres", []))
        attr_time_total += float(sm.get("attribution_time_seconds", 0.0))
    train_track_ids_list = train_track_ids.tolist()

    # Recompute top/bottom from merged scores
    mean_influence = scores_np.mean(axis=1)
    top_k_eff = min(top_k, num_train)
    top_idx = np.argsort(-mean_influence)[:top_k_eff]
    bot_idx = np.argsort(mean_influence)[:top_k_eff]

    test_track_ids = base_meta.get("test_track_ids", [])
    test_genres = base_meta.get("test_genres", [])

    metadata = {
        **base_meta,
        "num_train_samples": int(num_train),
        "num_test_samples": int(num_test),
        "scores_shape": [int(num_train), int(num_test)],
        "attribution_time_seconds": attr_time_total,
        "merged_from_shards": num_shards,
        "top_influential_indices": [int(train_indices[i]) for i in top_idx],
        "top_influential_sqlite_indices": [int(train_sqlite_indices_all[i]) for i in top_idx],
        "top_influential_track_ids": [str(train_track_ids_list[i]) for i in top_idx],
        "top_influential_scores": [float(mean_influence[i]) for i in top_idx],
        "bottom_influential_indices": [int(train_indices[i]) for i in bot_idx],
        "bottom_influential_sqlite_indices": [int(train_sqlite_indices_all[i]) for i in bot_idx],
        "bottom_influential_track_ids": [str(train_track_ids_list[i]) for i in bot_idx],
        "bottom_influential_scores": [float(mean_influence[i]) for i in bot_idx],
        "train_sqlite_indices": train_sqlite_indices_all,
        "train_track_ids": train_track_ids_list,
        "train_genres": train_genres_all,
    }

    # --- 4. Per-test rankings recomputation ---
    per_test_rank_k = min(top_k, num_train)
    per_test_rankings = {}
    per_test_rows = []
    for ti in range(num_test):
        col = scores_np[:, ti]
        top_pool = np.argsort(-col)[:per_test_rank_k]
        bot_pool = np.argsort(col)[:per_test_rank_k]

        def _row(rank_type, rank, pool_idx):
            return [
                ti,
                test_track_ids[ti] if ti < len(test_track_ids) else str(ti),
                test_genres[ti] if ti < len(test_genres) else "",
                rank_type,
                rank,
                int(pool_idx),
                int(train_indices[pool_idx]),
                int(train_sqlite_indices_all[pool_idx]),
                str(train_track_ids_list[pool_idx]),
                str(train_genres_all[pool_idx]) if pool_idx < len(train_genres_all) else "",
                float(col[pool_idx]),
            ]

        top_items = []
        for rank, pi in enumerate(top_pool, start=1):
            top_items.append({
                "rank": rank,
                "train_pool_index": int(pi),
                "train_ds_index": int(train_indices[pi]),
                "train_sqlite_idx": int(train_sqlite_indices_all[pi]),
                "train_track_id": str(train_track_ids_list[pi]),
                "train_genre": str(train_genres_all[pi]) if pi < len(train_genres_all) else "",
                "score": float(col[pi]),
            })
            per_test_rows.append(_row("top", rank, pi))

        bot_items = []
        for rank, pi in enumerate(bot_pool, start=1):
            bot_items.append({
                "rank": rank,
                "train_pool_index": int(pi),
                "train_ds_index": int(train_indices[pi]),
                "train_sqlite_idx": int(train_sqlite_indices_all[pi]),
                "train_track_id": str(train_track_ids_list[pi]),
                "train_genre": str(train_genres_all[pi]) if pi < len(train_genres_all) else "",
                "score": float(col[pi]),
            })
            per_test_rows.append(_row("bottom", rank, pi))

        per_test_rankings[str(ti)] = {
            "test_index": ti,
            "test_track_id": test_track_ids[ti] if ti < len(test_track_ids) else str(ti),
            "test_genre": test_genres[ti] if ti < len(test_genres) else "",
            "topk": top_items,
            "bottomk": bot_items,
        }

    rank_json = results_folder / f"{stage}_{method}_per_test_rankings.json"
    with open(str(rank_json), "w") as f:
        json.dump(per_test_rankings, f, indent=2)
    rank_tsv = results_folder / f"{stage}_{method}_per_test_rankings.tsv"
    with open(str(rank_tsv), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "test_index", "test_track_id", "test_genre",
            "rank_type", "rank",
            "train_pool_index", "train_ds_index", "train_sqlite_idx",
            "train_track_id", "train_genre", "score",
        ])
        w.writerows(per_test_rows)
    metadata["per_test_rank_k"] = per_test_rank_k
    metadata["per_test_rankings_json_path"] = str(rank_json)
    metadata["per_test_rankings_tsv_path"] = str(rank_tsv)

    with open(str(results_folder / f"{stage}_{method}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[merge] wrote metadata.json + per_test_rankings.{{json,tsv}}")
    print(f"[merge] DONE  final scores: ({num_train}, {num_test})  total compute: {attr_time_total:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True,
                   help="Root containing <stage>/ subdirs with shard files.")
    p.add_argument("--stage", required=True, choices=["semantic", "coarse", "fine"])
    p.add_argument("--method", required=True)
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    stage_dir = Path(args.results_root) / args.stage
    if not stage_dir.exists():
        raise FileNotFoundError(stage_dir)
    merge(stage_dir, args.stage, args.method, top_k=args.top_k)


if __name__ == "__main__":
    main()
