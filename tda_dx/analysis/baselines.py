

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from tda_dx.analysis.io_utils import *

# ─── Baseline generation ──────────────────────────────────────────────────────

def link_or_copy(src: Path, dst: Path) -> None:
    import os
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_run_id(condition: str, stage: str, method: str, seed: int) -> str:
    return "__".join([str(condition), str(stage), str(method), f"seed{int(seed)}"])


def write_per_test_rankings(
    output_dir: Path,
    stage: str,
    method: str,
    test_track_ids: list[str],
    test_genres: list[str],
    train_indices: np.ndarray,
    train_sqlite_indices: list[Any],
    train_track_ids: list[str],
    train_genres: list[str],
    scores: np.ndarray,
) -> tuple[Path, Path]:
    num_train, num_test = scores.shape
    per_test_rank_k = min(10, num_train)
    per_test_rankings: dict[str, Any] = {}
    per_test_rank_rows: list[list[Any]] = []
    for test_idx in range(num_test):
        col_scores = scores[:, test_idx]
        top_pool_indices = np.argsort(-col_scores)[:per_test_rank_k]
        bottom_pool_indices = np.argsort(col_scores)[:per_test_rank_k]
        top_items = []
        for rank, train_pool_idx in enumerate(top_pool_indices, start=1):
            train_ds_idx = int(train_indices[train_pool_idx])
            train_sqlite_idx = int(train_sqlite_indices[train_pool_idx])
            top_items.append({
                "rank": rank,
                "train_pool_index": int(train_pool_idx),
                "train_ds_index": train_ds_idx,
                "train_sqlite_idx": train_sqlite_idx,
                "train_track_id": train_track_ids[train_pool_idx],
                "train_genre": train_genres[train_pool_idx],
                "score": float(col_scores[train_pool_idx]),
            })
            per_test_rank_rows.append([
                test_idx, test_track_ids[test_idx], test_genres[test_idx], "top", rank,
                int(train_pool_idx), train_ds_idx, train_sqlite_idx,
                train_track_ids[train_pool_idx], train_genres[train_pool_idx],
                float(col_scores[train_pool_idx]),
            ])
        bottom_items = []
        for rank, train_pool_idx in enumerate(bottom_pool_indices, start=1):
            train_ds_idx = int(train_indices[train_pool_idx])
            train_sqlite_idx = int(train_sqlite_indices[train_pool_idx])
            bottom_items.append({
                "rank": rank,
                "train_pool_index": int(train_pool_idx),
                "train_ds_index": train_ds_idx,
                "train_sqlite_idx": train_sqlite_idx,
                "train_track_id": train_track_ids[train_pool_idx],
                "train_genre": train_genres[train_pool_idx],
                "score": float(col_scores[train_pool_idx]),
            })
            per_test_rank_rows.append([
                test_idx, test_track_ids[test_idx], test_genres[test_idx], "bottom", rank,
                int(train_pool_idx), train_ds_idx, train_sqlite_idx,
                train_track_ids[train_pool_idx], train_genres[train_pool_idx],
                float(col_scores[train_pool_idx]),
            ])
        per_test_rankings[str(test_idx)] = {
            "test_index": test_idx,
            "test_track_id": test_track_ids[test_idx],
            "test_genre": test_genres[test_idx],
            "topk": top_items,
            "bottomk": bottom_items,
        }
    per_test_rank_json_path = output_dir / f"{stage}_{method}_per_test_rankings.json"
    dump_json(per_test_rank_json_path, per_test_rankings)
    per_test_rank_tsv_path = output_dir / f"{stage}_{method}_per_test_rankings.tsv"
    with per_test_rank_tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "test_index", "test_track_id", "test_genre", "rank_type", "rank",
            "train_pool_index", "train_ds_index", "train_sqlite_idx",
            "train_track_id", "train_genre", "score",
        ])
        writer.writerows(per_test_rank_rows)
    return per_test_rank_json_path, per_test_rank_tsv_path


def _baseline_outputs_ready(output_dir: Path, stage: str, method: str, extra_files: list[str] | None = None) -> bool:
    scores_path = output_dir / f"{stage}_{method}_scores.npy"
    shards = find_sharded_npy(scores_path)
    scores_ok = (scores_path.exists() and scores_path.stat().st_size > 0) or any(s.stat().st_size > 0 for s in shards)
    required_single = [
        output_dir / f"{stage}_{method}_metadata.json",
        output_dir / "experiment_config.json",
        output_dir / "train_indices.npy",
    ]
    if extra_files:
        required_single.extend(output_dir / name for name in extra_files)
    return scores_ok and all(path.exists() for path in required_single)


def build_influence_summary(
    scores: np.ndarray,
    train_indices: np.ndarray,
    train_sqlite_indices: list[Any],
    train_track_ids: list[str],
    train_genres: list[str],
) -> dict[str, Any]:
    mean_influence = scores.mean(axis=1)
    top_k = min(10, len(mean_influence))
    top_indices = np.argsort(-mean_influence)[:top_k]
    bottom_indices = np.argsort(mean_influence)[:top_k]
    return {
        "top_influential_indices": train_indices[top_indices].tolist(),
        "top_influential_sqlite_indices": [train_sqlite_indices[i] for i in top_indices.tolist()],
        "top_influential_track_ids": [train_track_ids[i] for i in top_indices.tolist()],
        "top_influential_scores": mean_influence[top_indices].tolist(),
        "bottom_influential_indices": train_indices[bottom_indices].tolist(),
        "bottom_influential_sqlite_indices": [train_sqlite_indices[i] for i in bottom_indices.tolist()],
        "bottom_influential_track_ids": [train_track_ids[i] for i in bottom_indices.tolist()],
        "bottom_influential_scores": mean_influence[bottom_indices].tolist(),
    }


def ensure_random_baseline_for_stage(
    input_dir: Path,
    output_dir: Path,
    stage: str,
    seed: int,
    tracks_csv_override: Path | None,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_existing_stage_artifacts(input_dir, stage, None, tracks_csv_override)
    metadata = dict(artifacts["metadata"])
    experiment_config = dict(artifacts["experiment_config"])
    train_indices = load_npy_or_shards(artifacts["train_indices_path"])
    scores_shape = get_npy_shape(artifacts["scores_path"])
    if len(scores_shape) != 2:
        raise RuntimeError(f"Expected 2D score matrix for {input_dir}, got {scores_shape}")
    num_train, num_test = int(scores_shape[0]), int(scores_shape[1])
    method = str(artifacts["method"])
    output_scores_path = output_dir / f"{stage}_{method}_scores.npy"
    output_metadata_path = output_dir / f"{stage}_{method}_metadata.json"
    if _baseline_outputs_ready(output_dir, stage, method, extra_files=["test_losses.npy"]) and not force:
        return
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((num_train, num_test)).astype(np.float32, copy=False)
    np.save(str(output_scores_path), scores)
    np.save(str(output_dir / "test_losses.npy"), np.array([], dtype=np.float32))
    np.save(str(output_dir / "train_indices.npy"), train_indices)
    train_track_ids = [str(v) for v in metadata["train_track_ids"]]
    test_track_ids = [str(v) for v in metadata["test_track_ids"]]
    train_sqlite_indices = list(metadata["train_sqlite_indices"])
    train_genres = list(artifacts["train_genres"])
    test_genres = list(artifacts["test_genres"])
    summary = build_influence_summary(
        scores=scores, train_indices=train_indices,
        train_sqlite_indices=train_sqlite_indices,
        train_track_ids=train_track_ids, train_genres=train_genres,
    )
    run_id = build_run_id("random_baseline", stage, method, seed)
    metadata.update({
        "run_id": run_id, "condition": "random_baseline", "stage": stage, "method": method,
        "scores_shape": [num_train, num_test], "attribution_time_seconds": 0.0,
        "baseline_test_loss_mean": None, "baseline_test_loss_std": None,
        "train_genres": train_genres, "test_genres": test_genres,
        "tracks_csv": str(artifacts["tracks_csv_path"]) if artifacts["tracks_csv_path"] else metadata.get("tracks_csv", ""),
        "genre_column": str(artifacts["genre_column"]),
        **summary,
    })
    dump_json(output_metadata_path, metadata)
    experiment_config["run_id"] = run_id
    experiment_config["condition"] = "random_baseline"
    experiment_config["seed"] = int(seed)
    dump_json(output_dir / "experiment_config.json", experiment_config)


def ensure_shuffle_train_for_stage(
    input_dir: Path,
    output_dir: Path,
    stage: str,
    seed: int,
    tracks_csv_override: Path | None,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_existing_stage_artifacts(input_dir, stage, None, tracks_csv_override)
    metadata = dict(artifacts["metadata"])
    experiment_config = dict(artifacts["experiment_config"])
    method = str(artifacts["method"])
    output_metadata_path = output_dir / f"{stage}_{method}_metadata.json"
    output_scores_path = output_dir / f"{stage}_{method}_scores.npy"
    if _baseline_outputs_ready(output_dir, stage, method, extra_files=["train_permutation.npy"]) and not force:
        return
    # mmap to avoid loading the full attribution score matrix (can be 10s of GB) into RAM.
    # build_influence_summary only needs scores.mean(axis=1), which streams fine on a memmap.
    print(f"[baseline] {stage}: mmap-loading scores for shuffle_train ...", flush=True)
    scores = load_npy_or_shards(artifacts["scores_path"], mmap_mode="r")
    train_indices = load_npy_or_shards(artifacts["train_indices_path"])
    num_train, num_test = scores.shape
    train_sqlite_indices = list(metadata["train_sqlite_indices"])
    train_track_ids = [str(v) for v in metadata["train_track_ids"]]
    train_genres = list(artifacts["train_genres"])
    test_track_ids = [str(v) for v in metadata["test_track_ids"]]
    test_genres = list(artifacts["test_genres"])
    if not (len(train_sqlite_indices) == len(train_track_ids) == len(train_genres) == num_train):
        raise RuntimeError(f"Train metadata lengths do not match num_train in {input_dir}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_train)
    shuffled_train_sqlite_indices = [train_sqlite_indices[i] for i in permutation.tolist()]
    shuffled_train_track_ids = [train_track_ids[i] for i in permutation.tolist()]
    shuffled_train_genres = [train_genres[i] for i in permutation.tolist()]
    summary = build_influence_summary(
        scores=scores, train_indices=train_indices,
        train_sqlite_indices=shuffled_train_sqlite_indices,
        train_track_ids=shuffled_train_track_ids, train_genres=shuffled_train_genres,
    )
    run_id = build_run_id("shuffle_train", stage, method, seed)
    source_run_id = metadata.get("run_id", experiment_config.get("run_id"))
    source_condition = metadata.get("condition", experiment_config.get("condition"))
    metadata.update({
        "run_id": run_id, "condition": "shuffle_train",
        "train_sqlite_indices": shuffled_train_sqlite_indices,
        "train_track_ids": shuffled_train_track_ids,
        "train_genres": shuffled_train_genres, "test_genres": test_genres,
        "tracks_csv": str(artifacts["tracks_csv_path"]) if artifacts["tracks_csv_path"] else metadata.get("tracks_csv", ""),
        "genre_column": str(artifacts["genre_column"]),
        "source_results_dir": str(input_dir), "source_run_id": source_run_id,
        "source_condition": source_condition, "shuffle_seed": int(seed),
        "shuffle_permutation_path": str(output_dir / "train_permutation.npy"),
        **summary,
    })
    dump_json(output_metadata_path, metadata)
    experiment_config["run_id"] = run_id
    experiment_config["condition"] = "shuffle_train"
    experiment_config["shuffle_seed"] = int(seed)
    experiment_config["source_results_dir"] = str(input_dir)
    experiment_config["source_run_id"] = source_run_id
    experiment_config["source_condition"] = source_condition
    dump_json(output_dir / "experiment_config.json", experiment_config)
    np.save(output_dir / "train_permutation.npy", permutation)
    # Link scores instead of copying the full matrix (can be 100s of MB).
    src_scores = artifacts["scores_path"]
    if src_scores.exists():
        link_or_copy(src_scores, output_scores_path)
    else:
        for shard in find_sharded_npy(src_scores):
            link_or_copy(shard, output_dir / shard.name)
    np.save(str(output_dir / "train_indices.npy"), train_indices)
    if artifacts["test_losses_path"].exists():
        link_or_copy(artifacts["test_losses_path"], output_dir / "test_losses.npy")


def ensure_generated_baselines(
    input_results_dir: Path,
    tracks_csv_override: Path | None,
    seed: int,
    force: bool,
    generate_shuffle_train: bool = False,
) -> None:
    main_stage_dirs = discover_main_stage_dirs(input_results_dir)
    if not main_stage_dirs:
        raise RuntimeError(f"Could not find main stage result folders under {input_results_dir}")
    for stage, stage_dir in main_stage_dirs.items():
        ensure_random_baseline_for_stage(
            input_dir=stage_dir, output_dir=stage_dir / "random_baseline",
            stage=stage, seed=seed, tracks_csv_override=tracks_csv_override, force=force,
        )
        if not generate_shuffle_train:
            continue
        ensure_shuffle_train_for_stage(
            input_dir=stage_dir, output_dir=input_results_dir / "shuffle_train" / stage,
            stage=stage, seed=seed, tracks_csv_override=tracks_csv_override, force=force,
        )
