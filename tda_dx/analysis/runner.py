#!/usr/bin/env python3
"""TDA-Dx analysis runner — orchestrates per-condition stage analysis and writes paper tables."""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from tda_dx.analysis.io_utils import (
    _safe_float, _l1_normalize, _mean, _std, _pearson, _spearman, _rankdata,
    _gini, _entropy, _top_mass, _cosine_similarity,
    _looks_like_stage_dir, _parse_stage_method_filename, _select_header_column,
    load_json, dump_json, dump_tsv, find_one, find_main_scores_npy, find_sharded_npy,
    load_npy_or_shards, get_npy_shape, discover_stage_method_runs,
    select_primary_method, discover_main_stage_dirs, discover_condition_stage_dirs,
    load_track_to_genre_map, resolve_tracks_csv_override, resolve_track_genres,
    load_existing_stage_artifacts, normalize_score_matrix, normalize_score_column,
    aggregate_stage, load_stage_result,
)
from tda_dx.analysis._common import (
    Q1_BASELINE_CONDITIONS, DEFAULT_TOP_K, DEFAULT_PAIRWISE_TOP_K,
    TOP_TRAIN_TRACK_LIMIT, DEFAULT_RANK_SELECTION,
    FEATURE_DIMS, FEATURE_DIMS_V2, EVIDENCE_CHANNELS, EVIDENCE_CHANNELS_V2,
    N_BG_SAMPLES, MAIN_BUCKET, APPENDIX_BUCKET,
    _genre_match_ratio, _select_rank_indices,
)
from tda_dx.analysis.baselines import ensure_generated_baselines, STAGES
from tda_dx.analysis.summary import build_enrichment_summary
from tda_dx.analysis.homogeneity import (
    load_feature_table, build_feature_context,
    compute_track_similarity, compute_group_profile_strength,
)
from tda_dx.analysis.topk import build_topk_train_similarities
from tda_dx.analysis.reporting import generate_all_paper_tables
from tda_dx.analysis import summary as _summary_mod
from tda_dx.analysis import _common as _common_mod
from tda_dx.analysis import homogeneity as _homogeneity_mod
from tda_dx.analysis import topk as _topk_mod


# Modules that bind FEATURE_DIMS / EVIDENCE_CHANNELS by value at import time.
# Switching feature sets has to rebind the name inside each of them; patching
# only this module's globals leaves the consumers on the v1 lists.
_FEATURE_SET_HOLDERS = (_common_mod, _summary_mod, _homogeneity_mod, _topk_mod)


def _apply_feature_set_v2() -> None:
    global FEATURE_DIMS, EVIDENCE_CHANNELS
    FEATURE_DIMS = FEATURE_DIMS_V2
    EVIDENCE_CHANNELS = EVIDENCE_CHANNELS_V2
    for module in _FEATURE_SET_HOLDERS:
        if hasattr(module, "FEATURE_DIMS"):
            module.FEATURE_DIMS = FEATURE_DIMS_V2
        if hasattr(module, "EVIDENCE_CHANNELS"):
            module.EVIDENCE_CHANNELS = EVIDENCE_CHANNELS_V2


def build_stage_metrics(
    stage_data: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = stage_data["scores"]
    train_track_ids = stage_data["track_ids"]
    train_genres = stage_data["track_genres"]
    test_track_ids = stage_data["test_track_ids"]
    test_genres = stage_data["test_genres"]
    test_losses = stage_data["test_losses"]
    stage = stage_data["stage"]
    condition = stage_data["condition"]
    method = stage_data["method"]
    per_test_rows: list[dict[str, Any]] = []
    for test_idx, test_track_id in enumerate(test_track_ids):
        col_scores = np.asarray(scores[:, test_idx], dtype=np.float64)
        abs_scores = np.abs(col_scores).tolist()
        raw_scores = col_scores.tolist()
        abs_order = np.argsort(-np.abs(col_scores))
        raw_order = np.argsort(-col_scores)
        # concentration metrics use abs magnitude (sign-agnostic)
        # genre_match uses raw score order so only positively influential tracks are considered
        raw_ranked_items = [
            {"train_track_id": train_track_ids[i], "train_genre": train_genres[i], "score": float(col_scores[i])}
            for i in raw_order.tolist()
        ]
        top1_idx = int(abs_order[0]) if len(abs_order) else -1
        entropy, normalized_entropy = _entropy(abs_scores)
        gini = _gini(abs_scores)
        top5_mass = _top_mass(abs_scores, 5)
        top10_mass = _top_mass(abs_scores, 10)
        score_mean = _mean(raw_scores)
        score_std = _std(raw_scores, score_mean)
        test_genre = test_genres[test_idx] if test_idx < len(test_genres) else ""
        genre_match_top5 = _genre_match_ratio(raw_ranked_items, test_genre, 5)
        genre_match_top10 = _genre_match_ratio(raw_ranked_items, test_genre, 10)
        loss = float(test_losses[test_idx]) if test_idx < len(test_losses) else None
        per_test_rows.append({
            "condition": condition, "stage": stage, "method": method,
            "test_index": test_idx, "test_track_id": test_track_id, "test_genre": test_genre,
            "top1_train_track_id": train_track_ids[top1_idx] if top1_idx >= 0 else "",
            "top1_train_genre": train_genres[top1_idx] if top1_idx >= 0 else "",
            "top1_score": float(col_scores[top1_idx]) if top1_idx >= 0 else 0.0,
            "top5_mass": top5_mass, "top10_mass": top10_mass,
            "entropy": entropy, "normalized_entropy": normalized_entropy, "gini": gini,
            "positive_ratio": float(np.mean(col_scores > 0.0)),
            "score_mean": score_mean, "score_std": score_std,
            "score_range": float(np.max(col_scores) - np.min(col_scores)),
            "genre_match_top5": genre_match_top5, "genre_match_top10": genre_match_top10,
            "loss": loss,
        })
    summary = {
        "condition": condition, "stage": stage, "method": method,
        "test_count": len(per_test_rows),
        "track_count": int(scores.shape[0]),
        "mean_top5_mass": _mean([row["top5_mass"] for row in per_test_rows]),
        "mean_top10_mass": _mean([row["top10_mass"] for row in per_test_rows]),
        "mean_gini": _mean([row["gini"] for row in per_test_rows]),
        "mean_normalized_entropy": _mean([row["normalized_entropy"] for row in per_test_rows]),
        "mean_positive_ratio": _mean([row["positive_ratio"] for row in per_test_rows]),
        "mean_score_std": _mean([row["score_std"] for row in per_test_rows]),
        "mean_genre_match_top5": _mean([row["genre_match_top5"] for row in per_test_rows if row["genre_match_top5"] is not None]),
        "mean_genre_match_top10": _mean([row["genre_match_top10"] for row in per_test_rows if row["genre_match_top10"] is not None]),
        "mean_loss": _mean([row["loss"] for row in per_test_rows if row["loss"] is not None]),
    }
    return per_test_rows, summary


def load_self_sanity_artifacts(dataset_dir: Path, stage: str, method: str) -> dict[str, Any] | None:
    metrics_path = dataset_dir / f"{stage}_{method}_self_sanity_metrics.json"
    if not metrics_path.exists():
        return None
    payload: dict[str, Any] = {"metrics": load_json(metrics_path)}
    percentiles_path = dataset_dir / f"{stage}_{method}_self_diag_percentiles.npy"
    if percentiles_path.exists():
        payload["diag_percentiles"] = np.load(str(percentiles_path)).tolist()
    ranks_path = dataset_dir / f"{stage}_{method}_self_diag_ranks.npy"
    if ranks_path.exists():
        payload["diag_ranks"] = np.load(str(ranks_path)).tolist()
    return payload


def build_q1_self_sanity_summary(
    condition_outputs: list[dict[str, Any]],
    allowed_conditions: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in condition_outputs:
        if allowed_conditions is not None and output["condition"] not in allowed_conditions:
            continue
        for stage, method_map in output.get("stage_data_by_stage_and_method", {}).items():
            for method, stage_data in method_map.items():
                sanity = load_self_sanity_artifacts(Path(stage_data["dataset_dir"]), stage, method)
                if sanity is None:
                    continue
                metrics = sanity.get("metrics", {})
                diag_ranks = sanity.get("diag_ranks", [])
                diag_percentiles = sanity.get("diag_percentiles", [])
                top1_rate = _safe_float(metrics.get("top1_hit_rate_strict"))
                rows.append({
                    "condition": output["condition"], "stage": stage, "method": method,
                    "self_top1_rate": top1_rate,
                    "diag_rank_median": float(np.median(diag_ranks)) if diag_ranks else 0.0,
                    "diag_percentile_median": float(np.median(diag_percentiles)) if diag_percentiles else 0.0,
                    "sanity_verdict": "pass" if top1_rate >= 0.9 else "fail",
                })
    return rows


def build_q1_method_consistency(
    condition_outputs: list[dict[str, Any]],
    allowed_conditions: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in condition_outputs:
        if allowed_conditions is not None and output["condition"] not in allowed_conditions:
            continue
        for stage, method_map in output.get("stage_data_by_stage_and_method", {}).items():
            methods = sorted(method_map)
            if len(methods) < 2:
                continue
            for method_a, method_b in combinations(methods, 2):
                _, pair_summary_10, _ = build_pairwise_metrics(method_map[method_a], method_map[method_b], top_k=10)
                _, pair_summary, _ = build_pairwise_metrics(method_map[method_a], method_map[method_b], top_k=DEFAULT_PAIRWISE_TOP_K)
                rows.append({
                    "condition": output["condition"], "stage": stage,
                    "method_a": method_a, "method_b": method_b,
                    "mean_jaccard_at_10": pair_summary_10["mean_jaccard"],
                    "mean_overlap_ratio_a_at_10": pair_summary_10["mean_overlap_ratio_a"],
                    "mean_overlap_ratio_b_at_10": pair_summary_10["mean_overlap_ratio_b"],
                    "mean_jaccard_at_50": pair_summary["mean_jaccard"],
                    "mean_overlap_ratio_a_at_50": pair_summary["mean_overlap_ratio_a"],
                    "mean_overlap_ratio_b_at_50": pair_summary["mean_overlap_ratio_b"],
                    "mean_pearson": pair_summary["mean_pearson"],
                    "mean_spearman": pair_summary["mean_spearman"],
                })
    return rows


def _compare_stage_metrics(main_row: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "mean_top5_mass_delta": _safe_float(main_row.get("mean_top5_mass")) - _safe_float(baseline_row.get("mean_top5_mass")),
        "mean_gini_delta": _safe_float(main_row.get("mean_gini")) - _safe_float(baseline_row.get("mean_gini")),
        "mean_genre_match_top5_delta": _safe_float(main_row.get("mean_genre_match_top5")) - _safe_float(baseline_row.get("mean_genre_match_top5")),
        # reversed (baseline - main) so that positive = main is more concentrated (lower entropy = better signal)
        "mean_normalized_entropy_delta": _safe_float(baseline_row.get("mean_normalized_entropy")) - _safe_float(main_row.get("mean_normalized_entropy")),
    }
    positive = sum(1 for v in comparisons.values() if v > 0.0)
    negative = sum(1 for v in comparisons.values() if v < 0.0)
    comparisons["positive_metric_count"] = positive
    comparisons["negative_metric_count"] = negative
    comparisons["support_ratio"] = positive / (positive + negative) if (positive + negative) else 0.0
    return comparisons


def build_q1_baseline_comparison(
    condition_outputs: list[dict[str, Any]],
    available_conditions: list[str],
) -> list[dict[str, Any]]:
    outputs_by_condition = {payload["condition"]: payload for payload in condition_outputs}
    main_output = outputs_by_condition.get("main")
    if main_output is None:
        return []
    baseline_conditions = [c for c in available_conditions if c in Q1_BASELINE_CONDITIONS]
    main_stage_rows = {(row["stage"], row.get("method", "")): row for row in main_output.get("stage_metrics", [])}
    rows: list[dict[str, Any]] = []
    for baseline_condition in baseline_conditions:
        baseline_output = outputs_by_condition.get(baseline_condition)
        if baseline_output is None:
            continue
        baseline_stage_rows = {(row["stage"], row.get("method", "")): row for row in baseline_output.get("stage_metrics", [])}
        for stage, methods in main_output.get("stage_methods", {}).items():
            for method in methods:
                main_row = main_stage_rows.get((stage, method))
                baseline_row = baseline_stage_rows.get((stage, method))
                if main_row is None or baseline_row is None:
                    continue
                comparison = _compare_stage_metrics(main_row, baseline_row)
                rows.append({
                    "baseline_condition": baseline_condition, "stage": stage, "method": method,
                    "main_mean_top5_mass": _safe_float(main_row.get("mean_top5_mass")),
                    "baseline_mean_top5_mass": _safe_float(baseline_row.get("mean_top5_mass")),
                    "main_mean_gini": _safe_float(main_row.get("mean_gini")),
                    "baseline_mean_gini": _safe_float(baseline_row.get("mean_gini")),
                    "main_mean_normalized_entropy": _safe_float(main_row.get("mean_normalized_entropy")),
                    "baseline_mean_normalized_entropy": _safe_float(baseline_row.get("mean_normalized_entropy")),
                    "main_mean_genre_match_top5": _safe_float(main_row.get("mean_genre_match_top5")),
                    "baseline_mean_genre_match_top5": _safe_float(baseline_row.get("mean_genre_match_top5")),
                    **comparison,
                })
    return rows


# ─── Component 2: Cross-stage overlap & rank correlation ──────────────────────

def align_pair(stage_a: dict[str, Any], stage_b: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    tracks_a = stage_a["track_ids"]
    tracks_b = stage_b["track_ids"]
    common = sorted(set(tracks_a) & set(tracks_b), key=lambda v: int(v) if v.isdigit() else v)
    index_a = {track_id: idx for idx, track_id in enumerate(tracks_a)}
    index_b = {track_id: idx for idx, track_id in enumerate(tracks_b)}
    aligned_a = stage_a["scores"][[index_a[t] for t in common]]
    aligned_b = stage_b["scores"][[index_b[t] for t in common]]
    return common, aligned_a, aligned_b


def build_pairwise_metrics(
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
    top_k: int,
    rank_selection: str = DEFAULT_RANK_SELECTION,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    common_track_ids, aligned_a, aligned_b = align_pair(stage_a, stage_b)
    test_track_ids = stage_a["test_track_ids"]
    test_genres = stage_a["test_genres"]
    condition = stage_a["condition"]
    pair_rows: list[dict[str, Any]] = []
    for test_idx, test_track_id in enumerate(test_track_ids):
        xs = np.asarray(aligned_a[:, test_idx], dtype=np.float64)
        ys = np.asarray(aligned_b[:, test_idx], dtype=np.float64)
        order_a = _select_rank_indices(xs, len(xs), rank_selection)
        order_b = _select_rank_indices(ys, len(ys), rank_selection)
        top_a = set(order_a[:top_k].tolist())
        top_b = set(order_b[:top_k].tolist())
        inter = top_a & top_b
        union = top_a | top_b
        rank_pos_a = np.empty(len(order_a), dtype=np.int64)
        rank_pos_b = np.empty(len(order_b), dtype=np.int64)
        rank_pos_a[order_a] = np.arange(1, len(order_a) + 1)
        rank_pos_b[order_b] = np.arange(1, len(order_b) + 1)
        pair_rows.append({
            "condition": condition,
            "stage_a": stage_a["stage"], "stage_b": stage_b["stage"],
            "rank_selection": rank_selection,
            "test_index": test_idx, "test_track_id": test_track_id,
            "test_genre": test_genres[test_idx] if test_idx < len(test_genres) else "",
            "shared_track_count": int(len(common_track_ids)),
            "pearson": _pearson(xs.tolist(), ys.tolist()),
            "spearman": _spearman(xs.tolist(), ys.tolist()),
            "topk_jaccard": (len(inter) / len(union)) if union else 0.0,
            "overlap_ratio_a": (len(inter) / len(top_a)) if top_a else 0.0,
            "overlap_ratio_b": (len(inter) / len(top_b)) if top_b else 0.0,
            "topk_intersection": int(len(inter)),
            "sign_flip_ratio": float(np.mean(xs * ys < 0.0)),
            "normalized_rank_shift": (float(np.mean(np.abs(rank_pos_a - rank_pos_b))) / len(common_track_ids)) if common_track_ids else 0.0,
        })
    pearson_vals = [row["pearson"] for row in pair_rows if row["pearson"] is not None]
    spearman_vals = [row["spearman"] for row in pair_rows if row["spearman"] is not None]
    jaccard_vals = [row["topk_jaccard"] for row in pair_rows]
    overlap_a_vals = [row["overlap_ratio_a"] for row in pair_rows]
    overlap_b_vals = [row["overlap_ratio_b"] for row in pair_rows]
    rank_shift_vals = [row["normalized_rank_shift"] for row in pair_rows]
    summary = {
        "condition": condition,
        "stage_a": stage_a["stage"], "stage_b": stage_b["stage"],
        "rank_selection": rank_selection,
        "test_count": len(pair_rows),
        "shared_track_count": int(len(common_track_ids)),
        "mean_pearson": _mean(pearson_vals),
        "mean_spearman": _mean(spearman_vals),
        "median_spearman": float(np.median(spearman_vals)) if spearman_vals else 0.0,
        "mean_jaccard": _mean(jaccard_vals),
        "median_jaccard": float(np.median(jaccard_vals)) if jaccard_vals else 0.0,
        "mean_overlap_ratio_a": _mean(overlap_a_vals),
        "mean_overlap_ratio_b": _mean(overlap_b_vals),
        "mean_normalized_rank_shift": _mean(rank_shift_vals),
    }
    disagreement = sorted(
        pair_rows,
        key=lambda r: (r["topk_jaccard"], float(r["spearman"]) if r["spearman"] is not None else 1e9, r["test_track_id"]),
    )[:10]
    agreement = sorted(
        pair_rows,
        key=lambda r: (-r["topk_jaccard"], -(float(r["spearman"]) if r["spearman"] is not None else -1e9), r["test_track_id"]),
    )[:10]
    return pair_rows, summary, {"disagreement_examples": disagreement, "agreement_examples": agreement}

# ─── Main orchestration ───────────────────────────────────────────────────────

def analyze_condition(
    condition: str,
    stage_dirs: dict[str, Path],
    output_root: Path,
    tracks_csv_override: Path | None,
    feature_table: dict[str, dict[str, Any]],
    top_k: int,
    rank_selection: str,
    num_workers: int,
    compute_loo: bool,
    pairwise_top_k: int,
    available_conditions: list[str],
    track_aggregation: str,
    score_normalization: str,
    skip_appendix: bool = False,
) -> dict[str, Any]:
    stage_data_by_stage_and_method: dict[str, dict[str, dict[str, Any]]] = {}
    primary_stage_data: dict[str, dict[str, Any]] = {}
    stage_methods: dict[str, list[str]] = {}

    for stage in [s for s in STAGES if s in stage_dirs]:
        runs = discover_stage_method_runs(stage_dirs[stage], stage)
        methods = sorted(runs)
        if not methods:
            continue
        stage_methods[stage] = methods
        stage_data_by_stage_and_method[stage] = {}
        for method in methods:
            stage_data = load_stage_result(
                stage_dirs[stage],
                stage,
                condition,
                method,
                tracks_csv_override,
                track_aggregation=track_aggregation,
                score_normalization=score_normalization,
            )
            stage_data_by_stage_and_method[stage][method] = stage_data
        primary_method = select_primary_method(methods)
        if primary_method is not None:
            primary_stage_data[stage] = stage_data_by_stage_and_method[stage][primary_method]

    ordered_stages = [s for s in STAGES if s in primary_stage_data]
    main_condition_dir = output_root / MAIN_BUCKET / condition
    appendix_condition_dir = output_root / APPENDIX_BUCKET / condition
    main_condition_dir.mkdir(parents=True, exist_ok=True)
    if not skip_appendix:
        appendix_condition_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] Condition '{condition}' — stages: {ordered_stages}", flush=True)

    # ── Appendix-only: cross-stage overlap & rank correlation ─────────────────
    pairwise_metric_rows: list[dict[str, Any]] = []
    pairwise_summary_rows: list[dict[str, Any]] = []
    if not skip_appendix:
        print(f"[analyze]   Appendix: cross-stage overlap & rank correlation ...", flush=True)
        for stage_a, stage_b in combinations(ordered_stages, 2):
            pair_rows, pair_summary, _ = build_pairwise_metrics(
                primary_stage_data[stage_a], primary_stage_data[stage_b], top_k=pairwise_top_k, rank_selection=rank_selection,
            )
            pairwise_metric_rows.extend(pair_rows)
            pairwise_summary_rows.append(pair_summary)
        if pairwise_metric_rows:
            dump_tsv(appendix_condition_dir / "pairwise_stage_metrics.tsv", list(pairwise_metric_rows[0].keys()), pairwise_metric_rows)
        if pairwise_summary_rows:
            dump_tsv(appendix_condition_dir / "pairwise_stage_summary.tsv", list(pairwise_summary_rows[0].keys()), pairwise_summary_rows)

    # ── Main: set-level evidence alignment ────────────────────────────────────
    if feature_table:
        print(
            f"[analyze]   Main: set-level evidence alignment / enrichment z-scores "
            f"(rank_selection={rank_selection}) ...",
            flush=True,
        )
        feature_context = build_feature_context(feature_table)
        feat_sim_rows: list[dict[str, Any]] = []
        feat_align_summaries: list[dict[str, Any]] = []
        for stage in ordered_stages:
            n_test = len(primary_stage_data[stage].get("test_track_ids", []))
            loo_label = "on" if compute_loo else "off"
            print(f"[analyze]     stage={stage}, {n_test} test queries × {N_BG_SAMPLES} bg samples (workers={max(1, int(num_workers))}, loo={loo_label}) ...", flush=True)
            per_test_sim_rows, sim_summary = build_topk_train_similarities(
                primary_stage_data[stage], top_k=top_k, rank_selection=rank_selection,
                feature_table=feature_table, feature_context=feature_context,
                num_workers=num_workers, compute_loo=compute_loo,
            )
            print(f"[analyze]     stage={stage} done.", flush=True)
            feat_sim_rows.extend(per_test_sim_rows)
            feat_align_summaries.append(sim_summary)
        if feat_sim_rows:
            dump_tsv(main_condition_dir / "topk_train_similarities.tsv", list(feat_sim_rows[0].keys()), feat_sim_rows)
        if feat_align_summaries:
            dump_tsv(main_condition_dir / "feature_alignment_summary.tsv", list(feat_align_summaries[0].keys()), feat_align_summaries)

    return {
        "condition": condition,
        "available_stages": ordered_stages,
        "main_condition_dir": str(main_condition_dir),
        "appendix_condition_dir": str(appendix_condition_dir),
        "pairwise_summary": pairwise_summary_rows,
        "primary_stage_data": primary_stage_data,
        "stage_data_by_stage_and_method": stage_data_by_stage_and_method,
        "stage_methods": stage_methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze attribution results with main evidence outputs and appendix diagnostics."
    )
    parser.add_argument("--input-results-dir", required=True, help="Results root containing stage folders.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <input-results-dir>/analysis")
    parser.add_argument("--features-tsv", default=None, help="Path to features.tsv for musicological feature alignment (Component 3).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k for per-test signal summaries.")
    parser.add_argument(
        "--rank-selection",
        choices=("top", "bottom"),
        default=DEFAULT_RANK_SELECTION,
        help="Select highest-scoring tracks (top) or lowest-scoring tracks (bottom) while keeping the rest of the analyzer pipeline unchanged.",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Worker count for per-query feature alignment analysis. Deterministic across worker counts.")
    parser.add_argument("--skip-loo", action="store_true", help="Skip leave-one-out contribution analysis while preserving output columns as empty values.")
    parser.add_argument("--pairwise-top-k", type=int, default=DEFAULT_PAIRWISE_TOP_K, help="Top-k for cross-stage overlap metrics.")
    parser.add_argument("--tracks-csv", default=None, help="Optional tracks.csv override for genre lookup.")
    parser.add_argument(
        "--track-aggregation",
        choices=("sum", "mean"),
        default="sum",
        help="How to combine multiple sequence-level scores for the same train track.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=("none", "zscore_per_test", "l2_per_test", "minmax_per_test", "rank_per_test"),
        default="none",
        help="Optional normalization applied to each test column before track-level aggregation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random_baseline and shuffle_train generation.")
    parser.add_argument(
        "--feature-set",
        choices=["v1", "v2"],
        default="v1",
        help="Feature set: v1 = original 7 features, v2 = 9-feature set (onset/beat rhythm + chroma/tonnetz harmony + cqt/mfcc/spectral_centroid timbre).",
    )
    parser.add_argument("--skip-baseline-generation", action="store_true", help="Skip baseline materialization.")
    parser.add_argument("--force-baseline-regenerate", action="store_true", help="Regenerate baselines even if they already exist.")
    parser.add_argument("--skip-appendix", action="store_true", help="Skip all appendix computations (pairwise stage metrics, self-sanity, method consistency). Safe to use when only main paper tables are needed.")
    args = parser.parse_args()

    if args.feature_set == "v2":
        _apply_feature_set_v2()

    input_results_dir = Path(args.input_results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        input_results_dir / ("analysis-v2" if args.feature_set == "v2" else "analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv_override = Path(args.tracks_csv).resolve() if args.tracks_csv else None
    print(f"[analyze] Input:  {input_results_dir}", flush=True)
    print(f"[analyze] Output: {output_dir}", flush=True)
    if args.features_tsv:
        print(f"[analyze] Loading feature table from {args.features_tsv} ...", flush=True)
    feature_table = load_feature_table(Path(args.features_tsv).resolve()) if args.features_tsv else {}
    if feature_table:
        print(f"[analyze] Feature table loaded: {len(feature_table)} tracks", flush=True)
    elif args.features_tsv:
        print(f"[analyze] WARNING: feature table is empty (check path/format)", flush=True)

    if not args.skip_baseline_generation:
        print(f"[analyze] Generating baselines ...", flush=True)
        ensure_generated_baselines(
            input_results_dir=input_results_dir,
            tracks_csv_override=tracks_csv_override,
            seed=args.seed,
            force=args.force_baseline_regenerate,
        )

    condition_stage_dirs = discover_condition_stage_dirs(input_results_dir)
    if not condition_stage_dirs:
        raise RuntimeError(f"Could not find any condition/stage results under {input_results_dir}")
    available_conditions = ["main"] if "main" in condition_stage_dirs else []
    available_conditions.extend(sorted(c for c in condition_stage_dirs if c != "main"))
    print(f"[analyze] Conditions: {available_conditions}", flush=True)

    condition_outputs: list[dict[str, Any]] = []
    for condition in available_conditions:
        payload = analyze_condition(
            condition=condition,
            stage_dirs=condition_stage_dirs[condition],
            output_root=output_dir,
            tracks_csv_override=tracks_csv_override,
            feature_table=feature_table,
            top_k=args.top_k,
            rank_selection=args.rank_selection,
            num_workers=args.num_workers,
            compute_loo=not args.skip_loo,
            pairwise_top_k=args.pairwise_top_k,
            available_conditions=available_conditions,
            track_aggregation=args.track_aggregation,
            score_normalization=args.score_normalization,
            skip_appendix=args.skip_appendix,
        )
        condition_outputs.append(payload)

    appendix_root = output_dir / APPENDIX_BUCKET

    if not args.skip_appendix:
        # Appendix: sanity check (self top-1 rate)
        sanity_rows = build_q1_self_sanity_summary(condition_outputs, allowed_conditions={"main"})
        if sanity_rows:
            appendix_root.mkdir(parents=True, exist_ok=True)
            dump_tsv(appendix_root / "q1_self_sanity.tsv", list(sanity_rows[0].keys()), sanity_rows)

        # Appendix: cross-method consistency
        consistency_rows = build_q1_method_consistency(condition_outputs, allowed_conditions={"main"})
        if consistency_rows:
            appendix_root.mkdir(parents=True, exist_ok=True)
            dump_tsv(appendix_root / "q1_method_consistency.tsv", list(consistency_rows[0].keys()), consistency_rows)

    # Generate paper-ready tables if feature alignment was run
    if feature_table:
        print(f"[analyze] Generating paper tables ...", flush=True)
        generate_all_paper_tables(output_dir)

    print(f"[analyze] Writing summary files ...", flush=True)
    print(f"[analyze] ALL DONE", flush=True)
    print(f"  Input results:  {input_results_dir}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Conditions:     {', '.join(available_conditions)}")
    print(f"  Stages:         {', '.join(STAGES)}")
    print(f"  Rank select:    {args.rank_selection}")
    print(f"  Track agg:      {args.track_aggregation}")
    print(f"  Score norm:     {args.score_normalization}")
    if feature_table:
        print(f"  Feature table:  {len(feature_table)} tracks loaded")
    if feature_table:
        print("  Main outputs:         main/<condition>/topk_train_similarities.tsv, feature_alignment_summary.tsv")
    print("  Appendix outputs:     appendix/<condition>/pairwise_stage_metrics.tsv, pairwise_stage_summary.tsv")
    print("                        appendix/q1_self_sanity.tsv, appendix/q1_method_consistency.tsv")

