#!/usr/bin/env python3
"""
CLAP embedding alignment analysis for TRAK attribution.
Compares top-K attributed training audio against random samples
using CLAP cosine similarity, across all 3 model stages.

CLI: python analyzer/embedding_alignment.py [args]
Module: from analyzer.embedding_alignment import run_alignment_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow direct execution from repo root or analyzer/ directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ANALYZER = Path(__file__).resolve().parent
if str(_ANALYZER) not in sys.path:
    sys.path.insert(0, str(_ANALYZER))

from tda_dx.analysis.homogeneity import _compute_contrast_stats
from tda_dx.analysis.io_utils import _spearman, dump_json, load_stage_result, STAGES


# ─── CLAP I/O ─────────────────────────────────────────────────────────────────

def _load_clap_npz(path: Path) -> dict[str, np.ndarray]:
    """Load CLAP npz → {track_id: float64 embedding (D,)}."""
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k].astype(np.float64) for k in data.files}


def _build_normed_matrix(
    track_ids: list[str],
    clap_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, list[int]]:
    """Build L2-normalised embedding matrix for training tracks present in clap_dict.

    Returns (matrix (M, D), original_indices_into_track_ids).
    Tracks missing from clap_dict are silently skipped.
    """
    pairs = [(i, clap_dict[tid]) for i, tid in enumerate(track_ids) if tid in clap_dict]
    if not pairs:
        return None, []
    orig_idxs, embs = zip(*pairs)
    mat = np.stack(embs, axis=0)            # (M, D)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms, list(orig_idxs)


# ─── Core analysis ────────────────────────────────────────────────────────────

def compute_clap_alignment_zscores(
    stage_data: dict[str, Any],
    train_clap: dict[str, np.ndarray],
    conditioning_clap: dict[str, np.ndarray],
    generated_clap: dict[str, np.ndarray] | None = None,
    topk: int = 50,
    n_bg: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """For each test query compute CLAP alignment z-scores for TRAK top-k training tracks.

    group_value = mean cosine similarity (CLAP) of top-k training tracks to query audio.
    bg_values   = same metric for n_bg random sets of k training tracks (shared across queries).
    z-score via _compute_contrast_stats().

    Args:
        stage_data:        Output of load_stage_result() — contains scores, track_ids, test_track_ids.
        train_clap:        {track_id: embedding} for the full training set.
        conditioning_clap: {track_id: embedding} for test query conditioning audio.
        generated_clap:    {track_id: embedding} for decoded generated audio (optional).
        topk:              How many top TRAK training tracks to analyse.
        n_bg:              Number of random training subsets for the background distribution.
        seed:              RNG seed (background index sets are fixed and shared across queries).
    """
    scores = stage_data["scores"]           # (N_train, N_test) ndarray
    train_ids: list[str] = stage_data["track_ids"]
    test_ids: list[str] = stage_data["test_track_ids"]
    n_train = len(train_ids)
    effective_k = min(topk, n_train)

    # Build L2-normalised training embedding matrix once.
    train_normed, train_valid_orig_idxs = _build_normed_matrix(train_ids, train_clap)
    # Map original train index → row in train_normed
    orig_to_row: dict[int, int] = {orig: row for row, orig in enumerate(train_valid_orig_idxs)}

    # Pre-sample shared background index sets (each is a list of original train indices).
    rng = np.random.default_rng(seed)
    bg_index_sets: list[list[int]] = [
        rng.choice(n_train, size=effective_k, replace=False).tolist()
        for _ in range(n_bg)
    ]
    # Convert each bg set to row indices into train_normed (skip missing).
    bg_row_sets: list[list[int]] = [
        [orig_to_row[i] for i in idx_set if i in orig_to_row]
        for idx_set in bg_index_sets
    ]

    # Pre-compute within-group pairwise cosine similarity for each background group.
    # These are query-independent, so compute once and reuse across all test samples.
    def _mean_pairwise_cosine_sim(row_idxs: list[int]) -> float | None:
        if len(row_idxs) < 2 or train_normed is None:
            return None
        sub = train_normed[row_idxs]          # (k, D), already L2-normalised
        sim_mat = sub @ sub.T                  # (k, k) cosine similarity matrix
        k = len(row_idxs)
        triu = sim_mat[np.triu_indices(k, k=1)]
        return float(triu.mean())

    bg_within_sims: list[float] = [
        s for rows_ in bg_row_sets
        if (s := _mean_pairwise_cosine_sim(rows_)) is not None
    ]

    # Pre-compute centroid L2 norm for each background group (query-independent).
    # Centroid norm → 1 when all members point the same direction; → 0 when random.
    bg_centroid_norms: list[float] = [
        float(np.linalg.norm(train_normed[rows_].mean(axis=0)))
        for rows_ in bg_row_sets
        if rows_ and train_normed is not None
    ]
    _bg_centroid_arr = np.array(bg_centroid_norms) if bg_centroid_norms else None

    rows: list[dict[str, Any]] = []

    for test_idx, test_id in enumerate(test_ids):
        col_scores = np.asarray(scores[:, test_idx], dtype=np.float64)
        topk_orig_idxs = np.argsort(-col_scores)[:effective_k].tolist()
        topk_rows = [orig_to_row[i] for i in topk_orig_idxs if i in orig_to_row]

        row: dict[str, Any] = {
            "test_index": test_idx,
            "test_track_id": test_id,
            "topk": effective_k,
            "n_topk_with_clap": len(topk_rows),
        }

        def _add_comparison(label: str, ref_emb: np.ndarray) -> None:
            """Compute z-score metrics for one reference audio type and write into row."""
            ref_norm = np.linalg.norm(ref_emb)
            if ref_norm == 0.0 or train_normed is None:
                for k in (f"clap_sim_vs_{label}", f"z_vs_{label}",
                          f"bg_mean_vs_{label}", f"bg_std_vs_{label}", f"pct_vs_{label}"):
                    row[k] = None
                return

            q = ref_emb / ref_norm                                  # (D,) normalised query

            # Group value: mean CLAP cosine sim of top-k training tracks to query.
            group_sim: float | None = (
                float((train_normed[topk_rows] @ q).mean()) if topk_rows else None
            )

            # Background distribution (shared index sets, per-query similarity).
            bg_sims: list[float] = [
                float((train_normed[rows_] @ q).mean())
                for rows_ in bg_row_sets
                if rows_
            ]

            z, bg_mean, bg_std, pct = _compute_contrast_stats(group_sim, bg_sims)
            row[f"clap_sim_vs_{label}"] = group_sim
            row[f"z_vs_{label}"]        = z
            row[f"bg_mean_vs_{label}"]  = bg_mean
            row[f"bg_std_vs_{label}"]   = bg_std
            row[f"pct_vs_{label}"]      = pct
            row[f"delta_vs_{label}"]    = (
                (group_sim - bg_mean) if group_sim is not None and bg_mean is not None else None
            )

        # ── Within-group cohesion (query-independent) ─────────────────────────
        within_sim = _mean_pairwise_cosine_sim(topk_rows)
        _, bg_mean_w, bg_std_w, pct_w = _compute_contrast_stats(within_sim, bg_within_sims)
        row["within_clap_sim"]      = within_sim
        row["bg_mean_within_clap"]  = bg_mean_w
        row["bg_std_within_clap"]   = bg_std_w
        row["pct_within_clap"]      = pct_w
        row["delta_within_clap"]    = (
            (within_sim - bg_mean_w) if within_sim is not None and bg_mean_w is not None else None
        )

        # ── Centroid norm cohesion (non-parametric, query-independent) ─────────
        if topk_rows and train_normed is not None:
            centroid_norm = float(np.linalg.norm(train_normed[topk_rows].mean(axis=0)))
        else:
            centroid_norm = None
        _bg_centroid_mean = float(_bg_centroid_arr.mean()) if _bg_centroid_arr is not None else None
        row["centroid_norm_topk"]    = centroid_norm
        row["bg_mean_centroid_norm"] = _bg_centroid_mean
        row["pct_centroid_norm"]     = (
            float(np.mean(_bg_centroid_arr <= centroid_norm))
            if centroid_norm is not None and _bg_centroid_arr is not None else None
        )
        row["delta_centroid_norm"]   = (
            (centroid_norm - _bg_centroid_mean)
            if centroid_norm is not None and _bg_centroid_mean is not None else None
        )

        # ── Conditioning alignment ─────────────────────────────────────────────
        cond_emb = conditioning_clap.get(test_id)
        if cond_emb is not None:
            _add_comparison("conditioning", cond_emb)

            # Spearman: TRAK score rank ↔ CLAP similarity to conditioning query.
            if train_normed is not None:
                q = cond_emb / (np.linalg.norm(cond_emb) + 1e-12)
                all_clap_sims = (train_normed @ q).tolist()             # (M,)
                trak_scores_subset = [float(col_scores[i]) for i in train_valid_orig_idxs]
                row["spearman_trak_clap_conditioning"] = _spearman(trak_scores_subset, all_clap_sims)
            else:
                row["spearman_trak_clap_conditioning"] = None
        else:
            for k in ("clap_sim_vs_conditioning", "z_vs_conditioning",
                      "bg_mean_vs_conditioning", "bg_std_vs_conditioning",
                      "pct_vs_conditioning", "spearman_trak_clap_conditioning"):
                row[k] = None

        # ── Generated alignment (optional) ────────────────────────────────────
        if generated_clap is not None:
            gen_emb = generated_clap.get(test_id)
            if gen_emb is not None:
                _add_comparison("generated", gen_emb)
            else:
                for k in ("clap_sim_vs_generated", "z_vs_generated",
                          "bg_mean_vs_generated", "bg_std_vs_generated", "pct_vs_generated"):
                    row[k] = None

        rows.append(row)

    return rows


# ─── High-level pipeline ──────────────────────────────────────────────────────

def run_alignment_analysis(
    stage: str,
    trak_stage_dir: Path,
    train_clap_npz: Path,
    conditioning_clap_npz: Path,
    generated_clap_npz: Path | None,
    output_json: Path,
    topk: int = 50,
    n_bg: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Full pipeline for one stage.

    Loads TRAK results from trak_stage_dir, loads CLAP embeddings, runs
    compute_clap_alignment_zscores(), and writes results to output_json.
    """
    print(f"[embedding_alignment] stage={stage}  dir={trak_stage_dir}", flush=True)
    stage_data = load_stage_result(trak_stage_dir, stage, "main", None, None)
    n_test = len(stage_data["test_track_ids"])
    n_train = len(stage_data["track_ids"])
    print(f"[embedding_alignment]   N_train={n_train}  N_test={n_test}", flush=True)

    print(f"[embedding_alignment] Loading CLAP embeddings ...", flush=True)
    train_clap = _load_clap_npz(train_clap_npz)
    conditioning_clap = _load_clap_npz(conditioning_clap_npz)
    generated_clap = _load_clap_npz(generated_clap_npz) if generated_clap_npz else None
    print(
        f"[embedding_alignment]   train_clap={len(train_clap)}  "
        f"conditioning_clap={len(conditioning_clap)}  "
        f"generated_clap={len(generated_clap) if generated_clap else 'n/a'}",
        flush=True,
    )

    print(
        f"[embedding_alignment] Computing CLAP alignment "
        f"(topk={topk}, n_bg={n_bg}) ...",
        flush=True,
    )
    per_test = compute_clap_alignment_zscores(
        stage_data=stage_data,
        train_clap=train_clap,
        conditioning_clap=conditioning_clap,
        generated_clap=generated_clap,
        topk=topk,
        n_bg=n_bg,
        seed=seed,
    )

    # Summary statistics
    def _avg(key: str) -> float | None:
        vals = [r[key] for r in per_test if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary: dict[str, Any] = {
        "stage": stage,
        "n_test": n_test,
        "n_train": n_train,
        "topk": topk,
        "n_bg": n_bg,
        "mean_z_vs_conditioning":             _avg("z_vs_conditioning"),
        "mean_z_vs_generated":                _avg("z_vs_generated"),
        "mean_clap_sim_vs_conditioning":      _avg("clap_sim_vs_conditioning"),
        "mean_clap_sim_vs_generated":         _avg("clap_sim_vs_generated"),
        "mean_spearman_trak_clap_conditioning": _avg("spearman_trak_clap_conditioning"),
        "mean_within_clap_sim":               _avg("within_clap_sim"),
        "mean_pct_within_clap":               _avg("pct_within_clap"),
        "bg_mean_within_clap":                _avg("bg_mean_within_clap"),
        "mean_delta_within_clap":             _avg("delta_within_clap"),
        "mean_centroid_norm_topk":            _avg("centroid_norm_topk"),
        "mean_pct_centroid_norm":             _avg("pct_centroid_norm"),
        "bg_mean_centroid_norm":              _avg("bg_mean_centroid_norm"),
        "mean_delta_centroid_norm":           _avg("delta_centroid_norm"),
        "mean_delta_vs_conditioning":         _avg("delta_vs_conditioning"),
        "mean_delta_vs_generated":            _avg("delta_vs_generated"),
    }
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"[embedding_alignment]   {key} = {val:.4f}", flush=True)

    payload = {"summary": summary, "per_test": per_test}
    dump_json(output_json, payload)
    print(f"[embedding_alignment] Saved → {output_json}", flush=True)
    return payload


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CLAP embedding alignment analysis for TRAK attribution. "
            "For each stage, computes how well TRAK top-k training audio "
            "aligns with conditioning/generated audio in CLAP space, "
            "relative to a random baseline."
        )
    )
    p.add_argument(
        "--stage", nargs="+", choices=list(STAGES), required=True,
        help="One or more model stages to analyse.",
    )
    p.add_argument(
        "--trak_results_dir", type=Path, required=True,
        help=(
            "TRAK results root directory. Must contain sub-directories named "
            "after each stage (semantic/, coarse/, fine/) with "
            "{stage}_{method}_scores.npy and {stage}_{method}_metadata.json."
        ),
    )
    p.add_argument(
        "--train_clap", type=Path, required=True,
        help="CLAP npz for the full training set ({track_id: embedding}).",
    )
    p.add_argument(
        "--conditioning_clap", type=Path, required=True,
        help="CLAP npz for test query conditioning audio ({track_id: embedding}).",
    )
    p.add_argument(
        "--generated_clap", type=Path, default=None,
        help="CLAP npz for decoded generated audio (optional, {track_id: embedding}).",
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory for output JSON files ({stage}_clap_alignment.json).",
    )
    p.add_argument("--topk",  type=int, default=50,  help="Top-k TRAK training tracks (default: 50).")
    p.add_argument("--n_bg",  type=int, default=200, help="Background random sets (default: 200).")
    p.add_argument("--seed",  type=int, default=42,  help="RNG seed (default: 42).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stage in args.stage:
        trak_stage_dir = args.trak_results_dir / stage
        output_json = args.output_dir / f"{stage}_clap_alignment.json"
        run_alignment_analysis(
            stage=stage,
            trak_stage_dir=trak_stage_dir,
            train_clap_npz=args.train_clap,
            conditioning_clap_npz=args.conditioning_clap,
            generated_clap_npz=args.generated_clap,
            output_json=output_json,
            topk=args.topk,
            n_bg=args.n_bg,
            seed=args.seed,
        )

    print("[embedding_alignment] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
