"""lambda_val_test_split.py

Reviewer response: λ (top-K) selection bias fix.

Splits the 7148 test queries into val (1000) and test (6148) using a fixed
random seed.  For each available K, reads the *already-computed* per-test
topk_train_similarities.tsv files (written by analyze.py) and re-aggregates
enrichment z-scores on the val subset to select the best K.  Then reports
Table 1 metrics on the held-out test subset with the chosen K.

No re-running of analyze.py is required; this is a pure post-processing step
over existing output files.

Usage
-----
python lambda_val_test_split.py \\
    --results-dir  /path/to/large-norm-best/trak-proj4096-reg001-best \\
    --output-dir   /path/to/output \\
    [--val-size 1000] [--seed 42] [--stage fine]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


# ── constants matching analyze.py / reporting.py ─────────────────────────────
EVIDENCE_CHANNELS = ("rhythmic", "harmonic", "timbral")
STAGES = ("semantic", "coarse", "fine")
CONDITIONS = ("main", "random_baseline", "shuffle_train")
PRIMARY_CONDITION = "main"
BASELINE_CONDITIONS = ("random_baseline", "shuffle_train")


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _mean(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


def _frac_above(vals: list[float], threshold: float) -> float | None:
    valid = [v for v in vals if v is not None]
    return sum(1 for v in valid if v > threshold) / len(valid) if valid else None


def _std(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def _load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _dump_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fieldnames})


def _extract_top_k(path: Path) -> int | None:
    match = re.search(r"analysis-top(\d+)", str(path))
    return int(match.group(1)) if match else None


# ── core logic ────────────────────────────────────────────────────────────────

def discover_k_dirs(results_dir: Path) -> dict[int, Path]:
    """Return {K: analysis-topK-dir} sorted by K, for dirs that have the main TSV."""
    k_dirs: dict[int, Path] = {}
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        k = _extract_top_k(d)
        if k is None:
            continue
        tsv = d / "main" / "main" / "topk_train_similarities.tsv"
        if tsv.exists():
            k_dirs[k] = d
    return dict(sorted(k_dirs.items()))


def load_per_test_rows(analysis_dir: Path, condition: str) -> list[dict[str, str]]:
    """Load per-test rows for one condition from an analysis-topK directory."""
    tsv = analysis_dir / "main" / condition / "topk_train_similarities.tsv"
    return _load_tsv(tsv)


def filter_by_indices(rows: list[dict[str, str]], indices: set[int]) -> list[dict[str, str]]:
    return [r for r in rows if int(r["test_index"]) in indices]


def aggregate_enrichment(
    rows: list[dict[str, str]],
    condition: str,
    stage: str,
    method: str,
    top_k: int,
) -> dict[str, Any]:
    """Replicates build_enrichment_summary from summary.py over a filtered row list."""
    stage_rows = [r for r in rows if r.get("stage") == stage]
    summary: dict[str, Any] = {
        "condition": condition,
        "stage": stage,
        "method": method,
        "top_k": top_k,
        "test_count": len(stage_rows),
    }
    for ch in EVIDENCE_CHANNELS:
        zs = [_safe_float(r.get(f"channel_z_{ch}")) for r in stage_rows]
        zs = [v for v in zs if v is not None]
        summary[f"mean_channel_z_{ch}"] = _mean(zs)
        summary[f"std_channel_z_{ch}"] = _std(zs)
        summary[f"frac_pos_channel_z_{ch}"] = _frac_above(zs, 0.0)
        summary[f"frac_sig_channel_z_{ch}"] = _frac_above(zs, 1.96)
    means = [summary[f"mean_channel_z_{ch}"] for ch in EVIDENCE_CHANNELS if summary.get(f"mean_channel_z_{ch}") is not None]
    summary["mean_channel_z_all"] = _mean(means)
    return summary


def compute_val_score(summaries: list[dict[str, Any]]) -> float:
    """Aggregate λ-selection criterion: mean of mean_channel_z_all across stages (fine only by default, or all stages)."""
    vals = [s["mean_channel_z_all"] for s in summaries if s.get("mean_channel_z_all") is not None]
    return sum(vals) / len(vals) if vals else float("-inf")


def build_table1(
    main_rows: list[dict[str, str]],
    baseline_rows_by_cond: dict[str, list[dict[str, str]]],
    method: str,
    top_k: int,
    split_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replicate paper_table1_enrichment.tsv for a given split (val/test)."""
    out_rows: list[dict[str, Any]] = []
    # collapse across stages: only 'fine' stage or all? — match original: all stages collapsed
    for baseline_cond in BASELINE_CONDITIONS:
        baseline_rows = baseline_rows_by_cond[baseline_cond]
        row: dict[str, Any] = {
            "method": method,
            "vs_baseline": baseline_cond,
            "split": split_label,
            "top_k": top_k,
            "test_count": len(set(r["test_index"] for r in main_rows)),
        }
        all_main_means: list[float] = []
        all_base_means: list[float] = []
        for ch in EVIDENCE_CHANNELS:
            main_zs = [_safe_float(r.get(f"channel_z_{ch}")) for r in main_rows]
            main_zs = [v for v in main_zs if v is not None]
            base_zs = [_safe_float(r.get(f"channel_z_{ch}")) for r in baseline_rows]
            base_zs = [v for v in base_zs if v is not None]
            row[f"main_mean_z_{ch}"] = _mean(main_zs)
            row[f"baseline_mean_z_{ch}"] = _mean(base_zs)
            row[f"delta_mean_z_{ch}"] = (
                row[f"main_mean_z_{ch}"] - row[f"baseline_mean_z_{ch}"]
                if row[f"main_mean_z_{ch}"] is not None and row[f"baseline_mean_z_{ch}"] is not None
                else None
            )
            row[f"main_frac_pos_{ch}"] = _frac_above(main_zs, 0.0)
            row[f"main_frac_sig_{ch}"] = _frac_above(main_zs, 1.96)
            if row[f"main_mean_z_{ch}"] is not None:
                all_main_means.append(row[f"main_mean_z_{ch}"])
            if row[f"baseline_mean_z_{ch}"] is not None:
                all_base_means.append(row[f"baseline_mean_z_{ch}"])
        row["main_mean_z_all"] = _mean(all_main_means)
        row["baseline_mean_z_all"] = _mean(all_base_means)
        row["delta_mean_z_all"] = (
            row["main_mean_z_all"] - row["baseline_mean_z_all"]
            if row["main_mean_z_all"] is not None and row["baseline_mean_z_all"] is not None
            else None
        )
        out_rows.append(row)

    fieldnames = ["method", "vs_baseline", "split", "top_k", "test_count"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([
            f"main_mean_z_{ch}", f"baseline_mean_z_{ch}", f"delta_mean_z_{ch}",
            f"main_frac_pos_{ch}", f"main_frac_sig_{ch}",
        ])
    fieldnames.extend(["main_mean_z_all", "baseline_mean_z_all", "delta_mean_z_all"])
    return out_rows, fieldnames


def build_table2_stage(
    main_rows: list[dict[str, str]],
    method: str,
    top_k: int,
    split_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-stage breakdown (Table 2) for a given split."""
    out_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_rows = [r for r in main_rows if r.get("stage") == stage]
        row: dict[str, Any] = {
            "method": method,
            "stage": stage,
            "split": split_label,
            "top_k": top_k,
            "test_count": len(stage_rows),
        }
        means: list[float] = []
        for ch in EVIDENCE_CHANNELS:
            zs = [_safe_float(r.get(f"channel_z_{ch}")) for r in stage_rows]
            zs = [v for v in zs if v is not None]
            row[f"mean_z_{ch}"] = _mean(zs)
            row[f"frac_pos_{ch}"] = _frac_above(zs, 0.0)
            row[f"frac_sig_{ch}"] = _frac_above(zs, 1.96)
            if row[f"mean_z_{ch}"] is not None:
                means.append(row[f"mean_z_{ch}"])
        row["mean_z_all"] = _mean(means)
        out_rows.append(row)
    fieldnames = ["method", "stage", "split", "top_k", "test_count"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([f"mean_z_{ch}", f"frac_pos_{ch}", f"frac_sig_{ch}"])
    fieldnames.append("mean_z_all")
    return out_rows, fieldnames


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "λ (top-K) selection bias fix: split test queries into val/test, "
            "select K on val, report Table 1 metrics on test."
        )
    )
    parser.add_argument("--results-dir", required=True,
                        help="Path to the method results dir (e.g. .../trak-proj4096-reg001-best).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write output TSVs. Default: <results-dir>/val_test_split_analysis.")
    parser.add_argument("--val-size", type=int, default=1000,
                        help="Number of queries to reserve for λ-selection validation. Default: 1000.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for val/test split. Default: 42.")
    parser.add_argument("--method", default="trak",
                        help="Method name (used for labeling only). Default: trak.")
    parser.add_argument("--selection-stage", default="fine",
                        choices=list(STAGES) + ["all"],
                        help="Stage(s) used for K-selection criterion on val. 'all' = average across stages. Default: fine.")
    parser.add_argument("--fixed-k", type=int, default=None,
                        help=(
                            "Fix K to this value and skip val-based K-selection. "
                            "Use when the paper already reports a specific K (e.g. --fixed-k 100) "
                            "and you only need to show val/test split results are consistent."
                        ))
    parser.add_argument("--no-baseline", action="store_true",
                        help=(
                            "Skip Table 1 (which requires random_baseline/shuffle_train dirs). "
                            "Use for CLAP/CLEWS retrieval results that have no baseline conditions."
                        ))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        results_dir / "val_test_split_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover available K directories ─────────────────────────────────────
    k_dirs = discover_k_dirs(results_dir)
    if not k_dirs:
        raise RuntimeError(f"No analysis-topK directories with topk_train_similarities.tsv found under {results_dir}")
    print(f"[split] Found K values: {sorted(k_dirs)}")

    # ── determine total test count from one reference file ───────────────────
    ref_k = next(iter(k_dirs))
    ref_rows = load_per_test_rows(k_dirs[ref_k], PRIMARY_CONDITION)
    all_test_indices = sorted(set(int(r["test_index"]) for r in ref_rows))
    n_total = len(all_test_indices)
    print(f"[split] Total test queries: {n_total}")

    if args.val_size >= n_total:
        raise ValueError(f"--val-size {args.val_size} >= total queries {n_total}")

    # ── reproducible val/test split ───────────────────────────────────────────
    import random
    rng = random.Random(args.seed)
    shuffled = all_test_indices[:]
    rng.shuffle(shuffled)
    val_indices = set(shuffled[:args.val_size])
    test_indices = set(shuffled[args.val_size:])
    n_test = len(test_indices)
    print(f"[split] Val: {len(val_indices)} queries | Test: {n_test} queries (seed={args.seed})")

    # ── K selection / fixed-K ─────────────────────────────────────────────────
    val_selection_rows: list[dict[str, Any]] = []
    best_k: int | None = None

    if args.fixed_k is not None:
        # Fixed-K mode: paper already committed to this K; just verify no inflation.
        if args.fixed_k not in k_dirs:
            raise ValueError(
                f"--fixed-k {args.fixed_k} not found. Available: {sorted(k_dirs)}"
            )
        best_k = args.fixed_k
        print(f"\n[split] Fixed K={best_k} (--fixed-k mode; skipping val K-selection sweep)")
        print(f"[split] Val set used only to verify results are consistent with test set.")
    else:
        # Val-selection mode: find best K using only val queries.
        print(f"\n[split] === λ selection on val (stage={args.selection_stage}) ===")
        best_score = float("-inf")

        for k, analysis_dir in k_dirs.items():
            main_rows = load_per_test_rows(analysis_dir, PRIMARY_CONDITION)
            val_main = filter_by_indices(main_rows, val_indices)

            stages_for_selection = STAGES if args.selection_stage == "all" else (args.selection_stage,)
            summaries = [
                aggregate_enrichment(val_main, PRIMARY_CONDITION, stage, args.method, k)
                for stage in stages_for_selection
            ]
            score = compute_val_score(summaries)

            row: dict[str, Any] = {"top_k": k, "val_score_mean_z_all": score}
            for stage in STAGES:
                s = aggregate_enrichment(val_main, PRIMARY_CONDITION, stage, args.method, k)
                for ch in EVIDENCE_CHANNELS:
                    row[f"val_{stage}_mean_z_{ch}"] = s.get(f"mean_channel_z_{ch}")
                row[f"val_{stage}_mean_z_all"] = s.get("mean_channel_z_all")
            val_selection_rows.append(row)

            marker = "  <-- best so far" if score > best_score else ""
            print(f"  K={k:5d}  val_score={score:+.4f}{marker}")
            if score > best_score:
                best_score = score
                best_k = k

        print(f"\n[split] Selected K = {best_k}  (val_score = {best_score:+.4f})")

        # write val selection table
        if val_selection_rows:
            val_fields = ["top_k", "val_score_mean_z_all"]
            for stage in STAGES:
                for ch in EVIDENCE_CHANNELS:
                    val_fields.append(f"val_{stage}_mean_z_{ch}")
                val_fields.append(f"val_{stage}_mean_z_all")
            _dump_tsv(output_dir / "val_lambda_selection.tsv", val_fields, val_selection_rows)
            print(f"[split] Written: {output_dir / 'val_lambda_selection.tsv'}")

    # ── test evaluation for ALL K values ─────────────────────────────────────
    assert best_k is not None
    print(f"\n[split] === Test evaluation for all K values ===")

    # Accumulate cross-K summary rows for a single combined table
    all_t1_rows: list[dict[str, Any]] = []
    all_t2_rows: list[dict[str, Any]] = []
    t1_fields: list[str] = []
    t2_fields: list[str] = []

    for k, analysis_dir in k_dirs.items():
        print(f"[split]   K={k} ...", flush=True)
        main_rows_all = load_per_test_rows(analysis_dir, PRIMARY_CONDITION)
        baseline_rows_all: dict[str, list[dict[str, str]]] = {
            cond: load_per_test_rows(analysis_dir, cond) for cond in BASELINE_CONDITIONS
        }

        for split_label, m_rows, b_rows in [
            ("val",  filter_by_indices(main_rows_all, val_indices),
                     {c: filter_by_indices(r, val_indices) for c, r in baseline_rows_all.items()}),
            ("test", filter_by_indices(main_rows_all, test_indices),
                     {c: filter_by_indices(r, test_indices) for c, r in baseline_rows_all.items()}),
            ("full", main_rows_all, baseline_rows_all),
        ]:
            k_dir_out = output_dir / f"top{k}"
            if not args.no_baseline:
                t1_rows, t1_fields = build_table1(m_rows, b_rows, args.method, k, split_label)
                _dump_tsv(k_dir_out / f"table1_enrichment_{split_label}.tsv", t1_fields, t1_rows)
                all_t1_rows.extend(t1_rows)
            t2_rows, t2_fields = build_table2_stage(m_rows, args.method, k, split_label)
            _dump_tsv(k_dir_out / f"table2_stage_{split_label}.tsv", t2_fields, t2_rows)
            all_t2_rows.extend(t2_rows)

    # combined tables spanning all K (convenient for plotting / paper tables)
    if all_t1_rows:
        _dump_tsv(output_dir / "all_k_table1_enrichment.tsv", t1_fields, all_t1_rows)
        print(f"[split] Written: {output_dir / 'all_k_table1_enrichment.tsv'}  ({len(all_t1_rows)} rows)")
    if all_t2_rows:
        _dump_tsv(output_dir / "all_k_table2_stage.tsv", t2_fields, all_t2_rows)
        print(f"[split] Written: {output_dir / 'all_k_table2_stage.tsv'}  ({len(all_t2_rows)} rows)")

    # symlink / copy chosen-K files to top-level for convenience
    chosen_dir_out = output_dir / f"top{best_k}"
    for split_label in ("val", "test", "full"):
        for tname in ("table1_enrichment", "table2_stage"):
            src = chosen_dir_out / f"{tname}_{split_label}.tsv"
            dst = output_dir / f"{tname}_{split_label}.tsv"
            if src.exists():
                import shutil
                shutil.copy2(src, dst)
    print(f"[split] Copied chosen-K (K={best_k}) tables to top-level output dir.")

    # ── summary print ─────────────────────────────────────────────────────────
    mode_label = f"fixed K={best_k} (paper value)" if args.fixed_k else f"K={best_k} selected on val"
    print(f"\n{'='*60}")
    print(f"  Mode:         {mode_label}")
    print(f"  Val N:        {len(val_indices)} (seed={args.seed})")
    print(f"  Test N:       {n_test}")
    print(f"  Output:       {output_dir}")
    print(f"{'='*60}")
    if not args.no_baseline:
        print("Key metrics — val split:")
        for row in _load_tsv(output_dir / "table1_enrichment_val.tsv"):
            print(f"  vs {row['vs_baseline']:20s}  Δz_all={row.get('delta_mean_z_all', 'n/a')}")
        print("\nKey metrics — test split (held-out):")
        for row in _load_tsv(output_dir / "table1_enrichment_test.tsv"):
            print(f"  vs {row['vs_baseline']:20s}  Δz_all={row.get('delta_mean_z_all', 'n/a')}")
        print("\nFull set (original reporting, for comparison):")
        for row in _load_tsv(output_dir / "table1_enrichment_full.tsv"):
            print(f"  vs {row['vs_baseline']:20s}  Δz_all={row.get('delta_mean_z_all', 'n/a')}")
    else:
        print("(--no-baseline: Table 2 only, see all_k_table2_stage.tsv)")


if __name__ == "__main__":
    main()
