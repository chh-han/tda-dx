"""reporting.py — Main/appendix table generation for analyzer outputs."""
from __future__ import annotations

import csv
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any


EVIDENCE_CHANNELS = ("rhythmic", "harmonic", "timbral")
STAGES = ("semantic", "coarse", "fine")
BASELINE_CONDITIONS = ("random_baseline", "shuffle_train")
PRIMARY_CONDITION = "main"
MAIN_BUCKET = "main"
APPENDIX_BUCKET = "appendix"
_LOO_CHANNEL_DIMS: dict[str, tuple[str, ...]] = {
    "rhythmic": ("beat_pattern_sim", "onset_hist_sim"),
    "harmonic": ("chroma_sim", "tonnetz_sim", "chord_prog_sim"),
    "timbral": ("mfcc_sim", "cqt_sim"),
}


def _load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _dump_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fieldnames})


def _bucket_condition_path(output_dir: Path, bucket: str, condition: str, filename: str) -> Path:
    return output_dir / bucket / condition / filename


def _extract_top_k_from_path(path: Path) -> int | None:
    match = re.search(r"analysis-top(\d+)", str(path))
    return int(match.group(1)) if match else None


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _mean(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


def _std(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def _median(vals: list[float]) -> float | None:
    valid = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not valid:
        return None
    n = len(valid)
    mid = n // 2
    return valid[mid] if n % 2 else (valid[mid - 1] + valid[mid]) / 2


def _frac_above(vals: list[float], threshold: float) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(1 for v in valid if v > threshold) / len(valid) if valid else None


def _wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 5:
        return None
    diffs = [x - y for x, y in pairs if (x - y) != 0]
    if not diffs:
        return None
    n = len(diffs)
    abs_sorted = sorted(range(n), key=lambda k: abs(diffs[k]))
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(diffs[abs_sorted[j]]) == abs(diffs[abs_sorted[i]]):
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[abs_sorted[k]] = avg_rank
        i = j
    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    w = min(w_plus, w_minus)
    if n < 10:
        return None
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma <= 0:
        return None
    z = (w - mu) / sigma
    def _norm_sf_log(x: float) -> float:
        if x > 38:
            return -0.5 * x * x - math.log(x) - 0.5 * math.log(2 * math.pi)
        t = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        sf = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
        return math.log(max(sf, 1e-300))
    return max(min(1.0, 2 * math.exp(_norm_sf_log(abs(z)))), 1e-300)


def _load_topk_rows(output_dir: Path, condition: str) -> list[dict[str, str]]:
    return _load_tsv(_bucket_condition_path(output_dir, MAIN_BUCKET, condition, "topk_train_similarities.tsv"))


def _load_feature_summary(output_dir: Path, condition: str) -> list[dict[str, str]]:
    return _load_tsv(_bucket_condition_path(output_dir, MAIN_BUCKET, condition, "feature_alignment_summary.tsv"))


def _load_pairwise_summary(output_dir: Path, condition: str) -> list[dict[str, str]]:
    return _load_tsv(_bucket_condition_path(output_dir, APPENDIX_BUCKET, condition, "pairwise_stage_summary.tsv"))


def _discover_methods(output_dir: Path, condition: str) -> list[str]:
    rows = _load_feature_summary(output_dir, condition)
    methods = list(dict.fromkeys(r.get("method", "") for r in rows if r.get("method")))
    return methods or ["unknown"]


def _filter_method(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    if rows and "method" in rows[0]:
        return [r for r in rows if r.get("method") == method]
    return rows


def _filter_stage(rows: list[dict[str, str]], stage: str) -> list[dict[str, str]]:
    if stage == "all_stages":
        return rows
    if rows and "stage" in rows[0]:
        return [r for r in rows if r.get("stage") == stage]
    return rows


def _extract_channel_zs(rows: list[dict[str, str]], channel: str) -> list[float]:
    return [v for v in (_safe_float(r.get(f"channel_z_{channel}")) for r in rows) if v is not None]


def _compute_channel_loo_stats(rows: list[dict[str, str]], channel: str) -> tuple[float | None, float | None, float | None]:
    per_query_vals: list[float] = []
    for row in rows:
        vals = [_safe_float(row.get(f"loo_max_contribution_{dim}")) for dim in _LOO_CHANNEL_DIMS[channel]]
        vals = [v for v in vals if v is not None]
        if vals:
            per_query_vals.append(sum(vals) / len(vals))
    return _mean(per_query_vals), _std(per_query_vals), _frac_above(per_query_vals, 0.0)


def _find_one(rows: list[dict[str, Any]], **criteria: str) -> dict[str, Any] | None:
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
            return row
    return None


def combine_analysis_tables(
    analysis_dirs: list[Path],
    output_path: Path,
    source_relpath: str = "main/paper_table3_consistency.tsv",
    *,
    require_top_k: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Combine per-top-k analysis tables into one long-form TSV.

    Parameters
    ----------
    analysis_dirs:
        List of analysis directories such as ``analysis-top20`` and ``analysis-top100``.
    output_path:
        Destination TSV path.
    source_relpath:
        Relative path inside each analysis directory to combine.
    require_top_k:
        If True, directories must encode ``analysis-top{K}`` in their path.
    """
    rows_out: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None

    for analysis_dir in analysis_dirs:
        source_path = analysis_dir / source_relpath
        if not source_path.exists():
            continue

        top_k = _extract_top_k_from_path(analysis_dir)
        if require_top_k and top_k is None:
            raise ValueError(
                f"Could not infer top_k from analysis directory name: {analysis_dir}",
            )

        rows = _load_tsv(source_path)
        if not rows:
            continue

        if fieldnames is None:
            fieldnames = ["top_k", "source_path", *rows[0].keys()]

        for row in rows:
            merged = {
                "top_k": top_k,
                "source_path": str(source_path),
                **row,
            }
            rows_out.append(merged)

    if fieldnames is None:
        raise FileNotFoundError(
            f"No readable TSVs found for source_relpath={source_relpath!r} "
            f"under {len(analysis_dirs)} analysis directories.",
        )

    rows_out.sort(
        key=lambda row: (
            int(row["top_k"]) if row.get("top_k") not in (None, "") else -1,
            str(row.get("method", "")),
            str(row.get("stage", "")),
            str(row.get("vs_baseline", "")),
        ),
    )
    _dump_tsv(output_path, fieldnames, rows_out)
    return rows_out, fieldnames


def discover_analysis_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        [path for path in root_dir.iterdir() if path.is_dir() and re.fullmatch(r"analysis-top\d+", path.name)],
        key=lambda path: (_extract_top_k_from_path(path) or -1, path.name),
    )


def build_main_table1_enrichment(output_dir: Path, methods: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    rows_out: list[dict[str, Any]] = []
    for method in methods:
        main_rows = _filter_method(_load_topk_rows(output_dir, PRIMARY_CONDITION), method)
        for baseline_cond in BASELINE_CONDITIONS:
            baseline_rows = _filter_method(_load_topk_rows(output_dir, baseline_cond), method)
            row: dict[str, Any] = {"method": method, "vs_baseline": baseline_cond}
            all_main_means: list[float] = []
            all_base_means: list[float] = []
            for ch in EVIDENCE_CHANNELS:
                main_zs = _extract_channel_zs(main_rows, ch)
                base_zs = _extract_channel_zs(baseline_rows, ch)
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
            rows_out.append(row)
    fieldnames = ["method", "vs_baseline"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([
            f"main_mean_z_{ch}", f"baseline_mean_z_{ch}", f"delta_mean_z_{ch}",
            f"main_frac_pos_{ch}", f"main_frac_sig_{ch}",
        ])
    fieldnames.extend(["main_mean_z_all", "baseline_mean_z_all", "delta_mean_z_all"])
    return rows_out, fieldnames


def build_main_table2_stage(output_dir: Path, methods: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    rows_out: list[dict[str, Any]] = []
    for method in methods:
        method_rows = _filter_method(_load_topk_rows(output_dir, PRIMARY_CONDITION), method)
        stage_channel_zs: dict[str, dict[str, list[float]]] = {
            stage: {ch: _extract_channel_zs(_filter_stage(method_rows, stage), ch) for ch in EVIDENCE_CHANNELS}
            for stage in STAGES
        }
        for stage in STAGES:
            row: dict[str, Any] = {"method": method, "stage": stage}
            means: list[float] = []
            for ch in EVIDENCE_CHANNELS:
                zs = stage_channel_zs[stage][ch]
                row[f"mean_z_{ch}"] = _mean(zs)
                row[f"frac_pos_{ch}"] = _frac_above(zs, 0.0)
                row[f"frac_sig_{ch}"] = _frac_above(zs, 1.96)
                if row[f"mean_z_{ch}"] is not None:
                    means.append(row[f"mean_z_{ch}"])
                if stage != "semantic":
                    n = min(len(stage_channel_zs["semantic"][ch]), len(zs))
                    row[f"wilcoxon_p_vs_semantic_{ch}"] = _wilcoxon_p(stage_channel_zs["semantic"][ch][:n], zs[:n])
                else:
                    row[f"wilcoxon_p_vs_semantic_{ch}"] = None
            row["mean_z_all"] = _mean(means)
            valid = {ch: row.get(f"mean_z_{ch}") for ch in EVIDENCE_CHANNELS if row.get(f"mean_z_{ch}") is not None}
            row["dominant_channel"] = max(valid, key=valid.get) if valid else None
            rows_out.append(row)
    fieldnames = ["method", "stage"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([f"mean_z_{ch}", f"frac_pos_{ch}", f"frac_sig_{ch}", f"wilcoxon_p_vs_semantic_{ch}"])
    fieldnames.extend(["mean_z_all", "dominant_channel"])
    return rows_out, fieldnames


def build_main_table3_consistency(output_dir: Path, methods: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    rows_out: list[dict[str, Any]] = []
    for method in methods:
        for baseline_cond in BASELINE_CONDITIONS:
            for stage in list(STAGES) + ["all_stages"]:
                row: dict[str, Any] = {"method": method, "stage": stage, "vs_baseline": baseline_cond}
                for label, cond in (("main", PRIMARY_CONDITION), ("baseline", baseline_cond)):
                    stage_rows = _filter_stage(_filter_method(_load_topk_rows(output_dir, cond), method), stage)
                    for ch in EVIDENCE_CHANNELS:
                        zs = _extract_channel_zs(stage_rows, ch)
                        row[f"{label}_mean_z_{ch}"] = _mean(zs)
                        row[f"{label}_frac_pos_{ch}"] = _frac_above(zs, 0.0)
                        row[f"{label}_frac_sig_{ch}"] = _frac_above(zs, 1.96)
                rows_out.append(row)
    fieldnames = ["method", "stage", "vs_baseline"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([
            f"main_mean_z_{ch}", f"main_frac_pos_{ch}", f"main_frac_sig_{ch}",
            f"baseline_mean_z_{ch}", f"baseline_frac_pos_{ch}", f"baseline_frac_sig_{ch}",
        ])
    return rows_out, fieldnames


def build_cross_condition_enrichment(output_dir: Path, methods: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    rows_out: list[dict[str, Any]] = []
    for cond in [PRIMARY_CONDITION] + list(BASELINE_CONDITIONS):
        feat_rows = _load_feature_summary(output_dir, cond)
        for row in feat_rows:
            if methods and row.get("method") not in methods:
                continue
            out: dict[str, Any] = {
                "condition": cond,
                "stage": row.get("stage", ""),
                "method": row.get("method", ""),
                "test_count": row.get("test_count", ""),
            }
            for ch in EVIDENCE_CHANNELS:
                out[f"mean_channel_z_{ch}"] = _safe_float(row.get(f"mean_channel_z_{ch}"))
                out[f"std_channel_z_{ch}"] = _safe_float(row.get(f"std_channel_z_{ch}"))
                out[f"frac_pos_channel_z_{ch}"] = _safe_float(row.get(f"frac_pos_channel_z_{ch}"))
                out[f"frac_sig_channel_z_{ch}"] = _safe_float(row.get(f"frac_sig_channel_z_{ch}"))
            rows_out.append(out)
    fieldnames = ["condition", "stage", "method", "test_count"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([
            f"mean_channel_z_{ch}", f"std_channel_z_{ch}",
            f"frac_pos_channel_z_{ch}", f"frac_sig_channel_z_{ch}",
        ])
    return rows_out, fieldnames


def build_appendix_table1_loo(output_dir: Path, methods: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    rows_out: list[dict[str, Any]] = []
    for method in methods:
        for baseline_cond in BASELINE_CONDITIONS:
            row: dict[str, Any] = {"method": method, "vs_baseline": baseline_cond}
            for label, cond in (("main", PRIMARY_CONDITION), ("baseline", baseline_cond)):
                stage_rows = _filter_method(_load_topk_rows(output_dir, cond), method)
                for ch in EVIDENCE_CHANNELS:
                    m, s, fp = _compute_channel_loo_stats(stage_rows, ch)
                    row[f"{label}_mean_loo_{ch}"] = m
                    row[f"{label}_std_loo_{ch}"] = s
                    row[f"{label}_frac_pos_loo_{ch}"] = fp
            rows_out.append(row)
    fieldnames = ["method", "vs_baseline"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([
            f"main_mean_loo_{ch}", f"main_std_loo_{ch}", f"main_frac_pos_loo_{ch}",
            f"baseline_mean_loo_{ch}", f"baseline_std_loo_{ch}", f"baseline_frac_pos_loo_{ch}",
        ])
    return rows_out, fieldnames


def build_stage_significance(output_dir: Path, methods: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    rows_out: list[dict[str, Any]] = []
    for method in methods:
        method_rows = _filter_method(_load_topk_rows(output_dir, PRIMARY_CONDITION), method)
        stage_channel_zs: dict[str, dict[str, list[float]]] = {
            stage: {ch: _extract_channel_zs(_filter_stage(method_rows, stage), ch) for ch in EVIDENCE_CHANNELS}
            for stage in STAGES
        }
        for stage_a, stage_b in combinations(STAGES, 2):
            row: dict[str, Any] = {"method": method, "stage_a": stage_a, "stage_b": stage_b}
            for ch in EVIDENCE_CHANNELS:
                a_zs = stage_channel_zs[stage_a][ch]
                b_zs = stage_channel_zs[stage_b][ch]
                n = min(len(a_zs), len(b_zs))
                row[f"wilcoxon_p_{ch}"] = _wilcoxon_p(a_zs[:n], b_zs[:n])
                row[f"mean_z_a_{ch}"] = _mean(a_zs)
                row[f"mean_z_b_{ch}"] = _mean(b_zs)
            rows_out.append(row)
    fieldnames = ["method", "stage_a", "stage_b"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([f"wilcoxon_p_{ch}", f"mean_z_a_{ch}", f"mean_z_b_{ch}"])
    return rows_out, fieldnames


def build_paper_main_table_trak_core(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    enrich_rows, _ = build_main_table1_enrichment(output_dir, methods=["trak"])
    consistency_rows, _ = build_main_table3_consistency(output_dir, methods=["trak"])

    rows_out: list[dict[str, Any]] = []
    for baseline in BASELINE_CONDITIONS:
        enrich_row = _find_one(enrich_rows, method="trak", vs_baseline=baseline)
        consistency_row = _find_one(
            consistency_rows,
            method="trak",
            stage="all_stages",
            vs_baseline=baseline,
        )
        if enrich_row is None or consistency_row is None:
            continue
        row: dict[str, Any] = {
            "method": "trak",
            "baseline": baseline,
            "overall_mean": enrich_row.get("main_mean_z_all"),
        }
        for ch in EVIDENCE_CHANNELS:
            row[f"{ch}_mean"] = consistency_row.get(f"main_mean_z_{ch}")
            row[f"{ch}_pos_rate"] = consistency_row.get(f"main_frac_pos_{ch}")
            row[f"{ch}_sig_rate"] = consistency_row.get(f"main_frac_sig_{ch}")
        rows_out.append(row)

    fieldnames = ["method", "baseline", "overall_mean"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([f"{ch}_mean", f"{ch}_pos_rate", f"{ch}_sig_rate"])
    return rows_out, fieldnames


def build_paper_main_table_trak_stage(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    stage_rows, _ = build_main_table2_stage(output_dir, methods=["trak"])

    rows_out: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_row = _find_one(stage_rows, method="trak", stage=stage)
        if stage_row is None:
            continue
        row: dict[str, Any] = {
            "method": "trak",
            "stage": stage,
            "rhythmic_mean": stage_row.get("mean_z_rhythmic"),
            "harmonic_mean": stage_row.get("mean_z_harmonic"),
            "timbral_mean": stage_row.get("mean_z_timbral"),
            "dominant_channel": stage_row.get("dominant_channel"),
        }
        rows_out.append(row)

    fieldnames = [
        "method", "stage",
        "rhythmic_mean", "harmonic_mean", "timbral_mean",
        "dominant_channel",
    ]
    return rows_out, fieldnames


def build_paper_main_table_trak_stage_consistency(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    consistency_rows, _ = build_main_table3_consistency(output_dir, methods=["trak"])

    rows_out: list[dict[str, Any]] = []
    for stage in STAGES:
        row_src = _find_one(
            consistency_rows,
            method="trak",
            stage=stage,
            vs_baseline="random_baseline",
        )
        if row_src is None:
            continue
        row: dict[str, Any] = {
            "method": "trak",
            "stage": stage,
        }
        for ch in EVIDENCE_CHANNELS:
            row[f"{ch}_pos_rate"] = row_src.get(f"main_frac_pos_{ch}")
            row[f"{ch}_sig_rate"] = row_src.get(f"main_frac_sig_{ch}")
        rows_out.append(row)

    fieldnames = ["method", "stage"]
    for ch in EVIDENCE_CHANNELS:
        fieldnames.extend([f"{ch}_pos_rate", f"{ch}_sig_rate"])
    return rows_out, fieldnames


def generate_all_paper_tables(output_dir: Path, methods: list[str] | None = None) -> None:
    print(f"[reporting] Generating main/appendix tables in {output_dir} ...", flush=True)
    if methods is None:
        methods = _discover_methods(output_dir, PRIMARY_CONDITION)
    print(f"[reporting] Methods detected: {methods}", flush=True)

    main_dir = output_dir / MAIN_BUCKET
    appendix_dir = output_dir / APPENDIX_BUCKET

    rows, fields = build_main_table1_enrichment(output_dir, methods)
    if rows:
        _dump_tsv(main_dir / "paper_table1_enrichment.tsv", fields, rows)
        print(f"[reporting] main/paper_table1_enrichment.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_main_table2_stage(output_dir, methods)
    if rows:
        _dump_tsv(main_dir / "paper_table2_stage.tsv", fields, rows)
        print(f"[reporting] main/paper_table2_stage.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_main_table3_consistency(output_dir, methods)
    if rows:
        _dump_tsv(main_dir / "paper_table3_consistency.tsv", fields, rows)
        print(f"[reporting] main/paper_table3_consistency.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_cross_condition_enrichment(output_dir, methods)
    if rows:
        _dump_tsv(main_dir / "q1_cross_condition_enrichment.tsv", fields, rows)
        print(f"[reporting] main/q1_cross_condition_enrichment.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_paper_main_table_trak_core(output_dir)
    if rows:
        _dump_tsv(main_dir / "paper_main_table_trak_core.tsv", fields, rows)
        print(f"[reporting] main/paper_main_table_trak_core.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_paper_main_table_trak_stage(output_dir)
    if rows:
        _dump_tsv(main_dir / "paper_main_table_trak_stage.tsv", fields, rows)
        print(f"[reporting] main/paper_main_table_trak_stage.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_paper_main_table_trak_stage_consistency(output_dir)
    if rows:
        _dump_tsv(main_dir / "paper_main_table_trak_stage_consistency.tsv", fields, rows)
        print(f"[reporting] main/paper_main_table_trak_stage_consistency.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_appendix_table1_loo(output_dir, methods)
    if rows:
        _dump_tsv(appendix_dir / "paper_appendix_table1_loo.tsv", fields, rows)
        print(f"[reporting] appendix/paper_appendix_table1_loo.tsv ({len(rows)} rows)", flush=True)

    rows, fields = build_stage_significance(output_dir, methods)
    if rows:
        _dump_tsv(appendix_dir / "q1_stage_significance.tsv", fields, rows)
        print(f"[reporting] appendix/q1_stage_significance.tsv ({len(rows)} rows)", flush=True)

    print("[reporting] Done.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate main/appendix tables from analyzer output.")
    parser.add_argument("--output-dir", help="Analyzer output directory (same as analyze.py --output-dir).")
    parser.add_argument("--methods", nargs="*", default=None, help="Attribution methods to include (default: auto-detect).")
    parser.add_argument(
        "--aggregate-analysis-dirs",
        nargs="*",
        default=None,
        help="Optional explicit list of analysis-topK directories to combine into one TSV.",
    )
    parser.add_argument(
        "--aggregate-root-dir",
        default=None,
        help="Optional root directory under which analysis-topK directories will be auto-discovered.",
    )
    parser.add_argument(
        "--aggregate-source-relpath",
        default="main/paper_table3_consistency.tsv",
        help="Relative path inside each analysis-topK directory to combine.",
    )
    parser.add_argument(
        "--aggregate-output",
        default=None,
        help="Output TSV path for the combined table. Required when using aggregation mode unless --output-dir is set.",
    )
    args = parser.parse_args()

    if args.aggregate_analysis_dirs or args.aggregate_root_dir:
        analysis_dirs: list[Path] = []
        if args.aggregate_root_dir:
            analysis_dirs.extend(discover_analysis_dirs(Path(args.aggregate_root_dir)))
        if args.aggregate_analysis_dirs:
            analysis_dirs.extend(Path(path) for path in args.aggregate_analysis_dirs)

        deduped_dirs: list[Path] = []
        seen: set[Path] = set()
        for path in analysis_dirs:
            resolved = path.resolve()
            if resolved not in seen:
                deduped_dirs.append(path)
                seen.add(resolved)

        if not deduped_dirs:
            parser.error("Aggregation mode found no analysis directories.")

        if args.aggregate_output:
            output_path = Path(args.aggregate_output)
        elif args.output_dir:
            output_path = Path(args.output_dir) / "comparison_table3" / "combined_stage_topk.tsv"
        else:
            parser.error("Aggregation mode requires --aggregate-output or --output-dir.")

        rows, _ = combine_analysis_tables(
            deduped_dirs,
            output_path=output_path,
            source_relpath=args.aggregate_source_relpath,
        )
        print(
            f"[reporting] Wrote combined table to {output_path} "
            f"from {len(deduped_dirs)} analysis dirs ({len(rows)} rows)",
            flush=True,
        )
    else:
        if not args.output_dir:
            parser.error("--output-dir is required unless aggregation mode is used.")
        generate_all_paper_tables(Path(args.output_dir), methods=args.methods)
