"""embedding_val_test_split.py

Apply the same val/test split (seed=42, val=1000) to CLAP/CLEWS embedding alignment results.
Filter per_test entries and recompute val / test / full summaries, then save them as JSON and TSV.

Usage
-----
python embedding_val_test_split.py \\
    --embedding-dirs \\
        /path/to/embedding_alignment \\
        /path/to/embedding_alignment_clews_shs-clews \\
    --output-dir /path/to/output \\
    [--val-size 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


def _recompute_summary(per_test: list[dict[str, Any]], orig_summary: dict[str, Any]) -> dict[str, Any]:
    """Recompute aggregate summary from a (possibly filtered) per_test list."""
    summary = {
        "stage":  orig_summary.get("stage"),
        "n_test": len(per_test),
        "n_train": orig_summary.get("n_train"),
        "topk":   orig_summary.get("topk"),
        "n_bg":   orig_summary.get("n_bg"),
    }

    # scalar-mean fields: compute mean over per_test entries
    scalar_fields = [
        "within_clap_sim",
        "pct_within_clap",
        "centroid_norm_topk",
        "pct_centroid_norm",
        "clap_sim_vs_conditioning",
        "z_vs_conditioning",
        "pct_vs_conditioning",
        "spearman_trak_clap_conditioning",
        "clap_sim_vs_generated",
        "z_vs_generated",
        "pct_vs_generated",
        # CLEWS extras (may not exist in CLAP)
        "delta_within_clap",
        "delta_centroid_norm",
        "delta_vs_conditioning",
        "delta_vs_generated",
    ]
    mapping = {
        "within_clap_sim":              "mean_within_clap_sim",
        "pct_within_clap":              "mean_pct_within_clap",
        "centroid_norm_topk":           "mean_centroid_norm_topk",
        "pct_centroid_norm":            "mean_pct_centroid_norm",
        "clap_sim_vs_conditioning":     "mean_clap_sim_vs_conditioning",
        "z_vs_conditioning":            "mean_z_vs_conditioning",
        "pct_vs_conditioning":          "mean_pct_vs_conditioning",
        "spearman_trak_clap_conditioning": "mean_spearman_trak_clap_conditioning",
        "clap_sim_vs_generated":        "mean_clap_sim_vs_generated",
        "z_vs_generated":               "mean_z_vs_generated",
        "pct_vs_generated":             "mean_pct_vs_generated",
        "delta_within_clap":            "mean_delta_within_clap",
        "delta_centroid_norm":          "mean_delta_centroid_norm",
        "delta_vs_conditioning":        "mean_delta_vs_conditioning",
        "delta_vs_generated":           "mean_delta_vs_generated",
    }

    for per_field, sum_field in mapping.items():
        vals = [e[per_field] for e in per_test if per_field in e and e[per_field] is not None]
        summary[sum_field] = _mean(vals)

    # keep bg_mean fields from original (they depend on training pool, not test queries)
    for k, v in orig_summary.items():
        if k.startswith("bg_"):
            summary[k] = v

    return summary


def _dump_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fieldnames})


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ── core ──────────────────────────────────────────────────────────────────────

def process_embedding_dir(
    emb_dir: Path,
    val_indices: set[int],
    test_indices: set[int],
    output_dir: Path,
    method_name: str,
) -> None:
    """Process one embedding method directory (e.g. embedding_alignment/)."""
    # discover topk_* subdirs that contain JSON files
    topk_dirs = sorted(
        [d for d in emb_dir.iterdir() if d.is_dir() and re.fullmatch(r"topk_\d+", d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not topk_dirs:
        print(f"  [embed] No topk_* dirs found in {emb_dir}")
        return

    all_summary_rows: list[dict[str, Any]] = []

    for topk_dir in topk_dirs:
        k = int(topk_dir.name.split("_")[1])
        json_files = sorted(topk_dir.glob("*.json"))
        if not json_files:
            continue

        for json_path in json_files:
            stage = json_path.stem.split("_")[0]  # e.g. "fine" from "fine_clap_alignment.json"

            with json_path.open() as f:
                data = json.load(f)

            orig_summary = data.get("summary", {})
            per_test_all: list[dict[str, Any]] = data.get("per_test", [])
            if not per_test_all:
                print(f"  [embed] No per_test in {json_path.name}, skipping")
                continue

            per_test_val  = [e for e in per_test_all if int(e["test_index"]) in val_indices]
            per_test_test = [e for e in per_test_all if int(e["test_index"]) in test_indices]

            for split_label, per_test_subset in [
                ("val",  per_test_val),
                ("test", per_test_test),
                ("full", per_test_all),
            ]:
                new_summary = _recompute_summary(per_test_subset, orig_summary)
                new_summary["method"] = method_name
                new_summary["split"]  = split_label
                new_summary["k"]      = k

                # write per-K per-stage JSON (matches original format)
                out_json = output_dir / method_name / f"topk_{k}" / f"{split_label}_{json_path.name}"
                _dump_json(out_json, {"summary": new_summary, "per_test": per_test_subset})

                if split_label in ("test", "full"):
                    all_summary_rows.append(new_summary)

            print(f"  [embed] {method_name}  K={k:4d}  {stage}  "
                  f"val={len(per_test_val)}  test={len(per_test_test)}  full={len(per_test_all)}")

    # combined TSV across all K/stage (convenient for tables)
    if all_summary_rows:
        _dump_tsv(output_dir / method_name / "all_k_summary.tsv", all_summary_rows)
        print(f"  [embed] Written: {output_dir / method_name / 'all_k_summary.tsv'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply val/test split to CLAP/CLEWS embedding alignment results "
            "so N matches the TRAK test split."
        )
    )
    parser.add_argument("--embedding-dirs", nargs="+", required=True,
                        help="One or more embedding alignment root dirs (e.g. embedding_alignment/).")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write val/test split results.")
    parser.add_argument("--val-size", type=int, default=1000,
                        help="Validation split size (must match TRAK split). Default: 1000.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (must match TRAK split). Default: 42.")
    parser.add_argument("--n-total", type=int, default=7148,
                        help="Total number of test queries. Default: 7148.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # reproduce identical split used for TRAK
    all_indices = list(range(args.n_total))
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    val_indices  = set(all_indices[:args.val_size])
    test_indices = set(all_indices[args.val_size:])
    print(f"[embed] Val: {len(val_indices)} | Test: {len(test_indices)} (seed={args.seed})")

    for emb_dir_str in args.embedding_dirs:
        emb_dir = Path(emb_dir_str).resolve()
        method_name = emb_dir.name
        print(f"\n[embed] Processing: {method_name}")
        process_embedding_dir(emb_dir, val_indices, test_indices, output_dir, method_name)

    print(f"\n[embed] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
