"""Method-agnostic diagnostic on a saved (N_train, N_test) score matrix.

Reads `<root>/<stage>/<stage>_<method>_scores.npy` (memmap, fp32) and produces:

  * Score distribution: per-column mean / std / quantiles, sampled.
  * Column-mean (global popularity) energy: ||mu_col||^2 / mean(||col||^2).
    Quantifies the "every test retrieves the same train samples" failure mode.
  * Test-test column correlation matrix C (n_test, n_test):
      - mean off-diagonal correlation
      - sigma1^2(C) / ||C||_F^2 (rank-1 of correlation -> retrieval is essentially
        a single global axis modulated by test-specific scaling)
      - histogram of off-diagonal entries
  * (Optional) Score matrix spectrum via S^T S = (n_test, n_test):
      - Top singular values, cum-var fractions.

All heavy ops are bounded to (n_test, n_test) accumulators (~200 MB at n_test=7148).
Streaming reads `chunk_rows` rows at a time, so RAM footprint is bounded by chunk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def streaming_stats(score_path: Path, chunk_rows: int = 8192):
    """One pass over the (N, n_test) score matrix, accumulating:

      sum_col[j]    = sum_i S[i, j]
      sumsq_col[j]  = sum_i S[i, j]^2
      Gtt[j, k]     = sum_i S[i, j] * S[i, k]    (n_test x n_test, fp64)
      Stt_centered  = sum_i (S[i,j]-mu_j)(S[i,k]-mu_k) — derived after the pass:
                       = Gtt - N * mu mu^T
    """
    S = np.load(score_path, mmap_mode="r")  # (N, n_test) fp32
    N, n_test = S.shape
    print(f"  loaded mmap: shape={S.shape} dtype={S.dtype}", flush=True)

    sum_col = np.zeros(n_test, dtype=np.float64)
    sumsq_col = np.zeros(n_test, dtype=np.float64)
    Gtt = np.zeros((n_test, n_test), dtype=np.float64)

    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.asarray(S[s:e], dtype=np.float64)  # (b, n_test)
        sum_col += block.sum(axis=0)
        sumsq_col += (block * block).sum(axis=0)
        Gtt += block.T @ block
        if (s // chunk_rows) % 8 == 0:
            elapsed = time.time() - t0
            print(f"    rows {e}/{N}  elapsed {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  pass done in {elapsed:.1f}s", flush=True)
    return N, n_test, sum_col, sumsq_col, Gtt


def correlation_summary(N, n_test, sum_col, sumsq_col, Gtt, label):
    """Derive Pearson correlation matrix C from streaming stats.

    Per-column mean: mu_j = sum_col[j] / N
    Per-column std:  sigma_j = sqrt(sumsq_col[j]/N - mu_j^2)
    Centered Gram:   Gc = Gtt - N * mu mu^T   (sum_i (S_ij-mu_j)(S_ik-mu_k))
    Pearson:         C[j, k] = Gc[j, k] / (N * sigma_j * sigma_k)
    """
    mu = sum_col / N
    var = sumsq_col / N - mu * mu
    var = np.clip(var, 0.0, None)
    sigma = np.sqrt(var)

    # global popularity / column-mean energy
    norm_sq_col = sumsq_col  # sum_i S_ij^2 = ||col_j||^2
    mu_col_sq = N * (mu * mu)  # ||N * mu_j * 1_N||^2 / N = N * mu_j^2 (energy of mean component per col)
    # fraction of column energy explained by being a constant ≠ 0
    pop_frac_per_col = np.divide(mu_col_sq, np.maximum(norm_sq_col, 1e-30))
    pop_frac_mean = float(pop_frac_per_col.mean())
    pop_frac_median = float(np.median(pop_frac_per_col))

    # centered Gram and Pearson C
    Gc = Gtt - N * np.outer(mu, mu)
    Gc = 0.5 * (Gc + Gc.T)
    denom = np.outer(sigma, sigma) * N
    denom = np.where(denom > 0, denom, 1.0)
    C = Gc / denom
    np.fill_diagonal(C, 1.0)

    # off-diagonal stats
    iu = np.triu_indices(n_test, k=1)
    off = C[iu]
    off_abs = np.abs(off)

    # Spectrum of C
    eigvals_C = np.linalg.eigvalsh(0.5 * (C + C.T))[::-1]
    eigvals_C = np.clip(eigvals_C, 0.0, None)
    fro2_C = float(eigvals_C.sum())
    top1_C = float(eigvals_C[0])
    cum_C = np.cumsum(eigvals_C) / max(fro2_C, 1e-30)

    # Spectrum of raw S (= eigvals of Gtt)
    eigvals_S = np.linalg.eigvalsh(0.5 * (Gtt + Gtt.T))[::-1]
    eigvals_S = np.clip(eigvals_S, 0.0, None)
    fro2_S = float(eigvals_S.sum())
    top1_S = float(eigvals_S[0])
    cum_S = np.cumsum(eigvals_S) / max(fro2_S, 1e-30)

    return {
        "label": label,
        "N": int(N),
        "n_test": int(n_test),
        "score_global": {
            "mu_mean": float(mu.mean()),
            "mu_std": float(mu.std()),
            "mu_min": float(mu.min()),
            "mu_max": float(mu.max()),
            "sigma_mean": float(sigma.mean()),
            "sigma_median": float(np.median(sigma)),
            "sigma_min": float(sigma.min()),
            "sigma_max": float(sigma.max()),
        },
        "popularity_fraction": {
            # Alias for the paper's mean concentration ratio p (legacy name).
            "meaning": "per-column ||mu_col||^2 / ||col||^2 — share of column energy in the constant 'global' direction; high => same train samples scored highly across tests for trivial offset reasons",
            "mean": pop_frac_mean,
            "median": pop_frac_median,
            "p10": float(np.percentile(pop_frac_per_col, 10)),
            "p90": float(np.percentile(pop_frac_per_col, 90)),
        },
        "mean_concentration_ratio": {
            # Paper Eq. p (canonical name); identical values to popularity_fraction.
            "meaning": "Paper Eq. p — alias of popularity_fraction.",
            "mean": pop_frac_mean,
            "median": pop_frac_median,
            "p10": float(np.percentile(pop_frac_per_col, 10)),
            "p90": float(np.percentile(pop_frac_per_col, 90)),
        },
        "test_test_correlation": {
            "meaning": "Pearson correlation between test columns. mean|off| close to 1 => retrieval barely depends on the test query (degenerate). sigma1(C)/Frob(C) close to 1 => single-axis collapse.",
            "off_diag_mean": float(off.mean()),
            "off_diag_abs_mean": float(off_abs.mean()),
            "off_diag_std": float(off.std()),
            "off_diag_quantiles": {
                str(q): float(np.percentile(off, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)
            },
            "frac_above_0p5": float((off_abs > 0.5).mean()),
            "frac_above_0p9": float((off_abs > 0.9).mean()),
            "frac_above_0p99": float((off_abs > 0.99).mean()),
            "sigma1_sq_C": top1_C,
            "frob_sq_C": fro2_C,
            "sigma1_sq_over_frob_sq_C": float(top1_C / max(fro2_C, 1e-30)),
            "cum_var_top1_C": float(cum_C[0]),
            "cum_var_top5_C": float(cum_C[min(4, len(cum_C) - 1)]),
            "cum_var_top20_C": float(cum_C[min(19, len(cum_C) - 1)]),
        },
        "raw_score_spectrum": {
            "meaning": "eigvals of S^T S; sigma1 dominance = 'one-direction' rank-1 of raw scores",
            "sigma1_sq_S": top1_S,
            "frob_sq_S": fro2_S,
            "sigma1_sq_over_frob_sq_S": float(top1_S / max(fro2_S, 1e-30)),
            "cum_var_top1_S": float(cum_S[0]),
            "cum_var_top5_S": float(cum_S[min(4, len(cum_S) - 1)]),
            "cum_var_top20_S": float(cum_S[min(19, len(cum_S) - 1)]),
            "top10_eigvals_S": eigvals_S[:10].tolist(),
            # Paper Eq. r_i = sigma_i^2 / ||S||_F^2 for the first five ranks.
            "r_per_rank_S": [
                float(eigvals_S[i] / max(fro2_S, 1e-30))
                for i in range(min(5, len(eigvals_S)))
            ],
        },
    }


def diagnose_one(score_path: Path, chunk_rows: int):
    print(f"\n=== {score_path} ===", flush=True)
    if not score_path.exists():
        print("  MISSING — skip", flush=True)
        return None
    N, n_test, sum_col, sumsq_col, Gtt = streaming_stats(score_path, chunk_rows)
    summary = correlation_summary(N, n_test, sum_col, sumsq_col, Gtt, str(score_path))

    pop = summary["popularity_fraction"]
    cc = summary["test_test_correlation"]
    rs = summary["raw_score_spectrum"]
    print(f"  popularity frac (per-col): mean={pop['mean']:.4f} median={pop['median']:.4f} p10..p90={pop['p10']:.4f}..{pop['p90']:.4f}", flush=True)
    print(
        f"  test-test corr  off-diag:  mean={cc['off_diag_mean']:+.4f} |mean|={cc['off_diag_abs_mean']:.4f} std={cc['off_diag_std']:.4f} "
        f"q50={cc['off_diag_quantiles']['50']:+.4f} q95={cc['off_diag_quantiles']['95']:+.4f}  "
        f">|0.5|: {cc['frac_above_0p5']*100:.2f}%  >|0.9|: {cc['frac_above_0p9']*100:.2f}%  >|0.99|: {cc['frac_above_0p99']*100:.2f}%",
        flush=True,
    )
    print(
        f"  C spectrum:   sigma1^2/Frob^2={cc['sigma1_sq_over_frob_sq_C']:.4f}  "
        f"(top1 cumVar={cc['cum_var_top1_C']:.3f}, top5={cc['cum_var_top5_C']:.3f}, top20={cc['cum_var_top20_C']:.3f})",
        flush=True,
    )
    print(
        f"  S^TS spectrum: sigma1^2/Frob^2={rs['sigma1_sq_over_frob_sq_S']:.4f}  "
        f"(top1 cumVar={rs['cum_var_top1_S']:.3f}, top5={rs['cum_var_top5_S']:.3f}, top20={rs['cum_var_top20_S']:.3f})",
        flush=True,
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help=(
            "List of '<root>/<stage>/<stage>_<method>_scores.npy' files, OR "
            "'<root>:<method>' shortcuts that auto-expand to all 3 stages "
            "(semantic/coarse/fine)."
        ),
    )
    p.add_argument("--out", type=Path, required=True, help="output JSON")
    p.add_argument("--chunk_rows", type=int, default=8192)
    args = p.parse_args()

    score_files = []
    for t in args.targets:
        if ":" in t and not Path(t).exists():
            root, method = t.rsplit(":", 1)
            root = Path(root)
            for stage in ("semantic", "coarse", "fine"):
                f = root / stage / f"{stage}_{method}_scores.npy"
                score_files.append((stage, method, f))
        else:
            f = Path(t)
            # infer method from filename
            stage = f.parent.name
            method = f.stem.replace(f"{stage}_", "").replace("_scores", "")
            score_files.append((stage, method, f))

    results = []
    for stage, method, f in score_files:
        summary = diagnose_one(f, args.chunk_rows)
        if summary is None:
            continue
        summary["stage"] = stage
        summary["method"] = method
        results.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"runs": results}, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
