"""Top-K SVD analysis on attribution score matrices.

For each (method, stage), reads the (N, n_test) score matrix and computes:
  - top-K singular values  sigma_k
  - top-K right singular vectors  v_k     (each of length n_test)
  - top-K left  singular vectors  u_k     (each of length N) — derived as
                S @ v_k / sigma_k via a streaming pass
  - argmax/argmin of u_1..u_3 (top training samples on each end of each axis)
  - distribution stats of u_k and v_k

The intent is to characterise what the dominant retrieval axis encodes.
If u_1 corresponds to "silence / padding / long tracks" rather than music,
all methods that collapse to it produce nuisance retrieval. If u_1 differs
meaningfully across methods, that explains why z-scores diverge despite
identical sigma1^2/Frob^2 ratios.

Memory:
  S^T S accumulator        (n_test, n_test) fp64 ≈ 200 MB
  v_K matrix               (n_test, K) fp32  — kept in memory
  per-chunk U_k contribution (chunk, K) — bounded by chunk_rows
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def streaming_StS(score_path: Path, chunk_rows: int):
    S = np.load(score_path, mmap_mode="r")
    N, n_test = S.shape
    Gtt = np.zeros((n_test, n_test), dtype=np.float64)
    sum_col = np.zeros(n_test, dtype=np.float64)
    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.asarray(S[s:e], dtype=np.float64)
        Gtt += block.T @ block
        sum_col += block.sum(axis=0)
        if (s // chunk_rows) % 16 == 0:
            print(f"    pass1 rows {e}/{N}  elapsed {time.time()-t0:.1f}s", flush=True)
    return N, n_test, sum_col, Gtt


def topk_eigh(Gtt: np.ndarray, K: int):
    """Return top-K eigvals (descending) and corresponding eigvecs (n_test, K)."""
    G_sym = 0.5 * (Gtt + Gtt.T)
    # full eigh — n_test=7148 → fine on a 32-CPU node, ~30s
    eigvals, eigvecs = np.linalg.eigh(G_sym)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvals[:K].astype(np.float64), eigvecs[:, :K].astype(np.float32)


def streaming_topk_U(score_path: Path, V_top: np.ndarray, sigmas: np.ndarray,
                     chunk_rows: int, top_train_k: int):
    """Second streaming pass: compute U_top entries chunk by chunk, keep
    top/bottom-`top_train_k` train indices per axis. Returns:

      u_stats[k] = {
          "top_idx":  (top_train_k,) global train indices with largest u_k
          "top_val":  corresponding u_k values (= score on axis k)
          "bot_idx":  smallest u_k
          "bot_val":  corresponding values
          "mean":     mean of u_k over all train rows
          "std":      std
          "abs_mean": mean of |u_k|
          "frac_pos": fraction of u_k > 0
      }
    """
    S = np.load(score_path, mmap_mode="r")
    N, n_test = S.shape
    K = V_top.shape[1]

    # accumulators
    sum_uk = np.zeros(K, dtype=np.float64)
    sumsq_uk = np.zeros(K, dtype=np.float64)
    pos_count = np.zeros(K, dtype=np.int64)
    abs_sum_uk = np.zeros(K, dtype=np.float64)

    # heap-like top-K via partition sort over chunked candidates
    # For simplicity: keep top_train_k * 4 candidates per axis, prune as we go.
    keep = top_train_k
    top_buf = [np.empty(0, dtype=np.float64) for _ in range(K)]
    top_idx_buf = [np.empty(0, dtype=np.int64) for _ in range(K)]
    bot_buf = [np.empty(0, dtype=np.float64) for _ in range(K)]
    bot_idx_buf = [np.empty(0, dtype=np.int64) for _ in range(K)]

    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.asarray(S[s:e], dtype=np.float32)  # (b, n_test)
        # u_block[:, k] = S_block @ v_k / sigma_k
        Uk_block = block @ V_top  # (b, K)
        # divide by sigma to obtain unit-norm left vectors
        # but keep "score on axis" interpretation as Uk_block (=σ_k * u_k)
        # — i.e. keep raw S @ v_k (which has magnitude sigma_k).
        # That's fine for ranking: sigma_k > 0 so order preserved.

        # accumulate stats
        sum_uk += Uk_block.sum(axis=0)
        sumsq_uk += (Uk_block.astype(np.float64) ** 2).sum(axis=0)
        abs_sum_uk += np.abs(Uk_block).sum(axis=0)
        pos_count += (Uk_block > 0).sum(axis=0)

        # update top/bottom buffers per axis
        global_idx = np.arange(s, e, dtype=np.int64)
        for k in range(K):
            vals = Uk_block[:, k].astype(np.float64)
            # merge with current buffer, keep top `keep`
            cat_v = np.concatenate([top_buf[k], vals])
            cat_i = np.concatenate([top_idx_buf[k], global_idx])
            if cat_v.size > keep:
                part = np.argpartition(-cat_v, keep)[:keep]
                cat_v, cat_i = cat_v[part], cat_i[part]
            top_buf[k], top_idx_buf[k] = cat_v, cat_i

            cat_v = np.concatenate([bot_buf[k], vals])
            cat_i = np.concatenate([bot_idx_buf[k], global_idx])
            if cat_v.size > keep:
                part = np.argpartition(cat_v, keep)[:keep]
                cat_v, cat_i = cat_v[part], cat_i[part]
            bot_buf[k], bot_idx_buf[k] = cat_v, cat_i

        if (s // chunk_rows) % 16 == 0:
            print(f"    pass2 rows {e}/{N}  elapsed {time.time()-t0:.1f}s", flush=True)

    out = []
    for k in range(K):
        order = np.argsort(-top_buf[k])
        top_v = top_buf[k][order]
        top_i = top_idx_buf[k][order]
        order_b = np.argsort(bot_buf[k])
        bot_v = bot_buf[k][order_b]
        bot_i = bot_idx_buf[k][order_b]
        mean = float(sum_uk[k] / N)
        var = float(sumsq_uk[k] / N - mean * mean)
        std = float(max(var, 0.0) ** 0.5)
        abs_mean = float(abs_sum_uk[k] / N)
        frac_pos = float(pos_count[k] / N)
        out.append({
            "axis_k": k,
            "top_idx":  top_i.tolist(),
            "top_val":  [float(v) for v in top_v.tolist()],
            "bot_idx":  bot_i.tolist(),
            "bot_val":  [float(v) for v in bot_v.tolist()],
            "mean": mean,
            "std": std,
            "abs_mean": abs_mean,
            "frac_pos": frac_pos,
        })
    return out


def lookup_train_meta(meta_path: Path, indices: list[int]):
    """Return per-index dict with track_id, sqlite_idx, genre from metadata.json."""
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    tids = meta.get("train_track_ids", [])
    sids = meta.get("train_sqlite_indices", [])
    gens = meta.get("train_genres", [])

    def _g(lst, i):
        return lst[i] if i < len(lst) else None
    return [
        {
            "rank": rank,
            "train_pool_index": int(i),
            "train_track_id": _g(tids, i),
            "train_sqlite_idx": _g(sids, i),
            "train_genre": _g(gens, i),
        }
        for rank, i in enumerate(indices, start=1)
    ]


def diagnose_one(score_path: Path, meta_path: Path, K: int, chunk_rows: int,
                 top_train_k: int):
    print(f"\n=== {score_path} ===", flush=True)
    if not score_path.exists():
        print("  MISSING — skip", flush=True)
        return None
    t0 = time.time()
    N, n_test, sum_col, Gtt = streaming_StS(score_path, chunk_rows)
    print(f"  pass1 done. N={N} n_test={n_test}", flush=True)

    fro2 = float(np.trace(Gtt))
    eigvals, V_top = topk_eigh(Gtt, K)
    sigmas = np.sqrt(eigvals)  # singular values
    cum_var = np.cumsum(eigvals) / max(fro2, 1e-30)

    print(f"  top-{K} sigma^2/Frob^2: " + " ".join(
        f"{eigvals[k]/max(fro2,1e-30):.4f}" for k in range(K)
    ), flush=True)
    print(f"  cum var top1..topK: " + " ".join(
        f"{cum_var[k]:.4f}" for k in range(K)
    ), flush=True)

    # right singular vector stats (v_k) — sign asymmetry, etc.
    v_stats = []
    for k in range(K):
        v = V_top[:, k]
        v_stats.append({
            "axis_k": k,
            "v_mean": float(v.mean()),
            "v_std": float(v.std()),
            "v_abs_mean": float(np.abs(v).mean()),
            "v_frac_pos": float((v > 0).mean()),
            "v_min": float(v.min()),
            "v_max": float(v.max()),
        })

    u_stats = streaming_topk_U(score_path, V_top, sigmas, chunk_rows, top_train_k)

    # enrich top/bot entries with track meta
    for k_axis in range(min(3, K)):  # only enrich first 3 axes to keep JSON small
        u_stats[k_axis]["top_meta"] = lookup_train_meta(meta_path, u_stats[k_axis]["top_idx"])
        u_stats[k_axis]["bot_meta"] = lookup_train_meta(meta_path, u_stats[k_axis]["bot_idx"])

    elapsed = time.time() - t0
    summary = {
        "score_path": str(score_path),
        "N": int(N),
        "n_test": int(n_test),
        "frobenius_sq": fro2,
        "top_K": int(K),
        "top_eigvals": eigvals.tolist(),
        "top_sigma2_over_frob2": [float(eigvals[k] / max(fro2, 1e-30)) for k in range(K)],
        "cum_var_top": cum_var.tolist(),
        "v_stats": v_stats,
        "u_stats": u_stats,
        "elapsed_seconds": elapsed,
    }
    print(f"  done in {elapsed:.0f}s", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", required=True,
                   help="<root>:<method> shortcuts, expanded to all 3 stages")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--top_train_k", type=int, default=20,
                   help="How many top/bottom train indices to keep per axis")
    p.add_argument("--chunk_rows", type=int, default=8192)
    args = p.parse_args()

    runs = []
    for t in args.targets:
        if ":" in t:
            root, method = t.rsplit(":", 1)
            root = Path(root)
            for stage in ("semantic", "coarse", "fine"):
                f = root / stage / f"{stage}_{method}_scores.npy"
                meta = root / stage / f"{stage}_{method}_metadata.json"
                summary = diagnose_one(f, meta, args.K, args.chunk_rows, args.top_train_k)
                if summary is None:
                    continue
                summary["stage"] = stage
                summary["method"] = method
                runs.append(summary)
        else:
            raise ValueError("targets must use <root>:<method> form")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"runs": runs}, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
