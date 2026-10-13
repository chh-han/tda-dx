"""Rank-1 SVD removal on attribution score matrices.

For each (method, stage) input score matrix S of shape (N, n_test):
  Pass 1: Compute Gtt = S^T S; eigh -> top-1 right singular vector v_1
          and sigma_1 = sqrt(eigval_1).
  Pass 2: Stream rows; for each chunk B compute u_block = (B @ v_1) / sigma_1
          and write residual S_resid_block = B - sigma_1 * u_block[:, None] * v_1[None, :]
          to a fp16 npy in the output folder.

Saves output in the layout the analyze pipeline expects:
  <out_root>/<method>-r1/<stage>/<stage>_<method>_scores.npy   (fp16)
  <out_root>/<method>-r1/<stage>/<stage>_<method>_scores_with_ids.npz
  <out_root>/<method>-r1/<stage>/<stage>_<method>_metadata.json (cloned + updated)
  <out_root>/<method>-r1/<stage>/<stage>_<method>_per_test_rankings.{json,tsv}
  <out_root>/<method>-r1/<stage>/test_losses.npy             (copied)
  <out_root>/<method>-r1/<stage>/train_indices.npy           (copied)
  <out_root>/<method>-r1/<stage>/_rank1_info.json            (sigma_1, ||v_1||, ...)

We keep the method name in filenames identical to the original (e.g.
"<stage>_grad_cos_scores.npy") so the analyze pipeline reads them without
modification — only the parent folder differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np


def streaming_StS(score_path: Path, chunk_rows: int):
    S = np.load(score_path, mmap_mode="r")
    N, n_test = S.shape
    Gtt = np.zeros((n_test, n_test), dtype=np.float64)
    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.asarray(S[s:e], dtype=np.float64)
        Gtt += block.T @ block
        if (s // chunk_rows) % 16 == 0:
            print(f"    pass1 rows {e}/{N}  elapsed {time.time()-t0:.1f}s", flush=True)
    return N, n_test, Gtt


def top_eig(Gtt: np.ndarray):
    G_sym = 0.5 * (Gtt + Gtt.T)
    eigvals, eigvecs = np.linalg.eigh(G_sym)
    return float(eigvals[-1]), eigvecs[:, -1].astype(np.float64)


def stream_write_residual(score_path: Path, out_npy: Path, v1: np.ndarray,
                          sigma1: float, chunk_rows: int):
    S = np.load(score_path, mmap_mode="r")
    N, n_test = S.shape
    out_mm = np.lib.format.open_memmap(
        str(out_npy), mode="w+", dtype=np.float16, shape=(N, n_test)
    )
    v1_f32 = v1.astype(np.float32)
    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.array(S[s:e], dtype=np.float32, copy=True)  # writable copy
        u_block = block @ v1_f32                              # (b,)
        # subtract σ_1 * u * v^T. note u_block already includes σ_1 since
        # u (unit) = (S @ v) / σ_1, so σ_1 * u = S @ v = u_block_unscaled.
        # But we computed u_block = S @ v (no divide); so the rank-1 component
        # is u_block[:, None] * v1[None, :], no extra σ_1 factor needed.
        block -= u_block[:, None] * v1_f32[None, :]
        out_mm[s:e] = block.astype(np.float16)
        if (s // chunk_rows) % 16 == 0:
            print(f"    pass2 rows {e}/{N}  elapsed {time.time()-t0:.1f}s",
                  flush=True)
    out_mm.flush()
    del out_mm
    print(f"    wrote {out_npy}  ({time.time()-t0:.0f}s)", flush=True)


def compute_rankings_metadata(
    out_stage: Path,
    stage: str,
    method: str,
    orig_meta: dict,
    scores_path: Path,
    train_indices: np.ndarray,
    sigma1: float,
    v1_stats: dict,
    elapsed: float,
):
    print(f"  building rankings + metadata from {scores_path} ...", flush=True)
    scores = np.load(scores_path, mmap_mode="r")  # (N, n_test) fp16
    num_train, num_test = scores.shape
    per_test_rank_k = min(10, num_train)

    # global mean influence
    mean_inf = np.zeros(num_train, dtype=np.float32)
    chunk = 8192
    for s in range(0, num_train, chunk):
        e = min(s + chunk, num_train)
        block = np.asarray(scores[s:e], dtype=np.float32)
        mean_inf[s:e] = block.mean(axis=1)
    top_idx = np.argsort(-mean_inf)[:per_test_rank_k]
    bot_idx = np.argsort(mean_inf)[:per_test_rank_k]

    train_track_ids = list(orig_meta.get("train_track_ids", []))
    train_sqlite_indices = list(orig_meta.get("train_sqlite_indices", []))
    train_genres = list(orig_meta.get("train_genres", []))
    test_track_ids = list(orig_meta.get("test_track_ids", []))
    test_genres = list(orig_meta.get("test_genres", []))

    per_test_rankings = {}
    rows_tsv = []
    for j in range(num_test):
        col = np.asarray(scores[:, j], dtype=np.float32)
        topp = np.argsort(-col)[:per_test_rank_k]
        botp = np.argsort(col)[:per_test_rank_k]
        top_items, bot_items = [], []
        for rank, p in enumerate(topp.tolist(), start=1):
            tds = int(train_indices[p])
            tsq = int(train_sqlite_indices[p]) if p < len(train_sqlite_indices) else -1
            ttid = train_track_ids[p] if p < len(train_track_ids) else None
            tg = train_genres[p] if p < len(train_genres) else None
            sc = float(col[p])
            top_items.append({
                "rank": rank, "train_pool_index": int(p),
                "train_ds_index": tds, "train_sqlite_idx": tsq,
                "train_track_id": ttid, "train_genre": tg, "score": sc,
            })
            rows_tsv.append([
                j, test_track_ids[j] if j < len(test_track_ids) else None,
                test_genres[j] if j < len(test_genres) else None,
                "top", rank, int(p), tds, tsq, ttid, tg, sc,
            ])
        for rank, p in enumerate(botp.tolist(), start=1):
            tds = int(train_indices[p])
            tsq = int(train_sqlite_indices[p]) if p < len(train_sqlite_indices) else -1
            ttid = train_track_ids[p] if p < len(train_track_ids) else None
            tg = train_genres[p] if p < len(train_genres) else None
            sc = float(col[p])
            bot_items.append({
                "rank": rank, "train_pool_index": int(p),
                "train_ds_index": tds, "train_sqlite_idx": tsq,
                "train_track_id": ttid, "train_genre": tg, "score": sc,
            })
            rows_tsv.append([
                j, test_track_ids[j] if j < len(test_track_ids) else None,
                test_genres[j] if j < len(test_genres) else None,
                "bottom", rank, int(p), tds, tsq, ttid, tg, sc,
            ])
        per_test_rankings[str(j)] = {
            "test_index": j,
            "test_track_id": test_track_ids[j] if j < len(test_track_ids) else None,
            "test_genre": test_genres[j] if j < len(test_genres) else None,
            "topk": top_items,
            "bottomk": bot_items,
        }

    rk_json = out_stage / f"{stage}_{method}_per_test_rankings.json"
    with open(rk_json, "w") as f:
        json.dump(per_test_rankings, f, indent=2)
    rk_tsv = out_stage / f"{stage}_{method}_per_test_rankings.tsv"
    with open(rk_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "test_index", "test_track_id", "test_genre",
            "rank_type", "rank",
            "train_pool_index", "train_ds_index", "train_sqlite_idx",
            "train_track_id", "train_genre", "score",
        ])
        w.writerows(rows_tsv)

    new_meta = dict(orig_meta)
    new_meta["run_id"] = orig_meta.get("run_id", "") + "__r1removed"
    new_meta["scores_shape"] = [int(num_train), int(num_test)]
    new_meta["scores_dtype"] = "float16"
    new_meta["rank1_removed"] = True
    new_meta["rank1_sigma1"] = float(sigma1)
    new_meta["rank1_v1_stats"] = v1_stats
    new_meta["attribution_time_seconds"] = float(elapsed)
    new_meta["per_test_rankings_json_path"] = str(rk_json)
    new_meta["per_test_rankings_tsv_path"] = str(rk_tsv)
    new_meta["top_influential_indices"] = [int(train_indices[i]) for i in top_idx.tolist()]
    new_meta["top_influential_sqlite_indices"] = [
        int(train_sqlite_indices[i]) if i < len(train_sqlite_indices) else -1
        for i in top_idx.tolist()
    ]
    new_meta["top_influential_track_ids"] = [
        train_track_ids[i] if i < len(train_track_ids) else None
        for i in top_idx.tolist()
    ]
    new_meta["top_influential_scores"] = [float(mean_inf[i]) for i in top_idx.tolist()]
    new_meta["bottom_influential_indices"] = [int(train_indices[i]) for i in bot_idx.tolist()]
    new_meta["bottom_influential_sqlite_indices"] = [
        int(train_sqlite_indices[i]) if i < len(train_sqlite_indices) else -1
        for i in bot_idx.tolist()
    ]
    new_meta["bottom_influential_track_ids"] = [
        train_track_ids[i] if i < len(train_track_ids) else None
        for i in bot_idx.tolist()
    ]
    new_meta["bottom_influential_scores"] = [float(mean_inf[i]) for i in bot_idx.tolist()]
    meta_path = out_stage / f"{stage}_{method}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)


def process_one(method: str, orig_root: Path, stage: str, out_method_root: Path,
                chunk_rows: int):
    orig_stage = orig_root / stage
    score_path = orig_stage / f"{stage}_{method}_scores.npy"
    meta_path = orig_stage / f"{stage}_{method}_metadata.json"
    if not score_path.exists():
        print(f"  SKIP: {score_path} missing", flush=True)
        return
    out_stage = out_method_root / stage
    out_stage.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {method} / {stage} ===", flush=True)
    print(f"  src: {score_path}", flush=True)
    print(f"  dst: {out_stage}", flush=True)

    t0 = time.time()
    N, n_test, Gtt = streaming_StS(score_path, chunk_rows)
    fro2 = float(np.trace(Gtt))
    eigval1, v1 = top_eig(Gtt)
    sigma1 = float(eigval1) ** 0.5
    top1_ratio = float(eigval1 / max(fro2, 1e-30))
    print(f"  N={N} n_test={n_test}  σ_1²/Frob² = {top1_ratio:.4f}", flush=True)

    v1_stats = {
        "mean": float(v1.mean()),
        "std": float(v1.std()),
        "abs_mean": float(np.abs(v1).mean()),
        "frac_pos": float((v1 > 0).mean()),
        "min": float(v1.min()),
        "max": float(v1.max()),
    }

    out_score = out_stage / f"{stage}_{method}_scores.npy"
    stream_write_residual(score_path, out_score, v1, sigma1, chunk_rows)

    # copy small files
    for fn in ("test_losses.npy", "train_indices.npy", "experiment_config.json"):
        src = orig_stage / fn
        if src.exists():
            shutil.copy(src, out_stage / fn)
    train_indices = np.load(orig_stage / "train_indices.npy")
    if train_indices.shape[0] != N:
        raise RuntimeError(f"train_indices len {train_indices.shape[0]} vs N={N}")

    # metadata + rankings
    with open(meta_path) as f:
        orig_meta = json.load(f)
    compute_rankings_metadata(
        out_stage=out_stage, stage=stage, method=method,
        orig_meta=orig_meta, scores_path=out_score,
        train_indices=train_indices, sigma1=sigma1, v1_stats=v1_stats,
        elapsed=time.time() - t0,
    )

    # diagnostic info
    info = {
        "method": method, "stage": stage,
        "N": int(N), "n_test": int(n_test),
        "frob_sq": fro2,
        "sigma1_sq": float(eigval1),
        "sigma1_sq_over_frob_sq": top1_ratio,
        "v1_stats": v1_stats,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_stage / "_rank1_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # also write scores_with_ids.npz (used by some downstream tools)
    train_track_ids = list(orig_meta.get("train_track_ids", []))
    np.savez(
        str(out_stage / f"{stage}_{method}_scores_with_ids.npz"),
        scores=np.load(out_score, mmap_mode="r"),
        train_track_ids=np.array(train_track_ids, dtype=object),
        train_indices=train_indices,
    )
    print(f"  done in {time.time()-t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", action="append", required=True,
                   help='one or more "<method>:<orig_root>" — e.g. grad_cos:/path/to/grad_cos-proj4096')
    p.add_argument("--out_root", type=Path, required=True,
                   help="output base directory; per-method subdirs created")
    p.add_argument("--stages", nargs="+", default=["semantic", "coarse", "fine"])
    p.add_argument("--chunk_rows", type=int, default=8192)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    for tgt in args.target:
        method, orig = tgt.split(":", 1)
        orig_root = Path(orig)
        out_method_root = args.out_root / f"{method}-r1removed"
        out_method_root.mkdir(parents=True, exist_ok=True)
        for stage in args.stages:
            process_one(method, orig_root, stage, out_method_root, args.chunk_rows)
    print("\nALL DONE.")


if __name__ == "__main__":
    main()
