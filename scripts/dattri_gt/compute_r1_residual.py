"""Subtract the top-1 SVD component from a score matrix: S_resid = S - σ₁ u₁ v₁ᵀ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--out_score", required=True)
    ap.add_argument("--out_info", default=None)
    args = ap.parse_args()

    S = np.load(args.score).astype(np.float32)
    N, T = S.shape
    print(f"S shape: {S.shape}  dtype: {S.dtype}")

    Gtt = (S.astype(np.float64).T @ S.astype(np.float64))
    fro2 = float(np.trace(Gtt))
    eigvals, eigvecs = np.linalg.eigh(0.5 * (Gtt + Gtt.T))
    eigval1 = float(eigvals[-1])
    v1 = eigvecs[:, -1].astype(np.float64)
    sigma1 = float(eigval1) ** 0.5
    top1_ratio = float(eigval1 / max(fro2, 1e-30))
    print(f"sigma1^2/Frob^2 = {top1_ratio:.4f}  sigma1 = {sigma1:.4f}")

    # u = (S @ v) / sigma_1  →  σ_1 * u * v^T = (S @ v)[:, None] * v[None, :]
    u_unscaled = S.astype(np.float64) @ v1                      # shape (N,)
    resid = (S.astype(np.float64) - u_unscaled[:, None] * v1[None, :]).astype(np.float32)
    np.save(args.out_score, resid)
    print(f"saved residual: {args.out_score}  shape={resid.shape}  dtype={resid.dtype}")

    info = {
        "src": str(args.score),
        "out": str(args.out_score),
        "N": int(N), "n_test": int(T),
        "frob_sq": fro2,
        "sigma1_sq": eigval1,
        "sigma1_sq_over_frob_sq": top1_ratio,
        "v1_stats": {
            "mean": float(v1.mean()),
            "std": float(v1.std()),
            "abs_mean": float(np.abs(v1).mean()),
            "frac_pos": float((v1 > 0).mean()),
            "min": float(v1.min()),
            "max": float(v1.max()),
        },
    }
    if args.out_info:
        Path(args.out_info).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_info).write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
