"""Rank-1 (and rank-k) residual subtraction for segment-level score matrices.

Residual: S_res = S - sigma_1 * u_1 v_1^T, where (sigma_1, u_1, v_1) is the
top SVD component of S. Used in TDA-Dx paper Section 4.3 (Residual Analysis).

Only trust the residual when r_1 = sigma_1^2 / ||S||_F^2 is below ~0.9; for
larger r_1 the residual is dominated by numerical noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _top_svd_component(S: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return (sigma_1, u_1, v_1, frob_sq) using a Gram-matrix eigh on the
    smaller side. Both u_1 and v_1 are unit vectors.
    """
    S64 = np.asarray(S, dtype=np.float64)
    N, n_test = S64.shape
    if n_test <= N:
        Gtt = S64.T @ S64
        Gtt = 0.5 * (Gtt + Gtt.T)
        eigvals, eigvecs = np.linalg.eigh(Gtt)
        eigval1 = float(eigvals[-1])
        v1 = eigvecs[:, -1].astype(np.float64)
        sigma1 = float(max(eigval1, 0.0)) ** 0.5
        if sigma1 > 0:
            u1 = (S64 @ v1) / sigma1
        else:
            u1 = np.zeros(N, dtype=np.float64)
        frob_sq = float(np.trace(Gtt))
    else:
        Gnn = S64 @ S64.T
        Gnn = 0.5 * (Gnn + Gnn.T)
        eigvals, eigvecs = np.linalg.eigh(Gnn)
        eigval1 = float(eigvals[-1])
        u1 = eigvecs[:, -1].astype(np.float64)
        sigma1 = float(max(eigval1, 0.0)) ** 0.5
        if sigma1 > 0:
            v1 = (S64.T @ u1) / sigma1
        else:
            v1 = np.zeros(n_test, dtype=np.float64)
        frob_sq = float(np.trace(Gnn))
    return sigma1, u1, v1, frob_sq


def rank1_residual(S: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute S - sigma_1 * outer(u_1, v_1).

    Returns (residual, info) where info has keys
    sigma_1, r1, frob_sq, frob_residual_sq.
    """
    sigma1, u1, v1, frob_sq = _top_svd_component(S)
    rank1 = sigma1 * np.outer(u1, v1)
    residual = S.astype(np.result_type(S.dtype, np.float32), copy=True) - rank1.astype(
        np.result_type(S.dtype, np.float32)
    )
    frob_residual_sq = float((residual.astype(np.float64) ** 2).sum())
    r1 = float(sigma1 ** 2 / max(frob_sq, 1e-30))
    info = {
        "sigma_1": float(sigma1),
        "r1": r1,
        "frob_sq": float(frob_sq),
        "frob_residual_sq": frob_residual_sq,
    }
    return residual, info


def rank_k_residual(S: np.ndarray, k: int = 1) -> tuple[np.ndarray, dict]:
    """Subtract the top-k rank-1 components of S iteratively.

    Returns (residual, info) with keys sigma (list of singular values removed),
    r_k (cumulative energy fraction), frob_sq, frob_residual_sq.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    work = S.astype(np.result_type(S.dtype, np.float32), copy=True)
    sigmas: list[float] = []
    frob_sq = float((work.astype(np.float64) ** 2).sum())
    for _ in range(k):
        sigma_i, u_i, v_i, _ = _top_svd_component(work)
        sigmas.append(float(sigma_i))
        work -= (sigma_i * np.outer(u_i, v_i)).astype(work.dtype)
    frob_residual_sq = float((work.astype(np.float64) ** 2).sum())
    cum_energy = float(sum(s * s for s in sigmas) / max(frob_sq, 1e-30))
    info = {
        "k": int(k),
        "sigma": sigmas,
        "sigma_1": sigmas[0],
        "r1": float(sigmas[0] ** 2 / max(frob_sq, 1e-30)),
        "r_k": cum_energy,
        "frob_sq": frob_sq,
        "frob_residual_sq": frob_residual_sq,
    }
    return work, info


def write_residual_memmap(
    S: np.ndarray, output_path: str, dtype: str = "float32"
) -> dict:
    """Compute rank-1 residual and persist as a .npy memmap.

    Returns info dict with sigma_1, r1, frob_sq, frob_residual_sq, path, shape, dtype.
    """
    residual, info = rank1_residual(S)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        str(out), mode="w+", dtype=np.dtype(dtype), shape=residual.shape
    )
    mm[:] = residual.astype(np.dtype(dtype))
    mm.flush()
    del mm
    info["path"] = str(out)
    info["shape"] = list(residual.shape)
    info["dtype"] = str(np.dtype(dtype))
    return info
