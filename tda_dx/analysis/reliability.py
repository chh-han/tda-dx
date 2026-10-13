"""Score-matrix reliability diagnostics for TDA-Dx (paper section 3.3).

Computes singular-value energy fractions r_k = sigma_k^2 / ||S||_F^2,
popularity fraction p (column-mean energy share), the dominant left
singular vector u_1, and channel alignment alpha_c (max per-dim Pearson
and multivariate sqrt(R^2) from OLS).

All metrics operate on the raw segment score matrix S^seg (no per-query
normalization). For (N, n_test) matrices we route the eigendecomposition
through the smaller (n_test, n_test) Gram S^T S (matching the convention
used by the source diagnostic scripts).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


__all__ = [
    "compute_singular_energy",
    "compute_popularity_fraction",
    "compute_u1",
    "compute_alpha_max",
    "compute_alpha_reg",
    "compute_segment_reliability",
    "top_axis_indices",
]


def _topk_via_gram(S: np.ndarray, k: int):
    """Return (sigma[:k], U[:, :k], V[:, :k]) using the smaller side.

    For tall S this routes through S^T S so we never materialise an
    (N, N) matrix. Eigenvalues are clipped to >= 0.
    """
    S64 = np.asarray(S, dtype=np.float64)
    N, M = S64.shape
    k = min(k, min(N, M))
    if N >= M:
        # Right side: V via eigendecomposition of S^T S (M x M)
        G = S64.T @ S64
        G = 0.5 * (G + G.T)
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        eigvals = np.clip(eigvals, 0.0, None)
        sigmas = np.sqrt(eigvals[:k])
        V = eigvecs[:, :k]
        # U = S V / sigma
        U = np.zeros((N, k), dtype=np.float64)
        for j in range(k):
            s = max(sigmas[j], 1e-30)
            U[:, j] = (S64 @ V[:, j]) / s
        return sigmas, U, V, float(eigvals.sum())
    # Left side: U via eigendecomposition of S S^T (N x N)
    G = S64 @ S64.T
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    sigmas = np.sqrt(eigvals[:k])
    U = eigvecs[:, :k]
    V = np.zeros((M, k), dtype=np.float64)
    for j in range(k):
        s = max(sigmas[j], 1e-30)
        V[:, j] = (S64.T @ U[:, j]) / s
    return sigmas, U, V, float(eigvals.sum())


def compute_singular_energy(S: np.ndarray, k: int = 3) -> dict:
    """Return top-k singular values and their energy fractions r_k.

    r_k = sigma_k^2 / ||S||_F^2. Output keys: r1, r2, r3, sigma_1,
    sigma_2, sigma_3, frob_sq.
    """
    S64 = np.asarray(S, dtype=np.float64)
    frob_sq = float((S64 * S64).sum())
    sigmas, _U, _V, _eig_sum = _topk_via_gram(S64, max(k, 3))
    out = {"frob_sq": frob_sq}
    for i in range(3):
        sigma_i = float(sigmas[i]) if i < len(sigmas) else 0.0
        out[f"sigma_{i + 1}"] = sigma_i
        out[f"r{i + 1}"] = float(sigma_i * sigma_i / max(frob_sq, 1e-30))
    return out


def compute_popularity_fraction(S: np.ndarray) -> float:
    """p = mean over columns of (column_mean^2 * N) / column_energy.

    Per-column energy is sum_i S[i,j]^2; per-column mean-component
    energy is N * mu_j^2. The popularity fraction is the average of
    that ratio across columns.
    """
    S64 = np.asarray(S, dtype=np.float64)
    N = S64.shape[0]
    norm_sq_col = (S64 * S64).sum(axis=0)
    mu = S64.mean(axis=0)
    mu_col_sq = N * (mu * mu)
    pop_per_col = np.divide(mu_col_sq, np.maximum(norm_sq_col, 1e-30))
    return float(pop_per_col.mean())


def compute_u1(S: np.ndarray) -> np.ndarray:
    """Dominant left singular vector u_1 (unit-norm)."""
    _sigmas, U, _V, _ = _topk_via_gram(np.asarray(S, dtype=np.float64), 1)
    return U[:, 0]


def compute_alpha_max(u1: np.ndarray, feature_matrix: np.ndarray) -> float:
    """Max |Pearson(u1, feature_matrix[:, d])| over columns d.

    Rows with NaN in any feature column are dropped per-column. Columns
    with zero variance contribute 0.
    """
    u = np.asarray(u1, dtype=np.float64).reshape(-1)
    X = np.asarray(feature_matrix, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != u.shape[0] or X.shape[1] == 0:
        return 0.0
    best = 0.0
    for d in range(X.shape[1]):
        col = X[:, d]
        m = ~np.isnan(col)
        if m.sum() < 3:
            continue
        cu = u[m]
        cc = col[m]
        if cc.std() == 0 or cu.std() == 0:
            continue
        r, _ = pearsonr(cu, cc)
        if np.isnan(r):
            continue
        best = max(best, abs(float(r)))
    return best


def compute_alpha_reg(u1: np.ndarray, feature_matrix: np.ndarray) -> float:
    """sqrt(R^2) from OLS regressing u1 on the columns of feature_matrix.

    Both sides are mean-centered; rows with any NaN feature are dropped.
    """
    u = np.asarray(u1, dtype=np.float64).reshape(-1)
    X = np.asarray(feature_matrix, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != u.shape[0] or X.shape[1] == 0:
        return 0.0
    mask = ~np.isnan(X).any(axis=1)
    if mask.sum() < 3:
        return 0.0
    Xv = X[mask]
    uv = u[mask]
    u_c = uv - uv.mean()
    Xc = Xv - Xv.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Xc, u_c, rcond=None)
    pred = Xc @ coef
    ss_res = float(((u_c - pred) ** 2).sum())
    ss_tot = float((u_c ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    R2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return float(np.sqrt(R2))


def compute_segment_reliability(
    S_seg: np.ndarray,
    channel_features: dict[str, np.ndarray] | None = None,
) -> dict:
    """Bundle r1/r2/r3, p, and per-channel alpha for a raw segment matrix.

    `channel_features` maps channel name -> (N, D_c) feature matrix
    aligned with the rows of S_seg. Returns a dict with 'r1', 'r2',
    'r3', 'sigma_1..3', 'frob_sq', 'p', and (if features given) for
    each channel c: 'alpha_max_<c>', 'alpha_reg_<c>'.
    """
    out: dict = {}
    out.update(compute_singular_energy(S_seg, k=3))
    out["p"] = compute_popularity_fraction(S_seg)
    if channel_features:
        u1 = compute_u1(S_seg)
        for ch, feat in channel_features.items():
            out[f"alpha_max_{ch}"] = compute_alpha_max(u1, feat)
            out[f"alpha_reg_{ch}"] = compute_alpha_reg(u1, feat)
    return out


def top_axis_indices(
    S: np.ndarray,
    axis: int = 0,
    k: int = 1,
    top_n: int = 20,
) -> dict:
    """For SVD axis `axis` (0-indexed), return top-N / bot-N row indices.

    Ranks rows by S @ v_axis (proportional to u_axis since sigma > 0).
    `k` selects how many leading axes to compute; only `axis` is
    returned but `k` lets callers reuse the decomposition cost.
    """
    S64 = np.asarray(S, dtype=np.float64)
    K = max(k, axis + 1)
    sigmas, U, V, _eig_sum = _topk_via_gram(S64, K)
    sigma_a = float(sigmas[axis]) if axis < len(sigmas) else 0.0
    u_axis = U[:, axis] * sigma_a  # ranking-equivalent score on axis
    n = u_axis.shape[0]
    top_n_eff = min(top_n, n)
    top_idx = np.argpartition(-u_axis, top_n_eff - 1)[:top_n_eff]
    top_idx = top_idx[np.argsort(-u_axis[top_idx])]
    bot_idx = np.argpartition(u_axis, top_n_eff - 1)[:top_n_eff]
    bot_idx = bot_idx[np.argsort(u_axis[bot_idx])]
    return {
        "axis": int(axis),
        "sigma": sigma_a,
        "top_idx": top_idx.astype(np.int64).tolist(),
        "top_val": [float(v) for v in u_axis[top_idx]],
        "bot_idx": bot_idx.astype(np.int64).tolist(),
        "bot_val": [float(v) for v in u_axis[bot_idx]],
    }
