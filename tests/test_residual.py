"""Smoke test for tda_dx.analysis.residual.rank1_residual."""
import numpy as np

from tda_dx.analysis.residual import rank1_residual


def test_rank1_residual_shape_and_energy():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((30, 8)).astype(np.float64)
    residual, info = rank1_residual(S)
    assert residual.shape == S.shape
    for key in ("sigma_1", "r1", "frob_sq", "frob_residual_sq"):
        assert key in info
    assert info["frob_residual_sq"] < info["frob_sq"]
    assert info["frob_sq"] > 0.0
    assert 0.0 <= info["r1"] <= 1.0
