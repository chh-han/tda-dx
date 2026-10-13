"""Smoke test for tda_dx.analysis.reliability.compute_segment_reliability."""
import numpy as np

from tda_dx.analysis.reliability import compute_segment_reliability


def test_segment_reliability_keys_and_ordering():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((40, 12)).astype(np.float64)
    out = compute_segment_reliability(S)
    for key in ("r1", "r2", "r3", "p", "sigma_1", "frob_sq"):
        assert key in out, f"missing key: {key}"
    assert 0.0 <= out["r1"] <= 1.0
    assert out["r1"] >= out["r2"] >= out["r3"]
    assert 0.0 <= out["p"] <= 1.0
    assert out["frob_sq"] > 0.0
    assert out["sigma_1"] >= 0.0
