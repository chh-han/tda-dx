"""summary.py — Channel-first summary statistics for feature alignment results."""
from __future__ import annotations

import math
from typing import Any

# Matches constants defined in analyze.py
FEATURE_DIMS = (
    "beat_pattern_sim",
    "onset_hist_sim",
    "chroma_sim",
    "tonnetz_sim",
    "chord_prog_sim",
    "mfcc_sim",
    "cqt_sim",
)
EVIDENCE_CHANNELS: dict[str, tuple[str, ...]] = {
    "rhythmic": ("beat_pattern_sim", "onset_hist_sim"),
    "harmonic": ("chroma_sim", "tonnetz_sim", "chord_prog_sim"),
    "timbral": ("mfcc_sim", "cqt_sim"),
}

FEATURE_DIMS_V2 = (
    "onset_hist_sim",
    "beat_interval_sim",
    "cv_beat_sim",
    "mean_beat_sim",
    "chroma_sim",
    "tonnetz_sim",
    "cqt_sim",
    "mfcc_sim",
    "spectral_centroid_sim",
)
EVIDENCE_CHANNELS_V2: dict[str, tuple[str, ...]] = {
    "rhythmic": ("onset_hist_sim", "beat_interval_sim", "cv_beat_sim", "mean_beat_sim"),
    "harmonic": ("chroma_sim", "tonnetz_sim"),
    "timbral": ("cqt_sim", "mfcc_sim", "spectral_centroid_sim"),
}

def _safe_mean(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _safe_std(vals: list[float]) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def _safe_median(vals: list[float]) -> float | None:
    valid = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not valid:
        return None
    n = len(valid)
    mid = n // 2
    return valid[mid] if n % 2 else (valid[mid - 1] + valid[mid]) / 2


def _frac_above(vals: list[float], threshold: float) -> float | None:
    valid = [v for v in vals if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return sum(1 for v in valid if v > threshold) / len(valid)


def build_enrichment_summary(
    per_test_rows: list[dict[str, Any]],
    condition: str,
    stage: str,
    method: str,
) -> dict[str, Any]:
    """Build the primary, paper-facing summary row for one stage/method."""
    channel_z_lists: dict[str, list[float]] = {ch: [] for ch in EVIDENCE_CHANNELS}
    for row in per_test_rows:
        for ch in EVIDENCE_CHANNELS:
            v = row.get(f"channel_z_{ch}")
            if v is not None:
                try:
                    channel_z_lists[ch].append(float(v))
                except (TypeError, ValueError):
                    pass

    summary: dict[str, Any] = {
        "condition": condition,
        "stage": stage,
        "method": method,
        "test_count": len(per_test_rows),
        "feature_table_hit_count": sum(1 for r in per_test_rows if r.get("feature_table_hit")),
    }

    for ch in EVIDENCE_CHANNELS:
        zs = channel_z_lists[ch]
        summary[f"mean_channel_z_{ch}"] = _safe_mean(zs)
        summary[f"std_channel_z_{ch}"] = _safe_std(zs)
        summary[f"median_channel_z_{ch}"] = _safe_median(zs)
        summary[f"frac_pos_channel_z_{ch}"] = _frac_above(zs, 0.0)
        summary[f"frac_sig_channel_z_{ch}"] = _frac_above(zs, 1.96)

    # Per-feature summary stats (independent of channel grouping)
    for dim in FEATURE_DIMS:
        feat_zs: list[float] = []
        for row in per_test_rows:
            v = row.get(f"enrichment_z_{dim}")
            if v is not None:
                try:
                    feat_zs.append(float(v))
                except (TypeError, ValueError):
                    pass
        summary[f"mean_feat_z_{dim}"] = _safe_mean(feat_zs)
        summary[f"frac_pos_feat_z_{dim}"] = _frac_above(feat_zs, 0.0)
        summary[f"frac_sig_feat_z_{dim}"] = _frac_above(feat_zs, 1.96)

    return summary
