# Within-group musical homogeneity measurement for TDA-Dx.

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from tda_dx.analysis.io_utils import _safe_float, _l1_normalize, _mean, _std, _cosine_similarity
from tda_dx.analysis._common import (
    FEATURE_DIMS,
    _mean_var_vector,
    _beat_pattern_vector,
    _tempo_difference,
    _split_chord_token,
    _root_motion_tokens,
    _normalized_lcs_similarity,
    _normalized_lcs_similarity_cached,
    _mean_pairwise_similarity,
    _cosine_similarity_np,
    _dominant_ratio,
)


def load_feature_table(features_tsv_path: Path) -> dict[str, dict[str, Any]]:
    """Load per-track musicological features from features.tsv into {track_id -> feature dict}."""
    if not features_tsv_path.exists():
        return {}
    try:
        import pandas as pd
        try:
            import orjson as _json
            def _jloads(s): return _json.loads(s) if s and s != "[]" else []
        except ImportError:
            def _jloads(s): return json.loads(s) if s and s != "[]" else []

        _needed_cols = [
            "track_id", "genre_top", "key", "mode", "tempo_bpm",
            "beat_times", "mean_beat_interval", "median_beat_interval", "std_beat_interval", "iqr_beat_interval", "cv_beat_interval",
            "beat_class_hist", "beat_interval_hist", "onset_interval_hist",
            "spectral_centroid_mean",
            "stft_mean", "stft_var",
            "mfcc_mean", "mfcc_var",
            "cqt_mean", "cqt_var",
            "vqt_mean", "vqt_var",
            "chroma_mean", "tonnetz_mean", "chord_tokens", "chord_roots", "chord_types", "chord_sequence_collapsed",
        ]
        df = pd.read_csv(
            features_tsv_path, sep="\t", dtype=str, na_filter=False,
            usecols=lambda c: c in _needed_cols,
        )

        def _parse_float_list(s):
            try:
                parsed = _jloads(s)
                return [float(v) for v in parsed] if isinstance(parsed, list) else []
            except Exception:
                return []

        def _parse_str_list(s):
            try:
                parsed = _jloads(s)
                return [str(v) for v in parsed] if isinstance(parsed, list) else []
            except Exception:
                return []

        table: dict[str, dict[str, Any]] = {}
        for row in df.itertuples(index=False):
            raw_id = getattr(row, "track_id", "").strip()
            if not raw_id:
                continue
            try:
                track_id = str(int(raw_id))
            except (ValueError, TypeError):
                track_id = raw_id
            try:
                tempo = float(getattr(row, "tempo_bpm", "") or 0.0)
            except (ValueError, TypeError):
                tempo = 0.0
            table[track_id] = {
                "genre_top": getattr(row, "genre_top", "").strip(),
                "key": getattr(row, "key", "").strip(),
                "mode": getattr(row, "mode", "").strip(),
                "tempo_bpm": tempo,
                "beat_times": _parse_float_list(getattr(row, "beat_times", "[]")),
                "mean_beat_interval": float(getattr(row, "mean_beat_interval", "") or 0.0) if getattr(row, "mean_beat_interval", "") not in ("", None) else None,
                "median_beat_interval": float(getattr(row, "median_beat_interval", "") or 0.0) if getattr(row, "median_beat_interval", "") not in ("", None) else None,
                "std_beat_interval": float(getattr(row, "std_beat_interval", "") or 0.0) if getattr(row, "std_beat_interval", "") not in ("", None) else None,
                "iqr_beat_interval": float(getattr(row, "iqr_beat_interval", "") or 0.0) if getattr(row, "iqr_beat_interval", "") not in ("", None) else None,
                "cv_beat_interval": float(getattr(row, "cv_beat_interval", "") or 0.0) if getattr(row, "cv_beat_interval", "") not in ("", None) else None,
                "beat_class_hist": _parse_float_list(getattr(row, "beat_class_hist", "[]")),
                "beat_interval_hist": _l1_normalize(_parse_float_list(getattr(row, "beat_interval_hist", "[]"))),
                "onset_interval_hist": _l1_normalize(_parse_float_list(getattr(row, "onset_interval_hist", "[]"))),
                "stft_mean": _parse_float_list(getattr(row, "stft_mean", "[]")),
                "stft_var": _parse_float_list(getattr(row, "stft_var", "[]")),
                "mfcc_mean": _parse_float_list(getattr(row, "mfcc_mean", "[]")),
                "mfcc_var": _parse_float_list(getattr(row, "mfcc_var", "[]")),
                "cqt_mean": _parse_float_list(getattr(row, "cqt_mean", "[]")),
                "cqt_var": _parse_float_list(getattr(row, "cqt_var", "[]")),
                "vqt_mean": _parse_float_list(getattr(row, "vqt_mean", "[]")),
                "vqt_var": _parse_float_list(getattr(row, "vqt_var", "[]")),
                "spectral_centroid_mean": _safe_float(getattr(row, "spectral_centroid_mean", "")),
                "chroma_mean": _parse_float_list(getattr(row, "chroma_mean", "[]")),
                "tonnetz_mean": _parse_float_list(getattr(row, "tonnetz_mean", "[]")),
                "chord_tokens": set(_parse_str_list(getattr(row, "chord_tokens", "[]"))),
                "chord_roots": _parse_str_list(getattr(row, "chord_roots", "[]")),
                "chord_types": _parse_str_list(getattr(row, "chord_types", "[]")),
                "chord_sequence_collapsed": _parse_str_list(getattr(row, "chord_sequence_collapsed", "[]")),
            }
        return table
    except Exception:
        return {}


def build_feature_context(feature_table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Precompute reusable feature transforms once to avoid repeating work per query/sample."""
    def _zstats(vecs: list[list[float]]) -> tuple[list[float], list[float]] | None:
        if not vecs:
            return None
        dim_counts: dict[int, int] = defaultdict(int)
        for vec in vecs:
            dim_counts[len(vec)] += 1
        if not dim_counts:
            return None
        target_dim = max(dim_counts.items(), key=lambda item: item[1])[0]
        filtered = [v for v in vecs if len(v) == target_dim]
        if not filtered:
            return None
        d = target_dim
        g_mean = [sum(v[i] for v in filtered) / len(filtered) for i in range(d)]
        g_std = [
            math.sqrt(sum((v[i] - g_mean[i]) ** 2 for v in filtered) / len(filtered)) + 1e-8
            for i in range(d)
        ]
        return (g_mean, g_std)

    stft_vecs: list[list[float]] = []
    mfcc_vecs: list[list[float]] = []
    cqt_vecs: list[list[float]] = []
    vqt_vecs: list[list[float]] = []
    chroma_vecs: list[list[float]] = []
    tonnetz_vecs: list[list[float]] = []
    beat_pattern_vecs: list[list[float]] = []
    onset_hist_vecs: list[list[float]] = []
    for feat in feature_table.values():
        stft_vec = _mean_var_vector(feat.get("stft_mean", []), feat.get("stft_var", []))
        if stft_vec:
            stft_vecs.append(stft_vec)
        mean_v = feat.get("mfcc_mean", [])
        std_v = feat.get("mfcc_var", [])
        mfcc_vec = _mean_var_vector(mean_v, std_v)
        if mfcc_vec:
            mfcc_vecs.append(mfcc_vec)
        cqt_vec = _mean_var_vector(feat.get("cqt_mean", []), feat.get("cqt_var", []))
        if cqt_vec:
            cqt_vecs.append(cqt_vec)
        vqt_vec = _mean_var_vector(feat.get("vqt_mean", []), feat.get("vqt_var", []))
        if vqt_vec:
            vqt_vecs.append(vqt_vec)
        chroma_v = feat.get("chroma_mean", [])
        if chroma_v:
            chroma_vecs.append(chroma_v)
        tonnetz_v = feat.get("tonnetz_mean", [])
        if tonnetz_v:
            tonnetz_vecs.append(tonnetz_v)
        beat_pattern_v = _beat_pattern_vector(feat)
        if beat_pattern_v:
            beat_pattern_vecs.append(beat_pattern_v)
        onset_hist_v = feat.get("onset_interval_hist", [])
        if onset_hist_v:
            onset_hist_vecs.append(onset_hist_v)
    beat_interval_hist_vecs: list[list[float]] = [
        feat.get("beat_interval_hist", [])
        for feat in feature_table.values()
        if feat.get("beat_interval_hist")
    ]
    stft_zstats = _zstats(stft_vecs)
    mfcc_zstats = _zstats(mfcc_vecs)
    cqt_zstats = _zstats(cqt_vecs)
    vqt_zstats = _zstats(vqt_vecs)
    chroma_zstats = _zstats(chroma_vecs)
    tonnetz_zstats = _zstats(tonnetz_vecs)
    beat_hist_zstats = _zstats(beat_pattern_vecs)
    onset_hist_zstats = _zstats(onset_hist_vecs)
    beat_interval_hist_zstats = _zstats(beat_interval_hist_vecs)

    def _as_array(values: list[float]) -> np.ndarray | None:
        if not values:
            return None
        return np.asarray(values, dtype=np.float64)

    def _z_array(values: list[float], zstats: tuple[list[float], list[float]] | None) -> np.ndarray | None:
        if not values or zstats is None:
            return None
        means, stds = zstats
        if len(values) != len(means):
            return None
        arr = np.asarray(values, dtype=np.float64)
        return (arr - np.asarray(means, dtype=np.float64)) / np.asarray(stds, dtype=np.float64)

    prepared_tracks: dict[str, dict[str, Any]] = {}
    for track_id, feat in feature_table.items():
        stft_vec = _mean_var_vector(feat.get("stft_mean", []), feat.get("stft_var", []))
        mfcc_vec = _mean_var_vector(feat.get("mfcc_mean", []), feat.get("mfcc_var", []))
        cqt_vec = _mean_var_vector(feat.get("cqt_mean", []), feat.get("cqt_var", []))
        vqt_vec = _mean_var_vector(feat.get("vqt_mean", []), feat.get("vqt_var", []))
        prepared_tracks[track_id] = {
            "genre_top": feat.get("genre_top", ""),
            "key": feat.get("key", ""),
            "tempo_bpm": float(feat["tempo_bpm"]) if isinstance(feat.get("tempo_bpm"), (int, float)) and feat.get("tempo_bpm") > 0 else None,
            "beat_pattern_vec": _as_array(_beat_pattern_vector(feat)),
            "beat_pattern_vec_z": _z_array(_beat_pattern_vector(feat), beat_hist_zstats),
            "beat_interval_hist": _as_array(feat.get("beat_interval_hist", [])),
            "beat_interval_hist_z": _z_array(feat.get("beat_interval_hist", []), beat_interval_hist_zstats),
            "cv_beat_interval": feat.get("cv_beat_interval"),
            "mean_beat_interval": feat.get("mean_beat_interval"),
            "spectral_centroid_mean": feat.get("spectral_centroid_mean"),
            "onset_interval_hist": _as_array(feat.get("onset_interval_hist", [])),
            "onset_interval_hist_z": _z_array(feat.get("onset_interval_hist", []), onset_hist_zstats),
            "stft_vec": _as_array(stft_vec),
            "stft_vec_z": _z_array(stft_vec, stft_zstats),
            "mfcc_vec": _as_array(mfcc_vec),
            "mfcc_vec_z": _z_array(mfcc_vec, mfcc_zstats),
            "cqt_vec": _as_array(cqt_vec),
            "cqt_vec_z": _z_array(cqt_vec, cqt_zstats),
            "vqt_vec": _as_array(vqt_vec),
            "vqt_vec_z": _z_array(vqt_vec, vqt_zstats),
            "chroma_mean": _as_array(feat.get("chroma_mean", [])),
            "chroma_mean_z": _z_array(feat.get("chroma_mean", []), chroma_zstats),
            "tonnetz_mean": _as_array(feat.get("tonnetz_mean", [])),
            "tonnetz_mean_z": _z_array(feat.get("tonnetz_mean", []), tonnetz_zstats),
            "chord_tokens": feat.get("chord_tokens", set()) if isinstance(feat.get("chord_tokens", set()), set) else set(feat.get("chord_tokens", [])),
            "chord_roots": list(feat.get("chord_roots", [])),
            "chord_types": list(feat.get("chord_types", [])),
            "chord_sequence_collapsed": list(feat.get("chord_sequence_collapsed", [])),
            "chord_root_motion": _root_motion_tokens([_split_chord_token(t)[0] for t in feat.get("chord_sequence_collapsed", [])]),
        }

    return {
        "prepared_tracks": prepared_tracks,
        "stft_zstats": stft_zstats,
        "mfcc_zstats": mfcc_zstats,
        "cqt_zstats": cqt_zstats,
        "vqt_zstats": vqt_zstats,
        "chroma_zstats": chroma_zstats,
        "tonnetz_zstats": tonnetz_zstats,
        "beat_hist_zstats": beat_hist_zstats,
        "onset_hist_zstats": onset_hist_zstats,
        "beat_interval_hist_zstats": beat_interval_hist_zstats,
    }


def compute_track_similarity(
    feat_a: dict[str, Any],
    feat_b: dict[str, Any],
    stft_zstats: tuple[list[float], list[float]] | None = None,
    mfcc_zstats: tuple[list[float], list[float]] | None = None,
    cqt_zstats: tuple[list[float], list[float]] | None = None,
    vqt_zstats: tuple[list[float], list[float]] | None = None,
    chroma_zstats: tuple[list[float], list[float]] | None = None,
    tonnetz_zstats: tuple[list[float], list[float]] | None = None,
    beat_hist_zstats: tuple[list[float], list[float]] | None = None,
    onset_hist_zstats: tuple[list[float], list[float]] | None = None,
) -> dict[str, float | None]:
    """Compute pairwise similarity between two tracks across feature dimensions."""
    result: dict[str, float | None] = {}
    tempo_a = feat_a.get("tempo_bpm")
    tempo_b = feat_b.get("tempo_bpm")
    result["tempo_bpm_diff"] = _tempo_difference(
        float(tempo_a) if isinstance(tempo_a, (int, float)) else None,
        float(tempo_b) if isinstance(tempo_b, (int, float)) else None,
    )
    beat_summary_a = _beat_pattern_vector(feat_a)
    beat_summary_b = _beat_pattern_vector(feat_b)
    if beat_hist_zstats is not None:
        global_mean, global_std = beat_hist_zstats
        if len(beat_summary_a) == len(global_mean) and len(beat_summary_b) == len(global_mean):
            z_a = [(x - m) / s for x, m, s in zip(beat_summary_a, global_mean, global_std)]
            z_b = [(x - m) / s for x, m, s in zip(beat_summary_b, global_mean, global_std)]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z_a, z_b)))
            result["beat_pattern_sim"] = 1.0 / (1.0 + dist)
        else:
            result["beat_pattern_sim"] = _cosine_similarity(beat_summary_a, beat_summary_b)
    else:
        result["beat_pattern_sim"] = _cosine_similarity(beat_summary_a, beat_summary_b)
    onset_a = feat_a.get("onset_interval_hist", [])
    onset_b = feat_b.get("onset_interval_hist", [])
    if onset_hist_zstats is not None:
        global_mean, global_std = onset_hist_zstats
        if len(onset_a) == len(global_mean) and len(onset_b) == len(global_mean):
            z_a = [(x - m) / s for x, m, s in zip(onset_a, global_mean, global_std)]
            z_b = [(x - m) / s for x, m, s in zip(onset_b, global_mean, global_std)]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z_a, z_b)))
            result["onset_hist_sim"] = 1.0 / (1.0 + dist)
        else:
            result["onset_hist_sim"] = _cosine_similarity(onset_a, onset_b)
    else:
        result["onset_hist_sim"] = _cosine_similarity(onset_a, onset_b)
    chroma_a = feat_a.get("chroma_mean", [])
    chroma_b = feat_b.get("chroma_mean", [])
    if chroma_zstats is not None:
        global_mean, global_std = chroma_zstats
        if len(chroma_a) == len(global_mean) and len(chroma_b) == len(global_mean):
            z_a = [(x - m) / s for x, m, s in zip(chroma_a, global_mean, global_std)]
            z_b = [(x - m) / s for x, m, s in zip(chroma_b, global_mean, global_std)]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z_a, z_b)))
            result["chroma_sim"] = 1.0 / (1.0 + dist)
        else:
            result["chroma_sim"] = _cosine_similarity(chroma_a, chroma_b)
    else:
        result["chroma_sim"] = _cosine_similarity(chroma_a, chroma_b)
    for prefix, zstats in (
        ("mfcc", mfcc_zstats),
        ("cqt", cqt_zstats),
    ):
        mean_a = feat_a.get(f"{prefix}_mean", [])
        mean_b = feat_b.get(f"{prefix}_mean", [])
        var_a = feat_a.get(f"{prefix}_var", [])
        var_b = feat_b.get(f"{prefix}_var", [])
        vec_a = _mean_var_vector(mean_a, var_a)
        vec_b = _mean_var_vector(mean_b, var_b)
        sim_key = f"{prefix}_sim"
        if zstats is not None:
            global_mean, global_std = zstats
            if len(vec_a) == len(global_mean) and len(vec_b) == len(global_mean):
                z_a = [(x - m) / s for x, m, s in zip(vec_a, global_mean, global_std)]
                z_b = [(x - m) / s for x, m, s in zip(vec_b, global_mean, global_std)]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z_a, z_b)))
                result[sim_key] = 1.0 / (1.0 + dist)
                continue
        result[sim_key] = _cosine_similarity(mean_a, mean_b)

    tonnetz_a = feat_a.get("tonnetz_mean", [])
    tonnetz_b = feat_b.get("tonnetz_mean", [])
    if tonnetz_zstats is not None:
        global_mean, global_std = tonnetz_zstats
        if len(tonnetz_a) == len(global_mean) and len(tonnetz_b) == len(global_mean):
            z_a = [(x - m) / s for x, m, s in zip(tonnetz_a, global_mean, global_std)]
            z_b = [(x - m) / s for x, m, s in zip(tonnetz_b, global_mean, global_std)]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z_a, z_b)))
            result["tonnetz_sim"] = 1.0 / (1.0 + dist)
        else:
            tonnetz_sim = _cosine_similarity(tonnetz_a, tonnetz_b)
            result["tonnetz_sim"] = tonnetz_sim
    else:
        tonnetz_sim = _cosine_similarity(tonnetz_a, tonnetz_b)
        result["tonnetz_sim"] = tonnetz_sim
    motion_a = _root_motion_tokens([_split_chord_token(t)[0] for t in feat_a.get("chord_sequence_collapsed", [])])
    motion_b = _root_motion_tokens([_split_chord_token(t)[0] for t in feat_b.get("chord_sequence_collapsed", [])])
    result["chord_prog_sim"] = _normalized_lcs_similarity(motion_a, motion_b) if motion_a and motion_b else None
    genre_a, genre_b = feat_a.get("genre_top", ""), feat_b.get("genre_top", "")
    result["genre_match"] = (1.0 if genre_a == genre_b else 0.0) if genre_a and genre_b else None
    return result


def _scalar_coherence(values: list[float]) -> float | None:
    """1 / (1 + CV): 1 = perfectly uniform, decreases as spread grows.
    Matches analyze-feature-test.py scalar computation for cv_beat_interval,
    mean_beat_interval, spectral_centroid_mean.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    std = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return 1.0 / (1.0 + std / (abs(m) + 1e-8))


def _mean_pairwise_euclidean_sim(
    vectors: list,
    zstats: tuple[list[float], list[float]] | None = None,
) -> float | None:
    """Mean pairwise 1/(1+||z_a - z_b||) — fully vectorized with numpy."""
    valid = [v for v in vectors if v is not None and len(v) > 0]
    if len(valid) < 2:
        return None
    try:
        arr = np.asarray(valid, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if arr.ndim != 2:
        return None
    n, dim = arr.shape
    if zstats is not None:
        mean_vec, std_vec = zstats
        if len(mean_vec) == dim and len(std_vec) == dim:
            arr = (arr - np.asarray(mean_vec, dtype=np.float64)) / (np.asarray(std_vec, dtype=np.float64) + 1e-9)
    sq_norms = np.einsum("ij,ij->i", arr, arr)
    gram = arr @ arr.T
    sq_dists = np.clip(sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram, 0.0, None)
    idx = np.triu_indices(n, k=1)
    sims = 1.0 / (1.0 + np.sqrt(sq_dists[idx]))
    return float(sims.mean()) if sims.size > 0 else None


def _mean_pairwise_cosine_np(arrays: list) -> float | None:
    """Mean pairwise cosine similarity — vectorized with numpy gram matrix."""
    valid = [a for a in arrays if a is not None and len(a) > 0]
    if len(valid) < 2:
        return None
    try:
        arr = np.asarray(valid, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if arr.ndim != 2:
        return None
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = arr / norms
    gram = normed @ normed.T
    n = gram.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(gram[idx].mean()) if idx[0].size > 0 else None


def _mean_pairwise_jaccard(token_sets: list[set[str]]) -> float | None:
    if len(token_sets) < 2:
        return None
    sims: list[float] = []
    for a, b in combinations(token_sets, 2):
        union = a | b
        sims.append((len(a & b) / len(union)) if union else 0.0)
    return _mean(sims) if sims else None


def _pairwise_similarity_from_prepared(
    feat_a: dict[str, Any],
    feat_b: dict[str, Any],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}

    tempo_a = feat_a.get("tempo_bpm")
    tempo_b = feat_b.get("tempo_bpm")
    result["tempo_bpm_diff"] = _tempo_difference(tempo_a, tempo_b)

    beat_a_z = feat_a.get("beat_pattern_vec_z")
    beat_b_z = feat_b.get("beat_pattern_vec_z")
    if beat_a_z is not None and beat_b_z is not None and beat_a_z.shape == beat_b_z.shape:
        result["beat_pattern_sim"] = 1.0 / (1.0 + float(np.linalg.norm(beat_a_z - beat_b_z)))
    else:
        result["beat_pattern_sim"] = _cosine_similarity_np(feat_a.get("beat_pattern_vec"), feat_b.get("beat_pattern_vec"))

    onset_a_z = feat_a.get("onset_interval_hist_z")
    onset_b_z = feat_b.get("onset_interval_hist_z")
    if onset_a_z is not None and onset_b_z is not None and onset_a_z.shape == onset_b_z.shape:
        result["onset_hist_sim"] = 1.0 / (1.0 + float(np.linalg.norm(onset_a_z - onset_b_z)))
    else:
        result["onset_hist_sim"] = _cosine_similarity_np(feat_a.get("onset_interval_hist"), feat_b.get("onset_interval_hist"))

    chroma_a_z = feat_a.get("chroma_mean_z")
    chroma_b_z = feat_b.get("chroma_mean_z")
    if chroma_a_z is not None and chroma_b_z is not None and chroma_a_z.shape == chroma_b_z.shape:
        result["chroma_sim"] = 1.0 / (1.0 + float(np.linalg.norm(chroma_a_z - chroma_b_z)))
    else:
        result["chroma_sim"] = _cosine_similarity_np(feat_a.get("chroma_mean"), feat_b.get("chroma_mean"))

    for prefix in ("mfcc", "cqt"):
        vec_a_z = feat_a.get(f"{prefix}_vec_z")
        vec_b_z = feat_b.get(f"{prefix}_vec_z")
        sim_key = f"{prefix}_sim"
        if vec_a_z is not None and vec_b_z is not None and vec_a_z.shape == vec_b_z.shape:
            result[sim_key] = 1.0 / (1.0 + float(np.linalg.norm(vec_a_z - vec_b_z)))
        else:
            vec_a = feat_a.get(f"{prefix}_vec")
            vec_b = feat_b.get(f"{prefix}_vec")
            if vec_a is not None and vec_b is not None:
                half_a = vec_a.shape[0] // 2
                half_b = vec_b.shape[0] // 2
                result[sim_key] = _cosine_similarity_np(vec_a[:half_a], vec_b[:half_b])
            else:
                result[sim_key] = None

    tonnetz_a_z = feat_a.get("tonnetz_mean_z")
    tonnetz_b_z = feat_b.get("tonnetz_mean_z")
    if tonnetz_a_z is not None and tonnetz_b_z is not None and tonnetz_a_z.shape == tonnetz_b_z.shape:
        result["tonnetz_sim"] = 1.0 / (1.0 + float(np.linalg.norm(tonnetz_a_z - tonnetz_b_z)))
    else:
        result["tonnetz_sim"] = _cosine_similarity_np(feat_a.get("tonnetz_mean"), feat_b.get("tonnetz_mean"))

    result["chord_prog_sim"] = _normalized_lcs_similarity(
        feat_a.get("chord_root_motion", []),
        feat_b.get("chord_root_motion", []),
    )

    genre_a, genre_b = feat_a.get("genre_top", ""), feat_b.get("genre_top", "")
    result["genre_match"] = (1.0 if genre_a == genre_b else 0.0) if genre_a and genre_b else None
    return result


def compute_group_profile_strength(
    group_track_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
    stft_zstats: tuple[list[float], list[float]] | None = None,
    mfcc_zstats: tuple[list[float], list[float]] | None = None,
    cqt_zstats: tuple[list[float], list[float]] | None = None,
    vqt_zstats: tuple[list[float], list[float]] | None = None,
    chroma_zstats: tuple[list[float], list[float]] | None = None,
    tonnetz_zstats: tuple[list[float], list[float]] | None = None,
    beat_hist_zstats: tuple[list[float], list[float]] | None = None,
    onset_hist_zstats: tuple[list[float], list[float]] | None = None,
    beat_interval_hist_zstats: tuple[list[float], list[float]] | None = None,
    prepared_feature_table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {dim: None for dim in FEATURE_DIMS}
    if prepared_feature_table is not None:
        feats = [prepared_feature_table.get(track_id) for track_id in group_track_ids]
        feats = [feat for feat in feats if feat is not None]
        if not feats:
            return values

        beat_pattern_z_vectors = [feat["beat_pattern_vec_z"] for feat in feats if feat.get("beat_pattern_vec_z") is not None]
        if beat_pattern_z_vectors:
            values["beat_pattern_sim"] = _mean_pairwise_euclidean_sim(beat_pattern_z_vectors)
        else:
            beat_pattern_raw_vectors = [feat["beat_pattern_vec"] for feat in feats if feat.get("beat_pattern_vec") is not None]
            values["beat_pattern_sim"] = _mean_pairwise_euclidean_sim(beat_pattern_raw_vectors)

        onset_hist_vectors = [feat.get("onset_interval_hist_z") for feat in feats if feat.get("onset_interval_hist_z") is not None]
        if onset_hist_vectors:
            values["onset_hist_sim"] = _mean_pairwise_euclidean_sim(onset_hist_vectors)
        else:
            onset_hist_raw_vectors = [feat["onset_interval_hist"] for feat in feats if feat.get("onset_interval_hist") is not None]
            values["onset_hist_sim"] = _mean_pairwise_euclidean_sim(onset_hist_raw_vectors, zstats=None)

        chroma_vectors = [feat.get("chroma_mean_z") for feat in feats if feat.get("chroma_mean_z") is not None]
        if chroma_vectors:
            values["chroma_sim"] = _mean_pairwise_euclidean_sim(chroma_vectors)
        else:
            chroma_raw_vectors = [feat["chroma_mean"] for feat in feats if feat.get("chroma_mean") is not None]
            values["chroma_sim"] = _mean_pairwise_euclidean_sim(chroma_raw_vectors, zstats=None)

        tonnetz_vectors = [feat.get("tonnetz_mean_z") for feat in feats if feat.get("tonnetz_mean_z") is not None]
        if tonnetz_vectors:
            values["tonnetz_sim"] = _mean_pairwise_euclidean_sim(tonnetz_vectors)
        else:
            tonnetz_raw_vectors = [feat["tonnetz_mean"] for feat in feats if feat.get("tonnetz_mean") is not None]
            values["tonnetz_sim"] = _mean_pairwise_euclidean_sim(tonnetz_raw_vectors, zstats=None)

        chord_motions = [
            feat.get("chord_root_motion", [])
            for feat in feats
            if feat.get("chord_root_motion")
        ]
        values["chord_prog_sim"] = _mean_pairwise_similarity(
            chord_motions,
            lambda a, b: _normalized_lcs_similarity_cached(tuple(a), tuple(b)),
        )

        for prefix in ("mfcc", "cqt"):
            vectors = [feat.get(f"{prefix}_vec_z") for feat in feats if feat.get(f"{prefix}_vec_z") is not None]
            sim_key = f"{prefix}_sim"
            if vectors:
                values[sim_key] = _mean_pairwise_euclidean_sim(vectors)
            else:
                raw_vectors = [feat.get(f"{prefix}_vec") for feat in feats if feat.get(f"{prefix}_vec") is not None]
                if raw_vectors:
                    values[sim_key] = _mean_pairwise_euclidean_sim(
                        [vec[: vec.shape[0] // 2] for vec in raw_vectors],
                        zstats=None,
                    )

        bih_z_vecs = [feat.get("beat_interval_hist_z") for feat in feats if feat.get("beat_interval_hist_z") is not None]
        if bih_z_vecs:
            values["beat_interval_sim"] = _mean_pairwise_euclidean_sim(bih_z_vecs)
        else:
            bih_raw = [feat.get("beat_interval_hist") for feat in feats if feat.get("beat_interval_hist") is not None]
            values["beat_interval_sim"] = _mean_pairwise_euclidean_sim(bih_raw, zstats=None)
        values["cv_beat_sim"] = _scalar_coherence([feat.get("cv_beat_interval") for feat in feats])
        values["mean_beat_sim"] = _scalar_coherence([feat.get("mean_beat_interval") for feat in feats])
        values["spectral_centroid_sim"] = _scalar_coherence([feat.get("spectral_centroid_mean") for feat in feats])

        return values
    feats = [feature_table.get(track_id) for track_id in group_track_ids]
    feats = [feat for feat in feats if feat is not None]
    if not feats:
        return values

    beat_pattern_vectors = [_beat_pattern_vector(feat) for feat in feats]
    beat_pattern_vectors = [v for v in beat_pattern_vectors if v]
    values["beat_pattern_sim"] = _mean_pairwise_euclidean_sim(beat_pattern_vectors, zstats=beat_hist_zstats)
    onset_hist_vectors = [feat.get("onset_interval_hist", []) for feat in feats if feat.get("onset_interval_hist", [])]
    values["onset_hist_sim"] = _mean_pairwise_euclidean_sim(onset_hist_vectors, zstats=onset_hist_zstats)

    chroma_vectors = [feat.get("chroma_mean", []) for feat in feats if feat.get("chroma_mean", [])]
    values["chroma_sim"] = _mean_pairwise_euclidean_sim(chroma_vectors, zstats=chroma_zstats)
    tonnetz_vectors = [feat.get("tonnetz_mean", []) for feat in feats if feat.get("tonnetz_mean", [])]
    values["tonnetz_sim"] = _mean_pairwise_euclidean_sim(tonnetz_vectors, zstats=tonnetz_zstats)
    chord_sequences = [
        feat.get("chord_sequence_collapsed", [])
        for feat in feats
        if feat.get("chord_sequence_collapsed")
    ]
    chord_motions = [
        _root_motion_tokens([_split_chord_token(t)[0] for t in seq])
        for seq in chord_sequences
    ]
    chord_motions = [m for m in chord_motions if m]
    values["chord_prog_sim"] = _mean_pairwise_similarity(
        chord_motions,
        lambda a, b: _normalized_lcs_similarity_cached(tuple(a), tuple(b)),
    )

    for prefix, zstats in (
        ("mfcc", mfcc_zstats),
        ("cqt", cqt_zstats),
    ):
        vectors: list[list[float]] = []
        for feat in feats:
            vec = _mean_var_vector(feat.get(f"{prefix}_mean", []), feat.get(f"{prefix}_var", []))
            if vec:
                vectors.append(vec)
        values[f"{prefix}_sim"] = _mean_pairwise_euclidean_sim(vectors, zstats=zstats)

    bih_vecs = [feat.get("beat_interval_hist", []) for feat in feats if feat.get("beat_interval_hist")]
    values["beat_interval_sim"] = _mean_pairwise_euclidean_sim(bih_vecs, zstats=beat_interval_hist_zstats)
    values["cv_beat_sim"] = _scalar_coherence([feat.get("cv_beat_interval") for feat in feats])
    values["mean_beat_sim"] = _scalar_coherence([feat.get("mean_beat_interval") for feat in feats])
    values["spectral_centroid_sim"] = _scalar_coherence([feat.get("spectral_centroid_mean") for feat in feats])

    return values


def _compute_contrast_stats(group_value: float | None, bg_values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if group_value is None or not bg_values:
        return None, None, None, None
    bg_arr = np.asarray(bg_values, dtype=np.float64)
    bg_mean = float(bg_arr.mean())
    bg_std = float(bg_arr.std(ddof=1)) if len(bg_arr) > 1 else 0.0
    z = (float(group_value) - bg_mean) / (bg_std + 1e-9)
    pct = float(np.mean(bg_arr <= float(group_value)))
    return z, bg_mean, bg_std, pct


def _sample_random_profile_values(
    train_track_ids: list[str],
    group_size: int,
    num_samples: int,
    rng: np.random.Generator,
    feature_table: dict[str, dict[str, Any]],
    stft_zstats: tuple[list[float], list[float]] | None,
    mfcc_zstats: tuple[list[float], list[float]] | None,
    cqt_zstats: tuple[list[float], list[float]] | None,
    vqt_zstats: tuple[list[float], list[float]] | None,
    chroma_zstats: tuple[list[float], list[float]] | None,
    tonnetz_zstats: tuple[list[float], list[float]] | None,
    beat_hist_zstats: tuple[list[float], list[float]] | None,
    onset_hist_zstats: tuple[list[float], list[float]] | None,
    beat_interval_hist_zstats: tuple[list[float], list[float]] | None = None,
    prepared_feature_table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {dim: [] for dim in FEATURE_DIMS}
    n_train = len(train_track_ids)
    if n_train == 0:
        return out
    k = max(1, min(group_size, n_train))
    for _ in tqdm(range(num_samples), desc="bg samples", leave=False):
        sample_idx = rng.choice(n_train, size=k, replace=False)
        sample_track_ids = [train_track_ids[i] for i in sample_idx.tolist()]
        sample_profile = compute_group_profile_strength(
            sample_track_ids,
            feature_table=feature_table,
            stft_zstats=stft_zstats,
            mfcc_zstats=mfcc_zstats,
            cqt_zstats=cqt_zstats,
            vqt_zstats=vqt_zstats,
            chroma_zstats=chroma_zstats,
            tonnetz_zstats=tonnetz_zstats,
            beat_hist_zstats=beat_hist_zstats,
            onset_hist_zstats=onset_hist_zstats,
            beat_interval_hist_zstats=beat_interval_hist_zstats,
            prepared_feature_table=prepared_feature_table,
        )
        for dim in FEATURE_DIMS:
            value = sample_profile.get(dim)
            if value is not None:
                out[dim].append(float(value))
    return out
