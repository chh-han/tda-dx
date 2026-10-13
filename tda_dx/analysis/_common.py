# Common constants and helpers shared across TDA-Dx analysis modules.

import json
import math
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np

from tda_dx.analysis.io_utils import _mean


Q1_BASELINE_CONDITIONS = frozenset({"random_baseline", "shuffle_train"})
DEFAULT_TOP_K = 20
DEFAULT_PAIRWISE_TOP_K = 50
TOP_TRAIN_TRACK_LIMIT = 100
DEFAULT_RANK_SELECTION = "top"
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

N_BG_SAMPLES = 200

MAIN_BUCKET = "main"
APPENDIX_BUCKET = "appendix"


def _genre_match_ratio(items: list[dict[str, Any]], test_genre: str, top_k: int) -> float | None:
    if not test_genre:
        return None
    subset = items[:top_k]
    if not subset:
        return None
    comparable = [item for item in subset if item.get("train_genre")]
    if not comparable:
        return None
    hits = sum(1 for item in comparable if item.get("train_genre") == test_genre)
    return hits / len(comparable)


def _select_rank_indices(scores: np.ndarray, k: int, rank_selection: str) -> np.ndarray:
    effective_k = min(max(1, int(k)), len(scores))
    if rank_selection == "bottom":
        return np.argsort(scores)[:effective_k]
    return np.argsort(-scores)[:effective_k]


def _parse_json_list(s: str) -> list[float]:
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _parse_json_str_list(s: str) -> list[str]:
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _mean_var_vector(mean_values: list[float], var_values: list[float]) -> list[float]:
    std_values = [math.sqrt(max(v, 0.0)) for v in var_values]
    return mean_values + std_values


def _beat_pattern_vector(feat: dict[str, Any], bins: int = 24) -> list[float]:
    beat_times = feat.get("beat_times", [])
    if not isinstance(beat_times, list) or len(beat_times) < 3:
        return []
    try:
        beat_arr = np.asarray([float(v) for v in beat_times], dtype=np.float64)
    except (TypeError, ValueError):
        return []
    intervals = np.diff(beat_arr)
    if intervals.size < 2 or np.any(intervals <= 0):
        return []
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return []
    normalized = intervals / median_interval
    clipped = np.clip(normalized, 0.25, 4.0)
    hist_bins = np.geomspace(0.25, 4.0, bins + 1)
    counts, _ = np.histogram(clipped, bins=hist_bins)
    total = int(counts.sum())
    if total <= 0:
        return []
    return (counts / total).astype(np.float64).tolist()


def _fold_tempo_bpm(tempo: float | None, low: float = 60.0, high: float = 120.0) -> float | None:
    if tempo is None or tempo <= 0:
        return None
    value = float(tempo)
    while value > high:
        value /= 2.0
    while value < low:
        value *= 2.0
    return value


def _tempo_difference(tempo_a: float | None, tempo_b: float | None) -> float | None:
    if tempo_a is None or tempo_b is None or tempo_a <= 0 or tempo_b <= 0:
        return None
    folded_a = _fold_tempo_bpm(tempo_a)
    folded_b = _fold_tempo_bpm(tempo_b)
    if folded_a is None or folded_b is None:
        return None
    return abs(folded_a - folded_b)


def _ngram_counter(tokens: list[str], n: int) -> dict[str, float]:
    if n <= 0 or len(tokens) < n:
        return {}
    counts: dict[str, float] = defaultdict(float)
    for idx in range(len(tokens) - n + 1):
        key = " || ".join(tokens[idx : idx + n])
        counts[key] += 1.0
    return counts


def _counter_cosine_similarity(counter_a: dict[str, float], counter_b: dict[str, float]) -> float | None:
    if not counter_a or not counter_b:
        return None
    keys = set(counter_a) | set(counter_b)
    dot = sum(counter_a.get(k, 0.0) * counter_b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


def _token_hist_similarity(tokens_a: list[str], tokens_b: list[str]) -> float | None:
    clean_a = [str(token).strip() for token in tokens_a if str(token).strip()]
    clean_b = [str(token).strip() for token in tokens_b if str(token).strip()]
    if not clean_a or not clean_b:
        return None
    counter_a: dict[str, float] = defaultdict(float)
    counter_b: dict[str, float] = defaultdict(float)
    for token in clean_a:
        counter_a[token] += 1.0
    for token in clean_b:
        counter_b[token] += 1.0
    return _counter_cosine_similarity(counter_a, counter_b)


def _chord_profile_similarity(
    roots_a: list[str],
    roots_b: list[str],
    types_a: list[str],
    types_b: list[str],
) -> float | None:
    sims: list[float] = []
    root_sim = _token_hist_similarity(roots_a, roots_b)
    if root_sim is not None:
        sims.append(root_sim)
    type_sim = _token_hist_similarity(types_a, types_b)
    if type_sim is not None:
        sims.append(type_sim)
    return _mean(sims) if sims else None


def _split_chord_token(token: str) -> tuple[str, str]:
    token = str(token).strip()
    if not token:
        return "", ""
    if ":" in token:
        root, chord_type = token.split(":", 1)
        return root.strip(), chord_type.strip()
    return token, "maj"


def _root_to_pitch_class(root: str) -> int | None:
    lookup = {
        "C": 0, "B#": 0,
        "C#": 1, "Db": 1,
        "D": 2,
        "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4,
        "F": 5, "E#": 5,
        "F#": 6, "Gb": 6,
        "G": 7,
        "G#": 8, "Ab": 8,
        "A": 9,
        "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
    }
    return lookup.get(str(root).strip())


def _pitch_class_to_root(pc: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return names[int(pc) % 12]


def _transpose_root(root: str, shift: int) -> str:
    pc = _root_to_pitch_class(root)
    if pc is None:
        return str(root).strip()
    return _pitch_class_to_root(pc + shift)


def _transpose_chord_token(token: str, shift: int) -> str:
    root, chord_type = _split_chord_token(token)
    if not root:
        return str(token).strip()
    transposed_root = _transpose_root(root, shift)
    if chord_type:
        return f"{transposed_root}:{chord_type}"
    return transposed_root


def _normalized_lcs_similarity(sequence_a: list[str], sequence_b: list[str]) -> float | None:
    clean_a = [str(token).strip() for token in sequence_a if str(token).strip()]
    clean_b = [str(token).strip() for token in sequence_b if str(token).strip()]
    if not clean_a or not clean_b:
        return None
    m, n = len(clean_a), len(clean_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for idx_a in range(m):
        for idx_b in range(n):
            if clean_a[idx_a] == clean_b[idx_b]:
                dp[idx_a + 1][idx_b + 1] = dp[idx_a][idx_b] + 1
            else:
                dp[idx_a + 1][idx_b + 1] = max(dp[idx_a][idx_b + 1], dp[idx_a + 1][idx_b])
    return dp[m][n] / max(m, n)


def _root_motion_tokens(roots: list[str]) -> list[str]:
    pcs = [_root_to_pitch_class(root) for root in roots]
    pcs = [pc for pc in pcs if pc is not None]
    if len(pcs) < 2:
        return []
    return [str((pcs[idx + 1] - pcs[idx]) % 12) for idx in range(len(pcs) - 1)]


def _chord_progression_similarity(
    sequence_a: list[str],
    sequence_b: list[str],
) -> float | None:
    clean_a = [str(token).strip() for token in sequence_a if str(token).strip()]
    clean_b = [str(token).strip() for token in sequence_b if str(token).strip()]
    if not clean_a or not clean_b:
        return None
    roots_a = [_split_chord_token(t)[0] for t in clean_a]
    roots_b = [_split_chord_token(t)[0] for t in clean_b]
    motion_a = _root_motion_tokens(roots_a)
    motion_b = _root_motion_tokens(roots_b)
    if not motion_a or not motion_b:
        return None
    return _normalized_lcs_similarity(motion_a, motion_b)


@lru_cache(maxsize=32768)
def _chord_prog_sim_cached(a: tuple[str, ...], b: tuple[str, ...]) -> float | None:
    return _chord_progression_similarity(list(a), list(b))


@lru_cache(maxsize=131072)
def _normalized_lcs_similarity_cached(a: tuple[str, ...], b: tuple[str, ...]) -> float | None:
    if a > b:
        a, b = b, a
    return _normalized_lcs_similarity(list(a), list(b))


def _mean_pairwise_similarity(values: list[Any], sim_fn: Any) -> float | None:
    valid = [value for value in values if value is not None]
    if len(valid) < 2:
        return None
    sims: list[float] = []
    for a, b in combinations(valid, 2):
        sim = sim_fn(a, b)
        if sim is not None:
            sims.append(float(sim))
    return _mean(sims) if sims else None


def _dominant_ratio(values: list[str]) -> float | None:
    cleaned = [v for v in (s.strip() for s in values) if v]
    if not cleaned:
        return None
    counts: dict[str, int] = defaultdict(int)
    for value in cleaned:
        counts[value] += 1
    return max(counts.values()) / len(cleaned)


def _cosine_similarity_np(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return None
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)
