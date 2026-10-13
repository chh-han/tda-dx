

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

STAGES = ("semantic", "coarse", "fine")


# ─── I/O utilities ────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def dump_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _l1_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    total = float(sum(max(0.0, float(v)) for v in values))
    if total <= 0.0:
        return [0.0 for _ in values]
    return [max(0.0, float(v)) / total for v in values]


# ─── Math utilities ───────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean_value: float | None = None) -> float:
    if not values:
        return 0.0
    mu = _mean(values) if mean_value is None else mean_value
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    std_x = _std(xs, mean_x)
    std_y = _std(ys, mean_y)
    if std_x == 0.0 or std_y == 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / len(xs)
    return cov / (std_x * std_y)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _gini(values: list[float]) -> float:
    weights = sorted(max(0.0, float(v)) for v in values)
    total = sum(weights)
    if total <= 0.0 or not weights:
        return 0.0
    n = len(weights)
    weighted_sum = sum((2 * idx - n - 1) * v for idx, v in enumerate(weights, start=1))
    return weighted_sum / (n * total)


def _entropy(values: list[float]) -> tuple[float, float]:
    weights = [max(0.0, float(v)) for v in values]
    total = sum(weights)
    if total <= 0.0 or not weights:
        return 0.0, 0.0
    probs = [v / total for v in weights if v > 0.0]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(weights)) if len(weights) > 1 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0.0 else 0.0
    return entropy, normalized


def _top_mass(values: list[float], top_k: int) -> float:
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return sum(sorted(values, reverse=True)[:top_k]) / total


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


# ─── File / stage discovery ───────────────────────────────────────────────────

def find_one(dataset_dir: Path, suffix: str) -> Path | None:
    matches = sorted(dataset_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def find_main_scores_npy(dataset_dir: Path) -> Path | None:
    matches = sorted(dataset_dir.glob("*_scores.npy"))
    for path in matches:
        if path.name.endswith("_token_scores.npy"):
            continue
        return path
    return None


def find_sharded_npy(path: Path) -> list[Path]:
    """Returns sorted shard paths for a virtual .npy path that doesn't exist as a single file."""
    return sorted(path.parent.glob(f"{path.stem}_shard_*{path.suffix}"))


def load_npy_or_shards(path: Path, **kwargs) -> np.ndarray:
    """Load a .npy file, falling back to concatenating shards if the single file doesn't exist."""
    if path.exists():
        return np.load(str(path), **kwargs)
    shards = find_sharded_npy(path)
    if shards:
        return np.concatenate([np.load(str(s)) for s in shards], axis=0)
    raise FileNotFoundError(f"Could not find {path} or sharded version")


def get_npy_shape(path: Path) -> tuple[int, ...]:
    """Return the shape of a .npy file (or sharded equivalent) without loading the full array."""
    if path.exists():
        return tuple(np.load(str(path), mmap_mode="r").shape)
    shards = find_sharded_npy(path)
    if shards:
        first_shape = np.load(str(shards[0]), mmap_mode="r").shape
        total_rows = sum(np.load(str(s), mmap_mode="r").shape[0] for s in shards)
        return (total_rows,) + tuple(first_shape[1:])
    raise FileNotFoundError(f"Could not find {path} or sharded version")


def _parse_stage_method_filename(path: Path, suffix: str) -> tuple[str, str] | None:
    stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    stage, method = parts
    if not stage or not method:
        return None
    return stage, method


def discover_stage_method_runs(dataset_dir: Path, stage: str) -> dict[str, dict[str, Path]]:
    runs: dict[str, dict[str, Path]] = {}
    for metadata_path in sorted(dataset_dir.glob(f"{stage}_*_metadata.json")):
        parsed = _parse_stage_method_filename(metadata_path, "_metadata.json")
        if parsed is None:
            continue
        parsed_stage, method = parsed
        if parsed_stage != stage:
            continue
        scores_path = dataset_dir / f"{stage}_{method}_scores.npy"
        if not scores_path.exists() and not find_sharded_npy(scores_path):
            continue
        runs[method] = {"metadata_path": metadata_path, "scores_path": scores_path}
    if not runs:
        metadata_path = find_one(dataset_dir, "_metadata.json")
        scores_path = find_main_scores_npy(dataset_dir)
        if metadata_path is not None and scores_path is not None:
            method = _parse_stage_method_filename(metadata_path, "_metadata.json")
            fallback_method = method[1] if method else "trak"
            runs[fallback_method] = {"metadata_path": metadata_path, "scores_path": scores_path}
    return runs


def select_primary_method(methods: list[str]) -> str | None:
    if not methods:
        return None
    for preferred in ("trak", "grad_cos", "grad-cos"):
        if preferred in methods:
            return preferred
    return sorted(methods)[0]


def _looks_like_stage_dir(path: Path, stage: str) -> bool:
    return path.is_dir() and any(path.glob(f"{stage}_*_metadata.json"))


def discover_main_stage_dirs(root: Path) -> dict[str, Path]:
    stage_dirs: dict[str, Path] = {}
    for stage in STAGES:
        candidate = root / stage
        if _looks_like_stage_dir(candidate, stage):
            stage_dirs[stage] = candidate
    return stage_dirs


def discover_condition_stage_dirs(root: Path) -> dict[str, dict[str, Path]]:
    conditions: dict[str, dict[str, Path]] = defaultdict(dict)
    for stage in STAGES:
        stage_dir = root / stage
        if _looks_like_stage_dir(stage_dir, stage):
            conditions["main"][stage] = stage_dir
            for child in sorted(stage_dir.iterdir()):
                if _looks_like_stage_dir(child, stage):
                    conditions[child.name][stage] = child
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in STAGES or child.name.startswith("viewer_bundle"):
            continue
        for stage in STAGES:
            candidate = child / stage
            if _looks_like_stage_dir(candidate, stage):
                conditions[child.name][stage] = candidate
    return {
        condition: {stage: stage_map[stage] for stage in STAGES if stage in stage_map}
        for condition, stage_map in conditions.items()
        if stage_map
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def _select_header_column(level0: list[str], level1: list[str], target: str) -> int | None:
    target_lower = target.strip().lower()
    target_parts = [part.strip().lower() for part in target.replace(".", "|").split("|") if part.strip()]
    for idx, (h0, h1) in enumerate(zip(level0, level1)):
        left = h0.strip().lower()
        right = h1.strip().lower()
        if right == target_lower:
            return idx
        if f"{left}|{right}" == target_lower:
            return idx
        if f"{left}.{right}" == target_lower:
            return idx
        if len(target_parts) == 2 and [left, right] == target_parts:
            return idx
    return None


def load_track_to_genre_map(tracks_csv_path: Path, genre_column: str = "genre_top") -> dict[str, str]:
    if not tracks_csv_path.exists():
        return {}
    try:
        with tracks_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            level0 = next(reader)
            level1 = next(reader)
            _ = next(reader)
            if len(level0) != len(level1):
                return {}
            track_id_col = _select_header_column(level0, level1, "track_id")
            if track_id_col is None:
                track_id_col = 0
            genre_col = _select_header_column(level0, level1, genre_column)
            if genre_col is None and genre_column.strip().lower() == "genre_top":
                genre_col = _select_header_column(level0, level1, "track|genre_top")
            if genre_col is None:
                return {}
            track_to_genre: dict[str, str] = {}
            for row in reader:
                if len(row) <= max(track_id_col, genre_col):
                    continue
                raw_track = row[track_id_col].strip()
                if not raw_track:
                    continue
                try:
                    norm_track = str(int(raw_track))
                except (TypeError, ValueError):
                    norm_track = raw_track
                genre = row[genre_col].strip()
                if not genre or genre.lower() in {"nan", "none", "null", ""}:
                    continue
                track_to_genre[norm_track] = genre
        return track_to_genre
    except Exception:
        return {}


def resolve_tracks_csv_override(
    metadata: dict[str, Any],
    experiment_config: dict[str, Any],
    explicit_override: Path | None,
) -> Path | None:
    if explicit_override is not None:
        return explicit_override
    for value in (metadata.get("tracks_csv"), experiment_config.get("tracks_csv")):
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_track_genres(
    metadata: dict[str, Any],
    experiment_config: dict[str, Any],
    tracks_csv_override: Path | None,
) -> tuple[list[str], list[str], Path | None]:
    tracks_csv_path = resolve_tracks_csv_override(metadata, experiment_config, tracks_csv_override)
    genre_column = str(metadata.get("genre_column") or "genre_top")
    track_to_genre = load_track_to_genre_map(tracks_csv_path, genre_column=genre_column) if tracks_csv_path else {}
    train_track_ids = [str(v) for v in metadata.get("train_track_ids", [])]
    test_track_ids = [str(v) for v in metadata.get("test_track_ids", [])]
    raw_train_genres = metadata.get("train_genres", [])
    raw_test_genres = metadata.get("test_genres", [])
    train_genres: list[str] = []
    for idx, track_id in enumerate(train_track_ids):
        genre = str(raw_train_genres[idx] or "").strip() if idx < len(raw_train_genres) else ""
        if not genre:
            genre = track_to_genre.get(track_id, "")
        train_genres.append(genre)
    test_genres: list[str] = []
    for idx, track_id in enumerate(test_track_ids):
        genre = str(raw_test_genres[idx] or "").strip() if idx < len(raw_test_genres) else ""
        if not genre:
            genre = track_to_genre.get(track_id, "")
        test_genres.append(genre)
    return train_genres, test_genres, tracks_csv_path


def load_existing_stage_artifacts(
    dataset_dir: Path,
    stage: str,
    method: str | None,
    tracks_csv_override: Path | None,
) -> dict[str, Any]:
    runs = discover_stage_method_runs(dataset_dir, stage)
    resolved_method = method or select_primary_method(list(runs))
    if not resolved_method or resolved_method not in runs:
        raise FileNotFoundError(f"Could not find metadata in {dataset_dir}")
    metadata = load_json(runs[resolved_method]["metadata_path"])
    experiment_config_path = dataset_dir / "experiment_config.json"
    experiment_config = load_json(experiment_config_path) if experiment_config_path.exists() else {}
    train_indices_path = dataset_dir / "train_indices.npy"
    if not train_indices_path.exists() and not find_sharded_npy(train_indices_path):
        train_indices_path = None  # will be synthesized as np.arange(num_train) at load time
    train_genres, test_genres, tracks_csv_path = resolve_track_genres(metadata, experiment_config, tracks_csv_override)
    return {
        "metadata": metadata,
        "experiment_config": experiment_config,
        "scores_path": runs[resolved_method]["scores_path"],
        "train_indices_path": train_indices_path,
        "test_losses_path": dataset_dir / "test_losses.npy",
        "train_genres": train_genres,
        "test_genres": test_genres,
        "tracks_csv_path": tracks_csv_path,
        "genre_column": str(metadata.get("genre_column") or "genre_top"),
        "method": str(metadata.get("method") or experiment_config.get("method") or resolved_method),
        "stage": stage,
    }


def normalize_score_matrix(scores: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "none":
        return np.asarray(scores, dtype=np.float64)
    normalized = np.asarray(scores, dtype=np.float64).copy()
    if normalized.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape {normalized.shape}")
    if normalization == "zscore_per_test":
        means = normalized.mean(axis=0, keepdims=True)
        stds = normalized.std(axis=0, keepdims=True)
        stds[stds == 0.0] = 1.0
        return (normalized - means) / stds
    if normalization == "l2_per_test":
        norms = np.linalg.norm(normalized, axis=0, keepdims=True)
        norms[norms == 0.0] = 1.0
        return normalized / norms
    if normalization == "minmax_per_test":
        mins = normalized.min(axis=0, keepdims=True)
        maxs = normalized.max(axis=0, keepdims=True)
        spans = maxs - mins
        spans[spans == 0.0] = 1.0
        return (normalized - mins) / spans
    if normalization == "rank_per_test":
        # scipy.stats.rankdata is single-threaded; for large matrices we
        # parallelize over columns across the SLURM-allocated CPUs.
        import os
        from scipy.stats import rankdata
        n_train, n_test = normalized.shape
        n_workers = max(1, int(os.environ.get("ANALYZE_RANK_WORKERS",
                                               os.environ.get("SLURM_CPUS_PER_TASK",
                                                              os.cpu_count() or 1))))
        n_workers = min(n_workers, n_test)
        if n_workers <= 1 or n_test < 64:
            ranked = rankdata(normalized, method="average", axis=0) - 1.0
        else:
            # Threads (not processes): numpy sort releases the GIL and we
            # avoid pickling 10+ GB of scores between workers.
            from joblib import Parallel, delayed
            chunks = [c for c in np.array_split(np.arange(n_test), n_workers) if len(c) > 0]
            ranked_parts = Parallel(n_jobs=len(chunks), prefer="threads")(
                delayed(rankdata)(normalized[:, c], method="average", axis=0)
                for c in chunks
            )
            ranked = np.empty_like(normalized)
            for c, part in zip(chunks, ranked_parts):
                ranked[:, c] = part
            ranked -= 1.0
        return ranked / max(n_train - 1, 1)
    raise ValueError(f"Unsupported score normalization: {normalization}")


def normalize_score_column(col: np.ndarray, normalization: str) -> np.ndarray:
    col = np.asarray(col, dtype=np.float64)
    if normalization == "none":
        return col
    if normalization == "zscore_per_test":
        mean = col.mean()
        std = col.std()
        if std == 0.0:
            std = 1.0
        return (col - mean) / std
    if normalization == "l2_per_test":
        norm = float(np.linalg.norm(col))
        if norm == 0.0:
            norm = 1.0
        return col / norm
    if normalization == "minmax_per_test":
        min_val = col.min()
        max_val = col.max()
        span = float(max_val - min_val)
        if span == 0.0:
            span = 1.0
        return (col - min_val) / span
    if normalization == "rank_per_test":
        from scipy.stats import rankdata
        n = len(col)
        ranks = rankdata(col, method="average") - 1.0
        return ranks / max(n - 1, 1)
    raise ValueError(f"Unsupported score normalization: {normalization}")


def aggregate_stage(
    scores: np.ndarray,
    metadata: dict[str, Any],
    track_to_genre: dict[str, str],
    track_aggregation: str = "sum",
    score_normalization: str = "none",
) -> dict[str, Any]:
    train_track_ids = [str(v) for v in metadata["train_track_ids"]]
    test_track_ids = [str(v) for v in metadata["test_track_ids"]]
    train_genres: list[str] = []
    for idx, track_id in enumerate(train_track_ids):
        genre = ""
        raw_genres = metadata.get("train_genres", [])
        if idx < len(raw_genres):
            genre = str(raw_genres[idx] or "").strip()
        if not genre:
            genre = track_to_genre.get(track_id, "")
        train_genres.append(genre)
    test_genres: list[str] = []
    for idx, track_id in enumerate(test_track_ids):
        genre = ""
        raw_genres = metadata.get("test_genres", [])
        if idx < len(raw_genres):
            genre = str(raw_genres[idx] or "").strip()
        if not genre:
            genre = track_to_genre.get(track_id, "")
        test_genres.append(genre)
    unique_track_ids: list[str] = []
    unique_track_genres: list[str] = []
    track_index: dict[str, int] = {}
    sequence_counts: list[int] = []
    for track_id, genre in zip(train_track_ids, train_genres):
        if track_id not in track_index:
            track_index[track_id] = len(unique_track_ids)
            unique_track_ids.append(track_id)
            unique_track_genres.append(genre)
            sequence_counts.append(0)
        idx = track_index[track_id]
        sequence_counts[idx] += 1
        if not unique_track_genres[idx] and genre:
            unique_track_genres[idx] = genre
    normalized_scores = normalize_score_matrix(scores, score_normalization)
    aggregated_scores = np.zeros((len(unique_track_ids), normalized_scores.shape[1]), dtype=np.float64)
    for pool_idx, track_id in enumerate(train_track_ids):
        aggregated_scores[track_index[track_id]] += normalized_scores[pool_idx]
    if track_aggregation == "mean":
        aggregated_scores /= np.asarray(sequence_counts, dtype=np.float64)[:, None]
    elif track_aggregation != "sum":
        raise ValueError(f"Unsupported track aggregation: {track_aggregation}")
    return {
        "track_ids": unique_track_ids,
        "track_genres": unique_track_genres,
        "sequence_counts": sequence_counts,
        "scores": aggregated_scores,
        "test_track_ids": test_track_ids,
        "test_genres": test_genres,
        "track_aggregation": track_aggregation,
        "score_normalization": score_normalization,
    }


def load_stage_result(
    dataset_dir: Path,
    stage: str,
    condition: str,
    method: str | None,
    tracks_csv_override: Path | None,
    track_aggregation: str = "sum",
    score_normalization: str = "none",
) -> dict[str, Any]:
    runs = discover_stage_method_runs(dataset_dir, stage)
    resolved_method = method or select_primary_method(list(runs))
    if not resolved_method or resolved_method not in runs:
        raise FileNotFoundError(f"Could not find metadata in {dataset_dir}")
    metadata = load_json(runs[resolved_method]["metadata_path"])
    scores = load_npy_or_shards(runs[resolved_method]["scores_path"], allow_pickle=True)
    experiment_config_path = dataset_dir / "experiment_config.json"
    experiment_config = load_json(experiment_config_path) if experiment_config_path.exists() else {}
    test_losses_path = dataset_dir / "test_losses.npy"
    test_losses = np.load(str(test_losses_path)) if test_losses_path.exists() else np.array([], dtype=np.float32)
    train_genres, test_genres, _ = resolve_track_genres(metadata, experiment_config, tracks_csv_override)
    aggregated = aggregate_stage(
        scores,
        {**metadata, "train_genres": train_genres, "test_genres": test_genres},
        {},
        track_aggregation=track_aggregation,
        score_normalization=score_normalization,
    )
    aggregated["metadata"] = metadata
    aggregated["experiment_config"] = experiment_config
    aggregated["test_losses"] = test_losses
    aggregated["stage"] = stage
    aggregated["condition"] = condition
    aggregated["method"] = str(metadata.get("method") or experiment_config.get("method") or resolved_method)
    aggregated["dataset_dir"] = str(dataset_dir)
    return aggregated
