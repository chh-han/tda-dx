from __future__ import annotations

import csv
from pathlib import Path

from .constants import DEFAULT_SAMPLE_RATE, SAMPLE_RATE_PATTERN


def _load_track_rows(tracks_csv: Path) -> list[dict[str, str]]:
    with tracks_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        level0 = next(reader)
        level1 = next(reader)
        _ = next(reader)
        headers = []
        for left, right in zip(level0, level1):
            key = right.strip() or left.strip()
            if left.strip() and right.strip():
                key = f"{left.strip()}.{right.strip()}"
            headers.append(key)
        rows = []
        for row in reader:
            rows.append({header: row[idx] if idx < len(row) else "" for idx, header in enumerate(headers)})
    return rows


def _row_subset(row: dict[str, str]) -> str:
    return (row.get("set.subset") or row.get("subset") or "").strip().lower()


def _row_split(row: dict[str, str]) -> str:
    return (row.get("set.split") or row.get("split") or "").strip().lower()


def _infer_subset_from_audio_root(audio_root: Path) -> str | None:
    name = audio_root.name.lower()
    for subset in ("small", "medium", "large"):
        if f"fma_{subset}" in name:
            return subset
    return None


def _resolve_subset_filter(audio_root: Path, subset: str) -> str | None:
    normalized = subset.strip().lower()
    if normalized == "all":
        return None
    if normalized == "auto":
        return _infer_subset_from_audio_root(audio_root)
    return normalized


def _filter_track_rows(track_rows: list[dict[str, str]], subset_filter: str | None) -> list[dict[str, str]]:
    if subset_filter is None:
        return track_rows
    return [row for row in track_rows if _row_subset(row) == subset_filter]


def _resolve_split_filter(split: str) -> set[str] | None:
    normalized = split.strip().lower()
    if not normalized or normalized == "all":
        return None
    parts = [token.strip().lower() for token in normalized.split(",")]
    selected = {token for token in parts if token}
    return selected or None


def _filter_track_rows_by_split(track_rows: list[dict[str, str]], split_filter: set[str] | None) -> list[dict[str, str]]:
    if split_filter is None:
        return track_rows
    return [row for row in track_rows if _row_split(row) in split_filter]


def _track_audio_path(audio_root: Path, track_id: str, audio_format: str = "mp3") -> Path:
    numeric = int(track_id)
    ext = "npy" if audio_format == "npy" else "mp3"
    return audio_root / f"{numeric:06d}"[:3] / f"{numeric:06d}.{ext}"


def _infer_sample_rate_from_path(audio_root: Path) -> int | None:
    match = SAMPLE_RATE_PATTERN.search(audio_root.name)
    if not match:
        return None
    return int(round(float(match.group(1)) * 1000))


def _resolve_sample_rate(audio_root: Path, audio_format: str, sample_rate: int | None) -> int:
    if sample_rate is not None:
        return int(sample_rate)
    if audio_format == "npy":
        inferred_rate = _infer_sample_rate_from_path(audio_root)
        if inferred_rate is None:
            raise ValueError(
                f"sample rate must be provided for npy audio when it cannot be inferred from {audio_root}"
            )
        return inferred_rate
    return DEFAULT_SAMPLE_RATE
