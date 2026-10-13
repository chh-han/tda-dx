from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_HARMONIC_BACKEND,
    EXTRACTOR_DEPENDENCY_NAMES,
    FEATURE_SCHEMA_VERSION,
    HARMONIC_BACKEND_DEPENDENCY_NAMES,
    REQUIRED_CACHE_FIELDS,
    REQUIRED_CACHE_FIELDS_V1,
)


def _cache_row_needs_dependency_retry(record: dict[str, Any], dependencies: dict[str, bool]) -> bool:
    if record.get("record_status") == "dependency_error" and dependencies.get("librosa", False):
        return True
    extractor_status = record.get("extractor_status")
    if not isinstance(extractor_status, dict):
        return True
    for extractor_name, dependency_name in EXTRACTOR_DEPENDENCY_NAMES.items():
        if extractor_status.get(extractor_name) == "dependency_missing" and dependencies.get(dependency_name, False):
            return True
    harmonic_backend = str(record.get("harmonic_backend") or DEFAULT_HARMONIC_BACKEND).strip().lower()
    harmonic_dependency = HARMONIC_BACKEND_DEPENDENCY_NAMES.get(harmonic_backend)
    if harmonic_dependency is not None:
        if extractor_status.get("chord") == "dependency_missing" and dependencies.get(harmonic_dependency, False):
            return True
    return False


def _cache_row_is_reusable(
    record: dict[str, Any],
    *,
    expected_track_id: str,
    expected_split: str,
    expected_audio_path: Path,
    expected_harmonic_backend: str,
    expected_harmonic_model_path: Path | None,
    dependencies: dict[str, bool],
    extract_version: str = "all",
) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("schema_version") != FEATURE_SCHEMA_VERSION:
        return False
    if str(record.get("track_id", "")).strip() != expected_track_id:
        return False
    existing_split = str(record.get("split", "")).strip().lower()
    if existing_split and existing_split != expected_split:
        return False
    if str(record.get("audio_path", "")).strip() != str(expected_audio_path):
        return False
    existing_backend = str(record.get("harmonic_backend", DEFAULT_HARMONIC_BACKEND)).strip().lower()
    if existing_backend != expected_harmonic_backend.strip().lower():
        return False
    existing_model_path = str(record.get("harmonic_model_path", "")).strip()
    expected_model_path = str(expected_harmonic_model_path or "").strip()
    if existing_model_path != expected_model_path:
        return False
    cached_ev = record.get("extract_version", "all")
    if extract_version == "all" and cached_ev == "v1":
        return False

    required_fields = REQUIRED_CACHE_FIELDS_V1 if extract_version == "v1" else REQUIRED_CACHE_FIELDS
    if not required_fields.issubset(record):
        return False
    if not isinstance(record.get("extractor_status"), dict):
        return False

    v1_list_fields = (
        "mfcc_mean",
        "mfcc_var",
        "chroma_mean",
        "tonnetz_mean",
        "spectral_contrast_mean",
        "chord_tokens",
    )
    all_list_fields = (
        "mfcc_mean",
        "mfcc_var",
        "stft_mean",
        "stft_var",
        "cqt_mean",
        "cqt_var",
        "vqt_mean",
        "vqt_var",
        "chroma_mean",
        "chroma_var",
        "tonnetz_mean",
        "spectral_contrast_mean",
        "onset_times",
        "onset_interval_hist",
        "rms_segments",
        "chord_roots",
        "chord_types",
        "chord_tokens",
        "chord_tokens_raw",
        "chord_sequence",
        "chord_sequence_collapsed",
        "chord_times",
    )
    for list_field in (v1_list_fields if extract_version == "v1" else all_list_fields):
        if not isinstance(record.get(list_field), list):
            return False

    if extract_version != "v1":
        if not isinstance(record.get("invalid_chord_reason_counts"), dict):
            return False
        for numeric_field in (
            "raw_chord_event_count",
            "valid_chord_event_count",
            "invalid_chord_event_count",
            "valid_chord_ratio",
            "unique_chord_token_count",
            "collapsed_chord_token_count",
        ):
            if not isinstance(record.get(numeric_field), (int, float)):
                return False

    if record.get("record_status") == "missing_audio" and expected_audio_path.exists():
        return False
    if _cache_row_needs_dependency_retry(record, dependencies):
        return False
    return True
