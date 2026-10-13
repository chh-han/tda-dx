from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from .constants import (
    BEAT_INTERVAL_HIST_BINS,
    FEATURE_SCHEMA_VERSION,
    N_RMS_SEGMENTS,
    ONSET_INTERVAL_HIST_BINS,
)


def _empty_feature_payload() -> dict[str, Any]:
    n_beat_bins = len(BEAT_INTERVAL_HIST_BINS) - 1
    n_onset_bins = len(ONSET_INTERVAL_HIST_BINS) - 1
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "record_status": "ok",
        "record_error": "",
        "harmonic_backend": "",
        "harmonic_model_path": "",
        "extractor_status": {
            "key": "missing",
            "beat": "missing",
            "chord": "missing",
        },
        "key_error": "",
        "beat_error": "",
        "chord_error": "",
        "key": "",
        "mode": "",
        "key_strength": "",
        "mfcc_mean": [],
        "mfcc_var": [],
        "stft_mean": [],
        "stft_var": [],
        "cqt_mean": [],
        "cqt_var": [],
        "vqt_mean": [],
        "vqt_var": [],
        "chroma_mean": [],
        "chroma_var": [],
        "tonnetz_mean": [],
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_var": 0.0,
        "spectral_contrast_mean": [],
        "onset_times": [],
        "onset_interval_hist": [0.0] * n_onset_bins,
        "rms_segments": [0.0] * N_RMS_SEGMENTS,
        "tempo_bpm": "",
        "mean_beat_interval": "",
        "median_beat_interval": "",
        "std_beat_interval": "",
        "iqr_beat_interval": "",
        "cv_beat_interval": "",
        "median_beat_phase": "",
        "beat_count": 0,
        "beat_times": [],
        "beat_positions": [],
        "beat_class_hist": [],
        "beat_interval_hist": [0.0] * n_beat_bins,
        "chord_roots": [],
        "chord_types": [],
        "chord_tokens": [],
        "chord_sequence": [],
        "chord_tokens_raw": [],
        "chord_sequence_collapsed": [],
        "chord_times": [],
        "raw_chord_event_count": 0,
        "valid_chord_event_count": 0,
        "invalid_chord_event_count": 0,
        "invalid_chord_reason_counts": {},
        "valid_chord_ratio": 0.0,
        "unique_chord_token_count": 0,
        "collapsed_chord_token_count": 0,
    }


def _error_feature_payload(message: str, *, status: str = "error") -> dict[str, Any]:
    payload = _empty_feature_payload()
    payload["record_status"] = status
    payload["record_error"] = message
    return payload


def _status_summary(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record.get(field) or "missing") for record in records)
    return dict(sorted(counts.items()))


def _summarize_numeric_series(values: list[int] | list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    series = np.asarray(values, dtype=np.float64)
    return {
        "count": int(series.size),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(np.median(series)),
        "p10": float(np.percentile(series, 10)),
        "p90": float(np.percentile(series, 90)),
    }


def _build_table_row(record: dict[str, Any]) -> dict[str, Any]:
    beat_positions = list(record.get("beat_positions", []))
    median_beat_phase = record.get("median_beat_phase", "")
    return {
        "schema_version": record.get("schema_version", FEATURE_SCHEMA_VERSION),
        "track_id": record.get("track_id", ""),
        "split": record.get("split", ""),
        "genre_top": record.get("genre_top", ""),
        "audio_path": record.get("audio_path", ""),
        "harmonic_backend": record.get("harmonic_backend", ""),
        "harmonic_model_path": record.get("harmonic_model_path", ""),
        "record_status": record.get("record_status", "ok"),
        "record_error": record.get("record_error", ""),
        "tempo_bpm": record.get("tempo_bpm", ""),
        "mean_beat_interval": record.get("mean_beat_interval", ""),
        "median_beat_interval": record.get("median_beat_interval", ""),
        "std_beat_interval": record.get("std_beat_interval", ""),
        "iqr_beat_interval": record.get("iqr_beat_interval", ""),
        "cv_beat_interval": record.get("cv_beat_interval", ""),
        "median_beat_phase": median_beat_phase,
        "beat_count": record.get("beat_count", ""),
        "beat_times": json.dumps(record.get("beat_times", [])),
        "beat_positions": json.dumps(beat_positions),
        "beat_class_hist": json.dumps(record.get("beat_class_hist", [])),
        "beat_interval_hist": json.dumps(record.get("beat_interval_hist", [])),
        "onset_times": json.dumps(record.get("onset_times", [])),
        "onset_interval_hist": json.dumps(record.get("onset_interval_hist", [])),
        "rms_segments": json.dumps(record.get("rms_segments", [])),
        "key": record.get("key", ""),
        "mode": record.get("mode", ""),
        "key_strength": record.get("key_strength", ""),
        "mfcc_mean": json.dumps(record.get("mfcc_mean", [])),
        "mfcc_var": json.dumps(record.get("mfcc_var", [])),
        "stft_mean": json.dumps(record.get("stft_mean", [])),
        "stft_var": json.dumps(record.get("stft_var", [])),
        "cqt_mean": json.dumps(record.get("cqt_mean", [])),
        "cqt_var": json.dumps(record.get("cqt_var", [])),
        "vqt_mean": json.dumps(record.get("vqt_mean", [])),
        "vqt_var": json.dumps(record.get("vqt_var", [])),
        "chroma_mean": json.dumps(record.get("chroma_mean", [])),
        "chroma_var": json.dumps(record.get("chroma_var", [])),
        "tonnetz_mean": json.dumps(record.get("tonnetz_mean", [])),
        "spectral_centroid_mean": record.get("spectral_centroid_mean", 0.0),
        "spectral_centroid_var": record.get("spectral_centroid_var", 0.0),
        "spectral_contrast_mean": json.dumps(record.get("spectral_contrast_mean", [])),
        "chord_roots": json.dumps(record.get("chord_roots", [])),
        "chord_types": json.dumps(record.get("chord_types", [])),
        "chord_tokens": json.dumps(record.get("chord_tokens", [])),
        "chord_tokens_raw": json.dumps(record.get("chord_tokens_raw", record.get("chord_tokens", []))),
        "chord_sequence": json.dumps(record.get("chord_sequence", [])),
        "chord_sequence_collapsed": json.dumps(
            record.get("chord_sequence_collapsed", record.get("chord_sequence", []))
        ),
        "chord_times": json.dumps(record.get("chord_times", [])),
        "raw_chord_event_count": record.get("raw_chord_event_count", 0),
        "valid_chord_event_count": record.get("valid_chord_event_count", 0),
        "invalid_chord_event_count": record.get("invalid_chord_event_count", 0),
        "invalid_chord_reason_counts": json.dumps(record.get("invalid_chord_reason_counts", {}), sort_keys=True),
        "valid_chord_ratio": record.get("valid_chord_ratio", 0.0),
        "unique_chord_token_count": record.get("unique_chord_token_count", 0),
        "collapsed_chord_token_count": record.get("collapsed_chord_token_count", 0),
        "extractor_status": json.dumps(record.get("extractor_status", {}), sort_keys=True),
    }
