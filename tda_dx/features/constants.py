from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

TDADX_DATA_ROOT = Path(os.environ.get("TDADX_DATA_ROOT", "./data"))
TDADX_CHECKPOINT_ROOT = Path(os.environ.get("TDADX_CHECKPOINT_ROOT", "./checkpoints"))
DEFAULT_AUDIO_ROOT = Path(os.environ.get("TDADX_AUDIO_ROOT", str(TDADX_DATA_ROOT / "fma_large_48k_npy_norm")))
DEFAULT_METADATA_ROOT = Path(os.environ.get("TDADX_METADATA_ROOT", str(TDADX_DATA_ROOT / "fma_metadata")))
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_AUDIO_FORMAT = "npy"
DEFAULT_HARMONIC_CHECKPOINT_ROOT = TDADX_CHECKPOINT_ROOT
DEFAULT_OMNIZART_CHORD_MODEL_PATH = DEFAULT_HARMONIC_CHECKPOINT_ROOT / "omnizart" / "checkpoints" / "chord" / "chord_v1"
HARMONIC_BACKENDS = ("chordino", "omnizart")
DEFAULT_HARMONIC_BACKEND = "chordino"
FEATURE_SCHEMA_VERSION = 8
LOW_VALID_CHORD_RATIO_THRESHOLD = 0.5

BEAT_INTERVAL_HIST_BINS: list[float] = list(np.linspace(0.2, 2.0, 17).tolist())
ONSET_INTERVAL_HIST_BINS: list[float] = list(np.linspace(0.05, 1.0, 17).tolist())
N_RMS_SEGMENTS = 8

FEATURE_TABLE_FIELDS = [
    "schema_version",
    "track_id",
    "split",
    "genre_top",
    "audio_path",
    "harmonic_backend",
    "harmonic_model_path",
    "record_status",
    "record_error",
    "tempo_bpm",
    "mean_beat_interval",
    "median_beat_interval",
    "std_beat_interval",
    "iqr_beat_interval",
    "cv_beat_interval",
    "median_beat_phase",
    "beat_count",
    "beat_times",
    "beat_positions",
    "beat_class_hist",
    "beat_interval_hist",
    "onset_times",
    "onset_interval_hist",
    "rms_segments",
    "key",
    "mode",
    "key_strength",
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
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_contrast_mean",
    "chord_roots",
    "chord_types",
    "chord_tokens",
    "chord_tokens_raw",
    "chord_sequence",
    "chord_sequence_collapsed",
    "chord_times",
    "raw_chord_event_count",
    "valid_chord_event_count",
    "invalid_chord_event_count",
    "invalid_chord_reason_counts",
    "valid_chord_ratio",
    "unique_chord_token_count",
    "collapsed_chord_token_count",
    "extractor_status",
]

REQUIRED_CACHE_FIELDS = {
    "schema_version",
    "track_id",
    "audio_path",
    "harmonic_backend",
    "harmonic_model_path",
    "genre_top",
    "record_status",
    "record_error",
    "extractor_status",
    "key_error",
    "beat_error",
    "chord_error",
    "key",
    "mode",
    "key_strength",
    "beat_count",
    "median_beat_phase",
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
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_contrast_mean",
    "tempo_bpm",
    "mean_beat_interval",
    "median_beat_interval",
    "std_beat_interval",
    "iqr_beat_interval",
    "cv_beat_interval",
    "beat_times",
    "beat_positions",
    "beat_class_hist",
    "beat_interval_hist",
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
    "raw_chord_event_count",
    "valid_chord_event_count",
    "invalid_chord_event_count",
    "invalid_chord_reason_counts",
    "valid_chord_ratio",
    "unique_chord_token_count",
    "collapsed_chord_token_count",
}

REQUIRED_CACHE_FIELDS_V1: frozenset[str] = frozenset({
    "schema_version",
    "track_id",
    "audio_path",
    "genre_top",
    "record_status",
    "record_error",
    "extractor_status",
    "mfcc_mean",
    "mfcc_var",
    "chroma_mean",
    "tonnetz_mean",
    "spectral_contrast_mean",
    "chord_tokens",
    "key",
    "beat_interval_hist",
    "spectral_centroid_mean",
})

EXTRACTOR_DEPENDENCY_NAMES = {
    "key": "essentia",
    "beat": "BeatNet",
    "chord": "Chordino",
}
HARMONIC_BACKEND_DEPENDENCY_NAMES = {
    "chordino": "Chordino",
    "omnizart": "Omnizart",
}

SAMPLE_RATE_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)k(?:hz)?(?!\d)", re.IGNORECASE)
