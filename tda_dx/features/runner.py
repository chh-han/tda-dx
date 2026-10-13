from __future__ import annotations

import os
import warnings

# Suppress TensorFlow and absl noise before any imports that trigger them
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore", category=FutureWarning, module="BeatNet")
warnings.filterwarnings("ignore", category=UserWarning, module="madmom")
warnings.filterwarnings("ignore", message=".*SavedModel saved prior to TF 2.5.*")
warnings.filterwarnings("ignore", message=".*No training configuration found.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None  # type: ignore[assignment]

from .audio import (
    _coerce_audio_array,
    _compute_mustango_bpm,
    _detect_dependencies,
    _ensure_legacy_collections_aliases,
    _ensure_optional_pyaudio_stub,
    _extract_beat_arrays,
    _safe_import_librosa,
    _summarize_beats,
    _write_temp_wav,
)
from .cache import _cache_row_is_reusable
from .constants import (
    BEAT_INTERVAL_HIST_BINS,
    DEFAULT_AUDIO_ROOT,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_HARMONIC_BACKEND,
    DEFAULT_OMNIZART_CHORD_MODEL_PATH,
    HARMONIC_BACKENDS,
    DEFAULT_METADATA_ROOT,
    FEATURE_SCHEMA_VERSION,
    FEATURE_TABLE_FIELDS,
    N_RMS_SEGMENTS,
    ONSET_INTERVAL_HIST_BINS,
)
from .harmonic import _extract_harmonic_features
from .manifest import _manifest_payload
from .metadata import (
    _filter_track_rows,
    _filter_track_rows_by_split,
    _load_track_rows,
    _resolve_sample_rate,
    _resolve_split_filter,
    _resolve_subset_filter,
    _row_split,
    _track_audio_path,
)
from .records import (
    _build_table_row,
    _empty_feature_payload,
    _error_feature_payload,
)


def _summarize_transform(transform: np.ndarray | None) -> tuple[list[float], list[float]]:
    if transform is None:
        return [], []
    array = np.asarray(transform, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        return [], []
    return (
        np.mean(array, axis=1).astype(np.float32).tolist(),
        np.var(array, axis=1).astype(np.float32).tolist(),
    )


def _compute_cqt_summary(
    librosa_module: Any,
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray | None, list[float], list[float]]:
    try:
        cqt = np.abs(librosa_module.cqt(y=y, sr=sr))
        cqt_db = librosa_module.amplitude_to_db(cqt, ref=np.max)
    except Exception:
        return None, [], []
    cqt_mean, cqt_var = _summarize_transform(cqt_db)
    return cqt, cqt_mean, cqt_var


def extract_track_features(
    audio_path: Path,
    sample_rate: int,
    dependencies: dict[str, bool] | None = None,
    extract_version: str = "all",
    harmonic_backend: str = DEFAULT_HARMONIC_BACKEND,
    harmonic_model_path: Path | None = None,
) -> dict[str, Any]:
    is_npy = audio_path.suffix.lower() == ".npy"
    dependency_state = dependencies or _detect_dependencies()
    feature_row = _empty_feature_payload()
    temp_wav_path: Path | None = None

    if not dependency_state.get("librosa", False):
        return _error_feature_payload("librosa dependency unavailable", status="dependency_error")

    try:
        librosa = _safe_import_librosa()
        if is_npy:
            y = _coerce_audio_array(np.load(str(audio_path)))
            sr = sample_rate
        else:
            y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            y = _coerce_audio_array(y)
    except Exception as exc:
        return _error_feature_payload(f"audio_load_failed: {exc}")

    try:
        D = librosa.stft(y)
        S_linear = np.abs(D)
        S_power = S_linear ** 2
        log_mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S_power, sr=sr))
    except Exception:
        S_linear = None
        S_power = None
        log_mel = None

    stft_db: np.ndarray | None = None
    if S_power is not None:
        try:
            stft_db = librosa.power_to_db(S_power, ref=np.max)
        except Exception:
            stft_db = None

    try:
        mfcc = (
            librosa.feature.mfcc(S=log_mel, n_mfcc=13)
            if log_mel is not None
            else librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        )
    except Exception:
        mfcc = np.zeros((13, 1), dtype=np.float32)

    cqt: np.ndarray | None = None
    if extract_version != "v1":
        stft_mean, stft_var = _summarize_transform(stft_db)
        feature_row["stft_mean"] = stft_mean
        feature_row["stft_var"] = stft_var

        cqt, cqt_mean, cqt_var = _compute_cqt_summary(librosa, y, sr)
        feature_row["cqt_mean"] = cqt_mean
        feature_row["cqt_var"] = cqt_var

        try:
            vqt = np.abs(librosa.vqt(y=y, sr=sr))
            vqt_db = librosa.amplitude_to_db(vqt, ref=np.max)
        except Exception:
            vqt_db = None
        vqt_mean, vqt_var = _summarize_transform(vqt_db)
        feature_row["vqt_mean"] = vqt_mean
        feature_row["vqt_var"] = vqt_var

    try:
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr) if cqt is not None else librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        chroma = np.zeros((12, 1), dtype=np.float32)

    feature_row["mfcc_mean"] = np.mean(mfcc, axis=1).astype(np.float32).tolist()
    feature_row["mfcc_var"] = np.var(mfcc, axis=1).astype(np.float32).tolist()
    feature_row["chroma_mean"] = np.mean(chroma, axis=1).astype(np.float32).tolist()
    if extract_version != "v1":
        feature_row["chroma_var"] = np.var(chroma, axis=1).astype(np.float32).tolist()

    try:
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        feature_row["tonnetz_mean"] = np.mean(tonnetz, axis=1).astype(np.float32).tolist()
    except Exception:
        feature_row["tonnetz_mean"] = []

    if extract_version != "v1":
        try:
            centroid = (
                librosa.feature.spectral_centroid(S=S_linear, sr=sr)
                if S_linear is not None
                else librosa.feature.spectral_centroid(y=y, sr=sr)
            )
            feature_row["spectral_centroid_mean"] = float(np.mean(centroid))
            feature_row["spectral_centroid_var"] = float(np.var(centroid))
        except Exception:
            pass

    try:
        contrast = (
            librosa.feature.spectral_contrast(S=S_linear, sr=sr)
            if S_linear is not None
            else librosa.feature.spectral_contrast(y=y, sr=sr)
        )
        feature_row["spectral_contrast_mean"] = np.mean(contrast, axis=1).astype(np.float32).tolist()
    except Exception:
        feature_row["spectral_contrast_mean"] = []

    if extract_version != "v1":
        try:
            rms = librosa.feature.rms(y=y)[0]
            segs = np.array_split(rms, N_RMS_SEGMENTS)
            feature_row["rms_segments"] = [float(np.mean(s)) if len(s) > 0 else 0.0 for s in segs]
        except Exception:
            pass

    if extract_version != "v1":
        try:
            if log_mel is not None:
                onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr)
                onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            else:
                onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
            feature_row["onset_times"] = onset_times.tolist()
            if len(onset_times) > 1:
                ioi = np.diff(onset_times)
                counts, _ = np.histogram(ioi, bins=ONSET_INTERVAL_HIST_BINS)
                total = int(counts.sum())
                feature_row["onset_interval_hist"] = (
                    (counts / total).astype(np.float32).tolist() if total > 0 else [0.0] * len(counts)
                )
        except Exception:
            pass

    try:
        extractor_audio_path = audio_path
        if is_npy and (dependency_state.get("BeatNet", False) or harmonic_backend in HARMONIC_BACKENDS):
            temp_wav_path = _write_temp_wav(y, sample_rate)
            extractor_audio_path = temp_wav_path

        if dependency_state.get("essentia", False):
            try:
                from essentia.standard import KeyExtractor, MonoLoader

                if is_npy:
                    audio = y
                else:
                    audio = MonoLoader(filename=str(audio_path), sampleRate=sample_rate, resampleQuality=4)()
                key, scale, strength = KeyExtractor(sampleRate=sample_rate)(audio)[:3]
                feature_row["key"] = key
                feature_row["mode"] = scale
                if extract_version != "v1":
                    feature_row["key_strength"] = float(strength)
                feature_row["extractor_status"]["key"] = "ok"
            except Exception as exc:
                feature_row["key_error"] = str(exc)
                feature_row["extractor_status"]["key"] = "error"
        else:
            feature_row["extractor_status"]["key"] = "dependency_missing"

        if dependency_state.get("BeatNet", False):
            try:
                _ensure_legacy_collections_aliases()
                _ensure_optional_pyaudio_stub()
                from BeatNet.BeatNet import BeatNet

                estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
                beats = estimator.process(str(extractor_audio_path))
                beat_times, beat_positions = _extract_beat_arrays(beats)
                beat_summary = _summarize_beats(beat_times, beat_positions)
                if extract_version != "v1":
                    feature_row["beat_times"] = beat_times
                    feature_row["beat_positions"] = beat_positions
                    feature_row["beat_count"] = beat_summary.get("beat_count", len(beat_times))
                    feature_row["beat_class_hist"] = beat_summary.get("beat_class_hist", [])
                    feature_row["mean_beat_interval"] = beat_summary.get("mean_beat_interval")
                    feature_row["median_beat_interval"] = beat_summary.get("median_beat_interval")
                    feature_row["std_beat_interval"] = beat_summary.get("std_beat_interval")
                    feature_row["iqr_beat_interval"] = beat_summary.get("iqr_beat_interval")
                    feature_row["cv_beat_interval"] = beat_summary.get("cv_beat_interval")
                    feature_row["median_beat_phase"] = ""
                if len(beat_times) > 1:
                    intervals = np.diff(np.asarray(beat_times, dtype=np.float64))
                    if intervals.size:
                        counts, _ = np.histogram(intervals, bins=BEAT_INTERVAL_HIST_BINS)
                        total = int(counts.sum())
                        feature_row["beat_interval_hist"] = (
                            (counts / total).astype(np.float32).tolist() if total > 0 else [0.0] * len(counts)
                        )
                tempo_bpm = beat_summary.get("tempo_bpm")
                if tempo_bpm is not None:
                    feature_row["tempo_bpm"] = tempo_bpm
                feature_row["extractor_status"]["beat"] = "ok"
            except Exception as exc:
                feature_row["beat_error"] = str(exc)
                feature_row["extractor_status"]["beat"] = "error"
        else:
            feature_row["extractor_status"]["beat"] = "dependency_missing"

        harmonic_backend_normalized = str(harmonic_backend or DEFAULT_HARMONIC_BACKEND).strip().lower()
        harmonic_dependency_name = "Chordino" if harmonic_backend_normalized == "chordino" else "Omnizart"
        if dependency_state.get(harmonic_dependency_name, False):
            try:
                chord_data = _extract_harmonic_features(
                    extractor_audio_path,
                    backend=harmonic_backend_normalized,
                    model_path=harmonic_model_path,
                )
                if extract_version == "v1":
                    feature_row["chord_tokens"] = chord_data.get("chord_tokens", [])
                else:
                    feature_row.update(chord_data)
                feature_row["extractor_status"]["chord"] = "ok"
            except Exception as exc:
                feature_row["chord_error"] = str(exc)
                feature_row["extractor_status"]["chord"] = "error"
        else:
            feature_row["extractor_status"]["chord"] = "dependency_missing"
    finally:
        if temp_wav_path is not None:
            temp_wav_path.unlink(missing_ok=True)

    return feature_row


_UPGRADEABLE_FROM_VERSIONS: set[int] = {4, 5}


def _try_partial_upgrade(
    cached_record: dict[str, Any],
    audio_path: Path,
    sample_rate: int,
    dependencies: dict[str, bool],
) -> dict[str, Any] | None:
    if cached_record.get("schema_version") not in _UPGRADEABLE_FROM_VERSIONS:
        return None
    if cached_record.get("record_status") not in ("ok", None, ""):
        return None
    if not audio_path.exists():
        return None
    if not dependencies.get("librosa", False):
        return None

    upgraded: dict[str, Any] = dict(cached_record)
    upgraded["schema_version"] = FEATURE_SCHEMA_VERSION

    beat_times = cached_record.get("beat_times") or []
    beat_positions = cached_record.get("beat_positions") or []
    beat_summary = _summarize_beats(beat_times, beat_positions)
    n_beat_bins = len(BEAT_INTERVAL_HIST_BINS) - 1
    upgraded["tempo_bpm"] = beat_summary.get("tempo_bpm", upgraded.get("tempo_bpm"))
    upgraded["mean_beat_interval"] = beat_summary.get("mean_beat_interval", "")
    upgraded["median_beat_interval"] = beat_summary.get("median_beat_interval", "")
    upgraded["std_beat_interval"] = beat_summary.get("std_beat_interval", "")
    upgraded["iqr_beat_interval"] = beat_summary.get("iqr_beat_interval", "")
    upgraded["cv_beat_interval"] = beat_summary.get("cv_beat_interval", "")
    upgraded["beat_count"] = beat_summary.get("beat_count", len(beat_times))
    upgraded["beat_class_hist"] = beat_summary.get("beat_class_hist", [])
    upgraded["median_beat_phase"] = ""
    if len(beat_times) > 1:
        intervals = np.diff(np.asarray(beat_times, dtype=np.float64))
        if intervals.size > 0:
            counts, _ = np.histogram(intervals, bins=BEAT_INTERVAL_HIST_BINS)
            total = int(counts.sum())
            upgraded["beat_interval_hist"] = (
                (counts / total).astype(np.float32).tolist() if total > 0 else [0.0] * n_beat_bins
            )
        else:
            upgraded["beat_interval_hist"] = [0.0] * n_beat_bins
    else:
        upgraded["beat_interval_hist"] = [0.0] * n_beat_bins

    try:
        librosa = _safe_import_librosa()
        is_npy = audio_path.suffix.lower() == ".npy"
        if is_npy:
            y = _coerce_audio_array(np.load(str(audio_path)))
            sr = sample_rate
        else:
            y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            y = _coerce_audio_array(y)
    except Exception:
        return None

    try:
        D = librosa.stft(y)
        S_linear = np.abs(D)
        S_power = S_linear ** 2
        log_mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S_power, sr=sr))
    except Exception:
        S_linear = None
        S_power = None
        log_mel = None

    try:
        stft_db = librosa.power_to_db(S_power, ref=np.max) if S_power is not None else None
    except Exception:
        stft_db = None
    upgraded["stft_mean"], upgraded["stft_var"] = _summarize_transform(stft_db)

    cqt, upgraded["cqt_mean"], upgraded["cqt_var"] = _compute_cqt_summary(librosa, y, sr)

    try:
        vqt = np.abs(librosa.vqt(y=y, sr=sr))
        vqt_db = librosa.amplitude_to_db(vqt, ref=np.max)
    except Exception:
        vqt_db = None
    upgraded["vqt_mean"], upgraded["vqt_var"] = _summarize_transform(vqt_db)

    try:
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr) if cqt is not None else librosa.feature.chroma_cqt(y=y, sr=sr)
        upgraded["chroma_var"] = np.var(chroma, axis=1).astype(np.float32).tolist()
        try:
            tonnetz = librosa.feature.tonnetz(chroma=chroma)
            upgraded["tonnetz_mean"] = np.mean(tonnetz, axis=1).astype(np.float32).tolist()
        except Exception:
            upgraded.setdefault("tonnetz_mean", [])
    except Exception:
        upgraded.setdefault("chroma_var", [])
        upgraded.setdefault("tonnetz_mean", [])

    try:
        centroid = (
            librosa.feature.spectral_centroid(S=S_linear, sr=sr)
            if S_linear is not None
            else librosa.feature.spectral_centroid(y=y, sr=sr)
        )
        upgraded["spectral_centroid_mean"] = float(np.mean(centroid))
        upgraded["spectral_centroid_var"] = float(np.var(centroid))
    except Exception:
        upgraded.setdefault("spectral_centroid_mean", 0.0)
        upgraded.setdefault("spectral_centroid_var", 0.0)

    try:
        contrast = (
            librosa.feature.spectral_contrast(S=S_linear, sr=sr)
            if S_linear is not None
            else librosa.feature.spectral_contrast(y=y, sr=sr)
        )
        upgraded["spectral_contrast_mean"] = np.mean(contrast, axis=1).astype(np.float32).tolist()
    except Exception:
        upgraded.setdefault("spectral_contrast_mean", [])

    try:
        rms = librosa.feature.rms(y=y)[0]
        segs = np.array_split(rms, N_RMS_SEGMENTS)
        upgraded["rms_segments"] = [float(np.mean(s)) if len(s) > 0 else 0.0 for s in segs]
    except Exception:
        upgraded.setdefault("rms_segments", [0.0] * N_RMS_SEGMENTS)

    n_onset_bins = len(ONSET_INTERVAL_HIST_BINS) - 1
    try:
        if log_mel is not None:
            onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        else:
            onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        upgraded["onset_times"] = onset_times.tolist()
        if len(onset_times) > 1:
            ioi = np.diff(onset_times)
            counts, _ = np.histogram(ioi, bins=ONSET_INTERVAL_HIST_BINS)
            total = int(counts.sum())
            upgraded["onset_interval_hist"] = (
                (counts / total).astype(np.float32).tolist() if total > 0 else [0.0] * n_onset_bins
            )
        else:
            upgraded.setdefault("onset_interval_hist", [0.0] * n_onset_bins)
    except Exception:
        upgraded.setdefault("onset_times", [])
        upgraded.setdefault("onset_interval_hist", [0.0] * n_onset_bins)

    return upgraded


def _process_one_track(
    row: dict[str, str],
    audio_root: Path,
    cache_dir: Path,
    sample_rate: int,
    audio_format: str,
    dependencies: dict[str, bool],
    force: bool,
    extract_version: str = "all",
    harmonic_backend: str = DEFAULT_HARMONIC_BACKEND,
    harmonic_model_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    track_id = (row.get("track.track_id") or row.get("track_id") or row.get("") or "").strip()
    normalized_track_id = str(int(track_id))
    split_value = _row_split(row)
    genre_top = (row.get("track.genre_top") or row.get("genre_top") or "").strip()
    cache_path = cache_dir / f"{int(track_id):06d}.json"
    audio_path = _track_audio_path(audio_root, track_id, audio_format)

    base_record: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "track_id": normalized_track_id,
        "split": split_value,
        "audio_path": str(audio_path),
        "genre_top": genre_top,
        "harmonic_backend": harmonic_backend,
        "harmonic_model_path": str(harmonic_model_path or ""),
    }

    if cache_path.exists() and not force:
        try:
            cached_record = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached_record = None
        if cached_record is not None:
            if _cache_row_is_reusable(
                cached_record,
                expected_track_id=normalized_track_id,
                expected_split=split_value,
                expected_audio_path=audio_path,
                expected_harmonic_backend=harmonic_backend,
                expected_harmonic_model_path=harmonic_model_path,
                dependencies=dependencies,
                extract_version=extract_version,
            ):
                payload = dict(cached_record)
                if str(payload.get("split", "")).strip().lower() != split_value:
                    payload["split"] = split_value
                    cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
                return "cached", payload

            upgraded = _try_partial_upgrade(cached_record, audio_path, sample_rate, dependencies)
            if upgraded is not None:
                upgraded.update({key: base_record[key] for key in ("track_id", "split", "genre_top", "audio_path")})
                cache_path.write_text(json.dumps(upgraded, ensure_ascii=True), encoding="utf-8")
                return "upgraded", upgraded

    if not audio_path.exists():
        record = dict(base_record)
        record.update(_error_feature_payload("audio_file_missing", status="missing_audio"))
        cache_path.write_text(json.dumps(record, ensure_ascii=True), encoding="utf-8")
        return "missing", record

    record = dict(base_record)
    if extract_version == "all":
        payload = extract_track_features(
            audio_path,
            sample_rate,
            dependencies,
            extract_version=extract_version,
            harmonic_backend=harmonic_backend,
            harmonic_model_path=harmonic_model_path,
        )
    else:
        payload = extract_track_features(
            audio_path,
            sample_rate,
            dependencies,
            extract_version=extract_version,
            harmonic_backend=harmonic_backend,
            harmonic_model_path=harmonic_model_path,
        )
    record.update(payload)
    record["harmonic_backend"] = harmonic_backend
    record["harmonic_model_path"] = str(harmonic_model_path or "")
    if extract_version == "v1":
        record["extract_version"] = "v1"
    cache_path.write_text(json.dumps(record, ensure_ascii=True), encoding="utf-8")
    return record.get("record_status", "ok"), record


def _worker_fn(args: tuple) -> tuple[str, dict[str, Any]]:
    (
        row,
        audio_root,
        cache_dir,
        sample_rate,
        audio_format,
        dependencies,
        force,
        extract_version,
        harmonic_backend,
        harmonic_model_path,
    ) = args
    try:
        return _process_one_track(
            row,
            audio_root,
            cache_dir,
            sample_rate,
            audio_format,
            dependencies,
            force,
            extract_version=extract_version,
            harmonic_backend=harmonic_backend,
            harmonic_model_path=harmonic_model_path,
        )
    except Exception as exc:
        track_id = (row.get("track.track_id") or row.get("track_id") or row.get("") or "").strip()
        try:
            normalized_track_id = str(int(track_id))
        except (ValueError, TypeError):
            normalized_track_id = track_id
        audio_path = str(_track_audio_path(audio_root, track_id, audio_format)) if track_id else ""
        record: dict[str, Any] = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "track_id": normalized_track_id,
            "split": _row_split(row),
            "audio_path": audio_path,
            "genre_top": (row.get("track.genre_top") or row.get("genre_top") or "").strip(),
            "harmonic_backend": harmonic_backend,
            "harmonic_model_path": str(harmonic_model_path or ""),
        }
        record.update(_error_feature_payload(f"worker_exception: {exc}"))
        return "error", record


def run_extraction(
    audio_root: Path,
    metadata_root: Path,
    output_dir: Path,
    sample_rate: int | None,
    start: int | None = None,
    limit: int | None = None,
    force: bool = False,
    audio_format: str = "mp3",
    subset: str = "auto",
    split: str = "all",
    workers: int = 1,
    extract_version: str = "all",
    harmonic_backend: str = DEFAULT_HARMONIC_BACKEND,
    harmonic_model_path: Path | None = DEFAULT_OMNIZART_CHORD_MODEL_PATH,
) -> None:
    tracks_csv = metadata_root / "tracks.csv"
    track_rows = _load_track_rows(tracks_csv)
    subset_filter = _resolve_subset_filter(audio_root, subset)
    split_filter = _resolve_split_filter(split)
    track_rows = _filter_track_rows(track_rows, subset_filter)
    track_rows = _filter_track_rows_by_split(track_rows, split_filter)
    sample_rate = _resolve_sample_rate(audio_root, audio_format, sample_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "per_track"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dependencies = _detect_dependencies()

    valid_rows = [
        row for row in track_rows if (row.get("track.track_id") or row.get("track_id") or row.get("") or "").strip()
    ]
    if start is not None:
        valid_rows = valid_rows[start:]
    if limit is not None:
        valid_rows = valid_rows[:limit]

    total = len(valid_rows)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    t0 = time.monotonic()

    print(
        f"[feature_extraction] Starting: {total} tracks, {workers} worker(s), "
        f"extract_version={extract_version}, output -> {output_dir}",
        flush=True,
    )
    print(
        f"[feature_extraction] Harmonic backend: backend={harmonic_backend}, "
        f"model_path={str(harmonic_model_path or '')}",
        flush=True,
    )
    print(
        f"[feature_extraction] Dependencies: { {k: ('ok' if v else 'missing') for k, v in dependencies.items()} }",
        flush=True,
    )

    bar = _tqdm(total=total, unit="track", dynamic_ncols=True) if _tqdm is not None else None

    task_args = [
        (
            row,
            audio_root,
            cache_dir,
            sample_rate,
            audio_format,
            dependencies,
            force,
            extract_version,
            harmonic_backend,
            harmonic_model_path,
        )
        for row in valid_rows
    ]

    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for status, record in pool.imap_unordered(_worker_fn, task_args, chunksize=1):
                records.append(record)
                counts[status] = counts.get(status, 0) + 1
                if bar is not None:
                    bar.set_postfix(counts, track=record.get("track_id", ""))
                    bar.update(1)
    else:
        for args in task_args:
            status, record = _worker_fn(args)
            records.append(record)
            counts[status] = counts.get(status, 0) + 1
            if bar is not None:
                bar.set_postfix(counts, track=record.get("track_id", ""))
                bar.update(1)

    if bar is not None:
        bar.close()
    elapsed = time.monotonic() - t0
    summary_parts = ", ".join(f"{value} {key}" for key, value in sorted(counts.items()))
    print(f"[feature_extraction] Extraction done: {total} tracks - {summary_parts} - {elapsed:.1f}s", flush=True)

    def _sort_key(rec: dict[str, Any]) -> int:
        try:
            return int(rec.get("track_id") or 0)
        except (ValueError, TypeError):
            return 0

    records.sort(key=_sort_key)

    if start is not None:
        chunk_jsonl = output_dir / f"features_{start:06d}.jsonl"
        print(f"[feature_extraction] Chunk mode (start={start}): writing {chunk_jsonl} ...", flush=True)
        with chunk_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        print("[feature_extraction] Chunk done - run --aggregate-only to merge all chunks.", flush=True)
        return

    jsonl_path = output_dir / "features.jsonl"
    print(f"[feature_extraction] Writing {jsonl_path} ...", flush=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    tsv_path = output_dir / "features.tsv"
    print(f"[feature_extraction] Writing {tsv_path} ...", flush=True)
    table_rows = [_build_table_row(record) for record in records]
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_TABLE_FIELDS, delimiter="\t")
        writer.writeheader()
        if table_rows:
            writer.writerows(table_rows)

    manifest = _manifest_payload(
        audio_root=audio_root,
        metadata_root=metadata_root,
        output_dir=output_dir,
        sample_rate=sample_rate,
        subset_filter=subset_filter,
        split_filter=sorted(split_filter) if split_filter is not None else None,
        limit=limit,
        force=force,
        records=records,
        dependencies=dependencies,
    )
    manifest["missing_audio_count"] = sum(1 for record in records if record.get("record_status") == "missing_audio")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_elapsed = time.monotonic() - t0
    print(f"[feature_extraction] ALL DONE - {output_dir} ({total_elapsed:.1f}s total)", flush=True)


def aggregate_from_cache(
    output_dir: Path,
    audio_root: Path,
    metadata_root: Path,
    sample_rate: int,
) -> None:
    # Fast path: chunk jsonl files (features_######.jsonl) produced by array job
    chunk_files = sorted(
        output_dir.glob("features_*.jsonl"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if chunk_files:
        print(
            f"[aggregate] Fast path: merging {len(chunk_files)} chunk jsonl files "
            f"(skipping per_track reads) ...",
            flush=True,
        )
        records: list[dict[str, Any]] = []
        jsonl_path = output_dir / "features.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jsonl_out:
            for chunk in chunk_files:
                with chunk.open("r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        jsonl_out.write(line)          # raw copy — no re-serialization
                        records.append(json.loads(stripped))
        print(f"[aggregate] Loaded {len(records)} records from {len(chunk_files)} chunks", flush=True)
    else:
        # Fallback: read individual per_track/*.json files in parallel
        cache_dir = output_dir / "per_track"
        files = sorted(cache_dir.glob("*.json"), key=lambda file_path: int(file_path.stem))
        if not files:
            print(f"[aggregate] No chunk jsonl or per_track files found in {output_dir}", flush=True)
            return

        def _read_one(file_path: Path) -> tuple[int, dict[str, Any] | Exception]:
            try:
                return int(file_path.stem), json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return int(file_path.stem), exc

        n_workers = min(64, max(1, (os.cpu_count() or 1) * 4))
        print(f"[aggregate] Reading {len(files)} per_track files with {n_workers} threads ...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            raw = list(pool.map(_read_one, files))

        records = []
        for idx, result in sorted(raw, key=lambda x: x[0]):
            if isinstance(result, Exception):
                print(f"[aggregate] WARNING: skipping {idx}.json: {result}", flush=True)
            else:
                records.append(result)

        print(f"[aggregate] Loaded {len(records)} records from {cache_dir}", flush=True)

        jsonl_path = output_dir / "features.jsonl"
        print(f"[aggregate] Writing {jsonl_path} ...", flush=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    tsv_path = output_dir / "features.tsv"
    print(f"[aggregate] Writing {tsv_path} ...", flush=True)
    table_rows = [_build_table_row(record) for record in records]
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_TABLE_FIELDS, delimiter="\t")
        writer.writeheader()
        if table_rows:
            writer.writerows(table_rows)

    dependencies = _detect_dependencies()
    manifest = _manifest_payload(
        audio_root=audio_root,
        metadata_root=metadata_root,
        output_dir=output_dir,
        sample_rate=sample_rate,
        subset_filter=None,
        split_filter=None,
        limit=None,
        force=False,
        records=records,
        dependencies=dependencies,
    )
    manifest["aggregated_from_cache"] = True
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[aggregate] ALL DONE - {len(records)} records -> {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cached MIR features for FMA audio.")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--audio-format", choices=["mp3", "npy"], default=DEFAULT_AUDIO_FORMAT)
    parser.add_argument("--subset", choices=["auto", "all", "small", "medium", "large"], default="auto")
    parser.add_argument("--split", type=str, default="all", help="Track split filter (e.g. training,test or all)")
    parser.add_argument("--start", type=int, default=None, help="Skip the first N tracks (0-based index)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip extraction. Rebuild features.jsonl/tsv/manifest from all existing per_track/*.json files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). Values >1 use multiprocessing.Pool with spawn.",
    )
    parser.add_argument(
        "--extract-version",
        choices=["v1", "all"],
        default="all",
        help=(
            "v1: extract only fields used by analyze.py (tempo_bpm, beat_interval_hist, key, mode, "
            "mfcc_mean, mfcc_var, chroma_mean, tonnetz_mean, spectral_contrast_mean, chord_tokens). "
            "all (default): extract every field."
        ),
    )
    parser.add_argument(
        "--harmonic-backend",
        choices=list(HARMONIC_BACKENDS),
        default=DEFAULT_HARMONIC_BACKEND,
        help="Chord extraction backend. chordino keeps the old lightweight extractor; omnizart uses Harmony Transformer checkpoints.",
    )
    parser.add_argument(
        "--harmonic-model-path",
        type=Path,
        default=DEFAULT_OMNIZART_CHORD_MODEL_PATH,
        help="Local checkpoint path for the harmonic backend. For Omnizart this should point to open-musiclm/checkpoints/omnizart/checkpoints/chord/chord_v1.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.aggregate_only:
        sample_rate = _resolve_sample_rate(args.audio_root.resolve(), args.audio_format, args.sample_rate)
        aggregate_from_cache(
            output_dir=args.output_dir.resolve(),
            audio_root=args.audio_root.resolve(),
            metadata_root=args.metadata_root.resolve(),
            sample_rate=sample_rate,
        )
        return

    run_extraction(
        audio_root=args.audio_root.resolve(),
        metadata_root=args.metadata_root.resolve(),
        output_dir=args.output_dir.resolve(),
        sample_rate=args.sample_rate,
        start=args.start,
        limit=args.limit,
        force=args.force,
        audio_format=args.audio_format,
        subset=args.subset,
        split=args.split,
        workers=args.workers,
        extract_version=args.extract_version,
        harmonic_backend=args.harmonic_backend,
        harmonic_model_path=args.harmonic_model_path.resolve() if args.harmonic_model_path else None,
    )


if __name__ == "__main__":
    main()
