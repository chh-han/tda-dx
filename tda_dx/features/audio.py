from __future__ import annotations

import collections
import importlib
import importlib.util
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _detect_dependencies() -> dict[str, bool]:
    modules = {
        "librosa": "librosa",
        "essentia": "essentia",
        "BeatNet": "BeatNet",
        "Chordino": "chord_extractor",
    }
    state = {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}
    try:
        importlib.import_module("omnizart")
    except Exception:
        state["Omnizart"] = False
    else:
        state["Omnizart"] = True
    return state


def _safe_import_librosa() -> Any:
    import librosa

    return librosa


def _ensure_legacy_collections_aliases() -> None:
    import collections.abc

    legacy_aliases = {
        "MutableMapping": collections.abc.MutableMapping,
        "MutableSequence": collections.abc.MutableSequence,
        "MutableSet": collections.abc.MutableSet,
    }
    for name, value in legacy_aliases.items():
        if not hasattr(collections, name):
            setattr(collections, name, value)


def _ensure_optional_pyaudio_stub() -> None:
    if "pyaudio" in sys.modules:
        return
    if importlib.util.find_spec("pyaudio") is not None:
        return

    class _MissingPyAudio:
        paFloat32 = object()

        class PyAudio:
            def open(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("pyaudio is unavailable; BeatNet stream mode is unsupported in this environment")

    sys.modules["pyaudio"] = _MissingPyAudio()


def _coerce_audio_array(data: np.ndarray) -> np.ndarray:
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 0:
        return audio.reshape(1)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
    return np.ascontiguousarray(audio.astype(np.float32, copy=False))


def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return temp_path


def _extract_beat_arrays(beats: Any) -> tuple[list[float], list[float]]:
    beat_array = np.asarray(beats)
    if beat_array.ndim != 2 or beat_array.size == 0:
        return [], []
    beat_times = beat_array[:, 0].astype(np.float64).tolist()
    beat_positions = beat_array[:, 1].astype(np.float64).tolist() if beat_array.shape[1] > 1 else []
    return beat_times, beat_positions


def _compute_mustango_bpm(beat_times: list[float]) -> float | None:
    if len(beat_times) < 3:
        return None
    intervals = np.diff(np.asarray(beat_times, dtype=np.float64))
    if intervals.size == 0 or np.any(intervals <= 0):
        return None
    local_bpm = 60.0 / intervals
    padded = np.concatenate(([local_bpm[0]], local_bpm, [local_bpm[-1]]))
    filtered = padded.copy()
    for idx in range(len(filtered) - 1):
        filtered[idx + 1] = np.median(filtered[idx : idx + 3])
    return float(np.mean(filtered[1:-1]))


def _summarize_beats(beat_times: list[float], beat_positions: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "beat_count": len(beat_times),
        "tempo_bpm": None,
        "mean_beat_interval": None,
        "median_beat_interval": None,
        "std_beat_interval": None,
        "iqr_beat_interval": None,
        "cv_beat_interval": None,
        "beat_class_hist": [],
    }
    if beat_positions:
        beat_classes = [int(round(value)) for value in beat_positions if value is not None]
        if beat_classes:
            max_class = max(beat_classes)
            if max_class > 0:
                counts = np.bincount(np.asarray(beat_classes, dtype=np.int64), minlength=max_class + 1)[1:]
                total = int(counts.sum())
                if total > 0:
                    summary["beat_class_hist"] = (counts / total).astype(np.float32).tolist()
    if len(beat_times) < 2:
        return summary
    intervals = np.diff(np.asarray(beat_times, dtype=np.float64))
    if intervals.size == 0 or np.any(intervals <= 0):
        return summary
    summary["tempo_bpm"] = _compute_mustango_bpm(beat_times)
    summary["mean_beat_interval"] = float(np.mean(intervals))
    summary["median_beat_interval"] = float(np.median(intervals))
    summary["std_beat_interval"] = float(np.std(intervals))
    summary["iqr_beat_interval"] = float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    summary["cv_beat_interval"] = float(std_interval / mean_interval) if mean_interval > 0 else None
    return summary
