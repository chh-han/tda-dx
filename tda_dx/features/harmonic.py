from __future__ import annotations

import csv
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_CHORD_PATTERN = re.compile(r"^\s*([A-G](?:#|b)?)(?::?(.*))?\s*$")


def _normalize_chord_label(label: Any) -> dict[str, Any]:
    raw = str(label or "").strip()
    normalized = {
        "raw_label": raw,
        "root": "",
        "chord_type": "",
        "token": "",
        "is_valid": False,
        "invalid_reason": "",
    }
    if not raw:
        normalized["invalid_reason"] = "empty"
        return normalized
    if raw.lower() in {"n", "nan"}:
        normalized["invalid_reason"] = "no_chord"
        return normalized
    match = _CHORD_PATTERN.match(raw)
    if not match:
        normalized["invalid_reason"] = "unparseable"
        return normalized
    root = match.group(1) or ""
    chord_type = (match.group(2) or "maj").strip() or "maj"
    token = f"{root}:{chord_type}"
    normalized.update(
        {
            "root": root,
            "chord_type": chord_type,
            "token": token,
            "is_valid": True,
        }
    )
    return normalized


def _coerce_chord_event(chord: Any) -> tuple[Any, float, float | None]:
    if isinstance(chord, dict):
        chord_label = chord.get("label", "")
        start_time = chord.get("start", chord.get("timestamp", 0.0))
        end_time = chord.get("end")
        return chord_label, float(start_time or 0.0), float(end_time) if end_time is not None else None
    if hasattr(chord, "chord") and hasattr(chord, "timestamp"):
        return chord.chord, float(chord.timestamp), None
    chord_label = chord[0] if len(chord) > 0 else ""
    chord_time = float(chord[1]) if len(chord) > 1 else 0.0
    end_time = float(chord[2]) if len(chord) > 2 and chord[2] is not None else None
    return chord_label, chord_time, end_time


def _collapse_token_sequence(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or token != collapsed[-1]:
            collapsed.append(token)
    return collapsed


def _normalize_chord_events(chords: Any) -> dict[str, Any]:
    chord_items = list(chords or [])
    if len(chord_items) > 2:
        chord_items = chord_items[1:-1]
    roots: list[str] = []
    types: list[str] = []
    tokens: list[str] = []
    raw_tokens: list[str] = []
    chord_times: list[float] = []
    invalid_reasons: Counter[str] = Counter()
    for chord in chord_items:
        chord_label, chord_time, _ = _coerce_chord_event(chord)
        normalized = _normalize_chord_label(chord_label)
        raw_tokens.append(str(normalized["raw_label"]))
        if not normalized["is_valid"]:
            invalid_reasons[str(normalized["invalid_reason"] or "unknown")] += 1
            continue
        roots.append(str(normalized["root"]))
        types.append(str(normalized["chord_type"]))
        tokens.append(str(normalized["token"]))
        chord_times.append(float(chord_time))
    collapsed_tokens = _collapse_token_sequence(tokens)
    raw_chord_event_count = len(chord_items)
    valid_chord_event_count = len(tokens)
    invalid_chord_event_count = sum(invalid_reasons.values())
    valid_chord_ratio = valid_chord_event_count / raw_chord_event_count if raw_chord_event_count else 0.0
    return {
        "chord_roots": roots,
        "chord_types": types,
        "chord_tokens": tokens,
        "chord_tokens_raw": raw_tokens,
        "chord_sequence": list(tokens),
        "chord_sequence_collapsed": collapsed_tokens,
        "chord_times": chord_times,
        "raw_chord_event_count": raw_chord_event_count,
        "valid_chord_event_count": valid_chord_event_count,
        "invalid_chord_event_count": invalid_chord_event_count,
        "invalid_chord_reason_counts": dict(sorted(invalid_reasons.items())),
        "valid_chord_ratio": valid_chord_ratio,
        "unique_chord_token_count": len(set(tokens)),
        "collapsed_chord_token_count": len(collapsed_tokens),
    }


def _extract_chordino_events(audio_path: Path) -> list[Any]:
    from chord_extractor.extractors import Chordino

    return list(Chordino().extract(str(audio_path)))


def _resolve_omnizart_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _load_omnizart_csv(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"omnizart_csv_missing_header: {csv_path}")
        start_key = _resolve_omnizart_column(reader.fieldnames, ("start", "start_time", "start_sec", "onset"))
        end_key = _resolve_omnizart_column(reader.fieldnames, ("end", "end_time", "end_sec", "offset"))
        chord_key = _resolve_omnizart_column(reader.fieldnames, ("chord", "label", "name", "value"))
        if start_key is None or end_key is None or chord_key is None:
            raise RuntimeError(f"omnizart_csv_unrecognized_columns: {reader.fieldnames}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "label": row.get(chord_key, ""),
                    "start": float(row.get(start_key) or 0.0),
                    "end": float(row.get(end_key) or 0.0),
                }
            )
    return rows


def _find_omnizart_output_file(output_dir: Path) -> Path | None:
    csv_files = sorted(output_dir.rglob("*.csv"))
    if csv_files:
        return csv_files[0]

    # Omnizart sometimes writes a CSV payload without the ".csv" suffix.
    for candidate in sorted(output_dir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in {".mid", ".midi", ".wav", ".mp3", ".npy"}:
            continue
        try:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                header = handle.readline().strip().lower()
        except Exception:
            continue
        if "chord" in header and "start" in header and "end" in header:
            return candidate
    return None


_omnizart_app_cache: dict[str, Any] = {}


def _get_omnizart_app(model_path: Path) -> Any:
    """Return a cached ChordTranscription instance for this worker process.

    With multiprocessing.Pool(spawn), each worker is a fresh process, so the
    cache is populated once per worker (not once per track).  This avoids
    re-loading the TensorFlow model for every track, which was the dominant
    per-track overhead.
    """
    key = str(model_path)
    if key not in _omnizart_app_cache:
        try:
            from omnizart.chord.app import ChordTranscription
        except Exception as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(f"omnizart_import_failed: {exc}") from exc
        _omnizart_app_cache[key] = ChordTranscription()
    return _omnizart_app_cache[key]


def _extract_omnizart_events(audio_path: Path, model_path: Path) -> list[dict[str, Any]]:
    if not model_path.exists():
        raise RuntimeError(f"omnizart_model_missing: {model_path}")

    app = _get_omnizart_app(model_path)

    with tempfile.TemporaryDirectory(prefix="omnizart_chord_") as temp_dir:
        output_dir = Path(temp_dir)
        app.transcribe(str(audio_path), model_path=str(model_path), output=str(output_dir))
        csv_file = _find_omnizart_output_file(output_dir)
        if csv_file is None:
            raise RuntimeError(f"omnizart_csv_not_found: {output_dir}")
        return _load_omnizart_csv(csv_file)


def _extract_harmonic_features(
    audio_path: Path,
    backend: str,
    model_path: Path | None = None,
) -> dict[str, Any]:
    normalized_backend = str(backend or "chordino").strip().lower()
    if normalized_backend == "chordino":
        chord_events = _extract_chordino_events(audio_path)
    elif normalized_backend == "omnizart":
        if model_path is None:
            raise RuntimeError("omnizart_model_path_required")
        chord_events = _extract_omnizart_events(audio_path, model_path)
    else:
        raise RuntimeError(f"unsupported_harmonic_backend: {backend}")
    chord_data = _normalize_chord_events(chord_events)
    chord_data["harmonic_backend"] = normalized_backend
    chord_data["harmonic_model_path"] = str(model_path) if model_path is not None else ""
    return chord_data
