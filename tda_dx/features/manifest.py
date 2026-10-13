from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .constants import FEATURE_SCHEMA_VERSION, LOW_VALID_CHORD_RATIO_THRESHOLD
from .records import _status_summary, _summarize_numeric_series


def _manifest_payload(
    *,
    audio_root: Path,
    metadata_root: Path,
    output_dir: Path,
    sample_rate: int,
    subset_filter: str | None,
    split_filter: list[str] | None = None,
    limit: int | None,
    force: bool,
    records: list[dict[str, Any]],
    dependencies: dict[str, bool],
) -> dict[str, Any]:
    extractor_status_counts: dict[str, dict[str, int]] = {}
    for extractor_name in ("key", "beat", "chord"):
        extractor_status_counts[extractor_name] = _status_summary(
            [
                {"status": record.get("extractor_status", {}).get(extractor_name, "missing")}
                for record in records
            ],
            "status",
        )
    harmonic_qc_records = [
        record for record in records if record.get("extractor_status", {}).get("chord") == "ok"
    ]
    valid_chord_counts = [int(record.get("valid_chord_event_count", 0) or 0) for record in harmonic_qc_records]
    raw_chord_counts = [int(record.get("raw_chord_event_count", 0) or 0) for record in harmonic_qc_records]
    collapsed_lengths = [int(record.get("collapsed_chord_token_count", 0) or 0) for record in harmonic_qc_records]
    valid_chord_ratios = [float(record.get("valid_chord_ratio", 0.0) or 0.0) for record in harmonic_qc_records]
    invalid_reason_counts: Counter[str] = Counter()
    for record in harmonic_qc_records:
        reason_counts = record.get("invalid_chord_reason_counts", {})
        if not isinstance(reason_counts, dict):
            continue
        for reason, count in reason_counts.items():
            invalid_reason_counts[str(reason)] += int(count)
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "audio_root": str(audio_root),
        "metadata_root": str(metadata_root),
        "output_dir": str(output_dir),
        "sample_rate": sample_rate,
        "subset_filter": subset_filter or "all",
        "split_filter": split_filter or ["all"],
        "limit": limit,
        "force": force,
        "record_count": len(records),
        "record_status_counts": _status_summary(records, "record_status"),
        "extractor_status_counts": extractor_status_counts,
        "harmonic_qc": {
            "low_valid_chord_ratio_threshold": LOW_VALID_CHORD_RATIO_THRESHOLD,
            "eligible_record_count": len(harmonic_qc_records),
            "excluded_record_count": len(records) - len(harmonic_qc_records),
            "excluded_chord_extractor_status_counts": {
                status: count
                for status, count in extractor_status_counts["chord"].items()
                if status != "ok"
            },
            "rows_with_zero_valid_chord_tokens": sum(1 for value in valid_chord_counts if value == 0),
            "rows_with_low_valid_chord_ratio": sum(
                1 for value in valid_chord_ratios if value < LOW_VALID_CHORD_RATIO_THRESHOLD
            ),
            "rows_with_invalid_chord_events": sum(
                1 for record in harmonic_qc_records if int(record.get("invalid_chord_event_count", 0) or 0) > 0
            ),
            "raw_chord_event_count": _summarize_numeric_series(raw_chord_counts),
            "collapsed_chord_sequence_length": _summarize_numeric_series(collapsed_lengths),
            "valid_chord_ratio": _summarize_numeric_series(valid_chord_ratios),
            "invalid_chord_reason_counts": dict(sorted(invalid_reason_counts.items())),
        },
        "dependencies": dependencies,
    }
