"""jSymbolic MIDI features (strict jSymbolic 2.2 subset, McKay 2010 App. B).

Train side reads beat duration from the original MAESTRO MIDI tempo;
test side (generated_music) falls back to 120 BPM. See _record() for
the 12 features computed per segment."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pretty_midi

from dattri.benchmark.datasets.maestro import create_maestro_datasets
from dattri.benchmark.models.MusicTransformer.midi_processor.processor import (
    decode_midi,
    RANGE_TIME_SHIFT,
    START_IDX,
)
from dattri.benchmark.models.MusicTransformer.utilities.constants import TOKEN_PAD


N_PC_BINS  = 12
N_VI_BINS  = 12
MEL_INT_RANGE = 12
N_MEL_BINS = 2 * MEL_INT_RANGE + 1

# R-23 rhythmic value bin centres in beats: 32nd → double whole.
R23_CENTRES_BEATS = np.array(
    [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0], dtype=np.float64,
)
N_R23_BINS = len(R23_CENTRES_BEATS)

POLY_GRID_HZ = 50.0
DEFAULT_BEAT_DUR_SEC = 0.5  # 120 BPM fallback

_TIME_SHIFT_LO = START_IDX["time_shift"]
_TIME_SHIFT_HI = _TIME_SHIFT_LO + RANGE_TIME_SHIFT


def _segment_start_time(raw: np.ndarray, offset: int) -> float:
    if offset <= 0:
        return 0.0
    head = raw[:offset]
    mask = (head >= _TIME_SHIFT_LO) & (head < _TIME_SHIFT_HI)
    vals = head[mask].astype(np.int64) - _TIME_SHIFT_LO
    return float(np.sum(vals + 1) / 100.0)


def _decode_segment(raw: np.ndarray, offset: int, max_seq: int):
    seg = raw[offset : offset + max_seq + 1]
    seg = seg[(seg >= 0) & (seg < TOKEN_PAD)]
    if seg.size == 0:
        return [], 0.0
    pm = decode_midi(seg.tolist())
    notes = pm.instruments[0].notes if len(pm.instruments) > 0 else []
    if not notes:
        return [], 0.0
    seg_end = max(n.end for n in notes)
    return notes, float(seg_end)


def _slice_beat_dur(full_pm: pretty_midi.PrettyMIDI, t_start: float, t_end: float) -> Optional[float]:
    beats = full_pm.get_beats()
    beats = beats[(beats >= t_start) & (beats <= t_end)]
    if len(beats) < 2:
        return None
    diffs = np.diff(beats)
    diffs = diffs[diffs > 1e-4]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


# ---- harmonic features ----
def _pitch_class_hist(notes) -> List[float]:
    """jSymbolic P-2: count-based, normalised to sum=1."""
    if not notes:
        return [0.0] * N_PC_BINS
    pc = np.array([n.pitch % 12 for n in notes], dtype=np.int64)
    counts = np.bincount(pc, minlength=N_PC_BINS).astype(np.float64)
    s = counts.sum()
    return (counts / s).tolist() if s > 0 else counts.tolist()


def _vertical_interval_hist(notes, hz: float = POLY_GRID_HZ, seg_dur: float = 0.0) -> List[float]:
    """jSymbolic C-2 (wrapped). At each grid frame, enumerate all unordered
    pairs of simultaneously sounding pitches and bump the interval-class
    bin (|p1 - p2| mod 12). The histogram is summed over frames, then
    L1-normalised."""
    if not notes or seg_dur <= 0:
        return [0.0] * N_VI_BINS
    starts = np.array([n.start for n in notes], dtype=np.float64)
    ends   = np.array([n.end   for n in notes], dtype=np.float64)
    pitches = np.array([n.pitch for n in notes], dtype=np.int64)
    n_steps = max(1, int(np.ceil(seg_dur * hz)))
    counts = np.zeros(N_VI_BINS, dtype=np.float64)
    for k in range(n_steps):
        t = (k + 0.5) / hz
        active = pitches[(starts <= t) & (ends > t)]
        if active.size < 2:
            continue
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ic = int(abs(int(active[i]) - int(active[j])) % 12)
                counts[ic] += 1.0
    s = counts.sum()
    return (counts / s).tolist() if s > 0 else counts.tolist()


def _melodic_interval_hist(notes) -> List[float]:
    """jSymbolic M-1 (truncated to ±12). Intervals between consecutive notes
    sorted by onset; for polyphony we follow jSymbolic's flattening to a
    single onset-ordered stream (clipped to the truncation range)."""
    if len(notes) < 2:
        return [0.0] * N_MEL_BINS
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in sorted_notes], dtype=np.int64)
    iv = np.diff(pitches)
    iv = iv[(iv >= -MEL_INT_RANGE) & (iv <= MEL_INT_RANGE)]
    if iv.size == 0:
        return [0.0] * N_MEL_BINS
    counts = np.bincount(iv + MEL_INT_RANGE, minlength=N_MEL_BINS).astype(np.float64)
    s = counts.sum()
    return (counts / s).tolist() if s > 0 else counts.tolist()


# ---- rhythmic features ----
def _note_density(notes, seg_dur: float) -> float:
    """R-3: notes per second."""
    if not notes or seg_dur <= 1e-6:
        return 0.0
    return float(len(notes) / seg_dur)


def _mean_rhythmic_value(notes, beat_dur_sec: float) -> float:
    """R-15: average note duration in beats."""
    if not notes or beat_dur_sec <= 1e-6:
        return 0.0
    durs = np.array([max(n.end - n.start, 1e-6) for n in notes], dtype=np.float64)
    return float(np.mean(durs) / beat_dur_sec)


def _rhythmic_value_hist(notes, beat_dur_sec: float) -> List[float]:
    """R-23: histogram of note durations bucketed into the 11 canonical
    rhythmic values. Each note is assigned to the nearest bin centre in
    log space; bins are L1-normalised."""
    if not notes or beat_dur_sec <= 1e-6:
        return [0.0] * N_R23_BINS
    durs = np.array([max(n.end - n.start, 1e-6) for n in notes], dtype=np.float64) / beat_dur_sec
    log_durs = np.log(np.clip(durs, 1e-6, 64.0))
    log_centres = np.log(R23_CENTRES_BEATS)
    diffs = np.abs(log_durs[:, None] - log_centres[None, :])
    bin_idx = np.argmin(diffs, axis=1)
    counts = np.bincount(bin_idx, minlength=N_R23_BINS).astype(np.float64)
    s = counts.sum()
    return (counts / s).tolist() if s > 0 else counts.tolist()


# ---- dynamic features ----
def _avg_dvelocity(notes) -> float:
    """D-1: avg |Δvelocity| between notes sorted by onset."""
    if len(notes) < 2:
        return 0.0
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    vels = np.array([n.velocity for n in sorted_notes], dtype=np.float64)
    return float(np.mean(np.abs(np.diff(vels))))


def _var_dynamics(notes) -> float:
    """D-2: std of velocity."""
    if len(notes) < 2:
        return 0.0
    vels = np.array([n.velocity for n in notes], dtype=np.float64)
    return float(np.std(vels, ddof=1))


def _dynamic_range(notes) -> float:
    """D-3: velocity max − min."""
    if not notes:
        return 0.0
    vels = np.array([n.velocity for n in notes], dtype=np.float64)
    return float(vels.max() - vels.min())


# ---- texture features ----
def _polyphony_stats(notes, seg_dur: float, hz: float = POLY_GRID_HZ) -> tuple:
    """Returns (mean_polyphony, var_polyphony) over a fine time grid.
    Used for both T-2 (mean) and T-7 (std)."""
    if not notes or seg_dur <= 0:
        return (0.0, 0.0)
    starts = np.array([n.start for n in notes], dtype=np.float64)
    ends   = np.array([n.end   for n in notes], dtype=np.float64)
    n_steps = max(1, int(np.ceil(seg_dur * hz)))
    active = np.zeros(n_steps, dtype=np.int32)
    for s, e in zip(starts, ends):
        i0 = max(0, int(np.floor(s * hz)))
        i1 = min(n_steps, int(np.ceil(e * hz)))
        if i1 > i0:
            active[i0:i1] += 1
    return (float(active.mean()), float(active.std(ddof=1)) if n_steps > 1 else 0.0)


def _pitch_range(notes) -> float:
    """P-6: max − min pitch in semitones."""
    if not notes:
        return 0.0
    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    return float(pitches.max() - pitches.min())


# ---- master record ----
def _record(notes, seg_dur: float, beat_dur_sec: float) -> Dict[str, Any]:
    mean_poly, var_poly = _polyphony_stats(notes, seg_dur)
    return {
        # harmonic
        "pitch_class_hist":   _pitch_class_hist(notes),
        "vert_int_hist":      _vertical_interval_hist(notes, seg_dur=seg_dur),
        "mel_int_hist":       _melodic_interval_hist(notes),
        # rhythmic
        "note_density":       _note_density(notes, seg_dur),
        "mean_rhythmic_value": _mean_rhythmic_value(notes, beat_dur_sec),
        "rhythmic_value_hist": _rhythmic_value_hist(notes, beat_dur_sec),
        # dynamic
        "avg_dvelocity":      _avg_dvelocity(notes),
        "var_dynamics":       _var_dynamics(notes),
        "dynamic_range":      _dynamic_range(notes),
        # texture
        "mean_polyphony":     mean_poly,
        "var_polyphony":      var_poly,
        "pitch_range":        _pitch_range(notes),
        # provenance
        "n_notes":            int(len(notes)),
        "seg_dur_sec":        float(seg_dur),
        "beat_dur_sec":       float(beat_dur_sec),
        "valid":              bool(len(notes) > 0),
    }


def _build_original_midi_index(maestro_root: Path) -> Dict[str, Path]:
    manifest = maestro_root / "maestro-v2.0.0.json"
    if not manifest.exists():
        return {}
    with manifest.open() as f:
        meta = json.load(f)
    out = {}
    for piece in meta:
        rel = piece.get("midi_filename")
        if rel:
            out[Path(rel).name] = maestro_root / rel
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--maestro_root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    train_ds, test_ds = create_maestro_datasets(args.data_path, generated_music=True)
    ds = train_ds if args.split == "train" else test_ds
    assert ds.full_version

    midi_index: Dict[str, Path] = {}
    if args.split == "train":
        if args.maestro_root is None:
            args.maestro_root = str(Path(args.data_path) / "maestro-v2.0.0")
        midi_index = _build_original_midi_index(Path(args.maestro_root))
        print(f"original MIDI index size: {len(midi_index)}")

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"split={args.split}  n={n}  max_seq={ds.max_seq}")

    pm_cache: Dict[str, Optional[pretty_midi.PrettyMIDI]] = {}
    records: List[Dict[str, Any]] = []
    n_real_beats = 0

    for idx in range(n):
        pickle_file, offset = ds.length_dict[idx]
        with open(pickle_file, "rb") as f:
            raw = pickle.load(f)
        raw = np.asarray(raw, dtype=np.int64)
        notes, seg_dur = _decode_segment(raw, offset, ds.max_seq)

        beat_dur_sec = DEFAULT_BEAT_DUR_SEC
        if args.split == "train" and notes:
            stem = Path(pickle_file).name.replace(".pickle", "")
            full_path = midi_index.get(stem)
            if full_path is not None and full_path.exists():
                if str(full_path) not in pm_cache:
                    if len(pm_cache) > 64:
                        pm_cache.clear()
                    try:
                        pm_cache[str(full_path)] = pretty_midi.PrettyMIDI(str(full_path))
                    except Exception:
                        pm_cache[str(full_path)] = None
                full_pm = pm_cache[str(full_path)]
                if full_pm is not None:
                    t_start = _segment_start_time(raw, offset)
                    bd = _slice_beat_dur(full_pm, t_start, t_start + seg_dur)
                    if bd is not None:
                        beat_dur_sec = bd
                        n_real_beats += 1

        records.append(_record(notes, seg_dur, beat_dur_sec))

        if (idx + 1) % 200 == 0 or idx + 1 == n:
            valid = sum(1 for r in records if r["valid"])
            print(f"  [{idx + 1}/{n}]  valid={valid / (idx + 1):.3f}  "
                  f"real_beats={n_real_beats / (idx + 1):.3f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {out}  n_records={len(records)}")


if __name__ == "__main__":
    main()
