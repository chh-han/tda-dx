#!/usr/bin/env python3
"""
Phase 0 of contrastive learning: extract frozen MERT embeddings and save to cache.

Strategy: 10-second non-overlapping windows → MERT last_hidden_state →
          mean-pool over time axis → average all window embeddings → single track embedding.

Tracks shorter than --window-secs are processed as a single window.

By default only tracks referenced in training_tuples.json are embedded.
Pass --tracks-csv (FMA tracks.csv) with --fma-split to also embed every track
in that split — required for full-dataset random-pair baselines.

Usage:
  python extract_mert.py \\
    --mining-dir /path/to/small-omni/mining \\
    --audio-dir  /path/to/fma_large \\
    --output     /path/to/small-omni/mert_cache.npz \\
    [--model      m-a-p/MERT-v1-95M] \\
    [--device     cuda] \
    [--window-secs 10] \
    [--batch-size  16] \
    [--tracks-csv /path/to/fma_metadata/tracks.csv] \
    [--fma-split  small]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ─── FMA audio path helpers ───────────────────────────────────────────────────

def _fma_audio_path(audio_dir: Path, track_id: str | int, ext: str = "mp3") -> Path:
    """Return the expected audio path for a FMA track_id (zero-padded 6-digit)."""
    tid = str(int(track_id)).zfill(6)
    return audio_dir / tid[:3] / f"{tid}.{ext.lstrip('.')}"


def _collect_track_ids(mining_dir: Path) -> dict[str, set[str]]:
    """
    Walk all training_tuples.json files and return track IDs grouped by ML split.

    Expects paths like: mining_dir/<split>/phase2/training_tuples.json
    Returns {split_name: {track_id, ...}}
    """
    ids: dict[str, set[str]] = {}
    for tuples_path in mining_dir.rglob("training_tuples.json"):
        # .../mining/<split>/phase2/training_tuples.json  → parts[-3] == split
        split_name = tuples_path.parts[-3]
        split_set = ids.setdefault(split_name, set())
        with tuples_path.open(encoding="utf-8") as f:
            tuples = json.load(f)
        for t in tuples:
            split_set.add(str(t["anchor_track_id"]))
            split_set.add(str(t["positive_track_id"]))
            split_set.add(str(t["hard_negative_track_id"]))
            split_set.add(str(t["easy_negative_track_id"]))
    return ids


def _collect_track_ids_from_csv(
    tracks_csv: Path,
    fma_subset: str,
) -> dict[str, set[str]]:
    """
    Read FMA tracks.csv, filter tracks whose ('set','subset') matches `fma_subset`
    (e.g. 'small'), then group them by their ('set','split') value
    ('training' | 'validation' | 'test').

    Returns {ml_split: {track_id, ...}}
    """
    result: dict[str, set[str]] = {}
    with tracks_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        row0 = next(reader)   # top-level group names
        row1 = next(reader)   # field names within group

        subset_col: int | None = None
        split_col:  int | None = None
        for col_idx, (g, field) in enumerate(zip(row0, row1)):
            if g.strip().lower() == "set":
                fname = field.strip().lower()
                if fname == "subset":
                    subset_col = col_idx
                elif fname == "split":
                    split_col = col_idx

        for row in reader:
            if not row:
                continue
            track_id = row[0].strip()
            if not track_id:
                continue

            subset_val = (
                row[subset_col].strip().lower()
                if subset_col is not None and subset_col < len(row) else ""
            )
            split_val = (
                row[split_col].strip().lower()
                if split_col is not None and split_col < len(row) else "unknown"
            )

            if subset_val == fma_subset.lower():
                result.setdefault(split_val, set()).add(track_id)

    total = sum(len(v) for v in result.values())
    log.info(
        "tracks.csv: found %d tracks in subset='%s'  splits=%s",
        total, fma_subset, {k: len(v) for k, v in sorted(result.items())},
    )
    return result


# ─── Audio loading & windowing ────────────────────────────────────────────────

def _load_audio(path: Path, target_sr: int, source_sr: int | None = None) -> np.ndarray | None:
    """Load full audio, resample to target_sr, return float32 mono numpy array.

    Supports:
      .npy  — pre-extracted float32 mono waveform; source_sr must be provided.
      other — any format supported by torchaudio; source_sr is read from file header.
    """
    if path.suffix == ".npy":
        if source_sr is None:
            raise ValueError(f"--audio-sr is required when loading .npy files (got {path})")
        try:
            wav = np.load(str(path)).astype(np.float32)  # (T,) mono
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return None
        if source_sr != target_sr:
            wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=source_sr, new_freq=target_sr)
            wav = wav_t.squeeze(0).numpy()
        return wav.astype(np.float32)
    else:
        try:
            wav_t, sr = torchaudio.load(str(path))
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return None
        if sr != target_sr:
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
        return wav_t.mean(dim=0).numpy().astype(np.float32)  # mono, (T,)


def _split_windows(wav: np.ndarray, window_samples: int) -> list[np.ndarray]:
    """
    Split a 1-D audio array into non-overlapping windows of `window_samples`.
    Tracks shorter than one window are returned as a single window.
    The last partial window (if any) is discarded to keep uniform window length.
    """
    if len(wav) <= window_samples:
        return [wav]
    n_windows = len(wav) // window_samples
    return [wav[i * window_samples : (i + 1) * window_samples] for i in range(n_windows)]


# ─── MERT extraction ──────────────────────────────────────────────────────────

def _load_mert(model_name: str, device: torch.device):
    """Load MERT processor and model (frozen)."""
    from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Loaded %s  SR=%d", model_name, processor.sampling_rate)
    return processor, model


@torch.no_grad()
def _embed_windows(
    processor,
    model,
    windows: list[np.ndarray],
    target_sr: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Run MERT on a list of same-length windows in batches.
    Each window: (window_samples,) float32.
    Returns shape (D,) — mean-pooled track embedding.
    """
    window_embs: list[np.ndarray] = []

    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        inputs = processor(
            batch,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=False)
        # last_hidden_state: (B, T, D) — mean-pool over time axis
        embs = outputs.last_hidden_state.mean(dim=1)  # (B, D)
        window_embs.append(embs.cpu().float().numpy())

    # Average all window embeddings → single track embedding
    all_embs = np.concatenate(window_embs, axis=0)  # (W, D)
    return all_embs.mean(axis=0)                     # (D,)


@torch.no_grad()
def _embed_tracks_batched(
    todo: list[str],
    audio_dir: Path,
    audio_ext: str,
    audio_sr: int | None,
    processor,
    model,
    target_sr: int,
    window_samples: int,
    device: torch.device,
    batch_size: int,
    cache: dict[str, np.ndarray],
    skipped: list[str],
) -> None:
    """
    Process tracks in a cross-track batched fashion:
    windows from *multiple* tracks are accumulated until `batch_size` is
    reached, then a single MERT forward pass is run for all of them.
    This keeps the GPU busy regardless of how few windows each track has.

    Results are written directly into `cache`.  Unloadable tracks go into
    `skipped`.
    """
    # Each entry: (tid, window_index, window_array)
    pending_windows: list[tuple[str, int, np.ndarray]] = []
    # Track → total window count (so we know when all windows are ready)
    track_n_windows: dict[str, int] = {}
    # Accumulated per-track embeddings: tid → list of window emb vectors
    track_window_embs: dict[str, list[np.ndarray]] = {}

    try:
        from tqdm import tqdm
        iter_ids = tqdm(todo, desc="Extracting", unit="track")
    except ImportError:
        iter_ids = iter(todo)

    def _flush(batch: list[tuple[str, int, np.ndarray]]) -> None:
        """Run one MERT forward pass on `batch` and accumulate results."""
        windows = [w for _, _, w in batch]
        inputs = processor(
            windows,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=False)
        embs = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()  # (B, D)
        for (tid, _, _), emb in zip(batch, embs):
            track_window_embs[tid].append(emb)

    def _finalise_ready_tracks() -> None:
        """Mean-pool and commit any track whose all windows have been embedded."""
        done = [
            tid for tid, embs in track_window_embs.items()
            if len(embs) == track_n_windows[tid]
        ]
        for tid in done:
            cache[tid] = np.stack(track_window_embs.pop(tid), axis=0).mean(axis=0)
            del track_n_windows[tid]

    for tid in iter_ids:
        audio_path = _fma_audio_path(audio_dir, tid, audio_ext)
        if not audio_path.exists():
            log.warning("Missing audio: %s", audio_path)
            skipped.append(tid)
            continue

        wav = _load_audio(audio_path, target_sr, source_sr=audio_sr)
        if wav is None:
            skipped.append(tid)
            continue

        windows = _split_windows(wav, window_samples)
        track_n_windows[tid] = len(windows)
        track_window_embs[tid] = []

        for win_idx, win in enumerate(windows):
            pending_windows.append((tid, win_idx, win))
            if len(pending_windows) >= batch_size:
                _flush(pending_windows)
                pending_windows.clear()
                _finalise_ready_tracks()

    # Flush remaining windows
    if pending_windows:
        _flush(pending_windows)
        pending_windows.clear()
        _finalise_ready_tracks()

    # Finalise any stragglers (should be empty, but just in case)
    for tid, embs in track_window_embs.items():
        if embs:
            cache[tid] = np.stack(embs, axis=0).mean(axis=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen MERT track embeddings via 10s window average pooling."
    )
    parser.add_argument("--mining-dir", type=Path, default=None,
                        help="Root of the mining splits, e.g. small-omni/mining/ "
                             "(mutually exclusive with --track-ids-file).")
    parser.add_argument("--track-ids-file", type=Path, default=None,
                        help="Text file with one track_id per line. Alternative to --mining-dir.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Root FMA audio directory (contains 000/, 001/, ...).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .npz cache file path.")
    parser.add_argument("--model", type=str, default="m-a-p/MERT-v1-95M",
                        help="HuggingFace model name for MERT.")
    parser.add_argument("--audio-ext", type=str, default="mp3",
                        help="Audio file extension under --audio-dir (e.g. mp3, npy).")
    parser.add_argument("--audio-sr", type=int, default=None,
                        help="Source sample rate of audio files. Required for .npy; "
                             "ignored for formats where torchaudio reads SR from header.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-secs", type=float, default=10.0,
                        help="Window length in seconds for chunked MERT inference.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of windows per MERT forward pass.")
    parser.add_argument("--tracks-csv", type=Path, default=None,
                        help="Path to FMA tracks.csv. When provided, all tracks in "
                             "--fma-split are also embedded (needed for full-dataset "
                             "random-pair baselines).")
    parser.add_argument("--fma-split", type=str, default="small",
                        help="FMA split name to select from tracks.csv "
                             "(e.g. small, medium, large, training, validation, test). "
                             "Only used when --tracks-csv is given.")
    parser.add_argument("--global-mean", type=Path, default=None,
                        help="Path to a .npy file containing the global mean embedding "
                             "(computed from the training set). When provided, each embedding "
                             "is mean-centred before L2-normalisation to remove MERT anisotropy. "
                             "For the training-set job, omit this flag and the script saves the "
                             "computed mean to <output>.mean.npy automatically.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # ── Collect track IDs ─────────────────────────────────────────────────────
    tuple_ids_by_split: dict[str, set[str]] = {}
    dataset_only_ids_by_split: dict[str, set[str]] = {}

    if args.track_ids_file is not None:
        log.info("Loading track IDs from %s ...", args.track_ids_file)
        track_ids: set[str] = {
            line.strip() for line in args.track_ids_file.read_text().splitlines() if line.strip()
        }
        log.info("Loaded %d track IDs from file", len(track_ids))
    elif args.mining_dir is not None:
        log.info("Collecting tuple track IDs from %s ...", args.mining_dir)
        tuple_ids_by_split = _collect_track_ids(args.mining_dir)
        all_tuple_ids: set[str] = set().union(*tuple_ids_by_split.values())
        log.info(
            "Tuple IDs: %d total  splits=%s",
            len(all_tuple_ids),
            {k: len(v) for k, v in sorted(tuple_ids_by_split.items())},
        )

        dataset_only_ids_by_split = {}
        if args.tracks_csv is not None:
            log.info(
                "Collecting full-dataset IDs from %s (subset='%s') ...",
                args.tracks_csv, args.fma_split,
            )
            csv_ids_by_split = _collect_track_ids_from_csv(args.tracks_csv, args.fma_split)
            for split_name, split_ids in csv_ids_by_split.items():
                only = split_ids - all_tuple_ids
                dataset_only_ids_by_split[split_name] = only
            all_dataset_only = set().union(*dataset_only_ids_by_split.values())
            log.info(
                "Dataset-only IDs: %d  splits=%s",
                len(all_dataset_only),
                {k: len(v) for k, v in sorted(dataset_only_ids_by_split.items())},
            )

        track_ids = all_tuple_ids | set().union(*dataset_only_ids_by_split.values())
    else:
        raise ValueError("Either --mining-dir or --track-ids-file must be provided.")

    log.info("Loading MERT model: %s on %s", args.model, device)
    processor, model = _load_mert(args.model, device)
    target_sr: int = processor.sampling_rate
    window_samples = int(args.window_secs * target_sr)
    log.info("Window: %.1fs = %d samples @ %d Hz", args.window_secs, window_samples, target_sr)

    # Resume support: load existing cache
    cache: dict[str, np.ndarray] = {}
    if args.output.exists():
        log.info("Loading existing cache from %s ...", args.output)
        loaded = np.load(args.output, allow_pickle=False)
        cache = {k: loaded[k] for k in loaded.files}
        log.info("Resumed with %d cached embeddings", len(cache))

    todo = sorted(track_ids - set(cache.keys()))
    log.info("%d tracks to process", len(todo))

    skipped: list[str] = []

    _embed_tracks_batched(
        todo=todo,
        audio_dir=args.audio_dir,
        audio_ext=args.audio_ext,
        audio_sr=args.audio_sr,
        processor=processor,
        model=model,
        target_sr=target_sr,
        window_samples=window_samples,
        device=device,
        batch_size=args.batch_size,
        cache=cache,
        skipped=skipped,
    )

    # ── Mean-centre + L2-normalise ─────────────────────────────────────────────
    # MERT last_hidden_state mean-pool is anisotropic: all embeddings lean toward
    # a shared 'DC' direction, inflating cosine similarities to ~0.8.
    # Subtracting the global training mean removes this bias.
    if args.global_mean is not None:
        global_mean = np.load(str(args.global_mean)).astype(np.float32)
        log.info("Loaded global mean from %s  (norm=%.4f)", args.global_mean, np.linalg.norm(global_mean))
    else:
        # Compute mean from this batch (only correct when processing the full training set)
        all_vecs = np.stack(list(cache.values())).astype(np.float32)
        global_mean = all_vecs.mean(axis=0)
        mean_path = args.output.with_suffix(".mean.npy")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(mean_path), global_mean)
        log.info("Saved global mean → %s  (norm=%.4f)", mean_path, np.linalg.norm(global_mean))

    for tid in cache:
        v = cache[tid].astype(np.float32) - global_mean
        norm = np.linalg.norm(v)
        cache[tid] = v / norm if norm > 0.0 else v

    log.info("Saving cache (%d embeddings) to %s ...", len(cache), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.output), **cache)

    # ── Save companion index ───────────────────────────────────────────────────
    # Records which track IDs came from tuples vs full-dataset only,
    # grouped by ML split (training / validation / test).
    # Used by blind_spot_analysis.py (--cache-index) to correctly scope
    # random-full baselines to the same split being analysed.
    index_path = args.output.with_suffix(".index.json")
    embedded = set(cache.keys())

    def _filter_embedded(ids: set[str]) -> list[str]:
        return sorted(ids & embedded)

    index = {
        "cache_path": str(args.output),
        "tuple_ids": {
            split: _filter_embedded(ids)
            for split, ids in sorted(tuple_ids_by_split.items())
        },
        "dataset_only_ids": {
            split: _filter_embedded(ids)
            for split, ids in sorted(dataset_only_ids_by_split.items())
        },
        "skipped_ids": sorted(skipped),
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    log.info("Index saved to %s", index_path)
    for split in sorted(set(tuple_ids_by_split) | set(dataset_only_ids_by_split)):
        n_t = len(index["tuple_ids"].get(split, []))
        n_d = len(index["dataset_only_ids"].get(split, []))
        log.info("  %-12s  tuple=%d  dataset_only=%d", split, n_t, n_d)
    log.info("  skipped=%d", len(skipped))

    log.info("Done. Skipped %d tracks: %s", len(skipped), skipped[:20])


if __name__ == "__main__":
    main()
