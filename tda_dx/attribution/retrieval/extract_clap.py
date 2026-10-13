#!/usr/bin/env python3
"""
Extract frozen CLAP audio-branch embeddings and save to cache.

Strategy: 10-second non-overlapping windows (480 000 samples @ 48 kHz)
          → CLAP audio encoder → average all window embeddings → single track embedding.

Only the audio branch of CLAP is used; the text branch is discarded.
Output embedding dimension: 512 (audio_projection output).

Usage:
  python extract_clap.py \\
    --mining-dir  /path/to/small-omni/mining \\
    --audio-dir   /path/to/fma_small_48k_npy_norm \\
    --output      /path/to/small-omni/clap_cache.npz \\
    --checkpoint  /path/to/music_speech_audioset_epoch_15_esc_89.98.pt \\
    [--audio-ext  npy] \\
    [--audio-sr   48000] \\
    [--window-secs 10] \\
    [--batch-size  64] \\
    [--device      cuda] \\
    [--tracks-csv  /path/to/fma_metadata/tracks.csv] \\
    [--fma-split   small]
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import gc
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

# CLAP expects 48 kHz mono
CLAP_SR = 48_000

# ─── FMA helpers (identical to extract_mert.py) ───────────────────────────────

def _fma_audio_path(audio_dir: Path, track_id: str | int, ext: str = "mp3") -> Path:
    tid = str(int(track_id)).zfill(6)
    return audio_dir / tid[:3] / f"{tid}.{ext.lstrip('.')}"


def _collect_track_ids(mining_dir: Path) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = {}
    for tuples_path in mining_dir.rglob("training_tuples.json"):
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
    result: dict[str, set[str]] = {}
    with tracks_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        row0 = next(reader)
        row1 = next(reader)

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


# ─── Audio loading ─────────────────────────────────────────────────────────────

def _load_audio(path: Path, source_sr: int | None = None) -> np.ndarray | None:
    """Load audio, resample to CLAP_SR (48 kHz), return float32 mono (T,)."""
    if path.suffix == ".npy":
        if source_sr is None:
            raise ValueError(f"--audio-sr required for .npy files (got {path})")
        try:
            wav = np.load(str(path)).astype(np.float32)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return None
        if source_sr != CLAP_SR:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=source_sr, new_freq=CLAP_SR)
            wav = wav_t.squeeze(0).numpy()
        return wav.astype(np.float32)
    else:
        try:
            wav_t, sr = torchaudio.load(str(path))
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return None
        if sr != CLAP_SR:
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=CLAP_SR)
        return wav_t.mean(dim=0).numpy().astype(np.float32)


def _split_windows(wav: np.ndarray, window_samples: int) -> list[np.ndarray]:
    if len(wav) <= window_samples:
        return [np.array(wav, dtype=np.float32)]
    n = len(wav) // window_samples
    return [np.array(wav[i * window_samples : (i + 1) * window_samples], dtype=np.float32) for i in range(n)]


# ─── CLAP loading ──────────────────────────────────────────────────────────────

def _load_clap(checkpoint: Path, device: torch.device):
    """Load CLAP audio encoder (HTSAT-base backbone) from local checkpoint."""
    import laion_clap  # type: ignore

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")

    # Issue 1: PyTorch ≥2.6 changed torch.load default to weights_only=True;
    #           the CLAP .pt file contains numpy scalars which require weights_only=False.
    # Issue 2: The checkpoint was saved with an older transformers that registered
    #           position_ids as a buffer; current version dropped it → strict mismatch.
    # Fix both by monkey-patching torch.load and load_state_dict during load_ckpt.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(  # type: ignore[assignment]
        *a, **{**kw, "weights_only": False}
    )
    _orig_lsd = model.model.load_state_dict
    model.model.load_state_dict = lambda sd, strict=True, **kw: _orig_lsd(sd, strict=False, **kw)  # type: ignore[method-assign]
    try:
        model.load_ckpt(str(checkpoint))
    finally:
        torch.load = _orig_torch_load
        model.model.load_state_dict = _orig_lsd

    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Loaded CLAP checkpoint from %s", checkpoint)
    return model


# ─── Cross-track batched extraction ───────────────────────────────────────────

def _audio_loader(
    todo: list[str],
    audio_dir: Path,
    audio_ext: str,
    audio_sr: int | None,
    window_samples: int,
    q,          # queue.Queue
    sentinel,
    prefetch: int,
) -> None:
    """Producer thread: loads audio files and puts (tid, windows | None) into q."""
    for tid in todo:
        audio_path = _fma_audio_path(audio_dir, tid, audio_ext)
        if not audio_path.exists():
            q.put((tid, None))
            continue
        wav = _load_audio(audio_path, source_sr=audio_sr)
        if wav is None:
            q.put((tid, None))
            continue
        q.put((tid, _split_windows(wav, window_samples)))
    q.put(sentinel)


@torch.no_grad()
def _embed_tracks_batched(
    todo: list[str],
    audio_dir: Path,
    audio_ext: str,
    audio_sr: int | None,
    clap_model,
    window_samples: int,
    device: torch.device,
    batch_size: int,
    cache: dict[str, np.ndarray],
    skipped: list[str],
    prefetch: int = 4,
) -> None:
    """
    Accumulate windows from multiple tracks, forward batch_size windows at a time
    through the CLAP audio branch, then mean-pool per track.

    Audio loading runs in a background thread (prefetch) so GPU stays busy.
    """
    import queue
    import threading

    sentinel = object()
    q: queue.Queue = queue.Queue(maxsize=prefetch)

    loader = threading.Thread(
        target=_audio_loader,
        args=(todo, audio_dir, audio_ext, audio_sr, window_samples, q, sentinel, prefetch),
        daemon=True,
    )
    loader.start()

    pending: list[tuple[str, int, np.ndarray]] = []
    track_n_windows: dict[str, int] = {}
    track_window_embs: dict[str, list[np.ndarray]] = {}

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(todo), desc="Extracting", unit="track")
    except ImportError:
        pbar = None

    def _flush(batch: list[tuple[str, int, np.ndarray]]) -> None:
        wavs = [w for _, _, w in batch]
        max_len = max(len(w) for w in wavs)
        padded = np.zeros((len(wavs), max_len), dtype=np.float32)
        for i, w in enumerate(wavs):
            padded[i, : len(w)] = w
        embs = clap_model.get_audio_embedding_from_data(
            [padded[i] for i in range(len(padded))],
            use_tensor=False,
        )  # (B, 512) numpy
        for (tid, _, _), emb in zip(batch, embs):
            track_window_embs[tid].append(emb)

    def _finalise_ready() -> None:
        done = [tid for tid, embs in track_window_embs.items()
                if len(embs) == track_n_windows[tid]]
        for tid in done:
            cache[tid] = np.stack(track_window_embs.pop(tid), axis=0).mean(axis=0)
            del track_n_windows[tid]

    while True:
        item = q.get()
        if item is sentinel:
            break
        tid, windows = item
        if pbar is not None:
            pbar.update(1)
        if windows is None:
            log.warning("Missing or unreadable audio: %s", tid)
            skipped.append(tid)
            continue

        track_n_windows[tid] = len(windows)
        track_window_embs[tid] = []

        for win_idx, win in enumerate(windows):
            pending.append((tid, win_idx, win))
            if len(pending) >= batch_size:
                _flush(pending)
                pending.clear()
                _finalise_ready()

    if pending:
        _flush(pending)
        pending.clear()
        _finalise_ready()

    for tid, embs in track_window_embs.items():
        if embs:
            cache[tid] = np.stack(embs, axis=0).mean(axis=0)

    if pbar is not None:
        pbar.close()
    loader.join()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frozen CLAP audio-branch embeddings."
    )
    p.add_argument("--mining-dir", type=Path, default=None,
                   help="Mining dir with training_tuples.json files (mutually exclusive with --track-ids-file).")
    p.add_argument("--track-ids-file", type=Path, default=None,
                   help="Text file with one track_id per line. Alternative to --mining-dir.")
    p.add_argument("--audio-dir",  type=Path, required=True)
    p.add_argument("--output",     type=Path, required=True,
                   help="Output .npz cache path.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="CLAP checkpoint .pt file.")
    p.add_argument("--audio-ext",  type=str, default="mp3")
    p.add_argument("--audio-sr",   type=int, default=None,
                   help="Source sample rate (required for .npy).")
    p.add_argument("--window-secs", type=float, default=10.0)
    p.add_argument("--batch-size",  type=int, default=64,
                   help="Windows per CLAP forward pass.")
    p.add_argument("--save-every",  type=int, default=2000,
                   help="Save intermediate cache every N tracks (0 = disabled).")
    p.add_argument("--prefetch",    type=int, default=4,
                   help="Number of tracks to prefetch in the audio loader thread.")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tracks-csv", type=Path, default=None)
    p.add_argument("--fma-split",  type=str, default="small")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

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

        if args.tracks_csv is not None:
            log.info(
                "Collecting full-dataset IDs from %s (subset='%s') ...",
                args.tracks_csv, args.fma_split,
            )
            csv_ids_by_split = _collect_track_ids_from_csv(args.tracks_csv, args.fma_split)
            for split_name, split_ids in csv_ids_by_split.items():
                dataset_only_ids_by_split[split_name] = split_ids - all_tuple_ids
            all_dataset_only = set().union(*dataset_only_ids_by_split.values())
            log.info(
                "Dataset-only IDs: %d  splits=%s",
                len(all_dataset_only),
                {k: len(v) for k, v in sorted(dataset_only_ids_by_split.items())},
            )

        track_ids = all_tuple_ids | set().union(*dataset_only_ids_by_split.values())
    else:
        raise ValueError("Either --mining-dir or --track-ids-file must be provided.")

    # ── Load CLAP ─────────────────────────────────────────────────────────────
    log.info("Loading CLAP model on %s ...", device)
    clap_model = _load_clap(args.checkpoint, device)
    window_samples = int(args.window_secs * CLAP_SR)
    log.info("Window: %.1fs = %d samples @ %d Hz", args.window_secs, window_samples, CLAP_SR)

    # ── Resume ────────────────────────────────────────────────────────────────
    cache: dict[str, np.ndarray] = {}
    if args.output.exists():
        log.info("Loading existing cache from %s ...", args.output)
        loaded = np.load(args.output, allow_pickle=False)
        cache = {k: loaded[k] for k in loaded.files}
        log.info("Resumed with %d cached embeddings", len(cache))

    todo = sorted(track_ids - set(cache.keys()))
    log.info("%d tracks to process", len(todo))

    skipped: list[str] = []

    save_every = args.save_every if args.save_every > 0 else len(todo) + 1
    shard_paths: list[Path] = []

    for chunk_idx, chunk_start in enumerate(range(0, max(len(todo), 1), save_every)):
        chunk = todo[chunk_start : chunk_start + save_every]
        if not chunk:
            break

        chunk_cache: dict[str, np.ndarray] = {}
        _embed_tracks_batched(
            todo=chunk,
            audio_dir=args.audio_dir,
            audio_ext=args.audio_ext,
            audio_sr=args.audio_sr,
            clap_model=clap_model,
            window_samples=window_samples,
            device=device,
            batch_size=args.batch_size,
            cache=chunk_cache,
            skipped=skipped,
            prefetch=args.prefetch,
        )
        # Save shard and immediately free the chunk embeddings from RAM
        shard_path = args.output.with_suffix(f".shard{chunk_idx}.npz")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(shard_path), **chunk_cache)
        shard_paths.append(shard_path)
        log.info(
            "Shard %d saved (%d embeddings) → %s  [RAM freed]",
            chunk_idx, len(chunk_cache), shard_path,
        )
        del chunk_cache  # free embeddings from RAM
        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ── Merge shards + pre-existing cache into final npz ─────────────────────
    log.info("Merging %d shard(s) + %d pre-existing embeddings ...", len(shard_paths), len(cache))
    merged: dict[str, np.ndarray] = dict(cache)   # pre-existing (resume)
    for sp in shard_paths:
        shard_data = np.load(str(sp), allow_pickle=False)
        merged.update({k: shard_data[k] for k in shard_data.files})
    cache.update(merged)  # keep cache in sync for the index-writing step below

    # ── Save cache ────────────────────────────────────────────────────────────
    log.info("Saving cache (%d embeddings) to %s ...", len(cache), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.output), **cache)

    # ── Save companion index ──────────────────────────────────────────────────
    index_path = args.output.with_suffix(".index.json")
    embedded = set(cache.keys())

    def _fe(ids: set[str]) -> list[str]:
        return sorted(ids & embedded)

    index = {
        "cache_path": str(args.output),
        "encoder": "clap-audio",
        "checkpoint": str(args.checkpoint),
        "embedding_dim": int(next(iter(cache.values())).shape[0]) if cache else 512,
        "tuple_ids": {
            split: _fe(ids)
            for split, ids in sorted(tuple_ids_by_split.items())
        },
        "dataset_only_ids": {
            split: _fe(ids)
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
    log.info("Done.")


if __name__ == "__main__":
    main()
