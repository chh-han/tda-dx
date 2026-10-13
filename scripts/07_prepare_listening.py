#!/usr/bin/env python3
"""
Prepare attribution listening folder.

For each genre, randomly selects N test tracks, then for each track:
  - Saves the test audio (48kHz WAV from npy)
  - Decodes and saves the generated audio (EnCodec 24kHz tokens)
  - Saves top-10 training segments per attribution stage (semantic, coarse, fine)
    Each training entry saves only the exact segment window used during training,
    identified via train_pool_index → (sqlite_idx, start_sec) from the memmap pool.

Output structure:
  output_dir/
    {genre}_{track_id}/
      00_test_{track_id}.wav
      01_generated_{track_id}.wav
      semantic/
        rank01_score{score}_{train_track_id}_seg{N}s.wav
        ...
      coarse/
        ...
      fine/
        ...
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


# ─── Audio helpers ────────────────────────────────────────────────────────────

def get_audio_path(track_id: str, npy_base: str) -> str:
    padded = f"{int(track_id):06d}"
    return os.path.join(npy_base, padded[:3], f"{padded}.npy")


def load_npy_audio(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr.astype(np.float32)


def find_audio(track_id: str, npy_dirs: list) -> str | None:
    """Try multiple npy directories in order."""
    for d in npy_dirs:
        p = get_audio_path(track_id, d)
        if os.path.exists(p):
            return p
    return None


def save_wav(audio: np.ndarray, path: str, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


# ─── Token decoding ───────────────────────────────────────────────────────────

_encodec_model = None


def get_encodec_model():
    global _encodec_model
    if _encodec_model is None:
        from encodec import EncodecModel
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        _encodec_model = model
    return _encodec_model


def decode_tokens(npz_path: str) -> np.ndarray:
    """Decode coarse + fine tokens from npz → 24kHz waveform (numpy float32)."""
    model = get_encodec_model()
    d = np.load(npz_path)
    coarse = torch.from_numpy(d["coarse"].astype(np.int64))  # (T, 3)
    fine = torch.from_numpy(d["fine"].astype(np.int64))      # (T, 5)

    # Combine: (T, 8) → (1, T, 8)
    # Do NOT pad to model.quantizer.n_q (32); EnCodec decodes whatever K
    # quantizers you pass.  Padding with zeros would inject spurious codebook-0
    # embeddings from the unused quantizers and corrupt the output.
    combined = torch.cat([coarse, fine], dim=-1).unsqueeze(0)  # (1, T, 8)

    # EnCodec decode expects (B, n_q, T)
    codes = combined.permute(0, 2, 1)  # (1, 8, T)
    with torch.no_grad():
        wave = model.decode([(codes, None)])  # (1, 1, samples)
    return wave.squeeze().cpu().numpy().astype(np.float32)


# ─── Segment pool ─────────────────────────────────────────────────────────────

def build_segment_pool(mmap_dir: str, window_sec: int, stride_sec: int = None) -> list:
    """Build pool of (sqlite_idx, start_sec) indexed by exhaustive pool_index.

    Replicates MemmapPreprocessedDataset exhaustive traversal logic so we can
    map train_pool_index → exact audio segment without loading the full model.
    """
    if stride_sec is None:
        stride_sec = window_sec
    with open(os.path.join(mmap_dir, "idx_to_audio_seconds.json")) as f:
        secs = {int(k): int(v) for k, v in json.load(f).items()}
    with open(os.path.join(mmap_dir, "metadata.json")) as f:
        total = json.load(f)["total_count"]
    pool = []
    for sqlite_idx in range(total):
        length = secs.get(sqlite_idx, 30)
        max_start = max(0, length - window_sec)
        positions = list(range(0, max_start + 1, stride_sec)) or [0]
        for pos in positions:
            pool.append((sqlite_idx, pos))
    return pool


def load_idx_to_npy_path(mmap_dir: str) -> dict:
    """Returns {sqlite_idx: npy_path}."""
    with open(os.path.join(mmap_dir, "idx_to_path.json")) as f:
        return {int(k): v for k, v in json.load(f).items()}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tsv_genre_map(tsv_paths: list) -> dict:
    """Returns {track_id: genre}."""
    track_genre = {}
    for p in tsv_paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                track_genre[row["track_id"].strip()] = row["genre"].strip()
    return track_genre


def load_token_file_map(token_dir: str) -> dict:
    """Returns {track_id: npz_filename}."""
    m = {}
    for f in os.listdir(token_dir):
        if f.endswith("_full.npz"):
            track_id = f.split("_")[0]
            m[track_id] = f
    return m


def load_rankings(json_path: str) -> dict:
    """Returns {track_id: [topk entries]} from per_test_rankings.json."""
    with open(json_path) as f:
        raw = json.load(f)
    return {v["test_track_id"]: v["topk"] for v in raw.values()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True,
                   help="Root output directory for listening folders")
    p.add_argument("--samples-per-genre", type=int, default=5,
                   help="Number of test tracks to pick per genre (default: 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio-seconds", type=float, default=30.0,
                   help="Max seconds of audio to include (0 = all)")

    p.add_argument("--attr-dir", required=True,
                   help="attribution result root (e.g. .../trak-proj4096-reg001-best)")
    p.add_argument("--token-dir", required=True,
                   help="directory of generated tokens")
    p.add_argument("--tsv-files", nargs="+", required=True,
                   help="one or more selected_samples_shard_*.tsv files mapping test tracks to genres")

    p.add_argument("--npy-dirs", nargs="+", required=True,
                   help="one or more directories containing audio .npy files")
    p.add_argument("--mmap-dir", required=True,
                   help="Memmap preprocessed dataset folder (for segment pool lookup)")
    p.add_argument("--semantic-window-seconds", type=int, default=10)
    p.add_argument("--coarse-window-seconds", type=int, default=4)
    p.add_argument("--fine-window-seconds", type=int, default=2)
    p.add_argument("--audio-sr", type=int, default=48000,
                   help="Sample rate of source npy audio files")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading genre map from TSV files...")
    track_genre = load_tsv_genre_map(args.tsv_files)

    print("Loading attribution rankings...")
    sem_rankings = load_rankings(
        os.path.join(args.attr_dir, "semantic", "semantic_trak_per_test_rankings.json"))
    coarse_rankings = load_rankings(
        os.path.join(args.attr_dir, "coarse", "coarse_trak_per_test_rankings.json"))
    fine_rankings = load_rankings(
        os.path.join(args.attr_dir, "fine", "fine_trak_per_test_rankings.json"))

    print("Loading token file map...")
    token_map = load_token_file_map(args.token_dir)

    stage_windows = {
        "semantic": args.semantic_window_seconds,
        "coarse": args.coarse_window_seconds,
        "fine": args.fine_window_seconds,
    }
    print("Building segment pools from memmap...")
    seg_pools = {
        stage: build_segment_pool(args.mmap_dir, w)
        for stage, w in stage_windows.items()
    }
    for stage, pool in seg_pools.items():
        print(f"  {stage}: {len(pool):,} segments")
    idx_to_npy = load_idx_to_npy_path(args.mmap_dir)

    # ── Group test tracks by genre ──────────────────────────────────────────
    genre_buckets: dict[str, list] = {}
    skipped_no_genre = 0
    for track_id, genre in track_genre.items():
        if not genre:
            skipped_no_genre += 1
            continue
        genre_buckets.setdefault(genre, []).append(track_id)
    if skipped_no_genre:
        print(f"Skipped {skipped_no_genre} tracks with no genre label.")

    print(f"Genres found: {sorted(genre_buckets.keys())}")
    for g, bucket in sorted(genre_buckets.items()):
        print(f"  {g}: {len(bucket)} tracks")

    # ── Select N per genre ──────────────────────────────────────────────────
    selected: list[tuple[str, str]] = []  # (genre, track_id)
    for genre in sorted(genre_buckets.keys()):
        bucket = genre_buckets[genre]
        k = min(args.samples_per_genre, len(bucket))
        picked = rng.sample(bucket, k=k)
        for tid in picked:
            selected.append((genre, tid))

    print(f"\nSelected {len(selected)} test tracks total.")

    # ── Process each selected track ─────────────────────────────────────────
    max_samples_48k = int(args.audio_seconds * args.audio_sr) if args.audio_seconds > 0 else None
    max_samples_24k = int(args.audio_seconds * 24000) if args.audio_seconds > 0 else None

    for genre, track_id in selected:
        safe_genre = genre.replace("/", "-").replace(" ", "_")
        folder = output_dir / f"{safe_genre}_{track_id}"
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n── {genre} / track {track_id} → {folder.name}")

        # 1. Test audio
        audio_path = find_audio(track_id, args.npy_dirs)
        if audio_path:
            audio = load_npy_audio(audio_path)
            if max_samples_48k:
                audio = audio[:max_samples_48k]
            out_wav = str(folder / f"00_test_{track_id}.wav")
            save_wav(audio, out_wav, args.audio_sr)
            print(f"  [test audio] {out_wav} ({len(audio)/args.audio_sr:.1f}s)")
        else:
            print(f"  [test audio] MISSING for track {track_id}")

        # 2. Generated audio (decode from tokens)
        npz_file = token_map.get(track_id)
        if npz_file:
            npz_path = os.path.join(args.token_dir, npz_file)
            try:
                wave = decode_tokens(npz_path)
                if max_samples_24k:
                    wave = wave[:max_samples_24k]
                out_wav = str(folder / f"01_generated_{track_id}.wav")
                save_wav(wave, out_wav, 24000)
                print(f"  [generated] {out_wav} ({len(wave)/24000:.1f}s)")
            except Exception as e:
                print(f"  [generated] ERROR for {track_id}: {e}")
        else:
            print(f"  [generated] no token file for track {track_id}")

        # 3. Top-10 training tracks per stage
        stages = {
            "semantic": sem_rankings.get(track_id, []),
            "coarse": coarse_rankings.get(track_id, []),
            "fine": fine_rankings.get(track_id, []),
        }

        for stage_name, topk in stages.items():
            stage_dir = folder / stage_name
            stage_dir.mkdir(exist_ok=True)
            pool = seg_pools[stage_name]
            window_sec = stage_windows[stage_name]
            for entry in topk:
                rank = entry["rank"]
                score = entry["score"]
                train_id = entry["train_track_id"]
                pool_idx = entry["train_pool_index"]

                if pool_idx >= len(pool):
                    print(f"  [{stage_name}] pool_index {pool_idx} out of range (pool size {len(pool)})")
                    continue

                sqlite_idx, start_sec = pool[pool_idx]
                npy_path = idx_to_npy.get(sqlite_idx)
                if npy_path and os.path.exists(npy_path):
                    audio = load_npy_audio(npy_path)
                    start_sample = start_sec * args.audio_sr
                    end_sample = start_sample + window_sec * args.audio_sr
                    segment = audio[start_sample:end_sample]
                    score_str = f"{score:.4f}".replace(".", "p")
                    out_wav = str(stage_dir / f"rank{rank:02d}_score{score_str}_{train_id}_seg{start_sec}s.wav")
                    save_wav(segment, out_wav, args.audio_sr)
                    print(f"  [{stage_name}] rank{rank:02d} train {train_id} @{start_sec}s → {os.path.basename(out_wav)}")
                else:
                    print(f"  [{stage_name}] MISSING npy for sqlite_idx={sqlite_idx} (train {train_id})")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
