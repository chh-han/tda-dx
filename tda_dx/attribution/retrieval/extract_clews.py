#!/usr/bin/env python3
"""
Extract CLEWS audio embeddings and save to cache (.npz).

Loads a pre-trained CLEWS checkpoint (sony/clews), processes audio via the
model's internal CQT + shingling, and mean-pools shingle embeddings into a
single track-level embedding vector.

Requirements (add to your venv if missing):
  pip install nnAudio==0.3.3 omegaconf==2.3.0 einops

Checkpoint directory must contain 'configuration.yaml' alongside the .ckpt
file — this is already the layout inside the Zenodo clews.zip:
  shs-clews/checkpoint_best.ckpt
  shs-clews/configuration.yaml
  dvi-clews/checkpoint_best.ckpt
  dvi-clews/configuration.yaml

The CLEWS source repo (github.com/sony/clews) must be cloned locally;
pass its root with --clews-repo.

Usage:
  python extract_clews.py \\
    --track-ids-file  /path/to/train_track_ids.txt \\
    --audio-dir       /path/to/fma_large_48k_npy_norm \\
    --output          /path/to/clews_embeddings/fma_train_clews_2048.npz \\
    --checkpoint      /path/to/checkpoints/shs-clews/checkpoint_best.ckpt \\
    --clews-repo      /path/to/third_party/clews \\
    [--audio-ext      npy] \\
    [--audio-sr       48000] \\
    [--hop-secs       5.0] \\
    [--save-every     2000] \\
    [--device         cuda]
"""
from __future__ import annotations

import argparse
import importlib
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


# ─── FMA helpers ──────────────────────────────────────────────────────────────

def _fma_audio_path(audio_dir: Path, track_id: str | int, ext: str = "npy") -> Path:
    tid = str(int(track_id)).zfill(6)
    return audio_dir / tid[:3] / f"{tid}.{ext.lstrip('.')}"


# ─── Audio loading ────────────────────────────────────────────────────────────

def _load_audio_to_tensor(
    path: Path,
    target_sr: int,
    source_sr: int | None,
) -> torch.Tensor | None:
    """Load audio → float32 mono tensor (1, T) resampled to target_sr.

    Supports .npy (pre-extracted mono waveform; source_sr required) and any
    format that torchaudio can read.
    """
    try:
        if path.suffix == ".npy":
            if source_sr is None:
                raise ValueError("--audio-sr is required when loading .npy files")
            wav = np.load(str(path)).astype(np.float32)  # (T,)
            x = torch.from_numpy(wav).unsqueeze(0)        # (1, T)
            if source_sr != target_sr:
                x = torchaudio.functional.resample(
                    x, orig_freq=source_sr, new_freq=target_sr
                )
        else:
            x, sr = torchaudio.load(str(path))            # (C, T)
            x = x.mean(0, keepdim=True)                    # mono (1, T)
            if sr != target_sr:
                x = torchaudio.functional.resample(
                    x, orig_freq=sr, new_freq=target_sr
                )
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return None
    return x  # (1, T)


# ─── CLEWS model loading ──────────────────────────────────────────────────────

def _load_clews(
    checkpoint: Path,
    clews_repo: Path,
    device: torch.device,
):
    """Instantiate CLEWS model and load weights from a Fabric checkpoint.

    The CLEWS repo is added to sys.path so that 'models.clews', 'lib', and
    'utils' can be imported. Weights are loaded via torch.load without
    requiring Fabric at runtime (avoids the lightning install dependency).
    """
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        raise ImportError(
            "omegaconf not found. Install with: pip install omegaconf==2.3.0"
        )

    # Add CLEWS repo to sys.path so its local modules are importable
    repo_str = str(clews_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    log.info("CLEWS repo path: %s", repo_str)

    # configuration.yaml must live next to the .ckpt file
    conf_path = checkpoint.parent / "configuration.yaml"
    if not conf_path.exists():
        raise FileNotFoundError(
            f"configuration.yaml not found in {checkpoint.parent}.\n"
            "Extract clews.zip so that configuration.yaml sits next to checkpoint_best.ckpt.\n"
            "Expected structure:\n"
            "  checkpoints/shs-clews/checkpoint_best.ckpt\n"
            "  checkpoints/shs-clews/configuration.yaml"
        )
    conf = OmegaConf.load(str(conf_path))
    log.info(
        "CLEWS config: model=%s  sr=%d  zdim=%d  shingle_len=%.0fs  shingle_hop=%.0fs",
        conf.model.name, conf.data.samplerate, conf.model.zdim,
        conf.model.shingling.len, conf.model.shingling.hop,
    )

    # Instantiate the model (pure PyTorch — no Fabric needed here)
    module = importlib.import_module("models." + conf.model.name)
    model = module.Model(conf.model, sr=conf.data.samplerate)
    model.eval()

    # Load weights from checkpoint (saved by lightning.Fabric)
    log.info("Loading checkpoint weights from %s ...", checkpoint)
    ckpt_data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    raw_sd = ckpt_data["model"]

    # Fabric/DDP training may prefix parameter names — strip defensively
    for prefix in ("_forward_module.", "module."):
        if any(k.startswith(prefix) for k in raw_sd):
            log.info("Stripping '%s' prefix from checkpoint state_dict keys", prefix)
            raw_sd = {
                k[len(prefix):]: v
                for k, v in raw_sd.items()
                if k.startswith(prefix)
            }
            break

    missing, unexpected = model.load_state_dict(raw_sd, strict=False)
    if missing:
        log.warning("Missing keys  (%d): %s ...", len(missing), missing[:5])
    if unexpected:
        log.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])
    if not missing and not unexpected:
        log.info("Checkpoint loaded cleanly (no missing/unexpected keys)")

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %.1fM parameters", n_params / 1e6)

    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return model, conf


# ─── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def _embed_tracks(
    todo: list[str],
    audio_dir: Path,
    audio_ext: str,
    audio_sr: int | None,
    model,
    target_sr: int,
    hop_secs: float,
    device: torch.device,
    cache: dict[str, np.ndarray],
    skipped: list[str],
) -> None:
    """Process tracks one at a time; mean-pool shingle embeddings per track.

    CLEWS forward() returns (B, S, C) where S varies per track length.
    Mean-pooling over S gives a fixed-size (C,) track embedding.
    """
    try:
        from tqdm import tqdm
        iter_ids = tqdm(todo, desc="CLEWS embed", unit="track")
    except ImportError:
        iter_ids = iter(todo)

    for tid in iter_ids:
        audio_path = _fma_audio_path(audio_dir, tid, audio_ext)
        if not audio_path.exists():
            log.warning("Missing audio: %s", audio_path)
            skipped.append(tid)
            continue

        x = _load_audio_to_tensor(audio_path, target_sr=target_sr, source_sr=audio_sr)
        if x is None:
            skipped.append(tid)
            continue

        # x: (1, T) — CLEWS model expects (B, T)
        x = x.to(device)

        # z: (1, S, C) — S shingles of length shingling_len with hop_secs overlap
        z = model(x, shingle_hop=hop_secs)  # (1, S, C)

        # Mean-pool over shingle dimension → (1, C) → (C,)
        emb: np.ndarray = z.mean(dim=1).squeeze(0).cpu().float().numpy()
        cache[tid] = emb


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frozen CLEWS track embeddings."
    )
    p.add_argument(
        "--mining-dir", type=Path, default=None,
        help="Mining dir with training_tuples.json (mutually exclusive with --track-ids-file).",
    )
    p.add_argument(
        "--track-ids-file", type=Path, default=None,
        help="Text file with one track_id per line. Alternative to --mining-dir.",
    )
    p.add_argument("--audio-dir",  type=Path, required=True,
                   help="Root FMA audio directory (contains 000/, 001/, ...).")
    p.add_argument("--output",     type=Path, required=True,
                   help="Output .npz cache file path.")
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to checkpoint_best.ckpt. "
             "configuration.yaml must be in the same directory.",
    )
    p.add_argument(
        "--clews-repo", type=Path, required=True,
        help="Path to the cloned sony/clews repository root.",
    )
    p.add_argument("--audio-ext",  type=str, default="npy",
                   help="Audio file extension (npy, mp3, wav, flac, ...).")
    p.add_argument(
        "--audio-sr", type=int, default=None,
        help="Source sample rate of audio files. Required for .npy; "
             "ignored for formats with a header (mp3, wav, ...).",
    )
    p.add_argument(
        "--hop-secs", type=float, default=5.0,
        help="Shingle hop in seconds (default: 5.0 — same as CLEWS inference.py). "
             "Smaller values produce more shingles and a smoother mean-pool.",
    )
    p.add_argument(
        "--save-every", type=int, default=2000,
        help="Intermediate save every N tracks (0 = disabled).",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # ── Collect track IDs ─────────────────────────────────────────────────────
    if args.track_ids_file is not None:
        log.info("Loading track IDs from %s ...", args.track_ids_file)
        track_ids: set[str] = {
            line.strip()
            for line in args.track_ids_file.read_text().splitlines()
            if line.strip()
        }
        log.info("Loaded %d track IDs", len(track_ids))
    elif args.mining_dir is not None:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        from extract_mert import _collect_track_ids  # type: ignore
        all_split_ids = _collect_track_ids(args.mining_dir)
        track_ids = set().union(*all_split_ids.values())
        log.info("Collected %d track IDs from mining dir", len(track_ids))
    else:
        raise ValueError("Either --mining-dir or --track-ids-file must be provided.")

    # ── Load CLEWS model ──────────────────────────────────────────────────────
    model, conf = _load_clews(args.checkpoint, args.clews_repo, device)
    target_sr: int = conf.data.samplerate  # 16000 Hz for CLEWS
    zdim: int = conf.model.zdim            # 2048 for shs-clews / dvi-clews

    # ── Resume ────────────────────────────────────────────────────────────────
    cache: dict[str, np.ndarray] = {}
    if args.output.exists():
        log.info("Loading existing cache from %s ...", args.output)
        loaded = np.load(args.output, allow_pickle=False)
        cache = {k: loaded[k] for k in loaded.files}
        log.info("Resumed with %d cached embeddings (dim=%d)", len(cache), zdim)

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
        _embed_tracks(
            todo=chunk,
            audio_dir=args.audio_dir,
            audio_ext=args.audio_ext,
            audio_sr=args.audio_sr,
            model=model,
            target_sr=target_sr,
            hop_secs=args.hop_secs,
            device=device,
            cache=chunk_cache,
            skipped=skipped,
        )

        shard_path = args.output.with_suffix(f".shard{chunk_idx}.npz")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(shard_path), **chunk_cache)
        shard_paths.append(shard_path)
        log.info(
            "Saved shard %d  (%d tracks)  → %s", chunk_idx, len(chunk_cache), shard_path
        )

    # ── Merge shards + existing cache and write final .npz ────────────────────
    final_cache = dict(cache)
    for sp in shard_paths:
        loaded = np.load(str(sp), allow_pickle=False)
        final_cache.update({k: loaded[k] for k in loaded.files})
        sp.unlink()

    log.info("Saving %d embeddings (dim=%d) → %s", len(final_cache), zdim, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.output), **final_cache)
    log.info("Done. Skipped %d tracks.", len(skipped))


if __name__ == "__main__":
    main()
