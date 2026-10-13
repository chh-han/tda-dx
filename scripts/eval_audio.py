"""
Evaluate generated audio against reference audio via av-benchmark.

Computes FD-VGG/PANN/PASST, KL divergence, and Inception Score against
the first `audio_length` seconds of each conditioning reference track.

Required env vars:
  TDADX_DATA_ROOT       data root containing fma_large/, fma_large_48k_npy_norm/, exp_results/
  AV_BENCHMARK_PATH    path to a checkout of av-benchmark (must be importable)

Optional:
  TDADX_WORKSPACE       redirect torch/huggingface caches to the given dir
"""

import os
import sys

_workspace = os.environ.get("TDADX_WORKSPACE")
if _workspace:
    os.environ.setdefault("TORCH_HOME",            f"{_workspace}/.cache/torch")
    os.environ.setdefault("HF_HOME",               f"{_workspace}/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE",    f"{_workspace}/.cache/huggingface/transformers")
    os.environ.setdefault("HF_HUB_CACHE",          f"{_workspace}/.cache/huggingface/hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_workspace}/.cache/huggingface/hub")
    os.environ.setdefault("XDG_CACHE_HOME",        f"{_workspace}/.cache")
    os.environ.setdefault("TORCH_HUB_DIR",         f"{_workspace}/.cache/torch/hub")
    os.environ.setdefault("TORCHAUDIO_HOME",       f"{_workspace}/.cache/torchaudio")

_av_bench_path = os.environ.get("AV_BENCHMARK_PATH")
if _av_bench_path:
    sys.path.insert(0, _av_bench_path)

# In non-TTY environments (e.g. log files), tqdm's \r control chars accumulate and pollute logs.
# Must be set before importing av_bench so the env var takes effect.
if not sys.stderr.isatty():
    os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import tempfile
from pathlib import Path

import io
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from av_bench.extract import extract as av_extract
from av_bench.evaluate import evaluate as av_evaluate

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_DATA_ROOT = Path(os.environ.get("TDADX_DATA_ROOT", "./data"))
FMA_LARGE = _DATA_ROOT / "fma_large"
FMA_NPY_NORM = _DATA_ROOT / "fma_large_48k_npy_norm"
DEFAULT_GEN_TOKEN_DIR = _DATA_ROOT / "exp_results" / "gen_token"
NPY_SR = 48_000


def fma_path(track_id: int) -> Path:
    stem = f"{track_id:06d}"
    return FMA_LARGE / stem[:3] / f"{stem}.mp3"


def npy_norm_path(track_id: int, npy_dir: Path) -> Path:
    stem = f"{track_id:06d}"
    return npy_dir / stem[:3] / f"{stem}.npy"


def load_npy_norm(npy_path: Path) -> tuple[np.ndarray, int]:
    """Load a LUFS-normalized mono .npy file. Returns (1, samples) float32 at sr=48000."""
    arr = np.load(npy_path)          # (samples,) mono float32
    return arr[np.newaxis, :], NPY_SR


def load_mp3(mp3_path: Path) -> tuple[np.ndarray, int]:
    """Load MP3 via subprocess ffmpeg + soundfile, bypassing torchcodec entirely.
    Returns: wav (channels, samples) float32, sample_rate
    """
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-i', str(mp3_path), '-f', 'wav', '-acodec', 'pcm_f32le', 'pipe:1']
    result = subprocess.run(cmd, capture_output=True, check=True)
    wav_np, sr = sf.read(io.BytesIO(result.stdout), dtype='float32', always_2d=True)
    return wav_np.T, sr  # soundfile: (samples, ch) → (ch, samples)


def lufs_normalize_audio(wav_np: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize audio to a target integrated LUFS (ITU-R BS.1770-4).
    wav_np: (channels, samples) float32. Returns same shape."""
    import pyloudnorm as pyln
    audio = wav_np.T  # (samples, channels)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness) or loudness < -70.0:
        return wav_np  # silence / extreme low level: return as-is
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    normalized = np.clip(normalized, -1.0, 1.0)
    if normalized.ndim == 1:
        normalized = normalized[:, np.newaxis]
    return normalized.T  # (channels, samples)


def prepare_lufs_normalized_pred(pred_audio_dir: Path, out_dir: Path,
                                  target_lufs: float, audio_length_s: float) -> None:
    """Normalize WAV files in pred_audio_dir to target_lufs and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(pred_audio_dir.glob("*.wav"))
    total = len(pred_files)
    log_interval = max(1, total // 10)
    for i, f in enumerate(pred_files):
        wav_data, sr = sf.read(str(f), dtype='float32', always_2d=True)
        wav_np = wav_data.T  # (ch, samples)
        n = int(audio_length_s * sr)
        wav_np = wav_np[:, :n] if wav_np.shape[1] >= n else wav_np
        normalized = lufs_normalize_audio(wav_np, sr, target_lufs)
        sf.write(str(out_dir / f.name), normalized.T, sr)
        if (i + 1) % log_interval == 0:
            log.info(f"  LUFS norm pred: {i + 1}/{total}")
    log.info(f"LUFS-normalized pred: {total} files -> {out_dir}")


def find_audio_dir(model_dir: Path) -> Path:
    """
    Find the audio directory under {model_dir}/full/.
    Prefers the subdirectory with more .wav files (audio_full > audio when both exist).
    """
    full_dir = model_dir / "full"
    candidates = sorted(full_dir.glob("audio*"), key=lambda d: -len(list(d.glob("*.wav"))))
    if not candidates:
        raise FileNotFoundError(f"No audio* directory found under {full_dir}")
    chosen = candidates[0]
    log.info(f"  audio dir  : {chosen} ({len(list(chosen.glob('*.wav')))} wav files)")
    return chosen


def discover_models(gen_token_dir: Path) -> list[str]:
    """Return model names (subdirectory names) found in gen_token_dir."""
    return sorted(d.name for d in gen_token_dir.iterdir() if d.is_dir())


def load_manifests(model_dir: Path) -> pd.DataFrame:
    shards = sorted((model_dir / "full").glob("selected_samples_shard_*.tsv"))
    if not shards:
        raise FileNotFoundError(f"No manifest TSV files found in {model_dir / 'full'}")
    return pd.concat([pd.read_csv(s, sep="\t") for s in shards], ignore_index=True)


def prepare_gt_audio(df: pd.DataFrame, pred_audio_dir: Path, gt_dir: Path,
                     audio_length_s: float = 4.0, npy_dir: Path | None = None) -> int:
    """
    For each audio file in pred_audio_dir, extract the first audio_length_s seconds
    from the corresponding FMA reference track and save it with the SAME filename.
    npy_dir: directory of LUFS-normalized .npy files. If provided, load from npy instead of converting MP3.
    """
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Build '{track_id}_{sqlite_idx}' → track_id from manifest
    prefix_to_track = {
        f"{int(row['track_id'])}_{int(row['sqlite_idx'])}": int(row['track_id'])
        for _, row in df.iterrows()
    }

    pred_files = sorted(pred_audio_dir.glob("*.wav"))
    if not pred_files:
        raise FileNotFoundError(f"No .wav files found in {pred_audio_dir}")

    skipped = 0
    total = len(pred_files)
    log_interval = max(1, total // 10)  # log progress every 10%

    for i, audio_file in enumerate(pred_files):
        out_path = gt_dir / audio_file.name

        # Extract '{track_id}_{sqlite_idx}' prefix — works for both
        # '181_36.wav' and '181_36_full.wav'
        prefix = "_".join(audio_file.stem.split("_")[:2])
        track_id = prefix_to_track.get(prefix)
        if track_id is None:
            log.warning(f"No manifest entry for {audio_file.name} (prefix={prefix})")
            skipped += 1
            continue

        try:
            npy = npy_norm_path(track_id, npy_dir) if npy_dir else None
            if npy and npy.exists():
                wav_np, sr = load_npy_norm(npy)
            else:
                mp3 = fma_path(track_id)
                if not mp3.exists():
                    log.warning(f"FMA file not found: {mp3}")
                    skipped += 1
                    continue
                wav_np, sr = load_mp3(mp3)
            n_samples = int(audio_length_s * sr)
            if wav_np.shape[1] < n_samples:
                log.warning(f"Track {track_id} shorter than {audio_length_s}s, skipping")
                skipped += 1
                continue
            # soundfile expects (samples, channels) layout
            sf.write(str(out_path), wav_np[:, :n_samples].T, sr)
        except Exception as e:
            log.warning(f"Failed to load track {track_id}: {e}")
            skipped += 1

        if (i + 1) % log_interval == 0:
            log.info(f"  GT audio: {i + 1}/{total} copied (skipped={skipped})")

    log.info(f"GT audio prepared: {total - skipped}/{total} files (skipped {skipped})")
    return total - skipped


def _cache_complete(cache_dir: Path, skip_clap: bool = True) -> bool:
    required = ["pann_features.pth", "vggish_features.pth",
                "passt_features_embed.pth", "passt_logits.pth"]
    if not skip_clap:
        required += ["clap_laion_audio.pth", "clap_ms_audio.pth"]
    return all((cache_dir / f).exists() for f in required)


def compute_audio_clap_score(gt_cache_dir: Path, pred_cache_dir: Path) -> dict:
    """Cosine similarity between GT and pred CLAP audio embeddings (paired)."""
    import torch.nn.functional as F

    results = {}
    for key, filename in [("LAION-CLAP-Audio", "clap_laion_audio.pth"),
                           ("MS-CLAP-Audio",    "clap_ms_audio.pth")]:
        gt_feats   = torch.load(gt_cache_dir   / filename, weights_only=True)
        pred_feats = torch.load(pred_cache_dir / filename, weights_only=True)

        common = sorted(set(gt_feats.keys()) & set(pred_feats.keys()))
        if not common:
            log.warning(f"No common keys for {key}")
            continue

        gt_vecs   = F.normalize(torch.stack([gt_feats[k]   for k in common]), dim=-1)
        pred_vecs = F.normalize(torch.stack([pred_feats[k] for k in common]), dim=-1)
        results[key] = (gt_vecs * pred_vecs).sum(dim=-1).mean().item()

    return results


def filter_cache_by_track_ids(cache_dir: Path, stem_to_tid: dict,
                               allowed_tids: set, out_dir: Path) -> int:
    """Filter .pth files in cache_dir, keeping only stems whose track_id is in allowed_tids, and save to out_dir.
    stem_to_tid: mapping from file_stem (track_id_sqlite_idx) to track_id.
    Supports feature keys in both file_stem and file_stem_full formats.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    def tid_of(key):
        prefix = "_".join(key.split("_")[:2])
        return stem_to_tid.get(prefix)

    kept = None
    for pth in sorted(cache_dir.glob("*.pth")):
        feats = torch.load(pth, weights_only=True)
        filtered = {k: v for k, v in feats.items() if tid_of(k) in allowed_tids}
        torch.save(filtered, out_dir / pth.name)
        kept = len(filtered)
    return kept or 0


def run_eval(
    model_name: str,
    gen_token_dir: Path,
    audio_length: float,
    skip_clap: bool,
    device: str,
    force_reextract: bool = False,
    npy_dir: Path | None = None,
    restrict_track_ids: set | None = None,
    lufs_normalize_pred: bool = False,
    pred_lufs_target: float = -14.0,
) -> dict:
    model_dir      = gen_token_dir / model_name
    pred_audio_dir = find_audio_dir(model_dir)
    gt_cache_dir   = model_dir / "full" / "cache_gt"
    pred_cache_suffix = f"_lufs{int(abs(pred_lufs_target))}" if lufs_normalize_pred else ""
    pred_cache_dir = model_dir / "full" / f"cache_pred{pred_cache_suffix}"

    log.info(f"=== Evaluating: {model_name} ===")

    df = load_manifests(model_dir)
    log.info(f"  manifest   : {len(df)} samples")

    if not force_reextract and _cache_complete(gt_cache_dir, skip_clap):
        log.info(f"GT cache already exists, skipping extraction ({gt_cache_dir})")
    else:
        # AUDIO_EVAL_WAV_TMPDIR: dedicated path for temporary WAV files (e.g. a NOBACKUP volume with space)
        # Leave TMPDIR for unix sockets — Lustre-backed NOBACKUP volumes do not support unix sockets.
        wav_tmp_root = os.environ.get("AUDIO_EVAL_WAV_TMPDIR") or None
        with tempfile.TemporaryDirectory(dir=wav_tmp_root) as tmp_gt:
            gt_dir = Path(tmp_gt)
            log.info(f"Preparing GT audio clips (tmp={gt_dir})...")
            prepare_gt_audio(df, pred_audio_dir, gt_dir, audio_length, npy_dir=npy_dir)
            log.info("Extracting GT features...")
            av_extract(
                audio_path=gt_dir,
                output_path=gt_cache_dir,
                audio_length=audio_length,
                batch_size=32,
                num_workers=4,
                device=device,
                skip_video_related=True,
                skip_clap=skip_clap,
            )
        log.info("GT temp audio cleaned up.")

    if not force_reextract and _cache_complete(pred_cache_dir, skip_clap):
        log.info(f"Pred cache already exists, skipping extraction ({pred_cache_dir})")
    else:
        wav_tmp_root = os.environ.get("AUDIO_EVAL_WAV_TMPDIR") or None
        if lufs_normalize_pred:
            log.info(f"LUFS-normalizing pred audio to {pred_lufs_target} LUFS...")
            with tempfile.TemporaryDirectory(dir=wav_tmp_root) as tmp_pred:
                normalized_pred_dir = Path(tmp_pred)
                prepare_lufs_normalized_pred(pred_audio_dir, normalized_pred_dir,
                                             pred_lufs_target, audio_length)
                log.info("Extracting LUFS-normalized prediction features...")
                av_extract(
                    audio_path=normalized_pred_dir,
                    output_path=pred_cache_dir,
                    audio_length=audio_length,
                    batch_size=64,
                    num_workers=4,
                    device=device,
                    skip_video_related=True,
                    skip_clap=skip_clap,
                )
        else:
            log.info("Extracting prediction features...")
            av_extract(
                audio_path=pred_audio_dir,
                output_path=pred_cache_dir,
                audio_length=audio_length,
                batch_size=64,
                num_workers=4,
                device=device,
                skip_video_related=True,
                skip_clap=skip_clap,
            )

    log.info("Computing metrics...")
    # restrict_track_ids: compute FD only over track_ids common across all models (so sample counts match).
    if restrict_track_ids is not None:
        stem_to_tid = {row['file_stem']: int(row['track_id']) for _, row in df.iterrows()}
        wav_tmp_root = os.environ.get("AUDIO_EVAL_WAV_TMPDIR") or None
        filtered_base = Path(tempfile.mkdtemp(dir=wav_tmp_root, prefix=f"filtered_{model_name}_"))
        try:
            n_gt   = filter_cache_by_track_ids(gt_cache_dir,   stem_to_tid, restrict_track_ids, filtered_base / "gt")
            n_pred = filter_cache_by_track_ids(pred_cache_dir, stem_to_tid, restrict_track_ids, filtered_base / "pred")
            log.info(f"  Filtered to {n_gt} GT / {n_pred} pred features ({len(restrict_track_ids)} tracks)")
            eval_gt_cache   = filtered_base / "gt"
            eval_pred_cache = filtered_base / "pred"
        except Exception as e:
            log.warning(f"Filtering failed ({e}), falling back to full cache")
            import shutil; shutil.rmtree(filtered_base, ignore_errors=True)
            filtered_base = None
            eval_gt_cache, eval_pred_cache = gt_cache_dir, pred_cache_dir
    else:
        filtered_base = None
        eval_gt_cache, eval_pred_cache = gt_cache_dir, pred_cache_dir

    try:
        results = av_evaluate(
            gt_audio_cache=eval_gt_cache,
            pred_audio_cache=eval_pred_cache,
            is_paired=True,
            num_samples=1,
            skip_video_related=True,
            skip_clap=True,
        )
    finally:
        if filtered_base is not None:
            import shutil; shutil.rmtree(filtered_base, ignore_errors=True)

    if not skip_clap:
        log.info("Computing audio-audio CLAP similarity...")
        results.update(compute_audio_clap_score(eval_gt_cache, eval_pred_cache))

    out_json = model_dir / "full" / "eval_metrics.json"
    out_json.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved -> {out_json}")
    return results


def print_comparison(all_results: dict):
    metrics = sorted({m for r in all_results.values() for m in r})
    col_w = 22
    header = f"{'Metric':<30}" + "".join(f"{name:>{col_w}}" for name in all_results)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for m in metrics:
        row = f"{m:<30}"
        for r in all_results.values():
            val = r.get(m)
            row += f"{val:>{col_w}.4f}" if val is not None else f"{'N/A':>{col_w}}"
        print(row)
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated audio with av-benchmark")
    parser.add_argument("--gen_token_dir", type=Path, default=DEFAULT_GEN_TOKEN_DIR,
                        help="Root directory containing per-model gen_token subdirectories")
    parser.add_argument("--model", default=None,
                        help="Model name (subdirectory under gen_token_dir). Default: all discovered models")
    parser.add_argument("--audio_length", type=float, default=4.0, help="Clip length in seconds")
    parser.add_argument("--skip_clap", action="store_true", default=True,
                        help="Skip CLAP score (requires text captions to be meaningful)")
    parser.add_argument("--no_skip_clap", dest="skip_clap", action="store_false")
    parser.add_argument("--force_reextract", action="store_true",
                        help="Re-extract features even if cache already exists")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fma_npy_dir", type=Path, default=FMA_NPY_NORM,
                        help="directory of FMA .npy files")
    parser.add_argument("--lufs_models", nargs="*", default=[],
                        help="optional: list of model names whose ground truth should be loaded from npy.")
    parser.add_argument("--lufs_normalize_pred", action="store_true", default=False,
                        help="Optional: normalize predicted audio to pred_lufs_target LUFS before measurement (off by default; requires pyloudnorm).")
    parser.add_argument("--pred_lufs_target", type=float, default=-14.0,
                        help="Target LUFS when --lufs_normalize_pred is set (default: -14).")
    args = parser.parse_args()

    npy_dir = args.fma_npy_dir if args.fma_npy_dir and args.fma_npy_dir.exists() else None
    lufs_models = set(args.lufs_models or [])

    if args.model is not None:
        models = [args.model]
    else:
        models = discover_models(args.gen_token_dir)
        log.info(f"Discovered models: {models}")

    # When comparing multiple models, restrict FD to the intersection of track_ids (equal sample counts).
    if len(models) > 1:
        track_id_sets = [
            set(load_manifests(args.gen_token_dir / m)['track_id'].astype(int))
            for m in models
        ]
        restrict_track_ids = set.intersection(*track_id_sets)
        log.info(f"Common tracks across all models: {len(restrict_track_ids)} "
                 f"(intersection of {[len(s) for s in track_id_sets]})")
    else:
        restrict_track_ids = None

    all_results = {}
    for model_name in models:
        use_npy = npy_dir if model_name in lufs_models else None
        log.info(f"  GT source for {model_name}: {'npy (LUFS)' if use_npy else 'MP3'}")
        all_results[model_name] = run_eval(
            model_name=model_name,
            gen_token_dir=args.gen_token_dir,
            audio_length=args.audio_length,
            skip_clap=args.skip_clap,
            device=args.device,
            force_reextract=args.force_reextract,
            npy_dir=use_npy,
            restrict_track_ids=restrict_track_ids,
            lufs_normalize_pred=args.lufs_normalize_pred,
            pred_lufs_target=args.pred_lufs_target,
        )

    if len(all_results) > 1:
        print_comparison(all_results)
    else:
        name, results = next(iter(all_results.items()))
        print(f"\n{name} results:")
        for k, v in results.items():
            print(f"  {k:<30} {v:.4f}")


if __name__ == "__main__":
    main()
