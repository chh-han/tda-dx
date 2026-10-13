"""
Preprocess FMA audio files: audio -> target sample rate / channels float32 .npy
Saves each track as a .npy file for fast loading in downstream tasks.

Usage:
    python preprocess_audio_to_npy.py
    python preprocess_audio_to_npy.py --target_sample_hz 48000 --num_workers 15
"""

import os
import sys
import csv
import ast
import argparse
import gc
import math
import time
import multiprocessing as mp
import re
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

# Optional dependency: only required when --target_lufs is passed on the CLI.
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


# ---------------------------------------------------------------------------
# Audio processing (same as find_topk_similar_train.py)
# ---------------------------------------------------------------------------


class AudioValidationError(RuntimeError):
    """Raised when decoded or saved audio does not match pipeline expectations."""


def _minimum_lufs_num_samples(sample_hz: int, block_size_s: float = 0.400) -> int:
    return int(math.ceil(float(sample_hz) * float(block_size_s)))


def _validate_audio_tensor(
    data: torch.Tensor,
    *,
    sample_hz: int,
    stage: str,
    path: Path,
    require_channels: int | None,
    min_seconds: float,
    expected_seconds: float | None,
    expected_tolerance_seconds: float,
) -> None:
    if not isinstance(data, torch.Tensor):
        raise AudioValidationError(f'{stage}: expected torch.Tensor, got {type(data).__name__}')
    if data.ndim != 2:
        raise AudioValidationError(f'{stage}: expected shape [channels, samples], got {tuple(data.shape)}')
    if data.shape[0] < 1:
        raise AudioValidationError(f'{stage}: empty channel dimension for {path}')
    if require_channels is not None and data.shape[0] != require_channels:
        raise AudioValidationError(
            f'{stage}: expected {require_channels} channel audio, got shape {tuple(data.shape)}'
        )
    if data.shape[1] < 1:
        raise AudioValidationError(f'{stage}: zero samples for {path}')
    if not torch.isfinite(data).all():
        raise AudioValidationError(f'{stage}: non-finite values detected in {path}')

    min_samples = int(math.ceil(max(0.0, float(min_seconds)) * float(sample_hz)))
    if data.shape[1] < min_samples:
        raise AudioValidationError(
            f'{stage}: too short for {path} ({data.shape[1]} samples < {min_samples})'
        )

    if expected_seconds is not None and expected_seconds > 0:
        expected_samples = int(round(float(expected_seconds) * float(sample_hz)))
        tolerance_samples = int(round(max(0.0, float(expected_tolerance_seconds)) * float(sample_hz)))
        if abs(int(data.shape[1]) - expected_samples) > tolerance_samples:
            raise AudioValidationError(
                f'{stage}: unexpected duration for {path} '
                f'({data.shape[1]} samples, expected {expected_samples} +/- {tolerance_samples})'
            )


def _validate_audio_array(
    audio_np: np.ndarray,
    *,
    sample_hz: int,
    stage: str,
    path: Path,
    require_channels: int | None,
    min_seconds: float,
    expected_seconds: float | None,
    expected_tolerance_seconds: float,
) -> None:
    if not isinstance(audio_np, np.ndarray):
        raise AudioValidationError(f'{stage}: expected numpy.ndarray, got {type(audio_np).__name__}')
    if audio_np.ndim == 1:
        channels = 1
        n_samples = int(audio_np.shape[0])
    elif audio_np.ndim == 2:
        channels = int(audio_np.shape[0])
        n_samples = int(audio_np.shape[1])
    else:
        raise AudioValidationError(f'{stage}: expected 1D or 2D audio, got shape {audio_np.shape}')
    if require_channels is not None and channels != require_channels:
        raise AudioValidationError(f'{stage}: expected {require_channels} channels, got {channels} for {path}')
    if n_samples < 1:
        raise AudioValidationError(f'{stage}: zero samples for {path}')
    if not np.isfinite(audio_np).all():
        raise AudioValidationError(f'{stage}: non-finite values detected in {path}')

    min_samples = int(math.ceil(max(0.0, float(min_seconds)) * float(sample_hz)))
    if n_samples < min_samples:
        raise AudioValidationError(
            f'{stage}: too short for {path} ({n_samples} samples < {min_samples})'
        )

    if expected_seconds is not None and expected_seconds > 0:
        expected_samples = int(round(float(expected_seconds) * float(sample_hz)))
        tolerance_samples = int(round(max(0.0, float(expected_tolerance_seconds)) * float(sample_hz)))
        if abs(n_samples - expected_samples) > tolerance_samples:
            raise AudioValidationError(
                f'{stage}: unexpected duration for {path} '
                f'({n_samples} samples, expected {expected_samples} +/- {tolerance_samples})'
            )


def _convert_channels(data: torch.Tensor, target_channels: int) -> torch.Tensor:
    if data.ndim != 2:
        raise AudioValidationError(f'channel-convert: expected 2D [channels, samples], got {tuple(data.shape)}')
    if target_channels == 1:
        if data.shape[0] == 1:
            return data
        return torch.mean(data, dim=0, keepdim=True)
    if target_channels == 2:
        if data.shape[0] == 2:
            return data
        if data.shape[0] == 1:
            return data.repeat(2, 1)
        return data[:2]
    raise AudioValidationError(f'channel-convert: unsupported target_channels={target_channels}')


def _validate_existing_output(
    npy_path: Path,
    *,
    target_sample_hz: int,
    target_channels: int,
    min_output_seconds: float,
    expected_output_seconds: float | None,
    expected_duration_tolerance_seconds: float,
) -> int:
    audio_np = np.load(npy_path, allow_pickle=False)
    _validate_audio_array(
        audio_np,
        sample_hz=target_sample_hz,
        stage='existing-output',
        path=npy_path,
        require_channels=target_channels,
        min_seconds=min_output_seconds,
        expected_seconds=expected_output_seconds,
        expected_tolerance_seconds=expected_duration_tolerance_seconds,
    )
    return int(audio_np.shape[-1])

def _apply_lufs_normalization(
    data: torch.Tensor,
    sample_hz: int,
    target_lufs: float | None,
    *,
    min_peak: float = 1e-4,
    true_peak_limit: float = 0.99,
):
    if target_lufs is None:
        return data
    if pyln is None:
        raise RuntimeError(
            "--target_lufs requires the optional pyloudnorm package. "
            "Install with: pip install pyloudnorm"
        )

    audio_np = data.detach().cpu().numpy()
    loudness_np = np.transpose(audio_np, (1, 0)) if audio_np.ndim == 2 and audio_np.shape[0] > 1 else audio_np.squeeze(0)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if not np.isfinite(peak) or peak < min_peak:
        return data

    min_lufs_samples = _minimum_lufs_num_samples(sample_hz)
    if loudness_np.shape[0] < min_lufs_samples:
        raise AudioValidationError(
            f'lufs: too short for integrated loudness '
            f'({loudness_np.shape[0]} samples < {min_lufs_samples})'
        )

    meter = pyln.Meter(sample_hz)
    try:
        loudness = meter.integrated_loudness(loudness_np.astype(np.float64, copy=False))
    except ValueError as exc:
        raise AudioValidationError(f'lufs: {exc}') from exc
    if not np.isfinite(loudness):
        return data

    gain_db = float(target_lufs - loudness)
    gain = float(10.0 ** (gain_db / 20.0))
    normalized_np = audio_np * gain
    normalized_peak = float(np.max(np.abs(normalized_np))) if normalized_np.size else 0.0
    if np.isfinite(normalized_peak) and normalized_peak > true_peak_limit:
        normalized_np = normalized_np * (true_peak_limit / normalized_peak)

    normalized_np = normalized_np.astype(np.float32, copy=False)
    if normalized_np.ndim == 1:
        normalized_np = normalized_np[None, :]
    return torch.from_numpy(normalized_np)


def process_and_save(
    mp3_path,
    npy_path,
    target_sample_hz,
    target_channels,
    target_lufs=None,
    *,
    min_output_seconds: float = 0.400,
    expected_output_seconds: float | None = 30.0,
    expected_duration_tolerance_seconds: float = 1.0,
    verify_saved_output: bool = True,
):
    """Load audio, convert to target sample rate / channels float32, save as .npy"""
    data = None
    audio_np = None
    tmp_output_path = None
    try:
        try:
            data, sample_hz = torchaudio.load(mp3_path)
        except Exception as exc:
            raise AudioValidationError(f'decode: failed to load {mp3_path}: {exc}') from exc
        _validate_audio_tensor(
            data,
            sample_hz=int(sample_hz),
            stage='decoded',
            path=Path(mp3_path),
            require_channels=None,
            min_seconds=1e-4,
            expected_seconds=None,
            expected_tolerance_seconds=0.0,
        )

        # channel conversion
        data = _convert_channels(data, target_channels)
        _validate_audio_tensor(
            data,
            sample_hz=int(sample_hz),
            stage='channels',
            path=Path(mp3_path),
            require_channels=target_channels,
            min_seconds=1e-4,
            expected_seconds=None,
            expected_tolerance_seconds=0.0,
        )

        # resample
        if sample_hz != target_sample_hz:
            data = AF.resample(data, sample_hz, target_sample_hz)
        _validate_audio_tensor(
            data,
            sample_hz=int(target_sample_hz),
            stage='resampled',
            path=Path(mp3_path),
            require_channels=target_channels,
            min_seconds=min_output_seconds,
            expected_seconds=expected_output_seconds,
            expected_tolerance_seconds=expected_duration_tolerance_seconds,
        )

        # loudness normalization (optional)
        data = _apply_lufs_normalization(data, target_sample_hz, target_lufs)
        _validate_audio_tensor(
            data,
            sample_hz=int(target_sample_hz),
            stage='normalized',
            path=Path(mp3_path),
            require_channels=target_channels,
            min_seconds=min_output_seconds,
            expected_seconds=expected_output_seconds,
            expected_tolerance_seconds=expected_duration_tolerance_seconds,
        )

        # mono saves as (T,), stereo saves as (C, T)
        audio_np = data.numpy()
        if target_channels == 1:
            audio_np = audio_np.squeeze(0)
        _validate_audio_array(
            audio_np,
            sample_hz=int(target_sample_hz),
            stage='pre-save',
            path=Path(mp3_path),
            require_channels=target_channels,
            min_seconds=min_output_seconds,
            expected_seconds=expected_output_seconds,
            expected_tolerance_seconds=expected_duration_tolerance_seconds,
        )
        n_samples = int(audio_np.shape[-1])

        # save atomically, then optionally reload to verify the persisted file
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_path = npy_path.with_name(f'.{npy_path.stem}.tmp.{os.getpid()}.npy')
        np.save(tmp_output_path, audio_np)
        os.replace(tmp_output_path, npy_path)
        tmp_output_path = None
        if verify_saved_output:
            _validate_existing_output(
                npy_path,
                target_sample_hz=target_sample_hz,
                target_channels=target_channels,
                min_output_seconds=min_output_seconds,
                expected_output_seconds=expected_output_seconds,
                expected_duration_tolerance_seconds=expected_duration_tolerance_seconds,
            )
        return n_samples
    finally:
        if tmp_output_path is not None:
            tmp_output_path.unlink(missing_ok=True)
        # Release large per-track buffers as soon as the file is written.
        del audio_np
        del data
        gc.collect()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_WORKER_TARGET_HZ = None
_WORKER_TARGET_CHANNELS = None
_WORKER_TARGET_LUFS = None
_WORKER_DEBUG = False
_WORKER_MIN_OUTPUT_SECONDS = None
_WORKER_EXPECTED_OUTPUT_SECONDS = None
_WORKER_EXPECTED_DURATION_TOLERANCE_SECONDS = None
_WORKER_VERIFY_SAVED_OUTPUT = True

def _init_worker(
    target_sample_hz,
    target_channels,
    target_lufs,
    debug,
    min_output_seconds,
    expected_output_seconds,
    expected_duration_tolerance_seconds,
    verify_saved_output,
):
    global _WORKER_TARGET_HZ
    global _WORKER_TARGET_CHANNELS
    global _WORKER_TARGET_LUFS
    global _WORKER_DEBUG
    global _WORKER_MIN_OUTPUT_SECONDS
    global _WORKER_EXPECTED_OUTPUT_SECONDS
    global _WORKER_EXPECTED_DURATION_TOLERANCE_SECONDS
    global _WORKER_VERIFY_SAVED_OUTPUT
    _WORKER_TARGET_HZ = target_sample_hz
    _WORKER_TARGET_CHANNELS = target_channels
    _WORKER_TARGET_LUFS = target_lufs
    _WORKER_DEBUG = debug
    _WORKER_MIN_OUTPUT_SECONDS = min_output_seconds
    _WORKER_EXPECTED_OUTPUT_SECONDS = expected_output_seconds
    _WORKER_EXPECTED_DURATION_TOLERANCE_SECONDS = expected_duration_tolerance_seconds
    _WORKER_VERIFY_SAVED_OUTPUT = verify_saved_output
    torch.set_num_threads(1)
    if _WORKER_DEBUG:
        print(
            f'[worker-init pid={os.getpid()}] target_sample_hz={_WORKER_TARGET_HZ} '
            f'target_channels={_WORKER_TARGET_CHANNELS} '
            f'target_lufs={_WORKER_TARGET_LUFS} '
            f'min_output_seconds={_WORKER_MIN_OUTPUT_SECONDS} '
            f'expected_output_seconds={_WORKER_EXPECTED_OUTPUT_SECONDS} '
            f'verify_saved_output={_WORKER_VERIFY_SAVED_OUTPUT}',
            flush=True,
        )


def _process_one(args):
    task_idx, mp3_path_str, npy_path_str, delete_source = args
    global _WORKER_TARGET_HZ
    global _WORKER_TARGET_CHANNELS
    global _WORKER_TARGET_LUFS
    global _WORKER_DEBUG
    global _WORKER_MIN_OUTPUT_SECONDS
    global _WORKER_EXPECTED_OUTPUT_SECONDS
    global _WORKER_EXPECTED_DURATION_TOLERANCE_SECONDS
    global _WORKER_VERIFY_SAVED_OUTPUT
    mp3_path = Path(mp3_path_str)
    npy_path = Path(npy_path_str)

    if _WORKER_DEBUG and task_idx < 5:
        print(
            f'[worker-start pid={os.getpid()} task={task_idx}] '
            f'{mp3_path} -> {npy_path}',
            flush=True,
        )

    if npy_path.exists():
        n_samples = _validate_existing_output(
            npy_path,
            target_sample_hz=_WORKER_TARGET_HZ,
            target_channels=_WORKER_TARGET_CHANNELS,
            min_output_seconds=_WORKER_MIN_OUTPUT_SECONDS,
            expected_output_seconds=_WORKER_EXPECTED_OUTPUT_SECONDS,
            expected_duration_tolerance_seconds=_WORKER_EXPECTED_DURATION_TOLERANCE_SECONDS,
        )
        if delete_source:
            mp3_path.unlink(missing_ok=True)
        return mp3_path_str, n_samples, 'cached'

    try:
        n_samples = process_and_save(
            mp3_path,
            npy_path,
            _WORKER_TARGET_HZ,
            _WORKER_TARGET_CHANNELS,
            _WORKER_TARGET_LUFS,
            min_output_seconds=_WORKER_MIN_OUTPUT_SECONDS,
            expected_output_seconds=_WORKER_EXPECTED_OUTPUT_SECONDS,
            expected_duration_tolerance_seconds=_WORKER_EXPECTED_DURATION_TOLERANCE_SECONDS,
            verify_saved_output=_WORKER_VERIFY_SAVED_OUTPUT,
        )
    except AudioValidationError as exc:
        if _WORKER_DEBUG:
            print(
                f'[worker-skip pid={os.getpid()} task={task_idx}] '
                f'{mp3_path}: {exc}',
                flush=True,
            )
        return mp3_path_str, 0, f'validation_skip: {exc}'

    if delete_source:
        mp3_path.unlink(missing_ok=True)
    if _WORKER_DEBUG and task_idx < 5:
        print(
            f'[worker-done pid={os.getpid()} task={task_idx}] '
            f'n_samples={n_samples}',
            flush=True,
        )
    return mp3_path_str, n_samples, 'processed'


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _normalize_split_name(split_name):
    if split_name is None:
        return None
    value = str(split_name).strip().lower()
    if value in ('train', 'training'):
        return 'training'
    if value in ('val', 'valid', 'validation'):
        return 'validation'
    if value in ('test', 'testing'):
        return 'test'
    return value or None


def _normalize_subset_name(subset_name):
    if subset_name is None:
        return None
    value = str(subset_name).strip().lower()
    if value in ('small', 'medium', 'large'):
        return value
    return value or None


def _infer_subset_from_audio_root(audio_root):
    name = Path(audio_root).name.lower()
    for subset in ('small', 'medium', 'large'):
        if f'fma_{subset}' in name:
            return subset
    return None


def load_audio_entries(metadata_path):
    """
    Load audio entries from either:
    - FMA-style tracks.csv with a two-row header and track IDs
    - Jamendo-style TSV with a PATH column
    """
    metadata_path = Path(metadata_path)

    with open(metadata_path, newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.reader(f)
        first_row = next(reader)

        # FMA tracks.csv: multi-row header containing set/split and set/subset
        if any(cell.strip().lower() == 'track' for cell in first_row):
            second_row = next(reader)
            third_row = next(reader)
            split_idx = None
            subset_idx = None
            for i, (a, b, _c) in enumerate(zip(first_row, second_row, third_row)):
                if a.strip().lower() == 'set' and b.strip().lower() == 'split':
                    split_idx = i
                if a.strip().lower() == 'set' and b.strip().lower() == 'subset':
                    subset_idx = i
            if split_idx is None or subset_idx is None:
                raise ValueError(f'set.split / set.subset column not found in {metadata_path}')

            entries = []
            for row in reader:
                if len(row) <= max(split_idx, subset_idx):
                    continue
                track_id_raw = str(row[0]).strip()
                if not track_id_raw.isdigit():
                    continue
                split_name = _normalize_split_name(row[split_idx])
                subset_name = _normalize_subset_name(row[subset_idx])
                if split_name not in ('training', 'validation', 'test'):
                    continue
                track_id = int(track_id_raw)
                rel_audio_path = track_id_to_rel_audio_path(track_id)
                rel_output_path = track_id_to_rel_npy_path(track_id)
                entries.append({
                    'id': str(track_id),
                    'split': split_name,
                    'subset': subset_name,
                    'rel_audio_path': rel_audio_path,
                    'rel_output_path': rel_output_path,
                })
            return entries

    with open(metadata_path, newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t')
        dict_reader = csv.DictReader(f, dialect=dialect)
        if dict_reader.fieldnames is None:
            raise ValueError(f'could not read header from {metadata_path}')

        field_map = {name.strip().lower(): name for name in dict_reader.fieldnames}
        path_key = field_map.get('path')
        if path_key is None:
            raise ValueError(
                f'unsupported metadata format at {metadata_path}: expected FMA tracks.csv '
                f'or a table with a PATH column'
            )

        split_key = field_map.get('split')
        track_key = field_map.get('track_id')

        entries = []
        for idx, row in enumerate(dict_reader):
            rel_audio_path_raw = (row.get(path_key) or '').strip()
            if not rel_audio_path_raw:
                continue
            rel_audio_path = Path(rel_audio_path_raw)
            rel_output_path = rel_audio_path.with_suffix('.npy')
            split_name = _normalize_split_name(row.get(split_key)) if split_key else None
            track_name = (row.get(track_key) or rel_audio_path.stem or str(idx)).strip()
            entries.append({
                'id': track_name,
                'split': split_name,
                'subset': None,
                'rel_audio_path': rel_audio_path,
                'rel_output_path': rel_output_path,
            })
        return entries


def track_id_to_rel_audio_path(track_id):
    filename = f'{track_id:06d}.mp3'
    subdir = f'{track_id:06d}'[:3]
    return Path(subdir) / filename


def track_id_to_rel_npy_path(track_id):
    subdir = f'{track_id:06d}'[:3]
    filename = f'{track_id:06d}.npy'
    return Path(subdir) / filename


def load_fma_ignore_stems(metadata_path):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    assert pd is not None, 'pandas not found, please install pandas to filter fma'

    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    tracks_total = len(tracks)
    experimental_genre = 38
    experimental_tracks = tracks.loc[
        tracks['track', 'genres_all'].apply(lambda x: experimental_genre in ast.literal_eval(x))
    ]
    ignore_track_ids = list(
        experimental_tracks.loc[
            (experimental_tracks['track', 'listens'] <= 1000) |
            (experimental_tracks['track', 'favorites'] <= 5)
        ].index
    )
    ignore_stems = {f'{int(i):06d}' for i in ignore_track_ids}
    print(
        f'filter_fma enabled: tracks.csv total={tracks_total}, '
        f'experimental_genre_tracks={len(experimental_tracks)}',
        flush=True,
    )
    print(f'filter_fma enabled: filtered_out={len(ignore_stems)} low-quality experimental tracks', flush=True)
    print(f'filter_fma enabled: tracks_after_filter={tracks_total - len(ignore_stems)}', flush=True)
    return ignore_stems


def _normalize_dedupe_text(value):
    value = str(value).strip().lower()
    value = re.sub(r'\s+', ' ', value)
    return value


def load_fma_duplicate_stems(metadata_path):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    assert pd is not None, 'pandas not found, please install pandas to deduplicate fma'

    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    seen_keys = {}
    duplicate_stems = set()

    for track_id, row in tracks.iterrows():
        artist_name = _normalize_dedupe_text(row[('artist', 'name')])
        track_title = _normalize_dedupe_text(row[('track', 'title')])
        if not artist_name or not track_title:
            continue
        key = (artist_name, track_title)
        if key in seen_keys:
            duplicate_stems.add(f'{int(track_id):06d}')
        else:
            seen_keys[key] = int(track_id)

    print(
        f'dedupe_artist_title enabled: unique_keys={len(seen_keys)} '
        f'duplicates_filtered={len(duplicate_stems)}',
        flush=True,
    )
    return duplicate_stems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.shard_count < 1:
        raise ValueError('--shard_count must be >= 1')
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError('--shard_index must satisfy 0 <= shard_index < shard_count')
    if args.start_index < 0:
        raise ValueError('--start_index must be >= 0')
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError('--end_index must be >= start_index')

    print(f'target_sample_hz: {args.target_sample_hz}', flush=True)
    print(f'target_channels: {args.target_channels}', flush=True)
    print(f'num_workers: {args.num_workers}', flush=True)
    print(f'max_tasks_per_child: {args.max_tasks_per_child}', flush=True)
    print(f'shard_index: {args.shard_index}/{args.shard_count}', flush=True)
    print(f'start_index: {args.start_index}', flush=True)
    print(f'end_index: {args.end_index}', flush=True)
    print(f'delete_source: {args.delete_source}', flush=True)
    print(f'output_dir: {args.output_dir}', flush=True)
    print(f'target_lufs: {args.target_lufs}', flush=True)
    print(f'min_output_seconds: {args.min_output_seconds}', flush=True)
    print(f'expected_output_seconds: {args.expected_output_seconds}', flush=True)
    print(f'expected_duration_tolerance_seconds: {args.expected_duration_tolerance_seconds}', flush=True)
    print(f'verify_saved_output: {args.verify_saved_output}', flush=True)
    print(f'debug_logging: {args.debug_logging}', flush=True)

    entries = load_audio_entries(args.tracks_csv)
    print(f'total entries in metadata: {len(entries)}', flush=True)

    ignore_stems = None
    if args.filter_fma:
        ignore_stems = load_fma_ignore_stems(args.tracks_csv)
    else:
        print('filter_fma disabled', flush=True)

    duplicate_stems = None
    if args.dedupe_artist_title:
        duplicate_stems = load_fma_duplicate_stems(args.tracks_csv)
    else:
        print('dedupe_artist_title disabled', flush=True)

    requested_subset = args.subset
    if requested_subset == 'auto':
        requested_subset = _infer_subset_from_audio_root(args.audio_root)
    if requested_subset is not None:
        subset_aware_entries = [entry for entry in entries if entry.get('subset') is not None]
        if len(subset_aware_entries) == 0:
            print(
                f'warning: metadata has no subset column; ignoring --subset={requested_subset}',
                flush=True,
            )
        else:
            requested_subset = _normalize_subset_name(requested_subset)
            entries = [entry for entry in entries if entry.get('subset') == requested_subset]
            print(f'filtered to subset={requested_subset}: {len(entries)}', flush=True)

    if args.split:
        requested_split = _normalize_split_name(args.split)
        split_aware_entries = [entry for entry in entries if entry['split'] is not None]
        if len(split_aware_entries) == 0:
            print(
                f'warning: metadata has no split column; ignoring --split={requested_split}',
                flush=True,
            )
        else:
            entries = [entry for entry in entries if entry['split'] == requested_split]
            print(f'filtered to split={requested_split}: {len(entries)}', flush=True)

    if args.shard_count > 1:
        entries = [
            entry for entry_idx, entry in enumerate(entries)
            if entry_idx % args.shard_count == args.shard_index
        ]
        print(
            f'filtered to shard {args.shard_index}/{args.shard_count}: {len(entries)}',
            flush=True,
        )

    if args.start_index > 0 or args.end_index is not None:
        entries = entries[args.start_index:args.end_index]
        print(
            f'filtered to entry range [{args.start_index}, '
            f'{args.end_index if args.end_index is not None else "end"}): {len(entries)}',
            flush=True,
        )

    tasks = []
    missing = 0
    filtered = 0
    deduped = 0
    for task_idx, entry in enumerate(entries):
        rel_output_path = entry['rel_output_path']
        if ignore_stems is not None and rel_output_path.stem in ignore_stems:
            filtered += 1
            continue
        if duplicate_stems is not None and rel_output_path.stem in duplicate_stems:
            deduped += 1
            continue
        mp3_path = args.audio_root / entry['rel_audio_path']
        if not mp3_path.exists():
            missing += 1
            continue
        npy_path = args.output_dir / rel_output_path
        tasks.append((task_idx, str(mp3_path), str(npy_path), args.delete_source))

    print(
        f'files to process: {len(tasks)} '
        f'(missing mp3: {missing}, filtered: {filtered}, deduped: {deduped})',
        flush=True,
    )
    if len(tasks) > 0:
        preview_count = min(3, len(tasks))
        print('task preview:', flush=True)
        for task_idx, mp3_path, npy_path, _delete_source in tasks[:preview_count]:
            print(f'  [{task_idx}] {mp3_path} -> {npy_path}', flush=True)

    # check how many already done
    already_done = sum(1 for _, _, npy, _delete_source in tasks if Path(npy).exists())
    if already_done > 0:
        print(f'already preprocessed: {already_done}/{len(tasks)}', flush=True)

    # process with multiprocessing
    start_time = time.time()
    done = 0
    cached = 0
    validation_skipped = 0
    errors = 0
    submit_count = 0

    if args.num_workers <= 1:
        print('using sequential single-process mode for lowest memory usage', flush=True)
        _init_worker(
            args.target_sample_hz,
            args.target_channels,
            args.target_lufs,
            args.debug_logging,
            args.min_output_seconds,
            args.expected_output_seconds,
            args.expected_duration_tolerance_seconds,
            args.verify_saved_output,
        )
        submit_pbar = tqdm(total=len(tasks), desc='submit', unit='task', mininterval=1.0)
        done_pbar = tqdm(total=len(tasks), desc='process', unit='task', mininterval=1.0)
        try:
            for task in tasks:
                submit_count += 1
                submit_pbar.update(1)
                if submit_count <= 3 or submit_count % args.submit_log_every == 0:
                    print(f'submitted {submit_count}/{len(tasks)} tasks', flush=True)

                task_idx, mp3_path, _npy_path, _delete_source = task
                try:
                    path_str, n_samples, status = _process_one(task)
                    if status == 'cached':
                        cached += 1
                    elif str(status).startswith('validation_skip:'):
                        validation_skipped += 1
                        print(f'skipped: task={task_idx} file={mp3_path}: {status}', flush=True)
                    done += 1
                    done_pbar.update(1)
                except Exception as e:
                    errors += 1
                    done += 1
                    done_pbar.update(1)
                    print(f'error: task={task_idx} file={mp3_path}: {e}', flush=True)

                if done % args.log_every == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (len(tasks) - done) / rate if rate > 0 else float('inf')
                    pct = done / len(tasks) * 100
                    print(
                        f'progress: {pct:5.1f}%  {done}/{len(tasks)}  '
                        f'elapsed={elapsed:.0f}s  eta={remaining:.0f}s  '
                        f'errors={errors}  cached={cached}  validation_skipped={validation_skipped}',
                        flush=True,
                    )
        finally:
            submit_pbar.close()
            done_pbar.close()
    elif args.max_tasks_per_child is not None:
        print(
            f'using multiprocessing.Pool with maxtasksperchild={args.max_tasks_per_child}',
            flush=True,
        )
        submit_pbar = tqdm(total=len(tasks), desc='submit', unit='task', mininterval=1.0)
        done_pbar = tqdm(total=len(tasks), desc='process', unit='task', mininterval=1.0)
        try:
            # Pool lazily pulls from the task iterator and recycles workers after
            # the configured number of tracks, which is the most reliable way to
            # force OS-level memory release on this Python 3.10 runtime.
            with mp.Pool(
                processes=args.num_workers,
                initializer=_init_worker,
                initargs=(
                    args.target_sample_hz,
                    args.target_channels,
                    args.target_lufs,
                    args.debug_logging,
                    args.min_output_seconds,
                    args.expected_output_seconds,
                    args.expected_duration_tolerance_seconds,
                    args.verify_saved_output,
                ),
                maxtasksperchild=args.max_tasks_per_child,
            ) as pool:
                for task in tasks:
                    submit_count += 1
                    submit_pbar.update(1)
                    if submit_count <= 3 or submit_count % args.submit_log_every == 0:
                        print(
                            f'submitted {submit_count}/{len(tasks)} tasks',
                            flush=True,
                        )

                print('all tasks queued; waiting for worker results...', flush=True)

                for result in pool.imap_unordered(_process_one, tasks, chunksize=1):
                    path_str, n_samples, status = result
                    if status == 'cached':
                        cached += 1
                    elif str(status).startswith('validation_skip:'):
                        validation_skipped += 1
                        print(f'skipped: file={path_str}: {status}', flush=True)
                    done += 1
                    done_pbar.update(1)

                    if done % args.log_every == 0:
                        elapsed = time.time() - start_time
                        rate = done / elapsed if elapsed > 0 else 0
                        remaining = (len(tasks) - done) / rate if rate > 0 else float('inf')
                        pct = done / len(tasks) * 100
                        print(
                            f'progress: {pct:5.1f}%  {done}/{len(tasks)}  '
                            f'elapsed={elapsed:.0f}s  eta={remaining:.0f}s  '
                            f'errors={errors}  cached={cached}  validation_skipped={validation_skipped}',
                            flush=True,
                        )
        finally:
            submit_pbar.close()
            done_pbar.close()
    else:
        executor_kwargs = dict(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(
            args.target_sample_hz,
            args.target_channels,
            args.target_lufs,
            args.debug_logging,
                args.min_output_seconds,
                args.expected_output_seconds,
                args.expected_duration_tolerance_seconds,
                args.verify_saved_output,
            ),
        )

        executor = ProcessPoolExecutor(**executor_kwargs)

        try:
            max_pending = max(args.num_workers * args.max_pending_multiplier, args.num_workers)
            print(
                f'streaming {len(tasks)} tasks to process pool '
                f'(max_pending={max_pending})...',
                flush=True,
            )

            task_iter = iter(tasks)
            pending = {}
            submit_pbar = tqdm(total=len(tasks), desc='submit', unit='task', mininterval=1.0)
            done_pbar = tqdm(total=len(tasks), desc='process', unit='task', mininterval=1.0)

            def submit_next():
                nonlocal submit_count
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                future = executor.submit(_process_one, task)
                pending[future] = task
                submit_count += 1
                submit_pbar.update(1)
                if submit_count <= 3 or submit_count % args.submit_log_every == 0:
                    print(
                        f'submitted {submit_count}/{len(tasks)} tasks '
                        f'(pending={len(pending)})',
                        flush=True,
                    )
                return True

            while len(pending) < max_pending and submit_next():
                pass

            print('initial task window submitted; waiting for worker results...', flush=True)

            while pending:
                done_futures, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)

                for future in done_futures:
                    task = pending.pop(future)
                    while len(pending) < max_pending and submit_next():
                        pass

                    task_idx, mp3_path, _npy_path, _delete_source = task
                    try:
                        path_str, n_samples, status = future.result()
                        if status == 'cached':
                            cached += 1
                        elif str(status).startswith('validation_skip:'):
                            validation_skipped += 1
                            print(f'skipped: task={task_idx} file={mp3_path}: {status}', flush=True)
                        done += 1
                        done_pbar.update(1)
                    except Exception as e:
                        errors += 1
                        done += 1
                        done_pbar.update(1)
                        print(f'error: task={task_idx} file={mp3_path}: {e}', flush=True)

                    if done % args.log_every == 0:
                        elapsed = time.time() - start_time
                        rate = done / elapsed if elapsed > 0 else 0
                        remaining = (len(tasks) - done) / rate if rate > 0 else float('inf')
                        pct = done / len(tasks) * 100
                        print(
                            f'progress: {pct:5.1f}%  {done}/{len(tasks)}  '
                            f'elapsed={elapsed:.0f}s  eta={remaining:.0f}s  '
                            f'errors={errors}  cached={cached}  validation_skipped={validation_skipped}',
                            flush=True,
                        )
        finally:
            if 'submit_pbar' in locals():
                submit_pbar.close()
            if 'done_pbar' in locals():
                done_pbar.close()
            executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    print(f'\ndone: {done} files in {elapsed:.0f}s', flush=True)
    print(f'  processed: {done - cached - validation_skipped - errors}', flush=True)
    print(f'  cached: {cached}', flush=True)
    print(f'  validation_skipped: {validation_skipped}', flush=True)
    print(f'  errors: {errors}', flush=True)
    print(f'  output: {args.output_dir}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess audio metadata: audio -> target sample-rate/channel .npy')

    parser.add_argument('--tracks_csv', type=Path, required=True,
                        help='path to tracks.csv metadata (e.g. fma_metadata/tracks.csv)')
    parser.add_argument('--audio_root', type=Path, required=True,
                        help='root directory containing source audio (mp3/wav)')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='destination directory for resampled .npy files')
    parser.add_argument('--target_sample_hz', type=int, default=48000)
    parser.add_argument('--target_channels', type=int, choices=[1, 2], default=1)
    parser.add_argument('--target_lufs', type=float, default=None,
                        help='optional integrated loudness target in LUFS applied before saving .npy')
    parser.add_argument('--min_output_seconds', type=float, default=0.400,
                        help='minimum accepted output duration in seconds after resampling')
    parser.add_argument('--expected_output_seconds', type=float, default=30.0,
                        help='expected output duration in seconds; set <= 0 to disable duration checking')
    parser.add_argument('--expected_duration_tolerance_seconds', type=float, default=1.0,
                        help='allowed absolute duration deviation in seconds around expected_output_seconds')
    parser.add_argument('--verify_saved_output', action='store_true', default=True,
                        help='reload each saved .npy and validate it before marking the task successful')
    parser.add_argument('--no_verify_saved_output', action='store_false', dest='verify_saved_output',
                        help='skip reload verification of saved .npy files')
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--filter_fma', default=True, action=argparse.BooleanOptionalAction,
                        help='filter out low-quality experimental FMA tracks using tracks.csv metadata')
    parser.add_argument('--dedupe_artist_title', default=True, action=argparse.BooleanOptionalAction,
                        help='filter duplicate FMA tracks using normalized artist+title metadata keys')
    parser.add_argument('--split', type=str, default=None,
                        help='optional: only process "training", "validation", or "test" when metadata provides splits')
    parser.add_argument('--subset', choices=['auto', 'small', 'medium', 'large'], default=None,
                        help='optional: only process FMA subset entries; default disables subset filtering')
    parser.add_argument('--shard_count', type=int, default=1,
                        help='split work into this many shards and process only one shard in this run')
    parser.add_argument('--shard_index', type=int, default=0,
                        help='0-based shard index to process')
    parser.add_argument('--start_index', type=int, default=0,
                        help='start entry index after metadata filtering (inclusive)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='end entry index after metadata filtering (exclusive)')
    parser.add_argument('--delete_source', action='store_true',
                        help='delete the source mp3 after the output .npy is written successfully')
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--debug_logging', action='store_true',
                        help='print extra startup logs for the first few tasks and each worker init')
    parser.add_argument('--max_pending_multiplier', type=int, default=4,
                        help='keep at most num_workers * this many tasks pending in the process pool')
    parser.add_argument('--max_tasks_per_child', type=int, default=None,
                        help='restart each worker after this many tasks to force OS-level memory release')
    parser.add_argument('--submit_log_every', type=int, default=5000,
                        help='log every N task submissions while filling the process pool')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
