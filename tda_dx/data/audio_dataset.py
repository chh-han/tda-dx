import io
import json
import os
import random
import sqlite3
import time
from functools import partial, wraps
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from beartype.door import is_bearable
from beartype.typing import List, Literal, Optional, Tuple, Union

try:
    import pyloudnorm as pyln
    _PYLOUDNORM_AVAILABLE = True
except ImportError:
    _PYLOUDNORM_AVAILABLE = False
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchaudio.functional import resample

from tda_dx.utils import (beartype_jit, curtail_to_multiple, default,
                    float32_to_int16, int16_to_float32,
                    zero_mean_unit_var_norm)

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# sqlite helpers for preprocessing

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

@beartype_jit
def init_sqlite(db_path, readonly: bool = False, optimize_for_read: bool = False):
    """Connect to a sqlite database. Will create a new one if it doesn't exist."""
    connect_path = db_path
    uri = False
    if readonly:
        connect_path = f'file:{db_path}?mode=ro'
        uri = True

    conn = sqlite3.connect(
        connect_path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        uri=uri,
        cached_statements=512,
    )
    cursor = conn.cursor()

    if optimize_for_read:
        pragmas = [
            'PRAGMA query_only=ON',
            'PRAGMA temp_store=MEMORY',
            'PRAGMA cache_size=-200000',      # ~200MB page cache
            'PRAGMA mmap_size=268435456',     # 256MB mmap
        ]
        for pragma in pragmas:
            try:
                cursor.execute(pragma)
            except Exception:
                pass

    return conn, cursor

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]
FloatOrInt = Union[float, int]

# LUFS normalization helpers (inline, no npy intermediate required)

def _lufs_min_samples(sample_hz: int, block_size_s: float = 0.400) -> int:
    return int(np.ceil(float(sample_hz) * float(block_size_s)))


def _apply_lufs_normalization_np(
    audio: np.ndarray,
    sample_hz: int,
    target_lufs: float,
    min_peak: float = 1e-4,
    true_peak_limit: float = 0.99,
) -> np.ndarray:
    """Apply integrated-loudness normalisation to a numpy audio array.

    ``audio`` must be shaped ``(channels, samples)`` or ``(samples,)``.
    Returns the normalised array with the same shape and float32 dtype.
    If the clip is too short or silent the original array is returned unchanged.
    """
    if not _PYLOUDNORM_AVAILABLE:
        raise RuntimeError(
            'pyloudnorm is required for LUFS normalisation. '
            'Install it with: pip install pyloudnorm'
        )

    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if not np.isfinite(peak) or peak < min_peak:
        return audio

    # pyloudnorm expects (samples,) or (samples, channels)
    loudness_audio = audio.T if audio.ndim == 2 else audio
    if loudness_audio.shape[0] < _lufs_min_samples(sample_hz):
        return audio

    meter = pyln.Meter(sample_hz)
    try:
        loudness = meter.integrated_loudness(loudness_audio.astype(np.float64, copy=False))
    except ValueError as exc:
        if 'length greater than the block size' in str(exc):
            return audio
        raise
    if not np.isfinite(loudness):
        return audio

    gain = float(10.0 ** ((target_lufs - loudness) / 20.0))
    normalized = (audio * gain).astype(np.float32)
    norm_peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if np.isfinite(norm_peak) and norm_peak > true_peak_limit:
        normalized = normalized * (true_peak_limit / norm_peak)
    return np.ascontiguousarray(normalized)


# dataset functions

@beartype_jit
class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3'],
        max_length_seconds: Optional[Union[FloatOrInt, Tuple[Optional[FloatOrInt], ...]]] = 1,
        normalize: Union[bool, Tuple[bool, ...]] = False,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None,
        ignore_files: Optional[List[str]] = None,
        ignore_load_errors=True,
        random_crop=True,
        preserve_channels=False,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = []
        ignore_files = default(ignore_files, [])
        num_ignored = 0
        num_empty = 0

        # Allow extension-agnostic ignores (e.g., "000123", "000123.mp3", "000123.npy").
        ignore_file_set = set()
        ignore_stem_set = set()
        ignore_track_id_set = set()
        for ignore_file in ignore_files:
            name = Path(str(ignore_file)).name
            if len(name) == 0:
                continue
            ignore_file_set.add(name)

            stem = Path(name).stem
            ignore_stem_set.add(stem)
            try:
                ignore_track_id_set.add(str(int(stem)))
            except ValueError:
                pass
        for ext in exts:
            for file in path.glob(f'**/*.{ext}'):
                try:
                    if file.stat().st_size == 0:
                        num_empty += 1
                        continue
                except OSError:
                    if ignore_load_errors:
                        num_ignored += 1
                        continue
                    raise

                stem = file.stem
                try:
                    normalized_track_id = str(int(stem))
                except ValueError:
                    normalized_track_id = None

                if (
                    file.name in ignore_file_set or
                    stem in ignore_stem_set or
                    (normalized_track_id is not None and normalized_track_id in ignore_track_id_set)
                ):
                    num_ignored += 1
                    continue
                else:
                    files.append(file)
        assert len(files) > 0, 'no sound files found'
        if num_ignored > 0:
            print(f'skipped {num_ignored} ignored files')
        if num_empty > 0:
            print(f'skipped {num_empty} empty files')

        self.files = sorted(files)
        self.index_offset = 0
        self.ignore_load_errors = ignore_load_errors
        self.random_crop = random_crop
        self.preserve_channels = preserve_channels

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length_seconds = cast_tuple(max_length_seconds, num_outputs)
        self.max_length = tuple([int(s * hz) if exists(s) else None for s, hz in zip(self.max_length_seconds, self.target_sample_hz)])

        self.normalize = cast_tuple(normalize, num_outputs)

        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.max_length_seconds) == len(
            self.target_sample_hz) == len(self.seq_len_multiple_of) == len(self.normalize)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file = self.files[idx]
            data, sample_hz = torchaudio.load(file)
        except:
            if self.ignore_load_errors:
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                raise Exception(f'error loading file {file}')

        return self.process_audio(data, sample_hz, pad_to_target_length=True)

    def process_audio(self, data, sample_hz, pad_to_target_length=True):

        if not self.preserve_channels and data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        # recursively crop the audio at random in the order of longest to shortest max_length_seconds, padding when necessary.
        # e.g. if max_length_seconds = (10, 4), pick a 10 second crop from the original, then pick a 4 second crop from the 10 second crop
        # also use normalized data when specified

        temp_data = data
        need_normalized = any(self.normalize)
        temp_data_normalized = zero_mean_unit_var_norm(data) if need_normalized else None

        num_outputs = len(self.target_sample_hz)
        data = [None for _ in range(num_outputs)]

        sorted_max_length_seconds = sorted(
            enumerate(self.max_length_seconds),
            key=lambda t: (t[1] is not None, t[1])) # sort by max_length_seconds, while moving None to the beginning

        for unsorted_i, max_length_seconds in sorted_max_length_seconds:

            if exists(max_length_seconds):
                audio_length = temp_data.size(1)
                target_length = int(max_length_seconds * sample_hz)

                if audio_length > target_length:
                    max_start = audio_length - target_length
                    start = torch.randint(0, max_start, (1, )) if self.random_crop else 0

                    temp_data = temp_data[:, start:start + target_length]
                    if need_normalized:
                        temp_data_normalized = temp_data_normalized[:, start:start + target_length]
                else:
                    if pad_to_target_length:
                        temp_data = F.pad(temp_data, (0, target_length - audio_length), 'constant')
                        if need_normalized:
                            temp_data_normalized = F.pad(temp_data_normalized, (0, target_length - audio_length), 'constant')

            data[unsorted_i] = temp_data_normalized if self.normalize[unsorted_i] else temp_data
        # resample if target_sample_hz is not None in the tuple
        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))
        # quantize non-normalized audio to a valid waveform
        data_tuple = tuple(d if self.normalize[i] else int16_to_float32(float32_to_int16(d)) for i, d in enumerate(data_tuple))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            if exists(max_length) and pad_to_target_length:
                assert audio_length == max_length, f'audio length {audio_length} does not match max_length {max_length}.'

            if data.shape[0] == 1:
                data = rearrange(data, '1 ... -> ...')

            if exists(seq_len_multiple_of):
                if data.ndim == 1:
                    data = curtail_to_multiple(data, seq_len_multiple_of)
                else:
                    trimmed_length = (data.shape[-1] // seq_len_multiple_of) * seq_len_multiple_of
                    data = data[:, :trimmed_length]

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output


@beartype_jit
class NpySoundDataset(SoundDataset):
    def __init__(
        self,
        folder,
        source_sample_hz: int = 48000,
        exts: Optional[List[str]] = None,
        **kwargs
    ):
        self.source_sample_hz = source_sample_hz
        exts = default(exts, ['npy'])
        super().__init__(folder=folder, exts=exts, **kwargs)

    def __getitem__(self, idx):
        file = None
        try:
            file = self.files[idx]
            data = np.load(file)
        except:
            if self.ignore_load_errors:
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                raise Exception(f'error loading npy file {file}')

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        else:
            data = torch.tensor(data)

        if data.ndim == 1:
            data = data.unsqueeze(0)
        elif data.ndim == 2:
            if data.shape[0] != 1 and data.shape[1] == 1:
                data = data.transpose(0, 1)
        else:
            raise ValueError(f'expected 1D or 2D array in {file}, got shape={tuple(data.shape)}')

        data = data.to(torch.float32)

        return self.process_audio(data, self.source_sample_hz, pad_to_target_length=True)

@beartype_jit
class ChunkedSoundDataset(Dataset):
    """
    Enumerates every non-overlapping chunk of `chunk_seconds` across all files.
    Uses torchaudio.info() to scan durations without loading audio, then loads
    only the required frames per chunk via frame_offset / num_frames.
    The chunk index is cached to disk so scanning runs only once.
    """

    def __init__(
        self,
        folder,
        chunk_seconds: float,
        target_sample_hz: int,
        exts: List[str] = ['flac', 'wav', 'mp3'],
        seq_len_multiple_of: Optional[int] = None,
        ignore_load_errors: bool = True,
        ignore_files: Optional[List[str]] = None,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), f'folder does not exist: {folder}'

        self.chunk_seconds = chunk_seconds
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.ignore_load_errors = ignore_load_errors

        ignore_files = default(ignore_files, [])
        ignore_stem_set = set()
        ignore_track_id_set = set()
        for f in ignore_files:
            stem = Path(str(f)).stem
            ignore_stem_set.add(stem)
            try:
                ignore_track_id_set.add(str(int(stem)))
            except ValueError:
                pass

        files = []
        for ext in exts:
            for f in sorted(path.glob(f'**/*.{ext}')):
                try:
                    if f.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                stem = f.stem
                try:
                    norm_id = str(int(stem))
                except ValueError:
                    norm_id = None
                if stem in ignore_stem_set or (norm_id is not None and norm_id in ignore_track_id_set):
                    continue
                files.append(f)

        assert len(files) > 0, 'no sound files found'

        chunk_tag = str(chunk_seconds).replace('.', 'p')
        meta_path = path / f'.chunk_index_{chunk_tag}s.json'
        chunks_path = path / f'.chunk_index_{chunk_tag}s.npy'

        self.file_paths, self.file_sample_rates, self.chunks = \
            self._load_or_build_index(files, chunk_seconds, meta_path, chunks_path)

        print(f'ChunkedSoundDataset: {len(self.chunks)} chunks from {len(self.file_paths)} files')

    def _load_or_build_index(self, files, chunk_seconds, meta_path, chunks_path):
        current_file_strs = [str(f) for f in files]

        if meta_path.exists() and chunks_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if (
                    abs(meta.get('chunk_seconds', -1) - chunk_seconds) < 1e-6 and
                    meta.get('files') == current_file_strs
                ):
                    chunks = np.load(str(chunks_path))
                    file_paths = [Path(p) for p in meta['files']]
                    sample_rates = meta['sample_rates']
                    print(f'loaded chunk index from cache ({meta_path.name}): {len(chunks)} chunks')
                    return file_paths, sample_rates, chunks
            except Exception as e:
                print(f'chunk cache invalid, rebuilding ({e})')

        print(f'building chunk index for {len(files)} files (runs once per dataset)...')
        file_paths = []
        sample_rates = []
        all_chunks = []
        n_skipped = 0

        for i, f in enumerate(files):
            if i % 5000 == 0 and i > 0:
                print(f'  {i}/{len(files)} scanned, {len(file_paths)} valid so far...')
            try:
                data, source_hz = torchaudio.load(str(f))
                num_frames = data.shape[1]
                del data

                chunk_frames = int(chunk_seconds * source_hz)
                if chunk_frames == 0 or num_frames < chunk_frames:
                    n_skipped += 1
                    continue
                n_chunks = num_frames // chunk_frames
                file_idx = len(file_paths)
                file_paths.append(f)
                sample_rates.append(source_hz)
                for c in range(n_chunks):
                    all_chunks.append((file_idx, c))
            except Exception as e:
                if n_skipped < 3:
                    print(f'  warning: skipping {f.name}: {e}')
                n_skipped += 1

        if n_skipped > 0:
            print(f'  skipped {n_skipped} files (too short or unreadable)')

        chunks = np.array(all_chunks, dtype=np.int64).reshape(-1, 2) if all_chunks else np.empty((0, 2), dtype=np.int64)

        try:
            meta = {
                'chunk_seconds': chunk_seconds,
                'files': [str(p) for p in file_paths],
                'sample_rates': sample_rates,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            np.save(str(chunks_path), chunks)
            print(f'saved chunk index cache: {meta_path.name} + {chunks_path.name}')
        except Exception as e:
            print(f'warning: could not save chunk cache: {e}')

        return file_paths, sample_rates, chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        file_idx = int(self.chunks[idx, 0])
        chunk_idx = int(self.chunks[idx, 1])
        file_path = self.file_paths[file_idx]
        source_hz = self.file_sample_rates[file_idx]
        chunk_frames = int(self.chunk_seconds * source_hz)
        frame_offset = chunk_idx * chunk_frames

        try:
            data, sr = torchaudio.load(
                str(file_path),
                frame_offset=frame_offset,
                num_frames=chunk_frames,
            )
        except Exception:
            if self.ignore_load_errors:
                return self[torch.randint(0, len(self), (1,)).item()]
            raise

        if data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)

        if data.shape[1] < chunk_frames:
            data = F.pad(data, (0, chunk_frames - data.shape[1]))

        if sr != self.target_sample_hz:
            data = resample(data, sr, self.target_sample_hz)

        data = int16_to_float32(float32_to_int16(data))
        data = data.squeeze(0).float()

        if self.seq_len_multiple_of is not None:
            data = curtail_to_multiple(data, self.seq_len_multiple_of)

        return data


# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        data = list(filter(lambda x: x is not None, data))
        if len(data) == 0:
            return () # empty batch

        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)


@beartype_jit
class SoundDatasetForPreprocessing(SoundDataset):
    def __init__(
        self,
        folder,
        pad_to_seconds: int = 10,
        target_lufs: Optional[float] = None,
        **kwargs
    ):
        self.pad_to_seconds = pad_to_seconds
        self.target_lufs = target_lufs
        super().__init__(folder=folder, **kwargs)

    def __getitem__(self, idx):
        """
        override getitem to return data + audio file name, and None in the case of a load error.
        returns data + audio file name for preprocessing. supports batching via dataloader collate.
        """
        try:
            file = self.files[idx]
            data, sample_hz = torchaudio.load(file)
        except:
            if self.ignore_load_errors:
                return None
            else:
                raise Exception(f'error loading file {file}')

        # Apply LUFS normalisation before resampling/cropping so that loudness
        # measurement is done at the original sample rate on the full clip.
        if self.target_lufs is not None:
            data_np = _apply_lufs_normalization_np(data.numpy(), sample_hz, self.target_lufs)
            data = torch.from_numpy(data_np)

        # if audio length is less than pad to seconds, repeat pad
        # else pad audio to nearest second
        # this is so at least one semantic token sequence can be extracted from the audio
        max_len = self.pad_to_seconds * sample_hz
        if data.size(1) < max_len:
            n_repeat = int(max_len / data.size(1))
            data = data.repeat(1, n_repeat)
            data = F.pad(
                data,
                (0, max_len - data.size(1)),
                mode="constant",
                value=0,
            )
        else:
            data = F.pad(data, (0, sample_hz - data.size(1) % sample_hz), 'constant', value=0)

        data = self.process_audio(data, sample_hz, pad_to_target_length=False)
        # Convert tensors to numpy so DataLoader workers communicate via pickle
        # (no shared memory needed → works on NFS/Lustre where mmap sharing fails)
        data = tuple(t.numpy() if isinstance(t, torch.Tensor) else t for t in data)

        return {
            'idx': idx + self.index_offset,
            'data': data,
            'file_path': str(file)
        }


@beartype_jit
class NpySoundDatasetForPreprocessing(SoundDataset):
    def __init__(
        self,
        folder,
        pad_to_seconds: int = 10,
        source_sample_hz: int = 48000,
        exts: Optional[List[str]] = None,
        **kwargs
    ):
        self.pad_to_seconds = pad_to_seconds
        self.source_sample_hz = source_sample_hz
        exts = default(exts, ['npy'])
        super().__init__(folder=folder, exts=exts, **kwargs)

    def __getitem__(self, idx):
        """
        override getitem to load waveform from .npy and return data + file name.
        returns data + audio file name for preprocessing from .npy input.
        """
        file = None
        try:
            file = self.files[idx]
            data = np.load(file)
        except:
            if self.ignore_load_errors:
                return None
            else:
                raise Exception(f'error loading npy file {file}')

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        else:
            data = torch.tensor(data)

        if data.ndim == 1:
            data = data.unsqueeze(0)
        elif data.ndim == 2:
            # allow both (1, T) and (T, 1)
            if data.shape[0] != 1 and data.shape[1] == 1:
                data = data.transpose(0, 1)
        else:
            raise ValueError(f'expected 1D or 2D array in {file}, got shape={tuple(data.shape)}')

        data = data.to(torch.float32)
        sample_hz = self.source_sample_hz

        # Keep behavior aligned with SoundDatasetForPreprocessing:
        # repeat-pad short audio to at least pad_to_seconds, then pad to nearest second.
        max_len = self.pad_to_seconds * sample_hz
        if data.size(1) < max_len:
            n_repeat = int(max_len / data.size(1))
            data = data.repeat(1, n_repeat)
            data = F.pad(
                data,
                (0, max_len - data.size(1)),
                mode="constant",
                value=0,
            )
        else:
            data = F.pad(data, (0, sample_hz - data.size(1) % sample_hz), 'constant', value=0)

        data = self.process_audio(data, sample_hz, pad_to_target_length=False)
        # Convert tensors to numpy so DataLoader workers communicate via pickle
        # (no shared memory needed → works on NFS/Lustre where mmap sharing fails)
        data = tuple(t.numpy() if isinstance(t, torch.Tensor) else t for t in data)

        return {
            'idx': idx + self.index_offset,
            'data': data,
            'file_path': str(file)
        }

def sound_preprocessing_collate_fn(data):
    data = list(filter(lambda x: x is not None, data))
    if len(data) == 0:
        return None

    file_paths = [item['file_path'] for item in data]
    # Plain Python list — no tensor, no shared memory, safe across NFS workers
    indices = [int(item['idx']) for item in data]

    data_groups = list(zip(*[item['data'] for item in data]))

    padded_data = []
    lengths = []
    for group in data_groups:
        group = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in group]
        # numpy int array — pickleable without shared memory
        lengths.append(np.array([t.shape[0] for t in group], dtype=np.int64))
        # numpy float array — pickleable without shared memory
        padded_data.append(pad_sequence(group, batch_first=True).numpy())

    return {
        'idx': indices,
        'data': tuple(padded_data),
        'lengths': tuple(lengths),
        'file_path': file_paths,
    }

def get_sound_preprocessing_dataloader(ds, **kwargs):
    kwargs.setdefault('batch_size', 1)
    return DataLoader(ds, collate_fn=sound_preprocessing_collate_fn, **kwargs)

@beartype_jit
class PreprocessedDataset(Dataset):
    def __init__(
        self,
        folder,
        stage: Literal['semantic', 'coarse', 'fine'],
        semantic_window_seconds: int=10,
        coarse_window_seconds: int=4,
        fine_window_seconds: int=2,
        semantic_steps_per_second=50,
        acoustic_steps_per_second=75,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        self.stage = stage
        self.semantic_window_seconds = semantic_window_seconds
        self.coarse_window_seconds = coarse_window_seconds
        self.fine_window_seconds = fine_window_seconds
        self.semantic_steps_per_second = semantic_steps_per_second
        self.acoustic_steps_per_second = acoustic_steps_per_second
        self.db_path = path / 'preprocessed.db'
        self.conn = None
        self.cursor = None
        self.conn_pid = None
        self.ids = self._load_or_build_id_index(path)

    def __len__(self):
        return len(self.ids)

    def _cache_path_for_ids(self, folder_path: Path):
        db_stat = self.db_path.stat()
        return folder_path / f'ids_cache_{int(db_stat.st_mtime)}_{db_stat.st_size}.npy'

    def _stable_cache_path_for_ids(self, folder_path: Path):
        return folder_path / 'ids_cache.npy'

    def _stable_cache_meta_path_for_ids(self, folder_path: Path):
        return folder_path / 'ids_cache.meta.json'

    def _load_stable_cache_if_valid(self, folder_path: Path):
        cache_path = self._stable_cache_path_for_ids(folder_path)
        meta_path = self._stable_cache_meta_path_for_ids(folder_path)
        if not cache_path.exists() or not meta_path.exists():
            print(f'stable id cache not found ({cache_path.name}, {meta_path.name})')
            return None

        try:
            db_stat = self.db_path.stat()
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if (
                int(meta.get('db_mtime', -1)) != int(db_stat.st_mtime) or
                int(meta.get('db_size', -1)) != int(db_stat.st_size)
            ):
                print(
                    f'stable id cache stale: db_mtime/size changed '
                    f'(db mtime={int(db_stat.st_mtime)} size={int(db_stat.st_size)})'
                )
                return None

            t0 = time.time()
            cached_ids = np.load(str(cache_path))
            print(
                f'loaded stable preprocessed id cache: '
                f'{len(cached_ids)} tracks in {time.time() - t0:.2f}s'
            )
            return cached_ids.astype(np.int64).tolist()
        except Exception as exc:
            print(f'warning: failed to load stable id cache ({cache_path}): {exc}')
            return None

    def _write_id_cache_with_meta(self, folder_path: Path, ids):
        cache_path = self._stable_cache_path_for_ids(folder_path)
        meta_path = self._stable_cache_meta_path_for_ids(folder_path)
        hashed_cache_path = self._cache_path_for_ids(folder_path)

        db_stat = self.db_path.stat()
        meta = {
            'db_mtime': int(db_stat.st_mtime),
            'db_size': int(db_stat.st_size),
            'count': int(len(ids)),
        }

        try:
            cache_tmp = cache_path.with_name(f'{cache_path.name}.tmp.{os.getpid()}')
            meta_tmp = meta_path.with_name(f'{meta_path.name}.tmp.{os.getpid()}')
            with open(cache_tmp, 'wb') as f:
                np.save(f, np.array(ids, dtype=np.int64))
            with open(meta_tmp, 'w') as f:
                json.dump(meta, f)
            os.replace(cache_tmp, cache_path)
            os.replace(meta_tmp, meta_path)
            print(f'wrote stable preprocessed id cache -> {cache_path}')
        except Exception as exc:
            print(f'warning: failed to write stable id cache ({cache_path}): {exc}')

        # keep old hashed cache format for compatibility with previous runs
        try:
            hashed_tmp = hashed_cache_path.with_name(f'{hashed_cache_path.name}.tmp.{os.getpid()}')
            with open(hashed_tmp, 'wb') as f:
                np.save(f, np.array(ids, dtype=np.int64))
            os.replace(hashed_tmp, hashed_cache_path)
            print(f'wrote preprocessed id cache -> {hashed_cache_path}')
        except Exception as exc:
            print(f'warning: failed to write id cache ({hashed_cache_path}): {exc}')

    def _load_ids_from_infer_cache_if_valid(self, folder_path: Path):
        db_stat = self.db_path.stat()
        suffix = f'_{int(db_stat.st_mtime)}_{db_stat.st_size}.tsv'
        candidates = sorted(folder_path.glob(f'db_entries_*{suffix}'))
        if not candidates:
            print(f'no infer cache found matching db signature (*{suffix})')
            return None
        print(f'found infer cache candidates: {len(candidates)}')

        # Validate against full row count so split-specific caches (e.g. *_test_*.tsv)
        # are not accidentally used for training.
        conn, cursor = init_sqlite(str(self.db_path), readonly=True, optimize_for_read=True)
        n_rows = int(cursor.execute('SELECT COUNT(*) FROM tokens').fetchone()[0])
        conn.close()
        if n_rows == 0:
            return []

        for tsv_path in candidates:
            try:
                t0 = time.time()
                ids = []
                with open(tsv_path, 'r') as f:
                    for line in f:
                        line = line.rstrip('\n')
                        if not line:
                            continue
                        idx_str, _track_id = line.split('\t', 1)
                        ids.append(int(idx_str))
                if len(ids) != n_rows:
                    print(
                        f'skipping infer cache {tsv_path.name}: '
                        f'rows={len(ids)} != db_rows={n_rows}'
                    )
                    continue

                # Ensure uniqueness / deterministic order
                if len(set(ids)) != n_rows:
                    print(f'skipping infer cache {tsv_path.name}: duplicate idx detected')
                    continue
                ids = sorted(ids)
                print(
                    f'loaded ids from infer cache: {tsv_path} '
                    f'({len(ids)} tracks in {time.time() - t0:.2f}s)'
                )
                self._write_id_cache_with_meta(folder_path, ids)
                return ids
            except Exception as exc:
                print(f'warning: failed to read infer cache ({tsv_path}): {exc}')

        return None

    def _load_or_build_id_index(self, folder_path: Path):
        print(f'initializing preprocessed id index from {self.db_path}')
        stable_ids = self._load_stable_cache_if_valid(folder_path)
        if stable_ids is not None:
            return stable_ids

        infer_cache_ids = self._load_ids_from_infer_cache_if_valid(folder_path)
        if infer_cache_ids is not None:
            return infer_cache_ids

        cache_path = self._cache_path_for_ids(folder_path)
        if cache_path.exists():
            t0 = time.time()
            cached_ids = np.load(str(cache_path))
            print(
                f'loaded hashed preprocessed id cache ({cache_path.name}): '
                f'{len(cached_ids)} tracks in {time.time() - t0:.2f}s'
            )
            return cached_ids.astype(np.int64).tolist()
        print(f'no local id cache found ({cache_path.name}); scanning preprocessed.db...')

        conn, cursor = init_sqlite(str(self.db_path), readonly=True, optimize_for_read=True)
        # Fast path: if idx is dense (min..max contiguous), build ids without full scan.
        min_idx, max_idx, n_rows = cursor.execute('SELECT MIN(idx), MAX(idx), COUNT(*) FROM tokens').fetchone()
        if n_rows == 0:
            conn.close()
            return []

        if min_idx is not None and max_idx is not None and (int(max_idx) - int(min_idx) + 1) == int(n_rows):
            ids = list(range(int(min_idx), int(max_idx) + 1))
            conn.close()
            print(f'preprocessed.db id range is contiguous: [{min_idx}, {max_idx}] (tracks={n_rows})')
            print('building id cache from contiguous range (no full id scan needed)')
            self._write_id_cache_with_meta(folder_path, ids)
            return ids

        cursor.execute('SELECT idx from tokens ORDER BY idx')

        ids = []
        scanned = 0
        t0 = time.time()
        fetch_size = 100000
        print('cache not usable; running full id scan from DB...')
        while True:
            rows = cursor.fetchmany(fetch_size)
            if not rows:
                break
            ids.extend(int(i[0]) for i in rows)
            scanned += len(rows)
            if scanned % 1000000 == 0:
                print(f'preprocessed.db id scan progress: scanned={scanned}')

        conn.close()
        print(f'preprocessed.db id scan complete: total={len(ids)} elapsed={time.time() - t0:.2f}s')
        self._write_id_cache_with_meta(folder_path, ids)

        return ids

    def _ensure_sqlite_connection(self):
        cur_pid = os.getpid()
        if self.conn is not None and self.cursor is not None and self.conn_pid == cur_pid:
            return

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass

        self.conn, self.cursor = init_sqlite(str(self.db_path), readonly=True, optimize_for_read=True)
        self.conn_pid = cur_pid

    def get_and_assert_audio_length_from_tokens(self, clap_token_ids=None, semantic_token_ids=None, coarse_token_ids=None, fine_token_ids=None):
        """compute original audio length from tokens and assert that all provided tokens have the same audio length"""
        clap_audio_length = clap_token_ids.shape[0] + self.semantic_window_seconds - 1 if exists(clap_token_ids) else None
        semantic_audio_length = (semantic_token_ids.shape[1] + 1) // self.semantic_steps_per_second if exists(semantic_token_ids) else None
        coarse_audio_length = coarse_token_ids.shape[1] // self.acoustic_steps_per_second if exists(coarse_token_ids) else None
        fine_audio_length = fine_token_ids.shape[1] // self.acoustic_steps_per_second if exists(fine_token_ids) else None

        lengths = [clap_audio_length, semantic_audio_length, coarse_audio_length, fine_audio_length]
        lengths = [int(l) for l in lengths if exists(l)]
        assert len(set(lengths)) == 1, 'audio lengths are not equal'

        return lengths[0]

    def get_clap_tokens(self, clap_token_ids, start_idx):
        """aggregate clap token over entire context with a sliding"""
        return clap_token_ids[start_idx].unsqueeze(0)

    def crop_semantic_tokens(self, semantic_token_ids, start_idx, end_idx):
        # with start_idx = 0, end_idx = 2, semantic_steps_per_second=50
        # we return semantic_token_ids[:, 0:99]
        return semantic_token_ids[:, start_idx * self.semantic_steps_per_second: end_idx * self.semantic_steps_per_second - 1]

    def crop_acoustic_tokens(self, coarse_or_fine_ids, start_idx, end_idx):
        # with start_idx = 0, end_idx = 2, coarse_steps_per_second=75
        # we return coarse_token_ids[:, 0:150]
        return coarse_or_fine_ids[:, start_idx * self.acoustic_steps_per_second: end_idx * self.acoustic_steps_per_second]

    def compute_crop_indices(self, audio_length, outside_window, inside_window=None):
        outside_start_idx = random.randint(0, audio_length - outside_window)
        outside_end_idx = outside_start_idx + outside_window

        if exists(inside_window):
            inside_start_idx = random.randint(outside_start_idx, outside_end_idx - inside_window)
            inside_end_idx = inside_start_idx + inside_window

            return outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx
        else:
            return outside_start_idx, outside_end_idx, None, None

    def __getitem__(self, idx):
        self._ensure_sqlite_connection()
        sqlite_idx = self.ids[idx]

        # load and crop the tokens
        # 1) select a outer crop (in whole seconds) from clap tokens based on semantic window, taking the average if spans multiple windows
        # 2) select an inner crop inside this range
        # 3) return cropped tokens

        # c - -
        #   c - -
        #     c - -
        #       c - -
        #         c - -
        #           c - -    <- clap tokens with window size 3 computed in a sliding window
        # s s s s s s s s    <- semantic tokens where 's' represents self.semantic_steps_per_second tokens
        # c c c c c c c c    <- coarse tokens where 'c' represents self.acoustic_steps_per_second tokens
        #           [    ]  <- 1st crop: start_idx=5, end_idx=8
        #           [   ]    <- 2nd crop: start_idx=5, end_idx=7
        # 0 1 2 3 4 5 6 7

        if self.stage == 'semantic':
            clap_token_ids, semantic_token_ids = self.cursor.execute(f'SELECT clap, semantic FROM tokens where idx = ?', (sqlite_idx,)).fetchone()
            clap_token_ids, semantic_token_ids = torch.from_numpy(clap_token_ids.astype(np.int32)), torch.from_numpy(semantic_token_ids.astype(np.int32))

            audio_length = self.get_and_assert_audio_length_from_tokens(clap_token_ids=clap_token_ids, semantic_token_ids=semantic_token_ids)

            outside_start_idx, outside_end_idx, _, _ = self.compute_crop_indices(audio_length, self.semantic_window_seconds)

            clap_token_ids = self.get_clap_tokens(clap_token_ids, outside_start_idx)
            semantic_token_ids = self.crop_semantic_tokens(semantic_token_ids, outside_start_idx, outside_end_idx)

            return (clap_token_ids, semantic_token_ids)
        elif self.stage == 'coarse':
            clap_token_ids, semantic_token_ids, coarse_token_ids = self.cursor.execute(f'SELECT clap, semantic, coarse FROM tokens where idx = ?', (sqlite_idx,)).fetchone()
            clap_token_ids, semantic_token_ids, coarse_token_ids = torch.from_numpy(clap_token_ids.astype(np.int32)), torch.from_numpy(semantic_token_ids.astype(np.int32)), torch.from_numpy(coarse_token_ids.astype(np.int32))

            audio_length = self.get_and_assert_audio_length_from_tokens(clap_token_ids=clap_token_ids, semantic_token_ids=semantic_token_ids, coarse_token_ids=coarse_token_ids)

            outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx = self.compute_crop_indices(audio_length, self.semantic_window_seconds, self.coarse_window_seconds)

            clap_token_ids = self.get_clap_tokens(clap_token_ids, outside_start_idx)
            semantic_token_ids = self.crop_semantic_tokens(semantic_token_ids, inside_start_idx, inside_end_idx)
            coarse_token_ids = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)

            return (clap_token_ids, semantic_token_ids, coarse_token_ids)
        elif self.stage == 'fine':
            clap_token_ids, coarse_token_ids, fine_token_ids = self.cursor.execute(f'SELECT clap, coarse, fine FROM tokens where idx = ?', (sqlite_idx,)).fetchone()
            clap_token_ids, coarse_token_ids, fine_token_ids = torch.from_numpy(clap_token_ids.astype(np.int32)), torch.from_numpy(coarse_token_ids.astype(np.int32)), torch.from_numpy(fine_token_ids.astype(np.int32))

            audio_length = self.get_and_assert_audio_length_from_tokens(clap_token_ids=clap_token_ids, coarse_token_ids=coarse_token_ids, fine_token_ids=fine_token_ids)

            outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx = self.compute_crop_indices(audio_length, self.semantic_window_seconds, self.fine_window_seconds)

            clap_token_ids = self.get_clap_tokens(clap_token_ids, outside_start_idx)
            coarse_token_ids = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)
            fine_token_ids = self.crop_acoustic_tokens(fine_token_ids, inside_start_idx, inside_end_idx)

            return (clap_token_ids, coarse_token_ids, fine_token_ids)
        else:
            raise Exception(f'invalid stage {self.stage}')

@collate_one_or_multiple_tensors
def concatenate_fn(batch):
    return torch.cat(batch, dim=0)

def get_preprocessed_dataloader(ds, **kwargs):
    collate_fn = concatenate_fn
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)
