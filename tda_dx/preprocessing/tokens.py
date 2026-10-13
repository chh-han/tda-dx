import gc
import csv
import io
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator, DistributedType
from beartype.door import is_bearable
from beartype.typing import Dict, List, Literal, Optional, Union
from beartype.vale import Is
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typing_extensions import Annotated

from tda_dx.models.clap_quantized import ClapQuantized
from tda_dx.data.audio_dataset import (SoundDatasetForPreprocessing,
                   NpySoundDatasetForPreprocessing,
                   get_sound_preprocessing_dataloader, init_sqlite)
from tda_dx.models.hubert_kmeans import HfHubertWithKmeans, learn_kmeans
from tda_dx.models.types import NeuralCodec, Wav2Vec
from tda_dx.models.musiclm import (get_or_compute_acoustic_token_ids,
                           get_or_compute_clap_token_ids,
                           get_or_compute_semantic_token_ids)
from tda_dx.training.optimizer import get_linear_scheduler, get_optimizer
from tda_dx.utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, beartype_jit, ceil_div,
                    copy_file_to_folder, default, eval_decorator, exists,
                    generate_mask_with_prob, get_embeds, gumbel_sample,
                    mask_out_after_eos_id, round_down_nearest_multiple, top_k)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)


def noop(*args, **kwargs):
    pass

def without_none(arr):
    return list(filter(lambda x: x is not None, arr))

@beartype_jit
class DataPreprocessor(nn.Module):
    """
    Class to preprocess audio files for the single stage transformer trainer.

    Load audio and compute:
        1) clap tokens for the entire audio file, computed for 10 second sliding windows with 1 second interval
        2) semantic tokens for the entire audio file
        3) coarse+fine tokens for the entire audio file
    Run this once over the dataset and then use the preprocessed data for training.
    """

    def __init__(
        self,
        *,
        num_coarse_quantizers=3,
        num_fine_quantizers: Optional[int] = None,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        max_audio_length_seconds=180,
        random_crop=True,
        clap_audio_length_seconds=10,
        semantic_audio_length_seconds=10,
        clap_batch_size=32,
        preprocess_batch_size=1,
        num_crops=1,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors=True,
        replace_existing=False,
        folder=None,
        input_audio_format: Literal['audio', 'npy'] = 'audio',
        npy_sample_hz: int = 48000,
        target_lufs: Optional[float] = None,
        results_folder='./data/fma_preprocessed',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
        resume_from_db: bool = True,
        gc_every_n_steps: int = 50,
        sqlite_cache_size_kib: int = 65536,
        output_format: Literal['db', 'memmap'] = 'db',
        metadata_folder: Optional[str] = None,
        tracks_csv_path: Optional[str] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner
        self.neural_codec = neural_codec
        self.num_coarse_quantizers = num_coarse_quantizers
        self.num_fine_quantizers = num_fine_quantizers
        self.max_audio_length_seconds = max_audio_length_seconds
        self.clap_audio_length_seconds = int(clap_audio_length_seconds)
        self.semantic_audio_length_seconds = int(semantic_audio_length_seconds)
        # TODO: allow a smaller clap length than semantic length, and average the clap embeddings over the time period as in the paper
        assert self.clap_audio_length_seconds == self.semantic_audio_length_seconds, 'clap window must be equal to semantic window for now'
        self.clap_batch_size = clap_batch_size
        self.preprocess_batch_size = max(1, int(preprocess_batch_size))
        self.num_crops = num_crops
        self.replace_existing = replace_existing
        self.resume_from_db = bool(resume_from_db) and not self.replace_existing
        self.gc_every_n_steps = max(0, int(gc_every_n_steps))
        self.sqlite_cache_size_kib = max(0, int(sqlite_cache_size_kib))
        self.use_dataset_idx = (self.num_crops == 1)
        self.output_format = str(output_format).lower()
        self.use_memmap_output = (self.output_format == 'memmap')
        assert self.output_format in ('db', 'memmap'), f'invalid output_format: {self.output_format}'
        self.metadata_folder = metadata_folder
        self.tracks_csv_path = tracks_csv_path
        if self.tracks_csv_path is None and exists(self.metadata_folder):
            maybe_tracks_csv = Path(self.metadata_folder) / 'tracks.csv'
            if maybe_tracks_csv.exists():
                self.tracks_csv_path = str(maybe_tracks_csv)
        self.track_id_to_split = {}

        self.register_buffer('steps', torch.Tensor([0]))

        # create dataset

        assert exists(wav2vec) and exists(audio_conditioner) and exists(neural_codec)

        self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic', 'raw_wave_for_acoustic')

        self.clap_sample_hz = audio_conditioner.sample_rate
        self.semantic_sample_hz = wav2vec.target_sample_hz
        self.acoustic_sample_hz = neural_codec.sample_rate
        self.semantic_output_hz = wav2vec.output_hz
        self.acoustic_output_hz = neural_codec.output_hz

        target_sample_hz = (self.clap_sample_hz, self.semantic_sample_hz, self.acoustic_sample_hz)

        normalize = (False, True, False)

        seq_len_multiple_of = (None, wav2vec.seq_len_multiple_of, None)

        data_max_length_seconds = (max_audio_length_seconds, max_audio_length_seconds, max_audio_length_seconds)

        assert exists(folder), 'audio folder must be passed in for preprocessing'

        if input_audio_format == 'npy':
            self.ds = NpySoundDatasetForPreprocessing(
                folder,
                source_sample_hz=npy_sample_hz,
                pad_to_seconds=self.semantic_audio_length_seconds,
                max_length_seconds=data_max_length_seconds,
                random_crop=random_crop,
                normalize=normalize,
                target_sample_hz=target_sample_hz,
                seq_len_multiple_of=seq_len_multiple_of,
                ignore_load_errors=ignore_load_errors,
                ignore_files=ignore_files,
            )
        else:
            self.ds = SoundDatasetForPreprocessing(
                folder,
                pad_to_seconds=self.semantic_audio_length_seconds,
                max_length_seconds=data_max_length_seconds,
                random_crop=random_crop,
                normalize=normalize,
                target_sample_hz=target_sample_hz,
                seq_len_multiple_of=seq_len_multiple_of,
                ignore_load_errors=ignore_load_errors,
                ignore_files=ignore_files,
                target_lufs=target_lufs,
            )


        self.track_id_to_split = self._load_track_id_to_split_map(self.tracks_csv_path)

        self.results_folder = Path(results_folder)
        if self.is_main:
            self.results_folder.mkdir(parents=True, exist_ok=True)

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

        self.conn = None
        self.cursor = None
        self.mmap_writers = {}
        self.mmap_token_shapes = None
        self.id_to_position = {}
        self.idx_to_path = {}
        self.idx_to_track_id = {}
        self.idx_to_split = {}
        self.idx_to_audio_seconds = {}
        self.next_insert_idx = 0
        self.memmap_pos = 0
        self.memmap_capacity = max(1, int(self.num_crops * len(self.ds)))

        if self.use_memmap_output:
            if self.resume_from_db:
                if self.is_main:
                    self.print('memmap mode: forcing resume_from_db=False')
                self.resume_from_db = False

            if self.is_distributed:
                raise NotImplementedError('memmap preprocessing currently supports single-process execution only')

            output_files = [
                self.results_folder / 'clap_tokens.npy',
                self.results_folder / 'semantic_tokens.npy',
                self.results_folder / 'coarse_tokens.npy',
                self.results_folder / 'fine_tokens.npy',
                self.results_folder / 'id_mapping.npy',
                self.results_folder / 'idx_to_path.json',
                self.results_folder / 'idx_to_track_id.json',
                self.results_folder / 'idx_to_split.json',
                self.results_folder / 'idx_to_audio_seconds.json',
                self.results_folder / 'metadata.json',
            ]

            if self.replace_existing:
                for file_path in output_files:
                    if file_path.exists():
                        file_path.unlink()
            else:
                existing = [p.name for p in output_files if p.exists()]
                if len(existing) > 0:
                    raise FileExistsError(
                        f'memmap outputs already exist in {self.results_folder}: {existing}. '
                        f'Use replace_existing=True to overwrite.'
                    )
        else:
            self.conn, self.cursor = init_sqlite(str(self.results_folder / 'preprocessed.db'))
            self.cursor.execute("CREATE TABLE IF NOT EXISTS tokens(idx integer primary key, path text, clap array, semantic array, coarse array, fine array)")
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS idx_metadata("
                "idx integer primary key, path text, track_id text, split text)"
            )
            if self.sqlite_cache_size_kib > 0:
                self.cursor.execute(f"PRAGMA cache_size=-{self.sqlite_cache_size_kib}")
            self.cursor.execute("PRAGMA temp_store=FILE")

        self.resume_start_idx = 0
        if self.resume_from_db and not self.use_memmap_output:
            self.cursor.execute("SELECT COALESCE(MAX(idx), -1) FROM tokens")
            max_existing_idx = int(self.cursor.fetchone()[0])
            self.resume_start_idx = max_existing_idx + 1

            if self.is_main:
                self.print(f'resume enabled: found max existing idx={self.resume_start_idx - 1}')

        self.next_insert_idx = self.resume_start_idx

        if self.resume_from_db and (not self.use_memmap_output) and self.use_dataset_idx and hasattr(self.ds, 'files'):
            ds_files = getattr(self.ds, 'files', [])
            if len(ds_files) > 0:
                all_paths = [str(file_path) for file_path in ds_files]
                existing_paths = self._existing_path_set_chunked(all_paths)
                if len(existing_paths) > 0:
                    self.ds.files = [file_path for file_path in ds_files if str(file_path) not in existing_paths]
                skipped = len(ds_files) - len(self.ds.files)
                if self.is_main:
                    self.print(f'resume enabled: filtered {skipped} existing files, remaining {len(self.ds.files)}')

        self.accelerator.wait_for_everyone()

        # dataloader

        dataloader_kwargs = dict(
            batch_size=self.preprocess_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            dataloader_kwargs['persistent_workers'] = True
        self.dl = get_sound_preprocessing_dataloader(self.ds, **dataloader_kwargs)

        # prepare

        (
            self.audio_conditioner,
            self.wav2vec,
            self.neural_codec,
            self.dl
        ) = self.accelerator.prepare(
            self.audio_conditioner,
            self.wav2vec,
            self.neural_codec,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)


    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def _existing_idx_set(self, indices):
        if self.cursor is None:
            return set()
        if len(indices) == 0:
            return set()
        placeholders = ','.join(['?'] * len(indices))
        query = f"SELECT idx FROM tokens WHERE idx IN ({placeholders})"
        rows = self.cursor.execute(query, tuple(indices)).fetchall()
        return {int(row[0]) for row in rows}

    def _existing_path_set(self, paths):
        if self.cursor is None:
            return set()
        if len(paths) == 0:
            return set()
        placeholders = ','.join(['?'] * len(paths))
        query = f"SELECT path FROM tokens WHERE path IN ({placeholders})"
        rows = self.cursor.execute(query, tuple(paths)).fetchall()
        return {str(row[0]) for row in rows}

    def _existing_path_set_chunked(self, paths, chunk_size=900):
        found = set()
        for start in range(0, len(paths), chunk_size):
            chunk = paths[start:start + chunk_size]
            found.update(self._existing_path_set(chunk))
        return found

    def _extract_track_id_from_path(self, path_str: str) -> str:
        stem = Path(path_str).stem
        try:
            return str(int(stem))
        except ValueError:
            for part in reversed(stem.split('_')):
                try:
                    return str(int(part))
                except ValueError:
                    continue
        return stem

    def _load_track_id_to_split_map(self, tracks_csv_path: Optional[str]):
        if not exists(tracks_csv_path):
            return {}

        tracks_csv_path = Path(tracks_csv_path)
        if not tracks_csv_path.exists():
            self.print(f'warning: tracks metadata not found at {tracks_csv_path}; split metadata will be omitted')
            return {}

        if tracks_csv_path.suffix.lower() == '.tsv':
            return self._load_split_map_from_tsv(tracks_csv_path)

        # FMA-style CSV: two-row header, set/split column
        mapping = {}
        with open(tracks_csv_path, newline='') as f:
            reader = csv.reader(f)
            header1 = next(reader)
            header2 = next(reader)

            split_idx = None
            for i, (a, b) in enumerate(zip(header1, header2)):
                if a.strip().lower() == 'set' and b.strip().lower() == 'split':
                    split_idx = i
                    break
            if split_idx is None:
                raise ValueError(f'set.split column not found in {tracks_csv_path}')

            for row in reader:
                if len(row) <= split_idx:
                    continue
                try:
                    track_id = str(int(row[0]))
                except ValueError:
                    continue
                mapping[track_id] = row[split_idx].strip().lower()

        self.print(f'loaded track split metadata map: {len(mapping)} tracks from {tracks_csv_path}')
        return mapping

    def _load_split_map_from_tsv(self, tsv_path: Path) -> dict:
        """Load track_id → split mapping from a tab-delimited file.

        Expects a single header row and at minimum a ``split`` column plus
        either a ``path`` column (relative audio path, stem used as track_id)
        or a ``track_id`` column (e.g. ``track_0000214``).
        Column names are matched case-insensitively.
        """
        mapping = {}
        with open(tsv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if not reader.fieldnames:
                raise ValueError(f'could not read header from {tsv_path}')

            field_map = {name.strip().lower(): name for name in reader.fieldnames}
            split_key = field_map.get('split')
            path_key = field_map.get('path')
            track_id_key = field_map.get('track_id')

            if split_key is None:
                raise ValueError(
                    f'"split" column not found in {tsv_path}; '
                    f'available columns: {list(field_map.keys())}'
                )
            if path_key is None and track_id_key is None:
                raise ValueError(
                    f'neither "path" nor "track_id" column found in {tsv_path}; '
                    f'available columns: {list(field_map.keys())}'
                )

            for row in reader:
                split_val = (row.get(split_key) or '').strip().lower()
                if not split_val:
                    continue

                track_id = None
                if path_key:
                    path_val = (row.get(path_key) or '').strip()
                    if path_val:
                        track_id = self._extract_track_id_from_path(path_val)
                if not track_id and track_id_key:
                    raw = (row.get(track_id_key) or '').strip()
                    if raw:
                        track_id = self._extract_track_id_from_path(raw)
                if not track_id:
                    continue

                mapping[track_id] = split_val

        self.print(f'loaded track split metadata map: {len(mapping)} tracks from {tsv_path}')
        return mapping

    def _resolve_track_split(self, source_path: str):
        track_id = self._extract_track_id_from_path(source_path)
        split_name = self.track_id_to_split.get(track_id)
        if len(self.track_id_to_split) > 0 and not exists(split_name):
            raise RuntimeError(
                f'track_id={track_id} from path={source_path} was not found in tracks.csv split metadata'
            )
        return track_id, (split_name if exists(split_name) else 'unknown')


    def _expected_memmap_token_shapes(self, clap_arr, semantic_arr, coarse_arr, fine_arr):
        target_audio_seconds = max(1, int(self.max_audio_length_seconds))
        clap_steps = max(1, target_audio_seconds - self.clap_audio_length_seconds + 1)
        semantic_steps = max(1, target_audio_seconds * self.semantic_output_hz - 1)
        acoustic_steps = max(1, target_audio_seconds * self.acoustic_output_hz)

        if semantic_arr.ndim != 2:
            raise RuntimeError(f'expected semantic rank=2, got shape={semantic_arr.shape}')
        if coarse_arr.ndim != 3:
            raise RuntimeError(f'expected coarse rank=3, got shape={coarse_arr.shape}')
        if fine_arr.ndim != 3:
            raise RuntimeError(f'expected fine rank=3, got shape={fine_arr.shape}')

        return {
            'clap': (clap_steps, *tuple(clap_arr.shape[1:])),
            'semantic': (1, semantic_steps),
            'coarse': (1, acoustic_steps, int(coarse_arr.shape[-1])),
            'fine': (1, acoustic_steps, int(fine_arr.shape[-1])),
        }

    def _pad_or_trim_to_shape(self, arr: np.ndarray, target_shape: tuple):
        if tuple(arr.shape) == tuple(target_shape):
            return arr
        if arr.ndim != len(target_shape):
            raise RuntimeError(
                f'rank mismatch while fitting array for memmap: source={arr.shape}, target={target_shape}'
            )

        out = np.zeros(target_shape, dtype=arr.dtype)
        slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(arr.shape, target_shape))
        out[slices] = arr[slices]
        return out

    def _ensure_memmap_writers(self, clap_arr, semantic_arr, coarse_arr, fine_arr):
        if len(self.mmap_writers) > 0:
            return

        token_shapes = self._expected_memmap_token_shapes(clap_arr, semantic_arr, coarse_arr, fine_arr)
        self.mmap_token_shapes = token_shapes

        if self.is_main:
            self.print(
                f'initializing memmap outputs with capacity={self.memmap_capacity}, '
                f'shapes={token_shapes}'
            )

        self.mmap_writers = {
            'clap': np.memmap(
                self.results_folder / 'clap_tokens.npy',
                dtype=np.uint16, mode='w+',
                shape=(self.memmap_capacity, *token_shapes['clap']),
            ),
            'semantic': np.memmap(
                self.results_folder / 'semantic_tokens.npy',
                dtype=np.uint16, mode='w+',
                shape=(self.memmap_capacity, *token_shapes['semantic']),
            ),
            'coarse': np.memmap(
                self.results_folder / 'coarse_tokens.npy',
                dtype=np.uint16, mode='w+',
                shape=(self.memmap_capacity, *token_shapes['coarse']),
            ),
            'fine': np.memmap(
                self.results_folder / 'fine_tokens.npy',
                dtype=np.uint16, mode='w+',
                shape=(self.memmap_capacity, *token_shapes['fine']),
            ),
        }

    def _write_memmap_row(self, idx, clap_arr, semantic_arr, coarse_arr, fine_arr, source_path: Optional[str] = None, split_name: Optional[str] = None, audio_seconds: Optional[int] = None):
        self._ensure_memmap_writers(clap_arr, semantic_arr, coarse_arr, fine_arr)
        if self.memmap_pos >= self.memmap_capacity:
            raise RuntimeError(
                f'memmap capacity exceeded: pos={self.memmap_pos}, capacity={self.memmap_capacity}. '
                f'Increase capacity estimation before writing.'
            )

        clap_arr = self._pad_or_trim_to_shape(clap_arr, self.mmap_token_shapes['clap'])
        semantic_arr = self._pad_or_trim_to_shape(semantic_arr, self.mmap_token_shapes['semantic'])
        coarse_arr = self._pad_or_trim_to_shape(coarse_arr, self.mmap_token_shapes['coarse'])
        fine_arr = self._pad_or_trim_to_shape(fine_arr, self.mmap_token_shapes['fine'])

        pos = self.memmap_pos
        self.mmap_writers['clap'][pos] = clap_arr
        self.mmap_writers['semantic'][pos] = semantic_arr
        self.mmap_writers['coarse'][pos] = coarse_arr
        self.mmap_writers['fine'][pos] = fine_arr
        idx_int = int(idx)
        self.id_to_position[idx_int] = int(pos)
        if source_path is not None:
            source_path = str(source_path)
            self.idx_to_path[idx_int] = source_path
            self.idx_to_track_id[idx_int] = self._extract_track_id_from_path(source_path)
        if split_name is not None:
            self.idx_to_split[idx_int] = str(split_name)
        if audio_seconds is not None:
            self.idx_to_audio_seconds[idx_int] = int(audio_seconds)
        self.memmap_pos += 1

    def _finalize_memmap_outputs(self):
        if len(self.mmap_writers) == 0:
            raise RuntimeError('no memmap rows were written; cannot finalize memmap outputs')

        for mmap_arr in self.mmap_writers.values():
            mmap_arr.flush()

        metadata = {
            'total_count': int(self.memmap_pos),
            'clap_shape': list(self.mmap_token_shapes['clap']),
            'semantic_shape': list(self.mmap_token_shapes['semantic']),
            'coarse_shape': list(self.mmap_token_shapes['coarse']),
            'fine_shape': list(self.mmap_token_shapes['fine']),
            'dtype': 'uint16',
            'source': 'direct_preprocess_memmap',
            'max_audio_length_seconds': int(self.max_audio_length_seconds),
            'created_at_unix': int(time.time()),
            'idx_to_path_file': 'idx_to_path.json',
            'idx_to_track_id_file': 'idx_to_track_id.json',
            'idx_to_split_file': 'idx_to_split.json',
            'idx_to_audio_seconds_file': 'idx_to_audio_seconds.json',
        }

        np.save(self.results_folder / 'id_mapping.npy', self.id_to_position)
        with open(self.results_folder / 'idx_to_path.json', 'w') as f:
            json.dump({str(k): v for k, v in self.idx_to_path.items()}, f)
        with open(self.results_folder / 'idx_to_track_id.json', 'w') as f:
            json.dump({str(k): v for k, v in self.idx_to_track_id.items()}, f)
        with open(self.results_folder / 'idx_to_split.json', 'w') as f:
            json.dump({str(k): v for k, v in self.idx_to_split.items()}, f)
        with open(self.results_folder / 'idx_to_audio_seconds.json', 'w') as f:
            json.dump({str(k): int(v) for k, v in self.idx_to_audio_seconds.items()}, f)
        with open(self.results_folder / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.is_main:
            self.print(
                f'memmap preprocessing complete: written={self.memmap_pos}, '
                f'capacity={self.memmap_capacity}, folder={self.results_folder}'
            )

    def generate_tokens_from_batch(self, raw_wave_for_clap, raw_wave_for_semantic, raw_wave_for_acoustic):
        # split each waveform into clap windows, then batch windows across files for CLAP inference
        clap_split = raw_wave_for_clap.unfold(
            -1,
            self.audio_conditioner.sample_rate * self.clap_audio_length_seconds,
            self.audio_conditioner.sample_rate,
        )

        bs, num_windows, window_len = clap_split.shape
        clap_split = clap_split.reshape(bs * num_windows, window_len)

        window_batch_size = self.clap_batch_size
        clap_token_ids_all = []
        for i in range(0, clap_split.shape[0], window_batch_size):
            batch = clap_split[i:i + window_batch_size, :]
            clap_token_ids = get_or_compute_clap_token_ids(
                None,
                self.accelerator.unwrap_model(self.audio_conditioner),
                batch,
                None,
            )
            clap_token_ids_all.append(clap_token_ids)

        clap_token_ids = torch.cat(clap_token_ids_all, dim=0)
        clap_token_ids = clap_token_ids.reshape(bs, num_windows, *clap_token_ids.shape[1:])

        semantic_token_ids = get_or_compute_semantic_token_ids(
            None,
            raw_wave_for_semantic,
            self.accelerator.unwrap_model(self.wav2vec),
        )

        coarse_token_ids, fine_token_ids = get_or_compute_acoustic_token_ids(
            None,
            None,
            raw_wave_for_acoustic,
            self.accelerator.unwrap_model(self.neural_codec),
            self.num_coarse_quantizers,
            self.num_fine_quantizers,
        )

        return clap_token_ids, semantic_token_ids, (coarse_token_ids, fine_token_ids)

    def process(self, log_fn=noop):
        iters = math.ceil(
            self.num_crops * len(self.ds) /
            (self.accelerator.num_processes * self.preprocess_batch_size)
        )

        for step in tqdm(range(iters), desc='processing data', mininterval=5):
            inputs = next(self.dl_iter)
            if not exists(inputs):
                self.steps += 1
                continue

            local_batch = len(inputs['file_path'])

            if self.use_dataset_idx and exists(inputs.get('idx', None)):
                candidate_indices = [int(inputs['idx'][i]) for i in range(local_batch)]
            else:
                base_idx = (
                    step * self.accelerator.num_processes * self.preprocess_batch_size +
                    self.accelerator.process_index * self.preprocess_batch_size
                )
                candidate_indices = [int(base_idx + i) for i in range(local_batch)]

            existing_idx = set()
            existing_paths = set()
            if not self.replace_existing and not self.use_memmap_output:
                if self.resume_from_db:
                    existing_paths = self._existing_path_set(inputs['file_path'])
                    if len(existing_paths) == len(inputs['file_path']):
                        self.steps += 1
                        if self.gc_every_n_steps > 0 and (step + 1) % self.gc_every_n_steps == 0:
                            gc.collect()
                            self.cursor.execute('PRAGMA shrink_memory')
                        continue
                else:
                    existing_idx = self._existing_idx_set(candidate_indices)
                    if len(existing_idx) == len(candidate_indices):
                        self.steps += 1
                        if self.gc_every_n_steps > 0 and (step + 1) % self.gc_every_n_steps == 0:
                            gc.collect()
                            self.cursor.execute('PRAGMA shrink_memory')
                        continue

            data_kwargs = dict(zip(self.ds_fields, inputs['data']))
            # Convert numpy arrays (returned by collate to avoid NFS shared-memory issues)
            # to tensors and move to the inference device.
            data_kwargs = {
                k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v).to(self.device)
                for k, v in data_kwargs.items()
            }
            with torch.inference_mode():
                clap_token_ids, semantic_token_ids, (coarse_token_ids, fine_token_ids) = self.generate_tokens_from_batch(**data_kwargs)

            lengths = inputs.get('lengths', None)
            pending_rows = []
            pending_meta_rows = []

            for i, candidate_idx in enumerate(candidate_indices):
                if not self.replace_existing and not self.use_memmap_output:
                    if self.resume_from_db and inputs['file_path'][i] in existing_paths:
                        continue
                    if (not self.resume_from_db) and candidate_idx in existing_idx:
                        continue

                if self.use_memmap_output:
                    idx = int(self.next_insert_idx)
                    self.next_insert_idx += 1
                elif self.resume_from_db and not self.replace_existing:
                    idx = int(self.next_insert_idx)
                    self.next_insert_idx += 1
                else:
                    idx = int(candidate_idx)

                clap_tokens = clap_token_ids[i]
                semantic_tokens = semantic_token_ids[i]
                coarse_tokens = coarse_token_ids[i]
                fine_tokens = fine_token_ids[i]
                audio_seconds = max(1, int(clap_tokens.shape[0]) + self.clap_audio_length_seconds - 1)

                if exists(lengths):
                    clap_len = int(lengths[0][i].item())
                    semantic_len = int(lengths[1][i].item())
                    acoustic_len = int(lengths[2][i].item())

                    clap_seconds = clap_len // self.clap_sample_hz
                    semantic_seconds = semantic_len // self.semantic_sample_hz
                    acoustic_seconds = acoustic_len // self.acoustic_sample_hz
                    audio_seconds = min(clap_seconds, semantic_seconds, acoustic_seconds)

                    clap_steps = max(1, audio_seconds - self.clap_audio_length_seconds + 1)
                    semantic_steps = max(1, audio_seconds * self.semantic_output_hz - 1)
                    acoustic_steps = max(1, audio_seconds * self.acoustic_output_hz)

                    clap_tokens = clap_tokens[:clap_steps]
                    semantic_tokens = semantic_tokens[:semantic_steps]
                    coarse_tokens = coarse_tokens[:acoustic_steps]
                    fine_tokens = fine_tokens[:acoustic_steps]

                # Keep shape conventions consistent with legacy PreprocessedDataset:
                # semantic: (1, T), acoustic: (1, T, Q)
                if semantic_tokens.ndim == 1:
                    semantic_tokens = semantic_tokens.unsqueeze(0)
                if coarse_tokens.ndim == 2:
                    coarse_tokens = coarse_tokens.unsqueeze(0)
                if fine_tokens.ndim == 2:
                    fine_tokens = fine_tokens.unsqueeze(0)

                clap_arr = clap_tokens.detach().cpu().numpy().astype(np.uint16)
                semantic_arr = semantic_tokens.detach().cpu().numpy().astype(np.uint16)
                coarse_arr = coarse_tokens.detach().cpu().numpy().astype(np.uint16)
                fine_arr = fine_tokens.detach().cpu().numpy().astype(np.uint16)
                source_path = str(inputs['file_path'][i])
                track_id, split_name = self._resolve_track_split(source_path)

                pending_rows.append(
                    (idx, source_path, clap_arr, semantic_arr, coarse_arr, fine_arr, audio_seconds)
                )
                pending_meta_rows.append((idx, source_path, track_id, split_name))

            if pending_rows:
                if self.use_memmap_output:
                    for (idx, _path, clap_arr, semantic_arr, coarse_arr, fine_arr, audio_seconds), (_midx, _mpath, _mtrack_id, split_name) in zip(pending_rows, pending_meta_rows):
                        self._write_memmap_row(
                            idx,
                            clap_arr,
                            semantic_arr,
                            coarse_arr,
                            fine_arr,
                            source_path=_path,
                            split_name=split_name,
                            audio_seconds=audio_seconds,
                        )
                else:
                    db_rows = [(idx, path, clap, semantic, coarse, fine) for (idx, path, clap, semantic, coarse, fine, _audio_seconds) in pending_rows]
                    self.cursor.executemany(
                        "INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?)",
                        db_rows,
                    )
                    self.cursor.executemany(
                        "INSERT OR REPLACE INTO idx_metadata VALUES (?, ?, ?, ?)",
                        pending_meta_rows,
                    )
                    self.conn.commit()

            del pending_rows
            del pending_meta_rows
            del clap_token_ids, semantic_token_ids, coarse_token_ids, fine_token_ids

            self.steps += 1

            if self.gc_every_n_steps > 0 and (step + 1) % self.gc_every_n_steps == 0:
                gc.collect()
                if self.cursor is not None:
                    self.cursor.execute('PRAGMA shrink_memory')

        if self.use_memmap_output:
            self._finalize_memmap_outputs()
        else:
            self.conn.commit()
        self.print('processing complete')
