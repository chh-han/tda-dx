import json
import random
import time
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _concatenate_fn(batch):
    outputs = []
    for group in zip(*batch):
        outputs.append(torch.cat(group, dim=0))
    return tuple(outputs)


class MemmapPreprocessedDataset(Dataset):
    """Ultra-fast dataset based on memory-mapped files"""

    def __init__(
        self,
        folder: str,
        stage: Literal['semantic', 'coarse', 'fine'],
        semantic_window_seconds: int = 10,
        coarse_window_seconds: int = 4,
        fine_window_seconds: int = 2,
        semantic_steps_per_second: int = 50,
        acoustic_steps_per_second: int = 75,
        exhaustive_traversal: bool = True,
        exhaustive_stride: int = 1,
        jitter_seconds: float = 0.0,
    ):
        super().__init__()

        self.stage = stage
        self.semantic_window_seconds = semantic_window_seconds
        self.coarse_window_seconds = coarse_window_seconds
        self.fine_window_seconds = fine_window_seconds
        self.semantic_steps_per_second = semantic_steps_per_second
        self.acoustic_steps_per_second = acoustic_steps_per_second
        self.exhaustive_traversal = bool(exhaustive_traversal)
        self.exhaustive_stride = max(1, int(exhaustive_stride))
        self.jitter_seconds = max(0.0, float(jitter_seconds))

        folder_path = Path(folder)
        assert folder_path.exists(), f'Folder does not exist: {folder}'

        # Load metadata
        with open(folder_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load index mapping
        raw_mapping = np.load(folder_path / 'id_mapping.npy', allow_pickle=True).item()
        self.id_mapping = {int(k): int(v) for k, v in raw_mapping.items()}
        # Keep dataset axis deterministic across runs/filesystems.
        self.indices = sorted(self.id_mapping.keys())
        self.num_tracks = len(self.indices)

        # Load memory-mapped files (lazy loading)
        print(f"Loading memory-mapped files from {folder_path}")
        t0 = time.time()

        self.clap_mmap = np.memmap(
            folder_path / 'clap_tokens.npy',
            dtype=np.uint16,
            mode='r',
            shape=(self.metadata['total_count'], *self.metadata['clap_shape']),
        )

        self.semantic_mmap = np.memmap(
            folder_path / 'semantic_tokens.npy',
            dtype=np.uint16,
            mode='r',
            shape=(self.metadata['total_count'], *self.metadata['semantic_shape']),
        )

        if stage in ['coarse', 'fine']:
            self.coarse_mmap = np.memmap(
                folder_path / 'coarse_tokens.npy',
                dtype=np.uint16,
                mode='r',
                shape=(self.metadata['total_count'], *self.metadata['coarse_shape']),
            )

        if stage == 'fine':
            self.fine_mmap = np.memmap(
                folder_path / 'fine_tokens.npy',
                dtype=np.uint16,
                mode='r',
                shape=(self.metadata['total_count'], *self.metadata['fine_shape']),
            )

        self.audio_length_seconds = self._infer_audio_length_seconds_from_metadata()
        self.idx_to_audio_seconds = self._load_idx_to_audio_seconds(folder_path)
        self.track_audio_lengths = [self._resolve_audio_length_for_idx(idx) for idx in self.indices]

        if self.exhaustive_traversal:
            window = self._window_seconds_for_stage()
            stride = self.exhaustive_stride
            self.track_start_positions = []
            for length in self.track_audio_lengths:
                max_start = max(0, int(length) - int(window))
                positions = list(range(0, max_start + 1, stride))
                if len(positions) == 0:
                    positions = [0]
                self.track_start_positions.append(positions)
            self.track_start_counts = [len(p) for p in self.track_start_positions]
            self.track_start_offsets = np.zeros(self.num_tracks + 1, dtype=np.int64)
            if self.num_tracks > 0:
                self.track_start_offsets[1:] = np.cumsum(np.asarray(self.track_start_counts, dtype=np.int64))
            self.total_exhaustive_samples = int(self.track_start_offsets[-1])
        else:
            self.track_start_counts = None
            self.track_start_offsets = None
            self.total_exhaustive_samples = self.num_tracks

        if self.exhaustive_traversal:
            jitter_info = f', jitter=±{self.jitter_seconds}s' if self.jitter_seconds > 0 else ''
            print(
                f"Exhaustive traversal: stride={self.exhaustive_stride}s, "
                f"window={self._window_seconds_for_stage()}s{jitter_info}, "
                f"samples={self.total_exhaustive_samples:,} "
                f"({self.total_exhaustive_samples / max(1, self.num_tracks):.1f}/track)"
            )
        print(f"Loaded in {time.time() - t0:.2f}s")

    def __len__(self):
        if self.exhaustive_traversal:
            return self.total_exhaustive_samples
        return self.num_tracks

    def _window_seconds_for_stage(self) -> int:
        if self.stage == 'semantic':
            return int(self.semantic_window_seconds)
        if self.stage == 'coarse':
            return int(self.coarse_window_seconds)
        if self.stage == 'fine':
            return int(self.fine_window_seconds)
        raise ValueError(f'Invalid stage: {self.stage}')

    def _infer_audio_length_seconds_from_metadata(self) -> int:
        clap_shape = self.metadata.get('clap_shape')
        semantic_shape = self.metadata.get('semantic_shape')
        coarse_shape = self.metadata.get('coarse_shape')
        fine_shape = self.metadata.get('fine_shape')

        lengths = []
        if isinstance(clap_shape, list) and len(clap_shape) > 0:
            lengths.append(int(clap_shape[0]) + int(self.semantic_window_seconds) - 1)
        if isinstance(semantic_shape, list) and len(semantic_shape) > 0:
            semantic_tokens = int(semantic_shape[-1])
            lengths.append((semantic_tokens + 1) // int(self.semantic_steps_per_second))
        if isinstance(coarse_shape, list) and len(coarse_shape) > 0:
            coarse_tokens = int(coarse_shape[1] if len(coarse_shape) > 1 else coarse_shape[0])
            lengths.append(coarse_tokens // int(self.acoustic_steps_per_second))
        if isinstance(fine_shape, list) and len(fine_shape) > 0:
            fine_tokens = int(fine_shape[1] if len(fine_shape) > 1 else fine_shape[0])
            lengths.append(fine_tokens // int(self.acoustic_steps_per_second))

        lengths = [int(v) for v in lengths if int(v) > 0]
        if len(lengths) == 0:
            raise RuntimeError('Unable to infer audio length from metadata.json shapes')
        if len(set(lengths)) != 1:
            raise RuntimeError(f'Inconsistent audio lengths inferred from metadata: {lengths}')
        return lengths[0]

    def _load_idx_to_audio_seconds(self, folder: Path) -> Dict[int, int]:
        filename = self.metadata.get('idx_to_audio_seconds_file', 'idx_to_audio_seconds.json')
        path = folder / filename
        if not path.exists():
            return {}

        with open(path, 'r') as f:
            raw = json.load(f)

        mapping = {}
        for k, v in raw.items():
            try:
                idx = int(k)
                seconds = int(v)
            except Exception:
                continue
            if seconds > 0:
                mapping[idx] = seconds
        return mapping

    def _resolve_audio_length_for_idx(self, sqlite_idx: int) -> int:
        mapped = self.idx_to_audio_seconds.get(int(sqlite_idx), self.audio_length_seconds)
        length = max(1, int(mapped))
        return min(length, int(self.audio_length_seconds))

    def _ensure_rank2_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Legacy preprocessed pipeline expects semantic as (1, T).
        if token_ids.ndim == 1:
            return token_ids.unsqueeze(0)
        return token_ids

    def _ensure_rank3_acoustic(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Legacy preprocessed pipeline expects acoustic as (1, T, Q).
        if token_ids.ndim == 2:
            return token_ids.unsqueeze(0)
        return token_ids

    def get_clap_tokens(self, clap_token_ids, start_second):
        """Extract CLAP tokens (same as existing logic)"""
        return clap_token_ids[start_second].unsqueeze(0)

    def crop_semantic_tokens(self, semantic_token_ids, start_second, end_second):
        """Crop semantic tokens (same as existing logic)"""
        start_idx = start_second * self.semantic_steps_per_second
        end_idx = end_second * self.semantic_steps_per_second - 1
        return semantic_token_ids[:, start_idx:end_idx]

    def crop_acoustic_tokens(self, acoustic_token_ids, start_second, end_second):
        """Crop acoustic tokens (same as existing logic)"""
        start_idx = start_second * self.acoustic_steps_per_second
        end_idx = end_second * self.acoustic_steps_per_second
        return acoustic_token_ids[:, start_idx:end_idx]

    # ── Token-level crop methods (used with jitter) ──────────────────────

    def crop_semantic_tokens_by_idx(self, semantic_token_ids, start_token, num_tokens):
        """Crop semantic tokens by absolute token indices."""
        return semantic_token_ids[:, start_token:start_token + num_tokens]

    def crop_acoustic_tokens_by_idx(self, acoustic_token_ids, start_token, num_tokens):
        """Crop acoustic tokens by absolute token indices."""
        return acoustic_token_ids[:, start_token:start_token + num_tokens]

    def _apply_jitter(self, anchor_second, window_seconds, audio_length, hz, is_semantic=False):
        """Apply token-level jitter to an anchor start position.

        Returns (token_start, token_count, clap_second).
        - token_start: first token index for the crop
        - token_count: number of tokens in the crop (fixed per window)
        - clap_second: integer second for CLAP lookup (rounded from jittered position)
        """
        if is_semantic:
            # semantic window yields (window * hz - 1) tokens
            token_count = window_seconds * hz - 1
            total_tokens = audio_length * hz - 1
        else:
            token_count = window_seconds * hz
            total_tokens = audio_length * hz

        base_token = anchor_second * hz
        max_token_start = max(0, total_tokens - token_count)

        if self.jitter_seconds > 0:
            jitter_tokens = int(self.jitter_seconds * hz)
            jitter = random.randint(-jitter_tokens, jitter_tokens)
            token_start = base_token + jitter
        else:
            token_start = base_token

        token_start = max(0, min(token_start, max_token_start))

        # CLAP index: round jittered position to nearest integer second
        jittered_second = token_start / hz
        max_clap_second = max(0, audio_length - self.semantic_window_seconds)
        clap_second = max(0, min(round(jittered_second), max_clap_second))

        return token_start, token_count, clap_second

    def get_and_assert_audio_length_from_tokens(self, **kwargs):
        """Compute audio length (same as existing logic)"""
        clap_token_ids = kwargs.get('clap_token_ids')
        semantic_token_ids = kwargs.get('semantic_token_ids')
        coarse_token_ids = kwargs.get('coarse_token_ids')
        fine_token_ids = kwargs.get('fine_token_ids')

        clap_audio_length = clap_token_ids.shape[0] + self.semantic_window_seconds - 1 if clap_token_ids is not None else None
        semantic_audio_length = (semantic_token_ids.shape[1] + 1) // self.semantic_steps_per_second if semantic_token_ids is not None else None
        coarse_audio_length = coarse_token_ids.shape[1] // self.acoustic_steps_per_second if coarse_token_ids is not None else None
        fine_audio_length = fine_token_ids.shape[1] // self.acoustic_steps_per_second if fine_token_ids is not None else None

        lengths = [clap_audio_length, semantic_audio_length, coarse_audio_length, fine_audio_length]
        lengths = [int(l) for l in lengths if l is not None]
        assert len(set(lengths)) == 1, 'audio lengths are not equal'
        return lengths[0]

    def compute_crop_indices(self, audio_length, outside_window, inside_window=None):
        """Compute crop indices (same as existing logic)"""
        outside_start_idx = random.randint(0, max(0, audio_length - outside_window))
        outside_end_idx = outside_start_idx + outside_window

        if inside_window is not None:
            inside_start_idx = random.randint(
                outside_start_idx,
                max(outside_start_idx, outside_end_idx - inside_window),
            )
            inside_end_idx = inside_start_idx + inside_window
            return outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx
        return outside_start_idx, outside_end_idx, None, None

    def get_sqlite_idx_for_sample(self, sample_idx: int) -> int:
        sqlite_idx, _ = self._resolve_sample(sample_idx)
        return int(sqlite_idx)

    def _resolve_sample(self, sample_idx: int):
        if self.exhaustive_traversal:
            sample_idx = int(sample_idx)
            if sample_idx < 0 or sample_idx >= self.total_exhaustive_samples:
                raise IndexError(f'sample_idx out of range: {sample_idx}')
            track_pos = int(np.searchsorted(self.track_start_offsets, sample_idx, side='right') - 1)
            inner_offset = int(sample_idx - self.track_start_offsets[track_pos])
            # Map offset back to the actual start second via the positions list
            inner_start = int(self.track_start_positions[track_pos][inner_offset])
            sqlite_idx = int(self.indices[track_pos])
        else:
            sqlite_idx = int(self.indices[int(sample_idx)])
            inner_start = None

        return sqlite_idx, inner_start

    def __getitem__(self, idx):
        """Ultra-fast data loading - no SQLite queries!"""

        # 1) Find the actual position using index mapping
        sqlite_idx, forced_inner_start = self._resolve_sample(idx)
        pos = self.id_mapping[sqlite_idx]

        # 2) Read directly from memory-mapped files (extremely fast)
        clap_token_ids = torch.from_numpy(self.clap_mmap[pos].astype(np.int32))
        semantic_token_ids = torch.from_numpy(self.semantic_mmap[pos].astype(np.int32))
        semantic_token_ids = self._ensure_rank2_semantic(semantic_token_ids)

        inferred_audio_length = self.get_and_assert_audio_length_from_tokens(
            clap_token_ids=clap_token_ids,
            semantic_token_ids=semantic_token_ids,
        )
        audio_length = min(int(inferred_audio_length), self._resolve_audio_length_for_idx(sqlite_idx))

        # 3) Process by stage
        use_jitter = self.exhaustive_traversal and self.jitter_seconds > 0

        if self.stage == 'semantic':
            if self.exhaustive_traversal:
                if use_jitter:
                    sem_hz = self.semantic_steps_per_second
                    tok_start, tok_count, clap_sec = self._apply_jitter(
                        forced_inner_start, self.semantic_window_seconds,
                        audio_length, sem_hz, is_semantic=True,
                    )
                    clap_tokens = self.get_clap_tokens(clap_token_ids, clap_sec)
                    semantic_tokens = self.crop_semantic_tokens_by_idx(semantic_token_ids, tok_start, tok_count)
                else:
                    outside_start_idx = int(forced_inner_start)
                    outside_end_idx = outside_start_idx + self.semantic_window_seconds
                    clap_tokens = self.get_clap_tokens(clap_token_ids, outside_start_idx)
                    semantic_tokens = self.crop_semantic_tokens(semantic_token_ids, outside_start_idx, outside_end_idx)
            else:
                outside_start_idx, outside_end_idx, _, _ = self.compute_crop_indices(
                    audio_length,
                    self.semantic_window_seconds,
                )
                clap_tokens = self.get_clap_tokens(clap_token_ids, outside_start_idx)
                semantic_tokens = self.crop_semantic_tokens(semantic_token_ids, outside_start_idx, outside_end_idx)

            return (clap_tokens, semantic_tokens)

        if self.stage == 'coarse':
            coarse_token_ids = torch.from_numpy(self.coarse_mmap[pos].astype(np.int32))
            coarse_token_ids = self._ensure_rank3_acoustic(coarse_token_ids)

            inferred_audio_length = self.get_and_assert_audio_length_from_tokens(
                clap_token_ids=clap_token_ids,
                semantic_token_ids=semantic_token_ids,
                coarse_token_ids=coarse_token_ids,
            )
            audio_length = min(int(inferred_audio_length), self._resolve_audio_length_for_idx(sqlite_idx))

            if self.exhaustive_traversal:
                if use_jitter:
                    ac_hz = self.acoustic_steps_per_second
                    sem_hz = self.semantic_steps_per_second
                    # Inside window (coarse): token-level jitter
                    tok_start_ac, tok_count_ac, clap_sec = self._apply_jitter(
                        forced_inner_start, self.coarse_window_seconds,
                        audio_length, ac_hz, is_semantic=False,
                    )
                    # Semantic tokens for the same inside region
                    tok_start_sem, tok_count_sem, _ = self._apply_jitter(
                        forced_inner_start, self.coarse_window_seconds,
                        audio_length, sem_hz, is_semantic=True,
                    )
                    clap_tokens = self.get_clap_tokens(clap_token_ids, clap_sec)
                    semantic_tokens = self.crop_semantic_tokens_by_idx(semantic_token_ids, tok_start_sem, tok_count_sem)
                    coarse_tokens = self.crop_acoustic_tokens_by_idx(coarse_token_ids, tok_start_ac, tok_count_ac)
                else:
                    inside_start_idx = int(forced_inner_start)
                    inside_end_idx = inside_start_idx + self.coarse_window_seconds
                    max_clap_start = max(0, int(audio_length) - int(self.semantic_window_seconds))
                    outside_start_idx = min(inside_start_idx, max_clap_start)
                    clap_tokens = self.get_clap_tokens(clap_token_ids, outside_start_idx)
                    semantic_tokens = self.crop_semantic_tokens(semantic_token_ids, inside_start_idx, inside_end_idx)
                    coarse_tokens = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)
            else:
                outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx = self.compute_crop_indices(
                    audio_length,
                    self.semantic_window_seconds,
                    self.coarse_window_seconds,
                )
                clap_tokens = self.get_clap_tokens(clap_token_ids, outside_start_idx)
                semantic_tokens = self.crop_semantic_tokens(semantic_token_ids, inside_start_idx, inside_end_idx)
                coarse_tokens = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)

            return (clap_tokens, semantic_tokens, coarse_tokens)

        if self.stage == 'fine':
            coarse_token_ids = torch.from_numpy(self.coarse_mmap[pos].astype(np.int32))
            fine_token_ids = torch.from_numpy(self.fine_mmap[pos].astype(np.int32))
            coarse_token_ids = self._ensure_rank3_acoustic(coarse_token_ids)
            fine_token_ids = self._ensure_rank3_acoustic(fine_token_ids)

            inferred_audio_length = self.get_and_assert_audio_length_from_tokens(
                clap_token_ids=clap_token_ids,
                coarse_token_ids=coarse_token_ids,
                fine_token_ids=fine_token_ids,
            )
            audio_length = min(int(inferred_audio_length), self._resolve_audio_length_for_idx(sqlite_idx))

            if self.exhaustive_traversal:
                if use_jitter:
                    ac_hz = self.acoustic_steps_per_second
                    tok_start_ac, tok_count_ac, clap_sec = self._apply_jitter(
                        forced_inner_start, self.fine_window_seconds,
                        audio_length, ac_hz, is_semantic=False,
                    )
                    clap_tokens = self.get_clap_tokens(clap_token_ids, clap_sec)
                    coarse_tokens = self.crop_acoustic_tokens_by_idx(coarse_token_ids, tok_start_ac, tok_count_ac)
                    fine_tokens = self.crop_acoustic_tokens_by_idx(fine_token_ids, tok_start_ac, tok_count_ac)
                else:
                    inside_start_idx = int(forced_inner_start)
                    inside_end_idx = inside_start_idx + self.fine_window_seconds
                    max_clap_start = max(0, int(audio_length) - int(self.semantic_window_seconds))
                    outside_start_idx = min(inside_start_idx, max_clap_start)
                    clap_tokens = self.get_clap_tokens(clap_token_ids, outside_start_idx)
                    coarse_tokens = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)
                    fine_tokens = self.crop_acoustic_tokens(fine_token_ids, inside_start_idx, inside_end_idx)
            else:
                outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx = self.compute_crop_indices(
                    audio_length,
                    self.semantic_window_seconds,
                    self.fine_window_seconds,
                )
                clap_tokens = self.get_clap_tokens(clap_token_ids, outside_start_idx)
                coarse_tokens = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)
                fine_tokens = self.crop_acoustic_tokens(fine_token_ids, inside_start_idx, inside_end_idx)

            return (clap_tokens, coarse_tokens, fine_tokens)

        raise ValueError(f'Invalid stage: {self.stage}')


def get_fast_dataloader(dataset, batch_size=32, drop_last=True, **kwargs):
    """Optimized DataLoader settings"""
    kwargs.setdefault('shuffle', True)
    kwargs.setdefault('num_workers', 8)
    kwargs.setdefault('pin_memory', True)
    kwargs.setdefault('persistent_workers', kwargs['num_workers'] > 0)
    if kwargs['num_workers'] > 0:
        kwargs.setdefault('prefetch_factor', 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_concatenate_fn,
        drop_last=drop_last,
        **kwargs,
    )


# Usage example
if __name__ == '__main__':
    # Convert existing DB (run once)
    # python scripts/convert_db_to_memmap.py

    # Use fast dataset
    dataset = MemmapPreprocessedDataset('./data/fma_preprocessed_memmap', stage='semantic')
    dataloader = get_fast_dataloader(dataset, batch_size=32)

    print(f"Dataset size: {len(dataset)}")

    # Speed test
    import time

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Test only 10 batches
            break
        print(f"Batch {i}: {[x.shape for x in batch]}")

    print(f"10 batches loaded in {time.time() - t0:.2f}s")
