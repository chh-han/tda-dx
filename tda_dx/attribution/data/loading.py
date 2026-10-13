"""
attributor/data/loading.py

Dataset classes and loading utilities for attribution experiments.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tda_dx.data.memmap import MemmapPreprocessedDataset
from tda_dx.config import (
    load_model_config,
)
from tda_dx.attribution.utils.attribution_utils import (
    GeneratedTokenDataset,
    GeneratedTokenNpzDataset,
    DattriDatasetAdapter,
)
from tda_dx.attribution.utils.tracking import _validate_memmap_preprocessed_folder


DS_FIELDS = {
    'semantic': ('clap_token_ids', 'semantic_token_ids'),
    'coarse': ('clap_token_ids', 'semantic_token_ids', 'coarse_token_ids'),
    'fine': ('clap_token_ids', 'coarse_token_ids', 'fine_token_ids'),
}


def collate_token_tuples(batch):
    """Stack token tuples into batched tensors."""
    num_fields = len(batch[0])
    return tuple(torch.stack([sample[i] for sample in batch], dim=0) for i in range(num_fields))


class TokenExpandedGeneratedDataset(Dataset):
    """
    Expand test tracks into per-token samples for token-level grad_cos.

    Each item is: (*token_tuple, token_index)
    where token_index refers to a position in the prediction sequence.
    """

    def __init__(self, base_generated_dataset):
        self.base_dataset = base_generated_dataset
        self.entries = []
        for test_idx in range(len(self.base_dataset)):
            token_tuple = self.base_dataset[test_idx]
            pred_len = int(token_tuple[-1].numel())
            for token_pos in range(pred_len):
                self.entries.append((test_idx, token_pos))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        test_idx, token_pos = self.entries[idx]
        sample = self.base_dataset[test_idx]
        return tuple(sample) + (torch.tensor(token_pos, dtype=torch.long),)


def load_preprocessed_dataset(folder, stage, model_config, exhaustive_stride=None):
    folder = Path(folder)
    _validate_memmap_preprocessed_folder(stage, folder)

    window_seconds = {
        'semantic': int(model_config.global_cfg.semantic_audio_length_seconds),
        'coarse': int(model_config.global_cfg.coarse_audio_length_seconds),
        'fine': int(model_config.global_cfg.fine_audio_length_seconds),
    }
    # Default stride = window size (non-overlapping) for attribution
    if exhaustive_stride is None:
        exhaustive_stride = window_seconds[stage]

    return MemmapPreprocessedDataset(
        str(folder),
        stage=stage,
        semantic_window_seconds=window_seconds['semantic'],
        coarse_window_seconds=window_seconds['coarse'],
        fine_window_seconds=window_seconds['fine'],
        semantic_steps_per_second=model_config.hubert_kmeans_cfg.output_hz,
        acoustic_steps_per_second=model_config.acoustic_output_hz,
        exhaustive_stride=exhaustive_stride,
    )


__all__ = [
    'DS_FIELDS',
    'collate_token_tuples',
    'TokenExpandedGeneratedDataset',
    'load_preprocessed_dataset',
    'GeneratedTokenDataset',
    'GeneratedTokenNpzDataset',
    'DattriDatasetAdapter',
]
