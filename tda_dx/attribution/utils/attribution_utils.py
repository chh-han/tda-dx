"""
Utilities for integrating dattri data attribution with MusicLM.

Provides:
- create_functional_loss_fn: torch.func compatible loss function for dattri
- DattriDatasetAdapter: wraps PreprocessedDataset for dattri's expected format

Key design decisions:
- dattri internally uses torch.func.vmap(grad(loss_fn)), so loss_fn must accept
  a SINGLE sample (no batch dim) and use only vmap-compatible operations.
- No inplace operations (masked_fill_ → masked_fill).
- Batch dim is added inside loss_fn via unsqueeze(0) before calling the transformer.
"""

import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tda_dx.models.musiclm import TokenConditionedTransformerWrapper


def create_functional_loss_fn(wrapper: TokenConditionedTransformerWrapper):
    """
    Creates a dattri-compatible loss function: f(params, data_pair) -> scalar.

    The returned function works on a SINGLE sample (no batch dimension),
    which is required for torch.func.vmap compatibility inside dattri.

    It reproduces the loss computation from TokenConditionedTransformerWrapper.forward
    (open_musiclm.py:328-410) in a torch.func-compatible way:
    1. Flatten and append EOS to each token sequence
    2. Create labels (clone after EOS append)
    3. Remove last prediction token from input
    4. Build attention mask (zero out EOS in conditioning sequences)
    5. Forward through transformer via functional_call
    6. Compute weighted cross-entropy loss

    Args:
        wrapper: a TokenConditionedTransformerWrapper instance (used to extract
                 structural info: token_sequences, eos_ids, cross_entropy_loss_weights, etc.)

    Returns:
        loss_fn: f(params, data_pair) -> scalar loss
        transformer: the inner TokenConditionedTransformer (params correspond to this)
    """
    transformer = wrapper.transformer
    token_sequences = wrapper.token_sequences
    eos_ids = wrapper.eos_ids
    cross_entropy_loss_weights = wrapper.cross_entropy_loss_weights
    pad_id = wrapper.pad_id

    def loss_fn(params, data_pair):
        """
        Per-sample loss function compatible with torch.func (vmap + grad).

        Args:
            params: dict of transformer parameters (named_parameters or state_dict)
            data_pair: tuple of token tensors WITHOUT batch dim.
                       e.g. semantic stage: (clap_ids: (12,), semantic_ids: (499,))
                       Optional: append token index tensor at the end to compute
                       prediction-token-level loss for grad_cos attribution.
        """
        has_token_index = False
        token_index_tensor = None
        token_tensors = data_pair
        if len(data_pair) == (len(token_sequences) + 1):
            has_token_index = True
            token_index_tensor = data_pair[-1].to(torch.long).reshape(())
            token_tensors = data_pair[:-1]

        # cross_entropy target must be int64; normalize all token dtypes here
        all_ids = [ids.flatten().to(torch.long) for ids in token_tensors]
        device = all_ids[0].device
        dtype = all_ids[0].dtype

        # 1. Append EOS to each sequence
        all_ids_eos = []
        for ids, eos_id in zip(all_ids, eos_ids):
            eos_tensor = torch.tensor([eos_id], device=device, dtype=dtype)
            all_ids_eos.append(torch.cat([ids, eos_tensor]))

        # 2. Create labels before modifying input
        all_labels = [ids.clone() for ids in all_ids_eos]

        # 3. Remove last token from prediction sequence input (the appended EOS)
        all_ids_eos[-1] = all_ids_eos[-1][:-1]

        # 4. Build attention mask and zero out conditioning EOS/PAD positions
        #    (reproduces open_musiclm.py:358-371)
        #    - Conditioning sequences: mask out EOS and PAD, prepend True for start token
        #    - Prediction sequence: all True (start token + tokens)
        mask_parts = []
        for i in range(len(all_ids_eos) - 1):
            ids = all_ids_eos[i]
            m = (ids != pad_id) & (ids != eos_ids[i])
            # Non-inplace masked_fill (required for torch.func compatibility)
            all_ids_eos[i] = ids.masked_fill(~m, 0)
            # Prepend True for start token
            start_mask = torch.tensor([True], device=device)
            mask_parts.append(torch.cat([start_mask, m]))

        # Prediction sequence: start token + all prediction tokens → all True
        pred_len = all_ids_eos[-1].shape[0]
        mask_parts.append(torch.ones(pred_len + 1, dtype=torch.bool, device=device))

        combined_mask = torch.cat(mask_parts).unsqueeze(0)  # (1, total_seq_len)

        # 5. Add batch dim of 1 for transformer forward
        batched_ids = [ids.unsqueeze(0) for ids in all_ids_eos]

        # 6. Forward through transformer via functional_call
        all_logits = torch.func.functional_call(
            transformer,
            params,
            kwargs={'all_token_ids': batched_ids, 'self_attn_mask': combined_mask},
        )

        # 7. Compute weighted CE loss (reproduces open_musiclm.py:389-410)
        total_logits = 0
        running_loss = torch.tensor(0.0, device=device)
        token_level_loss = torch.tensor(0.0, device=device)
        used_token_level_head = False

        for logits, labels, weight, seq_info in zip(
            all_logits, all_labels, cross_entropy_loss_weights, token_sequences
        ):
            if weight > 0 and logits is not None:
                # logits: (1, seq_len, codebook_size+1)
                # Squeeze batch, then F.cross_entropy expects (N, C) input with (N,) target
                logits_2d = logits.squeeze(0)  # (seq_len, C)
                num = labels.numel()
                per_token_loss = F.cross_entropy(
                    logits_2d,
                    labels.to(torch.long),
                    reduction='none',
                )
                if has_token_index:
                    selected = torch.clamp(token_index_tensor, 0, per_token_loss.shape[0] - 1)
                    token_val = torch.gather(per_token_loss, 0, selected.unsqueeze(0)).squeeze(0)
                    token_level_loss = token_level_loss + token_val * weight
                    used_token_level_head = True
                loss = per_token_loss.mean()
                total_logits += num
                running_loss = running_loss + loss * num * weight

        if has_token_index:
            if not used_token_level_head:
                raise ValueError('token-level loss requested but no weighted prediction logits found')
            return token_level_loss

        if total_logits == 0:
            raise ValueError('No weighted logits contributed to loss; check cross_entropy_loss_weights')
        return running_loss / total_logits

    return loss_fn, transformer


class GeneratedTokenDataset(Dataset):
    """
    Load generated token tuples from a .pt file (output of infer_tokens.py).

    The .pt file contains:
      - 'tokens': list of tuples, each (clap_ids, semantic_ids, ...) already squeezed
      - 'stage': str
      - 'track_ids': list of str

    No DattriDatasetAdapter needed — tokens are already in the right shape.
    """

    def __init__(self, pt_path):
        data = torch.load(pt_path, weights_only=False)
        self.tokens = data['tokens']
        self.stage = data.get('stage', 'semantic')
        self.track_ids = data.get('track_ids', [])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def _parse_track_sqlite_from_stem(stem: str):
    parts = stem.split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return str(int(parts[0])), int(parts[1])

    ints = re.findall(r"\d+", stem)
    if len(ints) >= 2:
        return str(int(ints[0])), int(ints[1])
    if len(ints) == 1:
        return str(int(ints[0])), None
    return stem, None


def _npz_suffix_candidates(stage: str):
    stage = str(stage).strip().lower()
    if stage == 'semantic':
        return ('semantic', 'full', '')
    if stage == 'coarse':
        return ('coarse', 'full', '')
    if stage == 'fine':
        return ('fine', 'full', '')
    return ('full', 'semantic', 'coarse', 'fine', '')


_TOKEN_FILE_STEM_RE = re.compile(r'^(?P<track>\d+)_(?P<sqlite>\d+)(?:_(?P<tag>[A-Za-z0-9-]+))?$')


def _is_valid_npz_token_file(path: Path, stage: str):
    m = _TOKEN_FILE_STEM_RE.match(path.stem)
    if m is None:
        return False
    tag = (m.group('tag') or '').lower()
    allowed = {s for s in _npz_suffix_candidates(stage)}
    return tag in allowed


class GeneratedTokenNpzDataset(Dataset):
    """
    Load generated token tuples from per-track .npz files.

    Expected filename pattern:
      <track_id>_<sqlite_idx>_<stage>.npz
    e.g. 10250_411_full.npz

    .npz payload keys can include: clap, semantic, coarse, fine.
    Returned tuple format follows stage:
      - semantic: (clap, semantic)
      - coarse:   (clap, semantic, coarse)
      - fine:     (clap, coarse, fine)
    """

    REQUIRED_KEYS = {
        'semantic': ('clap', 'semantic'),
        'coarse': ('clap', 'semantic', 'coarse'),
        'fine': ('clap', 'coarse', 'fine'),
    }

    def __init__(self, tokens_dir, stage: str, manifest_path: str = None):
        self.tokens_dir = Path(tokens_dir)
        if not self.tokens_dir.exists():
            raise FileNotFoundError(f'generated tokens dir not found: {self.tokens_dir}')

        self.stage = str(stage).strip().lower()
        if self.stage not in self.REQUIRED_KEYS:
            raise ValueError(f'invalid stage for npz dataset: {stage!r}')

        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.files = self._resolve_files()
        if not self.files:
            raise RuntimeError(f'No token files found in {self.tokens_dir}')

        self.track_ids = []
        self.sqlite_indices = []
        for p in self.files:
            track_id, sqlite_idx = _parse_track_sqlite_from_stem(p.stem)
            self.track_ids.append(str(track_id))
            self.sqlite_indices.append(sqlite_idx)

        # fail fast on key mismatch
        with np.load(str(self.files[0]), allow_pickle=False) as probe:
            missing = [k for k in self.REQUIRED_KEYS[self.stage] if k not in probe]
            if missing:
                raise KeyError(
                    f'{self.files[0]} is missing required keys for stage={self.stage}: {missing}. '
                    f'Available keys: {list(probe.keys())}'
                )

    def _pick_candidate(self, file_stem: str):
        for suffix in _npz_suffix_candidates(self.stage):
            name = f'{file_stem}_{suffix}.npz' if suffix else f'{file_stem}.npz'
            candidate = self.tokens_dir / name
            if candidate.exists():
                return candidate

        matches = sorted(self.tokens_dir.glob(f'{file_stem}_*.npz'))
        if matches:
            return matches[0]

        plain = self.tokens_dir / f'{file_stem}.npz'
        if plain.exists():
            return plain
        return None

    def _resolve_files(self):
        if self.manifest_path:
            if not self.manifest_path.exists():
                raise FileNotFoundError(f'generated tokens manifest not found: {self.manifest_path}')

            resolved = []
            with open(self.manifest_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter='	')
                for row in reader:
                    file_stem = (row.get('file_stem') or '').strip()
                    if not file_stem:
                        track_id = (row.get('track_id') or '').strip()
                        sqlite_idx = (row.get('sqlite_idx') or '').strip()
                        if track_id and sqlite_idx:
                            file_stem = f'{track_id}_{sqlite_idx}'
                    if not file_stem:
                        continue

                    candidate = self._pick_candidate(file_stem)
                    if candidate is None:
                        raise FileNotFoundError(
                            f'No token file matched manifest file_stem={file_stem!r} in {self.tokens_dir}'
                        )
                    resolved.append(candidate)
            return resolved

        all_npz = sorted(self.tokens_dir.glob('*.npz'))
        valid = [p for p in all_npz if _is_valid_npz_token_file(p, self.stage)]
        valid_set = set(valid)
        invalid = [p for p in all_npz if p not in valid_set]
        if invalid:
            examples = ', '.join(p.name for p in invalid[:5])
            raise RuntimeError(
                f'Found {len(invalid)} unexpected .npz file(s) in {self.tokens_dir} '
                f'for stage={self.stage}. Examples: {examples}. '
                f'Use a clean token directory or pass --generated_tokens_manifest.'
            )
        return valid

    @staticmethod
    def _to_long_tensor(array):
        return torch.from_numpy(np.asarray(array)).to(torch.long)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(str(path), allow_pickle=False) as data:
            keys = self.REQUIRED_KEYS[self.stage]
            missing = [k for k in keys if k not in data]
            if missing:
                raise KeyError(
                    f'{path} is missing required keys for stage={self.stage}: {missing}. '
                    f'Available keys: {list(data.keys())}'
                )

            clap = self._to_long_tensor(data['clap'])
            if self.stage == 'semantic':
                semantic = self._to_long_tensor(data['semantic'])
                return clap, semantic

            if self.stage == 'coarse':
                semantic = self._to_long_tensor(data['semantic'])
                coarse = self._to_long_tensor(data['coarse'])
                return clap, semantic, coarse

            coarse = self._to_long_tensor(data['coarse'])
            fine = self._to_long_tensor(data['fine'])
            return clap, coarse, fine


class DattriDatasetAdapter(Dataset):
    """
    Wraps PreprocessedDataset output for dattri compatibility.

    PreprocessedDataset returns tensors with an extra batch dim of 1, e.g. (1, 12).
    This adapter squeezes that dim so the DataLoader can stack correctly:
      (12,) → DataLoader stacks → (B, 12) → dattri vmaps → (12,) per sample

    Stage token formats:
      - semantic: (clap_ids, semantic_ids)
      - coarse:   (clap_ids, semantic_ids, coarse_ids)
      - fine:     (clap_ids, coarse_ids, fine_ids)
    """

    def __init__(self, preprocessed_dataset):
        self.dataset = preprocessed_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.dataset[idx]
        # Squeeze the (1, ...) batch dim from PreprocessedDataset output
        return tuple(t.squeeze(0) for t in tokens)
