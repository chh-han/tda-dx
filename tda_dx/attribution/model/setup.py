"""
attributor/model/setup.py

Model loading, stage wrapper creation, and parameter configuration
utilities for attribution experiments.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from tda_dx.config import (
    load_model_config,
    create_semantic_transformer_from_config,
    create_coarse_transformer_from_config,
    create_fine_transformer_from_config,
)
from tda_dx.models.musiclm import (
    SemanticStage,
    CoarseStage,
    FineStage,
)
from tda_dx.attribution.data.loading import DS_FIELDS, collate_token_tuples


def create_transformer_from_checkpoint(stage, model_config, checkpoint_path, device):
    """Load a trained transformer from checkpoint."""
    if stage == 'semantic':
        return create_semantic_transformer_from_config(model_config, checkpoint_path, device)
    elif stage == 'coarse':
        return create_coarse_transformer_from_config(model_config, checkpoint_path, device)
    elif stage == 'fine':
        return create_fine_transformer_from_config(model_config, checkpoint_path, device)
    else:
        raise ValueError(f'invalid stage: {stage}')


def create_stage_wrapper(stage, transformer):
    """Build stage wrapper for forward pass with loss."""
    if stage == 'semantic':
        return SemanticStage(
            semantic_transformer=transformer,
            cross_entropy_loss_weights=[0.0, 1.0],
        )
    elif stage == 'coarse':
        return CoarseStage(
            coarse_transformer=transformer,
            cross_entropy_loss_weights=[0.0, 0.0, 1.0],
        )
    elif stage == 'fine':
        return FineStage(
            fine_transformer=transformer,
            cross_entropy_loss_weights=[0.0, 0.0, 1.0],
        )
    else:
        raise ValueError(f'invalid stage: {stage}')


def resolve_default_trak_layer_names(inner_transformer, stage_wrapper):
    """
    Resolve the prediction head parameter name for default TRAK attribution.

    For TokenConditionedTransformer, per-sequence output heads are stored in
    `logit_weights.{i}`. The final predicted sequence head is the last index.
    """
    token_sequences = stage_wrapper.transformer_wrapper.token_sequences
    predicted_seq_idx = len(token_sequences) - 1
    candidate = f'logit_weights.{predicted_seq_idx}'

    param_names = {name for name, p in inner_transformer.named_parameters() if p.requires_grad}
    if candidate in param_names:
        return [candidate]

    # Fallback: use the largest indexed logit head if naming differs unexpectedly.
    logit_heads = [name for name in param_names if name.startswith('logit_weights.')]
    if logit_heads:
        try:
            logit_heads = sorted(logit_heads, key=lambda n: int(n.split('.')[-1]))
        except ValueError:
            logit_heads = sorted(logit_heads)
        return [logit_heads[-1]]

    raise RuntimeError(
        'Could not resolve default TRAK last-layer name. '
        'Expected a parameter like logit_weights.<idx>.'
    )


def resolve_last_representation_layer_names(inner_transformer):
    """
    Resolve parameter names for the final representation block before output heads.

    In TokenConditionedTransformer.forward, output heads consume `pred_tokens`
    split from `self.transformer(...)` output. Therefore the pre-head representation
    comes from the final transformer block plus final norm.
    """
    trainable_names = [
        name for name, param in inner_transformer.named_parameters() if param.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError('No trainable parameters found on inner transformer')

    layer_pat = re.compile(r'^transformer\.layers\.(\d+)\.')
    layer_indices = sorted({
        int(m.group(1))
        for name in trainable_names
        for m in [layer_pat.match(name)]
        if m is not None
    })
    if not layer_indices:
        raise RuntimeError(
            'Could not find transformer block parameters (expected names like '
            '"transformer.layers.<idx>.*").'
        )

    last_layer_idx = layer_indices[-1]
    last_layer_prefix = f'transformer.layers.{last_layer_idx}.'
    final_norm_prefix = 'transformer.norm.'

    selected = sorted([
        name
        for name in trainable_names
        if name.startswith(last_layer_prefix) or name.startswith(final_norm_prefix)
    ])
    if not selected:
        raise RuntimeError(
            'Failed to resolve last representation layer params from '
            f'prefixes: {last_layer_prefix}, {final_norm_prefix}'
        )

    meta = {
        'last_transformer_block_index': last_layer_idx,
        'last_transformer_block_prefix': last_layer_prefix,
        'final_norm_prefix': final_norm_prefix,
        'num_selected_params': len(selected),
    }
    return selected, meta


def configure_trainable_params(inner_transformer, trainable_layer_names):
    """
    Attribution-only memory optimization:
    keep gradients only for requested parameter names.

    This does not change checkpoint values and only affects requires_grad flags
    in the current attribution process.
    """
    trainable_set = set(trainable_layer_names)
    total_params = 0
    trainable_params = 0

    for name, param in inner_transformer.named_parameters():
        total_params += param.numel()
        enable_grad = name in trainable_set
        param.requires_grad_(enable_grad)
        if enable_grad:
            trainable_params += param.numel()

    if trainable_params == 0:
        raise RuntimeError(
            f'No trainable params matched requested names: {trainable_layer_names}'
        )

    return total_params, trainable_params


@torch.no_grad()
def evaluate_per_sample_loss(transformer, stage, test_dataset, device):
    """
    Compute per-sample loss on the test set.

    Uses batch_size=1 to get exact per-sample losses.
    Returns np.array of shape (n_test,).
    """
    ds_fields = DS_FIELDS[stage]
    wrapper = create_stage_wrapper(stage, transformer)
    wrapper.eval()

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_token_tuples)

    losses = []
    for batch in loader:
        data_kwargs = {k: v.to(device) for k, v in zip(ds_fields, batch)}
        loss, _, _ = wrapper(**data_kwargs, return_loss=True)
        losses.append(loss.item())

    return np.array(losses)


__all__ = [
    'create_transformer_from_checkpoint',
    'create_stage_wrapper',
    'resolve_default_trak_layer_names',
    'resolve_last_representation_layer_names',
    'configure_trainable_params',
    'evaluate_per_sample_loss',
]
