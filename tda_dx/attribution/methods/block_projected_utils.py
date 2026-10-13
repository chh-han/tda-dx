"""
Shared helpers for block-projected attribution methods such as LoGra/FactGraSS.
"""

from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from tda_dx.attribution.model.setup import configure_trainable_params
from dattri.algorithm.block_projected_if.core.hook import HookManager


def resolve_block_projected_layer_names(model: nn.Module) -> List[str]:
    """Return MusicLM-safe trainable Linear module names inside transformer blocks."""
    layer_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("transformer.layers."):
            continue
        has_trainable_param = any(
            param.requires_grad for param in module.parameters(recurse=False)
        )
        if has_trainable_param:
            layer_names.append(name)
    return layer_names


def resolve_block_projected_param_names(
    model: nn.Module,
    layer_names: List[str],
) -> List[str]:
    """Map selected module names to the corresponding parameter names."""
    allowed_prefixes = tuple(f"{name}." for name in layer_names)
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(allowed_prefixes)
    ]


def configure_block_projected_params(
    model: nn.Module,
    method_label: str,
) -> List[str]:
    """Restrict gradients to supported transformer-block Linear parameters."""
    layer_names = resolve_block_projected_layer_names(model)
    if not layer_names:
        raise RuntimeError(
            f"{method_label} could not find any supported trainable nn.Linear "
            "layers inside transformer blocks."
        )

    supported_param_names = resolve_block_projected_param_names(model, layer_names)
    total_params_cnt, trainable_params_cnt = configure_trainable_params(
        model,
        supported_param_names,
    )
    print(
        f"{method_label}-supported params enabled: "
        f"{trainable_params_cnt:,} / {total_params_cnt:,} "
        f"({100.0 * trainable_params_cnt / total_params_cnt:.4f}%)"
    )
    print(f"{method_label} hooked linear layers: {len(layer_names)}")
    return layer_names


def validate_single_batch_contract(
    *,
    method_label: str,
    attributor,
    inner_transformer: nn.Module,
    layer_names: List[str],
    train_loader: DataLoader,
    device: str,
) -> None:
    """Fail fast if hooked layers do not emit per-sample compressed grads."""
    if attributor.compressors is None:
        attributor._setup_compressors(train_loader)

    batch = next(iter(train_loader))
    expected_batch_size = batch[0].shape[0]

    hook_manager = HookManager(
        inner_transformer,
        layer_names,
        device=device,
    )
    hook_manager.set_compressors(attributor.compressors)

    try:
        inner_transformer.zero_grad(set_to_none=True)
        loss = attributor._compute_batch_loss(batch)
        loss.backward()
        compressed = hook_manager.get_compressed_grads()
    finally:
        hook_manager.remove_hooks()
        inner_transformer.zero_grad(set_to_none=True)

    invalid = []
    for idx, grad in enumerate(compressed):
        if grad is None:
            invalid.append((layer_names[idx], None))
            continue
        if grad.dim() != 2 or grad.shape[0] != expected_batch_size:
            invalid.append((layer_names[idx], tuple(grad.shape)))

    if invalid:
        preview = ", ".join(f"{name}:{shape}" for name, shape in invalid[:10])
        raise RuntimeError(
            f"{method_label} per-sample gradient contract failed on the first batch. "
            f"Expected each hooked layer to emit shape ({expected_batch_size}, proj_dim). "
            f"Invalid layers (first 10): {preview}"
        )

    # LoGra/FactGraSS multiply backward activations by `batch_size` inside the
    # compressor to recover per-sample gradients (assumes loss reduction='mean').
    # If the user's loss_fn reduces with 'sum', scores would be silently scaled
    # by batch_size. Cross-check by comparing the projected per-sample-norm of
    # the first sample against an autograd-computed reference is expensive;
    # instead we emit a one-time reminder.
    print(
        f"{method_label}: assuming loss_fn uses reduction='mean' over the batch. "
        "If your loss is summed (or normalized differently), per-sample gradient "
        "scaling will be off by batch_size. See compressor.forward `* batch_size`."
    )
