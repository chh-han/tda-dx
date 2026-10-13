import math

from torch.optim import AdamW, Adam, lr_scheduler

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr = lr, betas = betas, eps = eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    return AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)

def get_linear_scheduler(
    optimizer,
    total_iters=10000,
    start_factor=1e-7,
):
    return lr_scheduler.LinearLR(optimizer=optimizer, start_factor=start_factor, end_factor=1., total_iters=total_iters)


def get_warmup_cosine_scheduler(
    optimizer,
    warmup_iters: int,
    total_iters: int,
    min_lr_factor: float = 0.01,
    warmup_start_factor: float = 1e-7,
):
    """Linear warmup followed by cosine annealing to ``min_lr_factor * base_lr``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_iters: Number of warmup steps (linear ramp from ``warmup_start_factor``).
        total_iters: Total number of training steps (including warmup).
        min_lr_factor: Minimum LR as a fraction of base LR at the end of cosine decay.
        warmup_start_factor: Starting LR multiplier for warmup phase.
    """
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=max(1, warmup_iters),
    )
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iters - warmup_iters),
        eta_min=0,  # placeholder, overridden by min_lr_factor via lambda below
    )
    # CosineAnnealingLR decays to eta_min=0 by default. We want to decay to
    # min_lr_factor * base_lr instead, so we wrap with a LambdaLR that applies
    # the cosine curve manually for full control.
    decay_iters = max(1, total_iters - warmup_iters)

    def cosine_with_min_lr(step):
        if step < warmup_iters:
            # During warmup: linear ramp
            if warmup_iters == 0:
                return 1.0
            return warmup_start_factor + (1.0 - warmup_start_factor) * step / warmup_iters
        # After warmup: cosine decay
        progress = (step - warmup_iters) / decay_iters
        progress = min(progress, 1.0)
        return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_min_lr)