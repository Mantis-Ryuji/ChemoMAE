from __future__ import annotations
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple

__all__ = ["build_optimizer", "build_scheduler"]


def _walk_to_module(root: nn.Module, param_name: str):
    """Return the modules owning a dotted parameter name, excluding its tensor name.

    For example, blocks.0.norm1.weight walks through root, blocks,
    blocks[0], and norm1. The walk includes every ancestor used by the
    LayerNorm exclusion rule.
    """
    parts = param_name.split(".")[:-1]
    m = root
    out = [m]
    for key in parts:
        m = m[int(key)] if key.isdigit() else getattr(m, key)
        out.append(m)
    return out


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.Optimizer:
    """Build AdamW with explicit name/module-based weight-decay exclusions.

    Parameters
    ----------
    model : torch.nn.Module
        Model to optimize. Move it to its training device and configure
        requires_grad before creating the optimizer.
    lr : float, default=1.5e-4
        Base learning rate before scheduler scaling.
    weight_decay : float, default=0.05
        AdamW decoupled weight decay applied to the decay group.
    betas : tuple of float, default=(0.9, 0.95)
        AdamW moment coefficients.
    eps : float, default=1e-8
        AdamW numerical stability term.

    Returns
    -------
    torch.optim.AdamW
        Optimizer containing the nonempty decay and no-decay groups.

    Notes
    -----
    Parameters with requires_grad=False are excluded. No decay is applied
    to names ending in ".bias", parameters whose module walk includes a
    LayerNorm, or names containing "cls_token" or "pos_embed". Remaining
    parameters receive weight_decay. Nonempty decay groups precede
    nonempty no-decay groups, whose weight_decay is zero.

    These are inspectable recipe choices, not a claim that one set of
    exclusions is scientifically optimal for every dataset or model.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_layernorm = any(isinstance(m, nn.LayerNorm) for m in _walk_to_module(model, name))
        is_special = ("cls_token" in name) or ("pos_embed" in name)

        if is_bias or is_layernorm or is_special:
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def build_warmup_cosine(
    optimizer: optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.0,
) -> LambdaLR:
    """Build a per-update LambdaLR with linear warmup and capped cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose group base rates are scaled.
    warmup_steps : int
        Number of scheduler indices allocated to warmup.
    total_steps : int
        Planned update count, including warmup.
    min_lr_scale : float, default=0.0
        Multiplier at scheduler index total_steps when warmup_steps < total_steps.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler with the same multiplier applied to every group base rate.

    Notes
    -----
    LambdaLR applies index zero during construction. With positive warmup,
    the first optimizer update uses the positive multiplier
    max(1e-8, 1/warmup_steps), not zero. Call scheduler.step() after an update:
    j uses index j-1, then prepares index j for the next update. The base
    rate appears at indices warmup_steps-1 and warmup_steps. For an ordinary
    budget with 0 <= warmup_steps < total_steps, the minimum is installed
    after the final planned update, rather than consumed by that update.

    The helper does not validate schedule budgets. The max(1, ...) guards
    avoid zero denominators; they do not make arbitrary budgets meaningful.
    """
    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_scale + 0.5 * (1 - min_lr_scale) * (1 + math.cos(math.pi * t))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_scheduler(
    optimizer: optim.Optimizer,
    *,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 1,
    min_lr_scale: float = 0.1,
) -> LambdaLR:
    """Build the warmup/cosine scheduler from epoch-sized update budgets.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule.
    steps_per_epoch : int
        Planned optimizer updates per epoch, usually len(train_loader).
    epochs : int
        Planned total epochs.
    warmup_epochs : int, default=1
        Number of epoch-sized warmup update budgets.
    min_lr_scale : float, default=0.1
        Multiplier installed at the planned final scheduler index.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler produced by build_warmup_cosine.

    Notes
    -----
    total_steps = steps_per_epoch * epochs and warmup_steps =
    steps_per_epoch * warmup_epochs. Index zero is applied on construction;
    the first update uses a positive warmup rate when warmup is enabled.
    With post-update stepping, the final minimum is installed after the
    last planned update. See build_warmup_cosine for indexing details.

    Trainer advances its scheduler only after successful optimizer
    updates. AMP-skipped attempts therefore do not advance this schedule.
    Epoch budgets still count attempts unless the caller's batching
    protocol explicitly supplies an update-based budget.
    """
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    return build_warmup_cosine(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_scale=min_lr_scale,
    )
