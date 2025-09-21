from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple

# -------------------------
# walk helper
# -------------------------
def _walk_to_module(root: nn.Module, param_name: str):
    """
    'blocks.0.norm1.weight' → [root, blocks, blocks[0], norm1]
    最後のテンソル名（weight/bias 等）は除く。
    """
    parts = param_name.split(".")[:-1]
    m = root
    out = [m]
    for key in parts:
        m = m[int(key)] if key.isdigit() else getattr(m, key)
        out.append(m)
    return out


# -------------------------
# Optimizer builder
# -------------------------
def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.Optimizer:
    """
    AdamW optimizer with typical weight decay exclusion:
      - all biases
      - LayerNorm (norm1/norm2)
      - cls_token and pos_embed
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


# -------------------------
# Scheduler builders
# -------------------------
def build_warmup_cosine(
    optimizer: optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.0,
) -> LambdaLR:
    """
    Return LambdaLR that does linear warmup then cosine decay to min_lr_scale.
    - min_lr_scale is relative to the base LR set in the optimizer.
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
    """
    Wrapper: linear warmup for warmup_epochs, then cosine decay.
    """
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    return build_warmup_cosine(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_scale=min_lr_scale,
    )
