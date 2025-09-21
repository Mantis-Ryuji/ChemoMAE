from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)


# -------------------------
# Param groups (WD 除外)
# -------------------------
def make_param_groups(
    model: nn.Module,
    *,
    weight_decay: float = 0.05,
    no_decay_bias_bn: bool = True,
) -> List[Dict]:
    """
    bias と Norm 層を weight_decay から除外する典型的な設定。
    """
    if not no_decay_bias_bn:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

    decay, no_decay = [], []
    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name.endswith("bias"):
                no_decay.append(p)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                no_decay.append(p)
            else:
                decay.append(p)

    # 残り（module に引っかからなかった param）も拾う
    seen = set(map(id, decay + no_decay))
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# -------------------------
# Optimizer builder
# -------------------------
def build_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    no_decay_bias_bn: bool = True,
) -> optim.Optimizer:
    """
    name ∈ {"adamw","adam","sgd"} をサポート（シンプル & 依存なし）。
    """
    pg = make_param_groups(model, weight_decay=weight_decay, no_decay_bias_bn=no_decay_bias_bn)

    n = name.lower()
    if n == "adamw":
        return optim.AdamW(pg, lr=lr, betas=betas, eps=eps)
    if n == "adam":
        return optim.Adam(pg, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if n == "sgd":
        return optim.SGD(pg, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    raise ValueError(f"unknown optimizer: {name}")


# -------------------------
# LR scheduler builders
# -------------------------
def _resolve_total_steps(
    total_steps: Optional[int],
    *,
    epochs: Optional[int],
    steps_per_epoch: Optional[int],
) -> int:
    if total_steps is not None:
        return int(total_steps)
    if epochs is None or steps_per_epoch is None:
        raise ValueError("total_steps が None の場合、epochs と steps_per_epoch が必要です。")
    return int(epochs * steps_per_epoch)


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    *,
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    イテレーション単位の Warmup + Cosine Decay。
    lr = base_lr * [ min_lr_ratio + (1-min_lr_ratio) * 0.5*(1+cos(pi * prog)) ]
    """
    T = _resolve_total_steps(total_steps, epochs=epochs, steps_per_epoch=steps_per_epoch)
    W = max(0, int(warmup_steps))
    M = float(min_lr_ratio)

    def lr_lambda(step: int):
        if step < W and W > 0:
            return (step + 1) / W  # 線形ウォームアップ
        t = min(max(step - W, 0), max(T - W, 1))
        prog = t / max(T - W, 1)
        return M + (1.0 - M) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_warmup_linear_scheduler(
    optimizer: optim.Optimizer,
    *,
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    イテレーション単位の Warmup + Linear Decay。
    """
    T = _resolve_total_steps(total_steps, epochs=epochs, steps_per_epoch=steps_per_epoch)
    W = max(0, int(warmup_steps))
    M = float(min_lr_ratio)

    def lr_lambda(step: int):
        if step < W and W > 0:
            return (step + 1) / W
        t = min(max(step - W, 0), max(T - W, 1))
        prog = t / max(T - W, 1)
        return M + (1.0 - M) * (1.0 - prog)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_scheduler(
    optimizer: optim.Optimizer,
    *,
    name: str = "cosine",  # "cosine" | "linear" | "step" | "multistep" | "cosine_restart"
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    min_lr_ratio: float = 0.0,
    # step/multistep
    step_size: int = 30,
    gamma: float = 0.1,
    milestones: Optional[Iterable[int]] = None,
    # cosine_restart
    T_0: int = 10,
    T_mult: int = 2,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Trainer の「イテレーションごと step()」前提。
    - cosine / linear: LambdaLR（total_steps 指定を推奨）
    - step / multistep: 反復ステップ単位で減衰（エポック単位にしたい場合は呼び出し側で epoch ごとに step）
    - cosine_restart: LinearLR (warmup) + CosineAnnealingWarmRestarts を SequentialLR で接続
    """
    n = name.lower()
    if n == "cosine":
        return build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr_ratio=min_lr_ratio,
        )
    if n == "linear":
        return build_warmup_linear_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr_ratio=min_lr_ratio,
        )
    if n == "step":
        # 反復単位の step スケジューラ
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if n == "multistep":
        if milestones is None:
            raise ValueError("multistep を使う場合は milestones を指定してください。")
        return MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
    if n == "cosine_restart":
        # warmup -> cosine restarts
        warm = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup_steps)) if warmup_steps > 0 else None
        cawr = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
        if warm is None:
            return cawr
        return SequentialLR(optimizer, schedulers=[warm, cawr], milestones=[warmup_steps])

    raise ValueError(f"unknown scheduler: {name}")
