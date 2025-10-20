from __future__ import annotations
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple

__all__ = ["build_optimizer", "build_scheduler"]


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


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.Optimizer:
    r"""
    Build an AdamW optimizer with standard weight-decay exclusions.

    概要
    ----
    - AdamW を構築する補助関数。
    - 通常の Transformer 系学習における慣習に従い、以下のパラメータは weight decay を除外する：
      - **バイアス項**（`.bias`）
      - **LayerNorm** の重み
      - 特殊トークン：`cls_token`, `pos_embed`

    Parameters
    ----------
    model : nn.Module
        最適化対象のモデル。
    lr : float, default=3e-4
        学習率。
    weight_decay : float, default=1e-4
        weight decay をかけるパラメータ群に対する係数。
    betas : Tuple[float, float], default=(0.9, 0.95)
        AdamW の β 値。
    eps : float, default=1e-8
        AdamW の数値安定化項。

    Returns
    -------
    optimizer : torch.optim.AdamW
        構築された AdamW Optimizer。

    Notes
    -----
    - `decay` group: 通常の重みパラメータ（weight_decay を適用）
    - `no_decay` group: bias, LayerNorm, cls_token, pos_embed（weight_decay=0.0）
    - これは BERT/ViT 系の学習レシピに基づく慣習で、汎用的に有効。

    使用例
    ------
    >>> model = ChemoMAE(seq_len=256)
    >>> optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-4)
    >>> for p in optimizer.param_groups:
    ...     print(len(p["params"]), p["weight_decay"])
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
    r"""
    Build a learning-rate scheduler: **linear warmup + cosine decay**.

    概要
    ----
    - 最初の `warmup_epochs` では学習率を 0 → base_lr へ線形にウォームアップ。
    - その後はコサイン曲線に従って `base_lr * min_lr_scale` まで減衰。
    - 典型的な Transformer 系のスケジューラ設計。

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        対象の Optimizer。
    steps_per_epoch : int
        1エポックあたりの更新ステップ数（= len(train_loader)）。
    epochs : int
        総エポック数。
    warmup_epochs : int, default=1
        ウォームアップを適用するエポック数。
    min_lr_scale : float, default=0.1
        最小学習率のスケール。最終的に `base_lr * min_lr_scale` に到達。

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR
        PyTorch の LambdaLR スケジューラ。

    Notes
    -----
    - 総ステップ数 = `steps_per_epoch * epochs`
    - ウォームアップステップ数 = `steps_per_epoch * warmup_epochs`
    - コサイン減衰はウォームアップ後の残りステップに適用される。

    使用例
    ------
    >>> optimizer = build_optimizer(model, lr=1e-3)
    >>> scheduler = build_scheduler(optimizer, steps_per_epoch=100, epochs=50, warmup_epochs=2)
    >>> for step in range(200):  # 学習ループ内で scheduler.step() を呼び出す
    ...     optimizer.step()
    ...     scheduler.step()
    """
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    return build_warmup_cosine(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_scale=min_lr_scale,
    )
