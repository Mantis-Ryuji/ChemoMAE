from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn


@dataclass
class EarlyStopping:
    """半分経過以降に監視を開始する EarlyStopping。"""
    patience: int = 20
    min_delta: float = 0.0
    start_epoch_ratio: float = 0.5  # 総エポックの何割経過後からカウント

    # 内部状態
    best: float = float("inf")
    best_epoch: int = -1
    _started: bool = False
    _count: int = 0
    _start_epoch: int = 1

    def setup(self, total_epochs: int):
        self._start_epoch = max(1, int(total_epochs * self.start_epoch_ratio))

    def step(self, epoch: int, val: float) -> bool:
        """True を返したら停止。"""
        improved = (val + self.min_delta) < self.best
        if improved:
            self.best = val
            self.best_epoch = epoch
            self._count = 0
        else:
            if epoch >= self._start_epoch:
                self._started = True
                self._count += 1
                if self._count >= self.patience:
                    return True
        return False

    @property
    def started(self) -> bool:
        return self._started


class EMACallback:
    """学習中に EMA を管理。 eval 前の一時適用用途にも利用可能。"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model: nn.Module):
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"decay": self.decay, "shadow": {k: v.detach().clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.decay = float(state.get("decay", self.decay))
        sh = state.get("shadow", {})
        self.shadow = {k: v.detach().clone() for k, v in sh.items()}
