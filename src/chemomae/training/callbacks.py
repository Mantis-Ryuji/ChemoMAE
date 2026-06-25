from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

__all__ = []

class EMACallback:
    r"""
    Exponential Moving Average (EMA) manager for model parameters.

    概要
    ----
    - 学習中にモデルパラメータの指数移動平均を追跡し、安定した推論や検証に利用できる。
    - EMA は過去の重みに滑らかに追従するため、学習中のばらつきを抑えて汎化性能を改善することがある。
    - `apply_to()` を使うことで一時的にモデルへ EMA 重みを適用可能（検証前など）。

    Parameters
    ----------
    model : nn.Module
        EMA 対象のモデル。
    decay : float, default=0.999
        EMA の減衰率。1.0 に近いほど過去の重みを強く保持する。

    Attributes
    ----------
    decay : float
        EMA 減衰率。
    shadow : dict[str, torch.Tensor]
        追跡中の EMA パラメータ（浮動小数のみ対象）。

    Methods
    -------
    register(model: nn.Module)
        モデルの現在の重みを EMA の初期値として登録。
    update(model: nn.Module)
        モデルの最新パラメータで EMA を更新。
    apply_to(model: nn.Module)
        EMA 重みをモデルに一時的に適用。
    state_dict() -> dict
        EMA の状態（decay, shadow）を返す。checkpoint 保存用。
    load_state_dict(state: dict)
        保存済み状態から EMA を復元。
    """

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
