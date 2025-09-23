from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn

__all__ = []

@dataclass
class EarlyStopping:
    r"""
    Early stopping utility with delayed start.

    概要
    ----
    - 通常の EarlyStopping と異なり、**総エポックの一定割合が経過してから**監視を開始する。
    - それまでは改善がなくてもカウントされないため、学習序盤の不安定さを無視できる。

    Parameters
    ----------
    patience : int, default=20
        監視開始後に改善が見られないエポック数がこの値に達すると停止。
    min_delta : float, default=0.0
        改善とみなすための最小変化幅。val_loss が `best - min_delta` より小さければ改善と判定。
    start_epoch_ratio : float, default=0.5
        総エポック数に対する監視開始時点の割合。
        例: 0.5 の場合は「学習の半分を過ぎてから early stopping を有効化」。

    Attributes
    ----------
    best : float
        これまでの最小検証損失。
    best_epoch : int
        最小検証損失が記録されたエポック。
    started : bool
        監視が開始されたかどうか。
    _count : int
        改善なしで経過したエポック数。
    _start_epoch : int
        実際に監視を開始するエポック番号。

    Methods
    -------
    setup(total_epochs: int)
        予定総エポック数を渡して監視開始エポックを設定。
    step(epoch: int, val: float) -> bool
        現在の val_loss を与えて early stop 判定を更新。
        - 改善があれば内部状態をリセット。
        - 停止条件を満たせば True を返す。

    使用例
    ------
    >>> es = EarlyStopping(patience=10, start_epoch_ratio=0.5)
    >>> es.setup(total_epochs=100)
    >>> for epoch in range(1, 101):
    ...     val_loss = validate(...)
    ...     if es.step(epoch, val_loss):
    ...         print(f"Stopped at epoch {epoch}, best={es.best:.4f} @ {es.best_epoch}")
    ...         break
    """
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

    使用例
    ------
    >>> ema = EMACallback(model, decay=0.999)
    >>> for x, y in loader:
    ...     loss = train_step(model, x, y)
    ...     ema.update(model)
    >>> ema.apply_to(model)   # 検証前に EMA 重みを適用
    >>> val_loss = validate(model)
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
