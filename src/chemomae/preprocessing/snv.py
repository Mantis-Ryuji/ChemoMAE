from __future__ import annotations
from dataclasses import dataclass
from typing import overload

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

__all__ = [
    "SNVScaler"
]

def _as_numpy(x):
    """Return (array, is_torch, torch_meta) where torch_meta=(device, dtype) if torch tensor."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy(), True, (x.device, x.dtype)
    return np.asarray(x), False, None


def _back_to_original_type(x_np: np.ndarray, is_torch: bool, torch_meta):
    if is_torch:
        device, dtype = torch_meta
        return torch.from_numpy(x_np).to(device=device, dtype=dtype)
    return x_np


def _std_unbiased(x: np.ndarray, *, axis=None, keepdims: bool = False) -> np.ndarray:
    """
    Unbiased std with ddof=1. If the sample length along axis is 1, fall back to ddof=0 to avoid NaN.
    """
    if axis is None:
        L = x.size
    else:
        L = x.shape[axis]
    ddof_eff = 1 if L >= 2 else 0
    return np.std(x, axis=axis, ddof=ddof_eff, keepdims=keepdims)


def _snv_numpy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    SNV in numpy (unbiased std, ddof=1 when sample length >= 2).
    - 1D: (L,) -> per-vector mean/std
    - 2D: (N, L) -> row-wise mean/std
    """
    if x.ndim == 1:
        mu = x.mean()
        sd = _std_unbiased(x)
        return (x - mu) / (sd + eps)
    if x.ndim == 2:
        mu = x.mean(axis=1, keepdims=True)
        sd = _std_unbiased(x, axis=1, keepdims=True)
        return (x - mu) / (sd + eps)
    raise ValueError(f"SNV expects 1D or 2D array, got shape={x.shape}.")


@overload
def snv(x: np.ndarray, eps: float = 1e-12) -> np.ndarray: ...
@overload
def snv(x: "torch.Tensor", eps: float = 1e-12) -> "torch.Tensor": ...

def snv(x, eps: float = 1e-12):
    """
    Functional SNV (stateless). Keeps the input framework (NumPy/Torch).
    Uses unbiased std (ddof=1) when a valid sample size is available.
    """
    x_np, is_torch, meta = _as_numpy(x)
    y = _snv_numpy(x_np.astype(np.float64, copy=False), eps=eps).astype(np.float32, copy=False)
    return _back_to_original_type(y, is_torch, meta)


@dataclass
class SNVScaler:
    r"""
    Standard Normal Variate (SNV) transformer for spectra (stateless).

    概要
    ----
    - 各行（サンプルごと）に対して平均・標準偏差で標準化を行う。
    - **完全 stateless**: fit() は存在せず、毎回 transform() が mean/std を計算。
    - NumPy / PyTorch いずれの入力でも動作し、同じ型で返却する。
    - 標準偏差は不偏推定（ddof=1、ただし行長=1 の場合は ddof=0）。
    - `transform_stats=True` の場合は `(y, mu, sd)` を返し、inverse_transform() に利用できる。

    Parameters
    ----------
    eps : float, default=1e-12
        数値安定化のために標準偏差に加える微小値。
    copy : bool, default=True
        True の場合は入力配列をコピーしてから変換。
    transform_stats : bool, default=False
        True の場合、`transform()` が `(y, mu, sd)` を返す。

    Methods
    -------
    transform(X)
        入力スペクトル X を SNV 変換する。
        - `transform_stats=False`: X と同じ型の標準化後データを返す。
        - `transform_stats=True`: (標準化後データ, 平均, 標準偏差) を返す。
          mu, sd は NumPy float32 配列で返る。
    inverse_transform(Y, mu, sd)
        `transform_stats=True` で得た mu, sd を使って逆変換する。

    Examples
    --------
    >>> scaler = SNVScaler(transform_stats=True)
    >>> X = np.array([[1.0, 2.0, 3.0],
    ...               [4.0, 5.0, 6.0]], dtype=np.float32)
    >>> Y, mu, sd = scaler.transform(X)
    >>> X_rec = scaler.inverse_transform(Y, mu=mu, sd=sd)
    """
    eps: float = 1e-12
    copy: bool = True
    transform_stats: bool = False  # True: transform() が (y, mu, sd) を返す

    def transform(self, X):
        X_np, is_torch, meta = _as_numpy(X)
        if self.copy:
            X_np = X_np.copy()

        y = _snv_numpy(X_np.astype(np.float64, copy=False), eps=self.eps)
        y = y.astype(np.float32, copy=False)

        if not self.transform_stats:
            return _back_to_original_type(y, is_torch, meta)

        # 統計を返す（後で inverse_transform に利用可）
        if y.ndim == 1:
            mu = X_np.mean()
            sd = _std_unbiased(X_np)
        else:
            mu = X_np.mean(axis=1, keepdims=True)
            sd = _std_unbiased(X_np, axis=1, keepdims=True)

        y_out = _back_to_original_type(y, is_torch, meta)
        # mu, sd は NumPy で返す（軽量・扱いやすい）
        return y_out, mu.astype(np.float32, copy=False), (sd + self.eps).astype(np.float32, copy=False)

    def inverse_transform(self, Y, *, mu: np.ndarray | float, sd: np.ndarray | float):
        """
        変換前統計（mu, sd）を用いて復元する。
        - Y: 1D or 2D
        - mu, sd: transform_stats=True で transform 時に得たものを想定（NumPy 推奨）
        """
        Y_np, is_torch, meta = _as_numpy(Y)
        if self.copy:
            Y_np = Y_np.copy()

        if Y_np.ndim == 1:
            y = Y_np * sd + mu
        elif Y_np.ndim == 2:
            y = Y_np * sd + mu  # (N,1) ブロードキャストを想定
        else:
            raise ValueError(f"SNVScaler.inverse_transform expects 1D or 2D, got {Y_np.shape}.")

        y = y.astype(np.float32, copy=False)
        return _back_to_original_type(y, is_torch, meta)
