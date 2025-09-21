from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, overload

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

__all__ = [
    "snv",
    "SNVScaler",
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


def _snv_numpy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    SNV in numpy.
    - 1D: (L,) -> per-vector mean/std
    - 2D: (N, L) -> row-wise mean/std
    """
    if x.ndim == 1:
        mu = x.mean()
        sd = x.std()
        return (x - mu) / (sd + eps)
    if x.ndim == 2:
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        return (x - mu) / (sd + eps)
    raise ValueError(f"SNV expects 1D or 2D array, got shape={x.shape}.")


@overload
def snv(x: np.ndarray, eps: float = 1e-12) -> np.ndarray: ...
@overload
def snv(x: "torch.Tensor", eps: float = 1e-12) -> "torch.Tensor": ...

def snv(x, eps: float = 1e-12):
    """
    Functional SNV (stateless). Keeps the input framework (NumPy/Torch).
    """
    x_np, is_torch, meta = _as_numpy(x)
    y = _snv_numpy(x_np.astype(np.float64, copy=False), eps=eps).astype(np.float32, copy=False)
    return _back_to_original_type(y, is_torch, meta)


@dataclass
class SNVScaler:
    """
    sklearn の StandardScaler に似たインターフェイスを持つ SNV 変換器（行単位）。

    特徴:
      - SNV は各サンプル（行）ごとに平均0・標準偏差1へ正規化するため、学習時に
        データ集合の統計量を保持しません（=基本的に stateless）。
      - そのため fit() はパイプライン互換のための No-Op とし、transform() で都度行単位に標準化します。
      - NumPy と PyTorch Tensor をどちらも受け付け、入力と同じ型で返します。

    制約:
      - inverse_transform は一般に定義できません（各行ごとの元の mean/std が必要）。
        代替として、transform_stats=True で transform 時に (y, mu, sd) を返し、
        それを inverse_transform に渡すことで元スケールに戻せます。
    """
    eps: float = 1e-12
    copy: bool = True
    transform_stats: bool = False  # True: transform() が (y, mu, sd) を返す

    # sklearn 互換の属性（利用側の安心感のため用意）
    n_features_in_: int | None = None
    fitted_: bool = False

    def fit(self, X, y=None):
        X_np, _, _ = _as_numpy(X)
        if X_np.ndim == 1:
            self.n_features_in_ = X_np.shape[0]
        elif X_np.ndim == 2:
            self.n_features_in_ = X_np.shape[1]
        else:
            raise ValueError(f"SNVScaler.fit expects 1D or 2D array, got shape={X_np.shape}.")
        self.fitted_ = True
        return self

    def _check_fitted(self):
        if not self.fitted_:
            # sklearn の慣習に合わせ fit を要求（実質 no-op だが Pipeline で一貫）
            raise RuntimeError("SNVScaler is not fitted yet. Call .fit(X) before .transform(X).")

    def transform(self, X):
        self._check_fitted()
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
            sd = X_np.std()
        else:
            mu = X_np.mean(axis=1, keepdims=True)
            sd = X_np.std(axis=1, keepdims=True)
        y_out = _back_to_original_type(y, is_torch, meta)
        # mu, sd は NumPy で返す（統計は軽量・取り回しやすい）
        return y_out, mu.astype(np.float32, copy=False), (sd + self.eps).astype(np.float32, copy=False)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Y, *, mu: np.ndarray | float, sd: np.ndarray | float):
        """
        変換前統計（mu, sd）を明示的に受け取って復元する。
        - Y: 1D or 2D
        - mu, sd: transform_stats=True で得たものを推奨（NumPy）
        """
        Y_np, is_torch, meta = _as_numpy(Y)
        if self.copy:
            Y_np = Y_np.copy()

        if Y_np.ndim == 1:
            y = Y_np * sd + mu
        elif Y_np.ndim == 2:
            # mu, sd の形は (N,1) を期待（ブロードキャスト可）
            y = Y_np * sd + mu
        else:
            raise ValueError(f"SNVScaler.inverse_transform expects 1D or 2D, got {Y_np.shape}.")

        y = y.astype(np.float32, copy=False)
        return _back_to_original_type(y, is_torch, meta)