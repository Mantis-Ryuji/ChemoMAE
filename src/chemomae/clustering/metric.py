from __future__ import annotations
import math
from numbers import Integral, Real
import numpy as np
import torch
from typing import Optional


@torch.no_grad()
def silhouette_samples_cosine_gpu(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    chunk: Optional[int] = 1000000,
    dtype: torch.dtype = torch.float32,
    return_numpy: bool = True,
    eps: float = 1e-12,
) -> np.ndarray | torch.Tensor:
    r"""
    Overview
    ----------
    GPU implementation of cosine-distance silhouette samples, equivalent to
    sklearn.metrics.silhouette_samples using cluster sums, with O(NKD) time.

    Parameters
    ----------
    X : (N, D) array-like
        Input features as a NumPy array or Torch tensor.
        Rows are L2-normalized internally, as for sklearn cosine distance.
        Zero vectors remain zero: dot product zero gives distance one.
    labels : (N,) array-like of int
        Cluster assignments; nonconsecutive labels are compressed to 0..K-1.

    device : str or torch.device, default="cuda"
        CPU or CUDA computation device.
    chunk : int | None
        Tile size for b_i computation (partitioning X @ M^T). Adjust to VRAM.
        None computes all rows together.
    dtype : torch.dtype
        Input/output dtype (float16 / bfloat16 / float32 / float64). Half inputs
        use float32 intermediates. NumPy output is always float32.
    return_numpy : bool
        Return a NumPy array when true, otherwise a Torch tensor.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    s : (N,) same type as `return_numpy`
        Per-sample silhouette values in [-1, 1].
        Samples in singleton clusters are assigned zero.

    Notes
    -----
    - Definition:
        - d(x,y) = 1 - cos(x,y)
        - a_i = mean within-cluster distance, excluding the sample itself
        - b_i = minimum mean distance to another cluster
        - s_i = (b_i - a_i) / max(a_i, b_i)
      Here cos is computed by dot products after row-wise L2 normalization.

    - Requires finite real features, integer labels, and 2 <= K < N.
      A zero-distance denominator and singleton clusters return zero.
    - Time is O(NKD), dominated by X @ M^T; no N-by-N distance matrix is built.
    - Chunking bounds the B-by-K similarity temporary only. Full features,
      labels, within-class work arrays, class sums, and output remain resident.
    - Requested precision overrides ambient AMP. Half intermediates use float32.
    """
    if chunk is not None and (
        isinstance(chunk, (bool, np.bool_)) or not isinstance(chunk, Integral) or chunk < 1
    ):
        raise ValueError("chunk must be a positive integer or None.")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("dtype must be float16, bfloat16, float32, or float64.")
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("Cosine silhouette supports CPU and CUDA devices.")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose device='cpu'.")
    for name, value in (("X", X), ("labels", labels)):
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise TypeError(f"{name} must be a NumPy array or Torch tensor.")
        if isinstance(value, torch.Tensor) and (
            value.layout != torch.strided or value.device.type == "meta"
        ):
            raise ValueError(f"{name} must be a dense tensor with stored values.")
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have nonempty shape (N, D) with N >= 2.")
    if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
        raise ValueError("labels must have shape (N,) matching X.")
    if isinstance(X, np.ndarray):
        if X.dtype.kind not in "iuf" or not np.isfinite(X).all():
            raise ValueError("X must contain finite real numeric values.")
    elif X.is_complex() or X.dtype == torch.bool or not bool(torch.isfinite(X).all()):
        raise ValueError("X must contain finite real numeric values.")
    if isinstance(labels, np.ndarray):
        if labels.dtype.kind not in "iu":
            raise TypeError("labels must have an integer dtype; floating labels are not truncated.")
        if labels.dtype.kind == "u" and labels.max() > np.iinfo(np.int64).max:
            raise ValueError("labels must fit in signed int64.")
    elif labels.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("labels must have an integer dtype; floating labels are not truncated.")

    # ---- Convert inputs to Torch ----
    x_is_numpy = isinstance(X, np.ndarray)
    l_is_numpy = isinstance(labels, np.ndarray)

    if x_is_numpy:
        X_t = torch.as_tensor(X, device=device, dtype=dtype)
    else:
        X_t = X.to(device=device, dtype=dtype, non_blocking=True)

    if l_is_numpy:
        y_t = torch.as_tensor(labels, device=device, dtype=torch.long)
    else:
        y_t = labels.to(device=device, dtype=torch.long, non_blocking=True)

    if not bool(torch.isfinite(X_t).all()):
        raise ValueError("X becomes nonfinite in the requested computation dtype.")
    N, D = X_t.shape

    # ---- Compress nonconsecutive labels to 0..K-1 ----
    uniq, inv = torch.unique(y_t, sorted=True, return_inverse=True)
    y_t = inv  # 0..K-1
    K = int(uniq.numel())
    if not 2 <= K < N:
        raise ValueError("Cosine silhouette requires 2 <= number of classes < N.")

    # Ambient AMP must not override the requested arithmetic.
    with torch.autocast(target_device.type, enabled=False):
        s = _silhouette_tensor(X_t, y_t, K, chunk=chunk, eps=float(eps))
    if return_numpy:
        return s.detach().to(dtype=torch.float32).cpu().numpy()
    return s


def _silhouette_tensor(
    X_t: torch.Tensor, y_t: torch.Tensor, K: int, *, chunk: int | None, eps: float,
) -> torch.Tensor:
    """Compute cluster-sum cosine distances without an N-by-N distance matrix."""
    N, D = X_t.shape
    device = X_t.device
    output_dtype = X_t.dtype
    if output_dtype in {torch.float16, torch.bfloat16}:
        X_t = X_t.float()

    # ---- Row-wise L2 normalization, retaining zero vectors ----
    # Zero rows have dot product zero, hence distance one, to any vector.
    norm = X_t.norm(p=2, dim=1, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise ValueError("Spectrum norms overflow the computation dtype.")
    safe = norm.clamp_min(eps)
    Xn = X_t / safe
    Xn = torch.where(norm > 0, Xn, torch.zeros_like(Xn))

    # ---- Per-cluster sum vectors S_k and sizes n_k ----
    S = torch.zeros(K, D, device=device, dtype=Xn.dtype)
    S.index_add_(0, y_t, Xn)  # scatter-add
    n = torch.bincount(y_t, minlength=K)
    n = torch.clamp(n, min=1)  # Avoid division by zero.
    M = S / n.unsqueeze(1)     # Cluster means need no unit normalization.

    # ---- a_i: within-cluster mean, excluding the sample itself ----
    S_c = S[y_t]            # (N, D)
    n_c = n[y_t]            # (N,)
    denom_a = torch.clamp(n_c - 1, min=1)      # Singleton values are overwritten below.
    mean_excl = (S_c - Xn) / denom_a.unsqueeze(1)
    cos_in = torch.sum(Xn * mean_excl, dim=1)  # x_i^T mean_excl
    a = 1.0 - cos_in.clamp(-1.0, 1.0)
    # Singleton clusters have a=0 and ultimately s=0.
    a = torch.where(n_c == 1, torch.zeros_like(a), a)

    # ---- b_i: minimum other-cluster mean distance = 1 - maximum mean similarity ----
    if chunk is None:
        sims = Xn @ M.t()  # (N, K)
        sims[torch.arange(N, device=device), y_t] = float("-inf")
        best_sim, _ = sims.max(dim=1)
        b = 1.0 - best_sim.clamp(-1.0, 1.0)
    else:
        b = torch.empty(N, device=device, dtype=Xn.dtype)
        start = 0
        while start < N:
            end = min(start + int(chunk), N)
            sims_blk = Xn[start:end] @ M.t()  # (B, K)
            sims_blk[torch.arange(end - start, device=device), y_t[start:end]] = float("-inf")
            best_sim_blk, _ = sims_blk.max(dim=1)
            b[start:end] = 1.0 - best_sim_blk.clamp(-1.0, 1.0)
            start = end

    # ---- silhouette s_i ----
    denom = torch.maximum(a, b).clamp_min(1e-12)
    s = (b - a) / denom
    # Singleton clusters have zero silhouette.
    s = torch.where(n_c == 1, torch.zeros_like(s), s)

    if not bool(torch.isfinite(s).all()):
        raise ValueError("Cosine silhouette arithmetic produced nonfinite values.")
    return s.to(output_dtype)


@torch.no_grad()
def silhouette_score_cosine_gpu(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    return_numpy: bool = True,
    **kwargs,
) -> float | torch.Tensor:
    """
    Return the mean per-sample silhouette, equivalent to silhouette_score.
    Forward kwargs unchanged to silhouette_samples_cosine_gpu.
    """
    # Force tensor output for aggregation.
    kw = dict(kwargs)
    kw["return_numpy"] = False
    s = silhouette_samples_cosine_gpu(X, labels, **kw)
    assert isinstance(s, torch.Tensor)

    # Compute the mean in FP32.
    score_t = s.float().mean(dtype=torch.float32)

    if return_numpy:
        return float(score_t.item())        # NumPy-compatible Python float.
    else:
        return score_t                      # torch.Tensor (float32 scalar)
