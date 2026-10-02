from __future__ import annotations
import numpy as np
import torch
from typing import Optional


@torch.no_grad()
def silhouette_samples_cosine_gpu(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    device: str = "cuda",
    chunk: Optional[int] = 1000000,
    dtype: torch.dtype = torch.float32,
    return_numpy: bool = True,
    eps: float = 1e-12,
) -> np.ndarray | torch.Tensor:
    r"""
    Overview
    ----------
    GPU implementation of cosine-distance silhouette samples, equivalent to
    sklearn.metrics.silhouette_samples (exact, O(NK)).

    Parameters
    ----------
    X : (N, D) array-like
        Input features as a NumPy array or Torch tensor.
        Rows are L2-normalized internally, as for sklearn cosine distance.
        Zero vectors remain zero: dot product zero gives distance one.
    labels : (N,) array-like of int
        Cluster assignments; nonconsecutive labels are compressed to 0..K-1.

    device : {"cuda","cpu",...}
        Computation device.
    chunk : int | None
        Tile size for b_i computation (partitioning X @ M^T). Adjust to VRAM.
        None computes all rows together.
    dtype : torch.dtype
        Computation dtype (float16 / bfloat16 / float32).
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

    - Complexity: O(ND + KD + NK_chunk), dominated by X @ M^T.
    - Memory depends on X, M, and the temporary tile; smaller chunks use less memory.
    """
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

    assert X_t.ndim == 2 and y_t.ndim == 1 and X_t.size(0) == y_t.size(0), "Shape mismatch"
    N, D = X_t.shape

    # ---- Compress nonconsecutive labels to 0..K-1 ----
    uniq, inv = torch.unique(y_t, sorted=True, return_inverse=True)
    y_t = inv  # 0..K-1
    K = int(uniq.numel())

    # ---- Row-wise L2 normalization, retaining zero vectors ----
    # Zero rows have dot product zero, hence distance one, to any vector.
    norm = X_t.norm(p=2, dim=1, keepdim=True)
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
    a = 1.0 - cos_in
    # Singleton clusters have a=0 and ultimately s=0.
    a = torch.where(n_c == 1, torch.zeros_like(a), a)

    # ---- b_i: minimum other-cluster mean distance = 1 - maximum mean similarity ----
    if chunk is None:
        sims = Xn @ M.t()  # (N, K)
        sims[torch.arange(N, device=device), y_t] = float("-inf")
        best_sim, _ = sims.max(dim=1)
        b = 1.0 - best_sim
    else:
        b = torch.empty(N, device=device, dtype=Xn.dtype)
        start = 0
        while start < N:
            end = min(start + int(chunk), N)
            sims_blk = Xn[start:end] @ M.t()  # (B, K)
            sims_blk[torch.arange(end - start, device=device), y_t[start:end]] = float("-inf")
            best_sim_blk, _ = sims_blk.max(dim=1)
            b[start:end] = 1.0 - best_sim_blk
            start = end

    # ---- silhouette s_i ----
    denom = torch.maximum(a, b).clamp_min(1e-12)
    s = (b - a) / denom
    # Singleton clusters have zero silhouette.
    s = torch.where(n_c == 1, torch.zeros_like(s), s)

    if return_numpy:
        return s.detach().to(dtype=torch.float32).cpu().numpy()
    return s


@torch.no_grad()
def silhouette_score_cosine_gpu(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    return_numpy: bool = True,
    **kwargs,
) -> float:
    """
    Return the mean per-sample silhouette, equivalent to silhouette_score.
    Forward kwargs unchanged to silhouette_samples_cosine_gpu.
    """
    # Force tensor output for aggregation.
    kw = dict(kwargs)
    kw["return_numpy"] = False
    s = silhouette_samples_cosine_gpu(X, labels, **kw)  # torch.Tensor
    
    # Compute the mean in FP32.
    score_t = s.float().mean(dtype=torch.float32)

    if return_numpy:
        return float(score_t.item())        # NumPy-compatible Python float.
    else:
        return score_t                      # torch.Tensor (float32 scalar)
