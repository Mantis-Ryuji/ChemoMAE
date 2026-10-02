"""Explicit-device cosine farthest-point sampling of spectral rows."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch

__all__ = ["cosine_fps_downsample"]


@torch.no_grad()
def cosine_fps_downsample(
    X: np.ndarray | torch.Tensor,
    *,
    ratio: float = 0.1,
    seed: int | None = None,
    init_index: int | None = None,
    return_numpy: bool = True,
    return_indices: bool = False,
    eps: float = 1e-12,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> np.ndarray | torch.Tensor | tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Select rows by farthest-point sampling under cosine geometry.

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor, shape (N, C)
        Finite real numerical rows. Selection normalizes rows internally;
        returned rows retain the original scale.
    ratio : float, default=0.1
        Positive finite fraction. The sample count is clipped to
        min(max(1, round(N*ratio)), N); ratios >= 1 select all rows.
    seed : int, optional
        Seed for a local initial-point stream; mutually exclusive with generator.
    init_index : int, optional
        Initial row index. When supplied, no random numbers are drawn.
    return_numpy : bool, default=True
        Return NumPy arrays, or Torch tensors when false.
    return_indices : bool, default=False
        Also return selected row indices, in selection order.
    eps : float, default=1e-12
        Positive finite value added to row norms for internal normalization.
    device : str or torch.device, optional
        CPU/CUDA computation device. None follows Torch input, otherwise CPU.
        CUDA is never selected merely because it is available.
    generator : torch.Generator, optional
        Caller-owned initial-point stream on the computation device. With neither
        seed nor generator, the ordinary device RNG draws the initial point.

    Returns
    -------
    numpy.ndarray or torch.Tensor, or tuple of these
        Selected rows and optionally indices. Torch input to Torch output keeps
        its original dtype/device. NumPy input to Torch output uses the compute
        device and original dtype. NumPy output preserves source dtype, except
        Torch bfloat16 is promoted to float32 for NumPy compatibility.

    Notes
    -----
    Zero rows remain zero; they have dissimilarity one to every direction.
    Ties choose the first remaining row. Already selected rows cannot repeat.
    Arithmetic uses float64 for float64 input and otherwise float32, with buffers
    of the same dtype. Selection costs O(N*k); full working rows remain resident.
    FPS is a discrete sampling operation and does not record autograd.
    """
    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise TypeError("ratio must be a positive finite real number")
    if not math.isfinite(float(ratio)) or ratio <= 0:
        raise ValueError("ratio must be positive and finite")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, Integral)
        or not -(2**63) <= seed < 2**64
    ):
        raise ValueError("seed must be an integer in Torch's seed range")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")
    if seed is not None and generator is not None:
        raise ValueError("seed and generator are mutually exclusive")
    if not isinstance(return_numpy, bool) or not isinstance(return_indices, bool):
        raise TypeError("return_numpy and return_indices must be boolean")

    is_numpy = isinstance(X, np.ndarray)
    if is_numpy:
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be 2D with at least one feature")
        if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
            raise ValueError("X must contain finite real numerical values")
        xt = torch.from_numpy(np.array(X, copy=True, order="C"))
    elif isinstance(X, torch.Tensor):
        if X.ndim != 2 or X.shape[1] == 0 or X.layout != torch.strided or X.device.type == "meta":
            raise ValueError("X must be a dense 2D tensor with at least one feature")
        if X.is_complex() or X.dtype == torch.bool or not torch.isfinite(X).all():
            raise ValueError("X must contain finite real numerical values")
        xt = X
    else:
        raise TypeError("X must be a NumPy array or Torch tensor")

    compute_device = torch.device(device) if device is not None else xt.device
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("FPS supports CPU and CUDA devices")
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    n = int(xt.shape[0])
    if init_index is not None and (
        isinstance(init_index, bool) or not isinstance(init_index, Integral)
        or not 0 <= init_index < n
    ):
        raise ValueError("init_index must be an integer within the input row range")
    if generator is not None and init_index is None:
        actual_generator_device = torch.device(generator.device)
        actual_index = actual_generator_device.index
        target_index = compute_device.index
        if compute_device.type == "cuda":
            actual_index = torch.cuda.current_device() if actual_index is None else actual_index
            target_index = torch.cuda.current_device() if target_index is None else target_index
        if actual_generator_device.type != compute_device.type or (
            compute_device.type == "cuda" and actual_index != target_index
        ):
            raise ValueError("generator must match the computation device")

    count = n if ratio >= 1 else min(max(1, round(n * float(ratio))), n)
    work_dtype = torch.float64 if xt.dtype == torch.float64 else torch.float32
    with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
        work = xt.to(device=compute_device, dtype=work_dtype)
        if not torch.isfinite(work).all():
            raise ValueError("X cannot be represented finitely in the computation dtype")
        indices = torch.empty(count, dtype=torch.long, device=compute_device)
        if count:
            norms = torch.linalg.vector_norm(work, dim=1, keepdim=True)
            if not torch.isfinite(norms).all():
                raise ValueError("row norms overflowed; use float64 input")
            denominator = norms + eps
            if not torch.isfinite(denominator).all() or not (denominator > 0).all():
                raise ValueError("normalization scale is invalid in the computation dtype")
            unit = work / denominator
            if init_index is None:
                if seed is not None:
                    generator = torch.Generator(device=compute_device).manual_seed(int(seed))
                first = int(torch.randint(n, (1,), device=compute_device, generator=generator).item())
            else:
                first = int(init_index)
            indices[0] = first
            minimum = (1.0 - unit @ unit[first]).clamp_min(0)
            minimum[first] = -torch.inf
            for position in range(1, count):
                selected = int(minimum.argmax().item())
                indices[position] = selected
                dissimilarity = 1.0 - (unit @ unit[selected]).clamp(-1, 1)
                minimum = torch.minimum(minimum, dissimilarity)
                minimum[selected] = -torch.inf

    if is_numpy:
        cpu_indices = indices.cpu().numpy()
        rows = X[cpu_indices]
        if return_numpy:
            return (rows, cpu_indices) if return_indices else rows
        rows_t = torch.from_numpy(rows).to(compute_device)
        return (rows_t, indices) if return_indices else rows_t
    rows_t = X.index_select(0, indices.to(X.device))
    if not return_numpy:
        original_indices = indices.to(X.device)
        return (rows_t, original_indices) if return_indices else rows_t
    compatible_rows = rows_t.float() if rows_t.dtype == torch.bfloat16 else rows_t
    rows = compatible_rows.detach().cpu().numpy()
    cpu_indices = indices.cpu().numpy()
    return (rows, cpu_indices) if return_indices else rows
