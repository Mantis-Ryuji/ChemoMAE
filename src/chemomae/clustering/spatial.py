"""Occupancy-corrected Local Label Agreement for a single spatial label map."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F


__all__ = ["LLAResult", "LLAWindowResult", "local_label_agreement"]


LLAUndefinedReason = Literal[
    "fewer_than_two_valid_pixels", "no_valid_neighbor_pairs", "single_class"
]


@dataclass(frozen=True)
class LLAWindowResult:
    """One neighborhood's corrected score and directed-pair diagnostics."""

    window: int
    score: float
    raw_agreement: float
    matching_pairs: int
    valid_pairs: int
    pixels_with_neighbors: int
    neighbor_pixel_fraction: float
    raw_undefined_reason: Literal["no_valid_neighbor_pairs"] | None
    undefined_reasons: tuple[LLAUndefinedReason, ...]


@dataclass(frozen=True)
class LLAResult:
    """One image's occupancy and separate window results; no image aggregation.

    ``class_labels`` contains sorted original label IDs, and ``class_counts``
    and ``occupancy`` follow that order. Isolated valid pixels contribute to
    occupancy. Undefined floating-point values are NaN; reasons are explicit.
    All diagnostics are CPU Python scalars, independent of the compute device.
    """

    valid_pixels: int
    class_labels: tuple[int, ...]
    class_counts: tuple[int, ...]
    occupancy: tuple[float, ...]
    used_classes: int
    maximum_occupancy: float
    chance_agreement: float
    windows: tuple[LLAWindowResult, ...]


def _integer_map(
    labels: np.ndarray | torch.Tensor, device: torch.device
) -> torch.Tensor:
    if isinstance(labels, np.ndarray):
        if labels.ndim != 2 or labels.dtype.kind not in "iu":
            raise ValueError("labels must be a two-dimensional integer array")
        if labels.dtype.kind == "u" and np.any(labels > np.iinfo(np.int64).max):
            raise ValueError("labels must be representable as signed 64-bit integers")
        # Copying supports read-only arrays and negative strides without
        # exposing the caller's storage to tensor operations.
        array = np.array(labels, dtype=np.int64, order="C", copy=True)
        return torch.as_tensor(array, device=device)
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a NumPy array or Torch tensor")
    if labels.ndim != 2 or labels.layout != torch.strided or labels.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("labels must be a two-dimensional integer tensor")
    return labels.detach().to(device=device, dtype=torch.int64)


def _valid_mask(
    valid_mask: np.ndarray | torch.Tensor,
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(valid_mask, np.ndarray):
        if (valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(shape)
                or valid_mask.dtype.kind not in "biu"
                or not np.all((valid_mask == 0) | (valid_mask == 1))):
            raise ValueError("valid_mask must be a matching binary or boolean array")
        array = np.array(valid_mask, dtype=np.bool_, order="C", copy=True)
        return torch.as_tensor(array, device=device)
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a NumPy array or Torch tensor")
    if (valid_mask.ndim != 2 or valid_mask.shape != shape
            or valid_mask.layout != torch.strided or valid_mask.dtype not in (
                torch.bool, torch.uint8, torch.int8, torch.int16,
                torch.int32, torch.int64
            )):
        raise ValueError("valid_mask must be a matching binary or boolean tensor")
    mask = valid_mask.detach().to(device=device)
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("valid_mask must contain only zero and one")
    return mask.to(dtype=torch.bool)


def _window_widths(windows: Sequence[int]) -> tuple[int, ...]:
    try:
        widths = tuple(windows)
    except TypeError as exc:
        raise ValueError("windows must be a nonempty sequence of positive odd integers") from exc
    if not widths or any(
        isinstance(width, bool) or not isinstance(width, Integral)
        or width <= 0 or width % 2 == 0 for width in widths
    ):
        raise ValueError("windows must be a nonempty sequence of positive odd integers")
    widths = tuple(int(width) for width in widths)
    if len(set(widths)) != len(widths):
        raise ValueError("windows must not contain duplicate widths")
    if any(width * width - 1 > 2**24 for width in widths):
        raise ValueError("window is too wide for exact FP32 local neighbor counts")
    return widths


def _finalize_scores(
    matching: int, pairs: int, counts: tuple[int, ...]
) -> tuple[float, float, tuple[LLAUndefinedReason, ...]]:
    """Apply finite-sample correction with unbounded Python integer products."""
    n = sum(counts)
    reasons: list[LLAUndefinedReason] = []
    if n < 2:
        reasons.append("fewer_than_two_valid_pixels")
    if pairs == 0:
        reasons.append("no_valid_neighbor_pairs")
    if len(counts) == 1:
        reasons.append("single_class")
    raw = matching / pairs if pairs else math.nan
    if reasons:
        return raw, math.nan, tuple(reasons)
    total = n * (n - 1)
    equal = sum(count * (count - 1) for count in counts)
    # This is (A - P) / (1 - P), without subtracting rounded near-one
    # probabilities or overflowing cubic-scale Torch int64 products.
    score = (matching * total - pairs * equal) / (pairs * (total - equal))
    return raw, score, ()


@torch.no_grad()
def local_label_agreement(
    labels: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    windows: Sequence[int] = (3, 5, 9),
    device: str | torch.device | None = None,
    class_chunk: int | None = 16,
) -> LLAResult:
    """Compute equation-(11) Local Label Agreement using binary convolutions.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor, shape (H, W)
        Integer class labels. Zero, negative and nonconsecutive valid labels
        are supported; no label value is reserved for background. NumPy values
        must fit in signed int64; Torch integer dtypes through int64 are accepted.
        Values outside ``valid_mask`` are ignored.
    valid_mask : numpy.ndarray or torch.Tensor, shape (H, W)
        Explicit boolean or binary-integer mask defining the evaluated region.
        An empty region is rejected rather than inferred from label values.
    windows : sequence of int, default (3, 5, 9)
        Distinct positive odd square widths, returned separately in this order.
        The center is excluded; width 1 therefore has no neighbor pairs.
    device : str or torch.device or None, default None
        CPU or CUDA compute device. None preserves the labels tensor's device,
        or the mask tensor's device for NumPy labels, and otherwise uses CPU.
    class_chunk : int or None, default 16
        Maximum number of binary class maps convolved together. None processes
        all present classes at once. This controls temporary memory, not scores.

    Returns
    -------
    LLAResult
        Python integer diagnostics and double-precision Python float scores.
        Each window's ``score`` is the corrected LLA; ``raw_agreement`` is the
        uncorrected matching-pair fraction. Undefined values are NaN and have
        explicit reasons. No averaging across windows or images is performed.

    Raises
    ------
    TypeError
        If inputs are neither NumPy arrays nor Torch tensors.
    ValueError
        For invalid shapes, dtypes, masks, window widths, chunks or devices.
    RuntimeError
        If convolution does not produce exact local integer counts.

    Notes
    -----
    Each valid directed neighbor pair has equal weight. Background, excluded
    pixels and out-of-image positions contribute zero, without wraparound or
    center self-agreement. Isolated valid pixels remain in the occupancy counts.
    Chance agreement is the finite-sample value sum(n_k*(n_k-1))/(N*(N-1)).
    Negative corrected scores are retained, and single-class maps are undefined.

    Binary conv2d runs in FP32 outside AMP, with TF32 disabled for cuDNN. Local
    counts must be exact integers and are reduced in int64. Compact totals are
    copied to the CPU for overflow-safe integer correction and final division.
    Thus counts, and the resulting Python float scores, must agree exactly
    across CPU/CUDA and class chunks when local-count verification passes;
    comparing to a separately rounded FP32 reference requires FP32 tolerance.

    Temporary storage is O(H*W*min(class_chunk, used_classes)) plus convolution
    workspace; no H*W*window**2 neighborhood tensor is materialized. Class
    chunking does not bound image size or backend convolution workspace.
    Inputs are not modified; no fitting, RNG, image aggregation or I/O occurs.
    """
    widths = _window_widths(windows)
    if class_chunk is not None and (
        isinstance(class_chunk, bool) or not isinstance(class_chunk, Integral)
        or class_chunk <= 0
    ):
        raise ValueError("class_chunk must be a positive integer or None")
    if device is None:
        if isinstance(labels, torch.Tensor):
            compute_device = labels.device
        elif isinstance(valid_mask, torch.Tensor):
            compute_device = valid_mask.device
        else:
            compute_device = torch.device("cpu")
    else:
        compute_device = torch.device(device)
    if compute_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; select device='cpu'")
    labels_t = _integer_map(labels, compute_device)
    valid = _valid_mask(valid_mask, labels_t.shape, compute_device)
    evaluated = labels_t[valid]
    n = evaluated.numel()
    if n == 0:
        raise ValueError("valid_mask must contain at least one valid pixel")
    if any(n * min(width * width - 1, n - 1) > torch.iinfo(torch.int64).max
           for width in widths):
        raise ValueError("directed pair counts exceed the int64 reduction limit")
    class_ids, inverse, class_counts_t = torch.unique(
        evaluated, sorted=True, return_inverse=True, return_counts=True
    )
    class_labels = tuple(int(label) for label in class_ids.cpu().tolist())
    counts = tuple(int(count) for count in class_counts_t.cpu().tolist())
    used = len(counts)
    chunk = used if class_chunk is None else int(class_chunk)
    dense = torch.full_like(labels_t, -1)
    dense[valid] = inverse
    occupancy = tuple(count / n for count in counts)
    chance = (
        sum(count * (count - 1) for count in counts) / (n * (n - 1))
        if n > 1 else math.nan
    )
    totals: list[torch.Tensor] = []
    noninteger = torch.zeros((), dtype=torch.bool, device=compute_device)
    with (
        torch.autocast(device_type=compute_device.type, enabled=False),
        torch.backends.cudnn.flags(allow_tf32=False),
    ):
        mask_map = valid.to(dtype=torch.float32)[None, None]
        for width in widths:
            kernel = torch.ones(
                (1, 1, width, width), device=compute_device, dtype=torch.float32
            )
            kernel[0, 0, width // 2, width // 2] = 0
            neighbors = F.conv2d(mask_map, kernel, padding=width // 2)
            rounded = neighbors.round()
            noninteger |= (neighbors != rounded).any()
            pair_count = rounded[0, 0][valid].to(dtype=torch.int64).sum()
            coverage = ((rounded[0, 0] > 0) & valid).sum(dtype=torch.int64)
            matching = torch.zeros((), dtype=torch.int64, device=compute_device)
            for start in range(0, used, chunk):
                ids = torch.arange(start, min(start + chunk, used), device=compute_device)
                class_maps = (dense[None] == ids[:, None, None])[:, None]
                neighbors = F.conv2d(
                    class_maps.to(dtype=torch.float32), kernel, padding=width // 2
                )
                rounded = neighbors.round()
                noninteger |= (neighbors != rounded).any()
                matching += rounded[class_maps].to(dtype=torch.int64).sum()
            totals.extend((matching, pair_count, coverage))
    compact = torch.stack((*totals, noninteger.to(dtype=torch.int64))).cpu().tolist()
    if compact[-1]:
        raise RuntimeError("binary convolution produced noninteger neighbor counts")
    rows: list[LLAWindowResult] = []
    for index, width in enumerate(widths):
        matching, pairs, coverage = (int(value) for value in compact[3 * index:3 * index + 3])
        raw, score, reasons = _finalize_scores(matching, pairs, counts)
        rows.append(LLAWindowResult(
            window=width, score=score, raw_agreement=raw,
            matching_pairs=matching, valid_pairs=pairs,
            pixels_with_neighbors=coverage, neighbor_pixel_fraction=coverage / n,
            raw_undefined_reason="no_valid_neighbor_pairs" if pairs == 0 else None,
            undefined_reasons=reasons,
        ))
    return LLAResult(
        valid_pixels=n, class_labels=class_labels, class_counts=counts,
        occupancy=occupancy, used_classes=used, maximum_occupancy=max(occupancy),
        chance_agreement=chance, windows=tuple(rows),
    )
