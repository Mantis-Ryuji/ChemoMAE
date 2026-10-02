from __future__ import annotations
import torch

__all__ = ["masked_sse", "masked_mse"]


def masked_sse(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "batch_mean",
) -> torch.Tensor:
    r"""
    Aggregate squared reconstruction errors at selected positions.

    True mask entries select the loss region. For masked reconstruction,
    invert the model's visible mask with ``mask = ~visible``.

    Parameters
    ----------
    x_recon : torch.Tensor, shape (B, L)
        Reconstructed spectra.
    x : torch.Tensor, shape (B, L)
        Target spectra.
    mask : torch.Tensor, shape (B, L), dtype=bool
        True includes a position in the loss; false excludes it.
    reduction : {"sum", "mean", "batch_mean"}, default="batch_mean"
        ``sum`` returns total selected SSE. ``mean`` divides by the number
        of selected elements. ``batch_mean`` divides selected SSE by the
        batch size, so its value still depends on the selected channel count.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Notes
    -----
    An empty selection returns zero for every reduction. The empty ``mean``
    fallback creates a constant Tensor without a gradient connection.
    Otherwise gradients can reach both reconstruction and target; disable
    target gradients explicitly when they are not needed. Supply finite
    inputs over the full spectrum: errors are squared before boolean selection.
    """
    diff2 = (x_recon - x).pow(2)[mask]
    if reduction == "sum":
        return diff2.sum()
    if reduction == "mean":
        return diff2.mean() if diff2.numel() > 0 else diff2.new_tensor(0.0)
    if reduction == "batch_mean":
        B = x.size(0)
        return diff2.sum() / max(B, 1)
    raise ValueError(f"unknown reduction: {reduction}")


def masked_mse(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    Aggregate squared errors, averaging selected elements by default.

    True mask entries select the loss region. The default ``mean`` reduction
    computes MSE over those selected elements.

    Parameters
    ----------
    x_recon : torch.Tensor, shape (B, L)
        Reconstructed spectra.
    x : torch.Tensor, shape (B, L)
        Target spectra.
    mask : torch.Tensor, shape (B, L), dtype=bool
        True includes a position in the loss; false excludes it.
    reduction : {"mean", "sum", "batch_mean"}, default="mean"
        ``mean`` returns selected SSE divided by the selected element count.
        ``sum`` returns total selected SSE. ``batch_mean`` divides selected
        SSE by the batch size. This matches :func:`masked_sse` with the same
        reduction; only the default reduction differs.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Notes
    -----
    An empty selection returns zero for every reduction. The empty ``mean``
    fallback creates a constant Tensor without a gradient connection.
    Invert a visible mask when selecting hidden positions, for example
    ``masked_mse(x_recon, x, ~visible)``. Supply finite inputs over the full
    spectrum: errors are squared before boolean selection.
    """
    diff2 = (x_recon - x).pow(2)[mask]
    if reduction == "sum":
        return diff2.sum()
    if reduction == "mean":
        return diff2.mean() if diff2.numel() > 0 else diff2.new_tensor(0.0)
    if reduction == "batch_mean":
        B = x.size(0)
        return diff2.sum() / max(B, 1)
    raise ValueError(f"unknown reduction: {reduction}")
