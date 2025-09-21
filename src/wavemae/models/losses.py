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
    """
    Sum of squared errors on masked positions.

    Args:
        x_recon: (B, L) 再構成スペクトル
        x:       (B, L) 元スペクトル
        mask:    (B, L) bool, True=masked
        reduction:
          - "sum": 全マスク要素の SSE 合計
          - "mean": マスク要素ごとの平均
          - "batch_mean": バッチあたり平均 (sum / B)

    Returns:
        torch.Tensor: 損失値
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
    """
    Mean squared error on masked positions.

    Args:
        x_recon: (B, L)
        x:       (B, L)
        mask:    (B, L) bool, True=masked
        reduction:
          - "mean": マスク要素ごとの平均（一般的な MSE）
          - "sum": SSE と同じ（全マスク要素合計）
          - "batch_mean": バッチごとに平均化 (SSE/B)

    Returns:
        torch.Tensor: 損失値
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
