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
    Masked Sum of Squared Errors (SSE).

    概要
    ----
    `mask=True` の位置（＝**隠す領域**）に対してのみ二乗誤差を集計します。
    通常は可視マスク `visible` から `mask = ~visible` を作って渡します。

    Parameters
    ----------
    x_recon : torch.Tensor, shape (B, L)
        再構成系列。
    x : torch.Tensor, shape (B, L)
        元系列。
    mask : torch.Tensor, shape (B, L), dtype=bool
        True=損失を計算する位置（=隠す領域）。False=可視で損失対象外。
    reduction : {"sum", "mean", "batch_mean"}, default="batch_mean"
        集約方法を指定。
        - "sum": すべてのマスク要素の SSE 合計
        - "mean": マスク要素数で割った平均（要素平均）
        - "batch_mean": バッチ平均（SSE / B）。マスク要素数の違いに依存しないスケーリング

    Returns
    -------
    torch.Tensor
        スカラー損失。

    Notes
    -----
    - 空マスク（`mask.sum()==0`）のとき：
      - "sum" と "batch_mean" は 0 を返す
      - "mean" は 0 を返す（ゼロ割りを避けて 0 にフォールバック）
    - 勾配は `x_recon` と `x` の双方に流れます（必要に応じて `x` の `requires_grad` を切る）。

    Examples
    --------
    >>> x = torch.randn(2, 4)
    >>> x_rec = x + 0.1
    >>> visible = torch.tensor([[1,1,0,0],[1,0,1,0]], dtype=torch.bool)
    >>> loss = masked_sse(x_rec, x, ~visible, reduction="batch_mean")
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
    Masked Mean Squared Error (MSE).

    概要
    ----
    `mask=True` の位置（＝**隠す領域**）に対してのみ二乗誤差を集計します。
    「MSE」のデフォルト挙動は **マスク要素の平均**（"mean"）です。

    Parameters
    ----------
    x_recon : torch.Tensor, shape (B, L)
        再構成系列。
    x : torch.Tensor, shape (B, L)
        元系列。
    mask : torch.Tensor, shape (B, L), dtype=bool
        True=損失を計算する位置（=隠す領域）。False=可視で損失対象外。
    reduction : {"mean", "sum", "batch_mean"}, default="mean"
        集約方法を指定。
        - "mean": マスク要素数で割った平均（一般的な MSE）
        - "sum": すべてのマスク要素の SSE 合計（SSE と同義）
        - "batch_mean": バッチ平均（SSE / B）

    Returns
    -------
    torch.Tensor
        スカラー損失。

    Notes
    -----
    - 空マスク（`mask.sum()==0`）のとき：
      - "mean" は 0 を返す
      - "sum" / "batch_mean" も 0 を返す
    - 可視マスクを使う場合は `masked_mse(x_rec, x, ~visible)` のように反転して渡してください。

    Examples
    --------
    >>> x = torch.randn(2, 4)
    >>> x_rec = torch.randn(2, 4)
    >>> visible = torch.tensor([[1,1,0,0],[1,0,1,0]], dtype=torch.bool)
    >>> mse = masked_mse(x_rec, x, ~visible, reduction="mean")
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
