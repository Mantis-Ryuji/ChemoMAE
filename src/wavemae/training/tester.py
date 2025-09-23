from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn

from ..models.losses import masked_sse, masked_mse


class Tester:
    r"""
    Evaluate a trained WaveMAE model on a dataset.

    概要
    ----
    - 学習済み WaveMAE を評価し、masked SSE/MSE の平均損失を算出する。
    - 評価結果は JSON ログ (`out_dir/training_history.json`) に追記保存される。
    - `fixed_visible` を指定すれば、常に同じ可視マスクでの評価も可能。

    Parameters
    ----------
    model : nn.Module
        評価対象の WaveMAE モデル。
    device : str | torch.device, default="cuda"
        評価に用いるデバイス。
    out_dir : str | Path, default="runs"
        評価ログを保存するディレクトリ。

    Methods
    -------
    run(data_loader, *, criterion="sse", reduction="batch_mean", fixed_visible=None) -> float
        DataLoader 全体に対して評価を実行し、平均損失を返す。

    run() 引数
    -----------
    data_loader : iterable
        入力バッチを返す DataLoader。
    criterion : {"sse","mse"}, default="sse"
        損失種別。
    reduction : {"sum","mean","batch_mean"}, default="batch_mean"
        損失の集約方法（`masked_sse` / `masked_mse` に渡される）。
    fixed_visible : torch.Tensor, optional
        固定の可視マスク (True=使う)。未指定なら model.forward 内で自動生成。

    Returns
    -------
    avg_loss : float
        データローダ全体での平均損失。

    Notes
    -----
    - モデルは `(x_recon, z, visible_mask)` を返す前提。
    - 損失は `~visible_mask`（True=masked）上で計算される。
    - ログはリスト形式で追記されるため、繰り返し実行しても履歴が保持される。

    例
    --
    >>> tester = Tester(model, device="cuda", out_dir="runs")
    >>> avg_loss = tester.run(test_loader, criterion="mse")
    >>> print(f"Test loss: {avg_loss:.4f}")
    """
    def __init__(self, model: nn.Module, *, device: str | torch.device = "cuda", out_dir: str | Path = "runs"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / "training_history.json"
        try:
            self.items = json.loads(self.history_path.read_text(encoding="utf-8"))
            if not isinstance(self.items, list):
                self.items = []
        except Exception:
            self.items = []

    def _append_history(self, rec: Dict):
        self.items.append(rec)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.history_path)

    @torch.no_grad()
    def run(
        self,
        data_loader,
        *,
        criterion: str = "sse",      # "sse" | "mse"
        reduction: str = "batch_mean",
        fixed_visible: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Parameters
        ----------
        data_loader : iterable
            入力バッチを返す DataLoader。
        criterion : {"sse","mse"}
            損失種別。
        reduction : str
            損失の集約方法。
        fixed_visible : torch.Tensor, optional
            固定の可視マスク (True=使う)。未指定なら model.forward 内で生成。
        """
        self.model.eval()
        total, count = 0.0, 0

        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)

            if fixed_visible is None:
                # モデルに任せて visible_mask を生成
                x_recon, _, visible_mask = self.model(x)
            else:
                B, L = x.shape
                visible_mask = fixed_visible.to(self.device)
                visible_mask = visible_mask.expand(B, L) if visible_mask.dim() == 1 else visible_mask
                z = self.model.encoder(x, visible_mask)
                x_recon = self.model.decoder(z)

            if criterion == "sse":
                loss = masked_sse(x_recon, x, ~visible_mask, reduction=reduction)
            elif criterion == "mse":
                loss = masked_mse(x_recon, x, ~visible_mask, reduction=reduction)
            else:
                raise ValueError(f"unknown criterion: {criterion}")

            total += float(loss.item()) * x.size(0)
            count += x.size(0)

        avg = total / max(1, count)
        self._append_history({"phase": "test", "test_loss": float(avg), "criterion": criterion})
        return avg
