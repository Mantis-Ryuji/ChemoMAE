from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn

from ..models.losses import masked_sse, masked_mse


class Tester:
    """学習済み WaveMAE を評価（平均 masked SSE/MSE）。履歴に追記ログ。"""

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
        fixed_mask: Optional[torch.Tensor] = None,
    ) -> float:
        self.model.eval()
        total, count = 0.0, 0

        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)

            if fixed_mask is None:
                x_recon, _, mask = self.model(x)          # モデルから mask を得る
            else:
                B, L = x.shape
                mask = fixed_mask.to(self.device)
                mask = mask.expand(B, L) if mask.dim() == 1 else mask
                z = self.model.encoder(x, ~mask)
                x_recon = self.model.decoder(z)

            if criterion == "sse":
                loss = masked_sse(x_recon, x, mask, reduction=reduction)
            elif criterion == "mse":
                loss = masked_mse(x_recon, x, mask, reduction=reduction)
            else:
                raise ValueError(f"unknown criterion: {criterion}")

            total += float(loss.item()) * x.size(0)
            count += x.size(0)

        avg = total / max(1, count)
        self._append_history({"phase": "test", "test_loss": float(avg), "criterion": criterion})
        return avg
