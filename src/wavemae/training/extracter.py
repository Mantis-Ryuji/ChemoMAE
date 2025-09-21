from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.wave_mae import WaveMAE  # 型ヒント用


@dataclass
class ExtractConfig:
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    save_path: Optional[str | Path] = None  # ".npy" または ".pt"
    return_numpy: bool = False              # True: np.ndarray 返却


class Extracter:
    """
    WaveMAE の encode() を使って、全可視（all-visible）で潜在表現 Z を一括抽出するヘルパー。
    - モデルのマスクは使わず、常に visible_mask=True のみで encode します。
    - AMP 対応（bf16/fp16）。Z は CPU に集約。
    - 保存: save_path が ".npy" なら np.save、その他は torch.save。
    """

    def __init__(self, model: WaveMAE, cfg: ExtractConfig = ExtractConfig()):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def _autocast(self):
        if not self.cfg.amp or self.device.type != "cuda":
            from contextlib import nullcontext
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)

    @torch.no_grad()
    def __call__(self, loader: Iterable) -> torch.Tensor | np.ndarray:
        self.model.eval().to(self.device)
        feats = []

        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)  # (B, L)
            B, L = x.shape
            visible = torch.ones(B, L, dtype=torch.bool, device=self.device)

            with self._autocast():
                z = self.model.encode(x, visible)  # (B, D)

            feats.append(z.detach().cpu())

        Z = torch.cat(feats, dim=0) if feats else torch.empty(
            0, self.model.encoder.to_latent.out_features
        )

        # 保存が指定されていれば書き出し
        if self.cfg.save_path is not None:
            path = Path(self.cfg.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".npy":
                np.save(path.as_posix(), Z.numpy())
            else:
                torch.save(Z, path.as_posix())

        return Z.numpy() if self.cfg.return_numpy else Z
