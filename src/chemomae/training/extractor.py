from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Literal
from tqdm import tqdm

import numpy as np
import torch

from ..models.chemo_mae import ChemoMAE


@dataclass
class ExtractorConfig:
    r"""
    Configuration for latent feature extraction with `Extractor`.

    概要
    ----
    - 学習済み ChemoMAE から **全可視 (visible=True)** で潜在表現 z を一括抽出する際の設定。
    - AMP や出力の保存形式・返却形式を制御する。

    Attributes
    ----------
    device : str | torch.device, default="cuda"
        推論に用いるデバイス（"cuda" / "cpu" など）。
    amp : bool, default=True
        AMP (Automatic Mixed Precision) を使用するか。
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        AMP の精度種別。GPU に応じて選択（A100/H100 などは bf16 が安定）。
    save_path : str | Path | None, default=None
        抽出した潜在表現 Z の保存先。拡張子で書式を自動判定：
        - ".npy" → `np.save`（numpy array で保存）
        - その他 → `torch.save`（torch.Tensor で保存）
        None の場合は保存しない。
    return_numpy : bool, default=False
        `True` の場合は `np.ndarray` を返す。`False` なら `torch.Tensor` を返す。

    Notes
    -----
    - `Extractor` は常に **全可視** で `model.encoder(x, visible)` を呼び出すため、
      乱数マスクに依存しない **決定的な特徴抽出** を行う。
    - 保存と返却形式は独立：`save_path=".npy"` かつ `return_numpy=False` のような組み合わせも可。
    """
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    save_path: Optional[str | Path] = None  # ".npy" または ".pt"
    return_numpy: bool = False              # True: np.ndarray 返却


class Extractor:
    r"""
    Helper to extract latent features Z from a trained ChemoMAE in all-visible mode.

    概要
    ----
    - `ChemoMAE.encoder` を **全可視マスク (visible_mask=True)** で呼び出し、
      潜在表現 Z を一括で抽出する。
    - 推論時は AMP (bf16/fp16) に対応し、結果は CPU に集約される。
    - `ExtractorConfig.save_path` が指定されていれば自動保存される。

    Parameters
    ----------
    model : ChemoMAE
        学習済み ChemoMAE モデル。
    cfg : ExtractorConfig, default=ExtractorConfig()
        抽出処理の設定（デバイス、AMP、保存先、返却形式など）。

    Usage
    -----
    >>> model = ChemoMAE(...)
    >>> cfg = ExtractorConfig(device="cuda", save_path="latent.npy", return_numpy=True)
    >>> extractor = Extractor(model, cfg)
    >>> Z = extractor(loader)   # -> np.ndarray shape (N, D)

    Notes
    -----
    - **マスクは一切使わない** ため、乱数に依存しない決定的な潜在表現が得られる。
    - `save_path`:
        * 拡張子が ".npy" の場合 → `np.save` で保存。
        * それ以外 → `torch.save` で保存。
    - 返り値の型は `cfg.return_numpy` に依存する。
    """
    def __init__(self, model: ChemoMAE, cfg: ExtractorConfig = ExtractorConfig()):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def _autocast(self):
        if not self.cfg.amp or self.device.type != "cuda":
            from contextlib import nullcontext
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)

    def __call__(self, loader: Iterable) -> torch.Tensor | np.ndarray:
        self.model.eval().to(self.device)
        feats = []

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Extracting", unit="batch"):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device, non_blocking=True)  # (B, L)
                B, L = x.shape
                visible_mask = torch.ones(B, L, dtype=torch.bool, device=self.device)

                with self._autocast():
                    z = self.model.encoder(x, visible_mask)  # (B, D)

                feats.append(z.detach().cpu())

        Z = torch.cat(feats, dim=0) if feats else torch.empty(
            0, self.model.encoder.to_latent.out_features
        )

        if self.cfg.save_path is not None:
            path = Path(self.cfg.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".npy":
                np.save(path.as_posix(), Z.numpy())
            else:
                torch.save(Z, path.as_posix())

        return Z.numpy() if self.cfg.return_numpy else Z
