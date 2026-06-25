from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..models.chemo_mae import ChemoMAE
from .augmenter import SpectraAugmenter


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
      ChemoMAE 側の乱数マスクには依存しない。
    - `augmenter` が指定された場合は、抽出前に入力スペクトルへ augmentation を適用する。
      stochastic augmentation を使うため、同じ入力でも抽出結果は乱数に依存しうる。
    - 保存と返却形式は独立：`save_path=".npy"` かつ `return_numpy=False` のような組み合わせも可。
    """

    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    save_path: Optional[str | Path] = None
    return_numpy: bool = False


class Extractor:
    r"""
    Helper to extract latent features Z from a trained ChemoMAE in all-visible mode.

    概要
    ----
    - `ChemoMAE.encoder` を **全可視マスク (visible_mask=True)** で呼び出し、
      潜在表現 Z を一括で抽出する。
    - 推論時は AMP (bf16/fp16) に対応し、結果は CPU に集約される。
    - `ExtractorConfig.save_path` が指定されていれば自動保存される。
    - optional `SpectraAugmenter` を指定した場合、特徴抽出前の入力に augmentation を適用する。

    Parameters
    ----------
    model : ChemoMAE
        学習済み ChemoMAE モデル。
    cfg : ExtractorConfig | None, default=None
        抽出処理の設定（デバイス、AMP、保存先、返却形式など）。
        None の場合は `ExtractorConfig()` を使う。
    augmenter : SpectraAugmenter | None, default=None
        特徴抽出前に入力へ適用する optional augmenter。
        None の場合は augmentation なしで抽出する。

    Notes
    -----
    - model は常に `eval()` に設定する。
    - `SpectraAugmenter` は `eval()` では恒等写像になるため、
      augmenter が指定された場合は、抽出中のみ `augmenter.train()` に設定して適用する。
    - 抽出後、augmenter の元の training/eval 状態は復元する。
    - `save_path`:
        * 拡張子が ".npy" の場合 → `np.save` で保存。
        * それ以外 → `torch.save` で保存。
    - 返り値の型は `cfg.return_numpy` に依存する。
    """

    def __init__(
        self,
        model: ChemoMAE,
        cfg: ExtractorConfig | None = None,
        *,
        augmenter: SpectraAugmenter | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else ExtractorConfig()
        self.device = torch.device(self.cfg.device)
        self.augmenter = augmenter

    def _autocast(self):
        if not self.cfg.amp or self.device.type != "cuda":
            from contextlib import nullcontext

            return nullcontext()

        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)

    def _to_x(self, batch: object) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"batch must contain a torch.Tensor, got {type(x)}.")
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, L), got shape={tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating tensor.")

        return x.to(self.device, non_blocking=True)

    def _empty_features(self) -> torch.Tensor:
        out_features = int(self.model.encoder.to_latent.out_features)
        return torch.empty(0, out_features)

    def __call__(self, loader: Iterable) -> torch.Tensor | np.ndarray:
        self.model.to(self.device)
        self.model.eval()

        augmenter_was_training: bool | None = None
        if self.augmenter is not None:
            self.augmenter.to(self.device)
            augmenter_was_training = self.augmenter.training
            self.augmenter.train()

        feats: list[torch.Tensor] = []

        try:
            with torch.inference_mode():
                for batch in tqdm(loader, desc="Extracting", unit="batch"):
                    x = self._to_x(batch)
                    x_input = self.augmenter(x) if self.augmenter is not None else x

                    batch_size, num_features = x_input.shape
                    visible_mask = torch.ones(
                        batch_size,
                        num_features,
                        dtype=torch.bool,
                        device=self.device,
                    )

                    with self._autocast():
                        z = self.model.encoder(x_input, visible_mask)

                    feats.append(z.detach().float().cpu())
        finally:
            if self.augmenter is not None and augmenter_was_training is not None:
                self.augmenter.train(augmenter_was_training)

        z_all = torch.cat(feats, dim=0) if feats else self._empty_features()

        if self.cfg.save_path is not None:
            path = Path(self.cfg.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".npy":
                np.save(path.as_posix(), z_all.numpy())
            else:
                torch.save(z_all, path.as_posix())

        return z_all.numpy() if self.cfg.return_numpy else z_all