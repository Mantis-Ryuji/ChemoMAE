from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ..models.losses import masked_mse, masked_sse
from .augmenter import SpectraAugmenter


@dataclass
class TesterConfig:
    r"""
    訓練済み ChemoMAE モデルを評価するための設定クラス。

    Attributes
    ----------
    out_dir : str | Path, default="runs"
        評価履歴を保存するディレクトリ。
    device : str | torch.device, default="cuda"
        評価に使用するデバイス。
    amp : bool, default=True
        AMP (Automatic Mixed Precision) を有効にするかどうか。
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        AMP で使用する dtype。近年の CUDA GPU では bf16 を推奨する。
    loss_type : {"sse", "mse"}, default="mse"
        再構成誤差の種類。
        - "sse": masked sum of squared error
        - "mse": masked mean squared error
    reduction : {"sum", "mean", "batch_mean"}, default="mean"
        `masked_sse` / `masked_mse` に渡す reduction。
    fixed_visible : torch.Tensor[bool] | None, default=None
        固定の可視マスク。True が visible を表す。
        shape は (L,) または (B, L) を受け付ける。
        None の場合はモデル内部で通常どおり mask を生成する。
    log_history : bool, default=True
        True の場合、評価結果を JSON 履歴ファイルに追記する。
    history_filename : str, default="test_history.json"
        `out_dir` 配下に保存する履歴ファイル名。

    Notes
    -----
    - `Tester` はモデル選択を行わない。
    - `augmenter` が `Tester` に渡された場合、モデル入力には augmentation 後の
      スペクトルを使い、再構成ターゲットには元の入力スペクトルを使う。
    - Trainer の `training_history.json` と混在しないよう、デフォルトの履歴ファイル名は
      `test_history.json` とする。
    """

    out_dir: str | Path = "runs"
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    loss_type: Literal["sse", "mse"] = "mse"
    reduction: Literal["sum", "mean", "batch_mean"] = "mean"
    fixed_visible: Optional[torch.Tensor] = None

    log_history: bool = True
    history_filename: str = "test_history.json"


class Tester:
    r"""
    ChemoMAE の masked reconstruction loss を評価するための helper。

    概要
    ----
    指定された DataLoader 全体を走査し、訓練済み ChemoMAE-like model の
    masked reconstruction loss を計算する。

    Parameters
    ----------
    model : nn.Module
        訓練済み ChemoMAE モデル。
        `model(x)` が `(x_recon, z, visible_mask)` を返すことを想定する。
    cfg : TesterConfig | None, default=None
        評価設定。None の場合は `TesterConfig()` を使う。
    augmenter : SpectraAugmenter | None, default=None
        評価前に入力へ適用する optional augmenter。
        None の場合は既存挙動どおり augmentation なしで評価する。

    Notes
    -----
    - model は常に `eval()` に設定して評価する。
    - `SpectraAugmenter` は `eval()` 状態では恒等写像として振る舞う。
      そのため、augmenter が指定された場合は、評価中のみ augmenter だけを
      `train()` に切り替えて augmentation を実際に適用する。
    - 評価後、augmenter の元の train/eval 状態は復元する。
    - augmenter を指定した場合の入出力関係は次のとおり。

      ```python
      x_input = augmenter(x)
      x_recon = model(x_input)
      loss = loss_fn(x_recon, x, masked)
      ```

      つまり、モデル入力は augmentation 後のスペクトルだが、
      再構成ターゲットは元の入力スペクトルである。
      これは Trainer の denoising-style objective と整合する。
    """

    __test__ = False

    def __init__(
        self,
        model: nn.Module,
        cfg: TesterConfig | None = None,
        *,
        augmenter: SpectraAugmenter | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else TesterConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device).eval()
        self.augmenter = augmenter

        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / self.cfg.history_filename

        # 既存履歴をロードする。壊れている場合は安全に無視する。
        self._history: list[Dict[str, Any]] = []
        if self.cfg.log_history and self.history_path.exists():
            try:
                data = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._history = data
            except Exception:
                self._history = []

    def _append_history(self, rec: Dict[str, Any]) -> None:
        """評価履歴を JSON ファイルへ atomic write で追記する。

        Parameters
        ----------
        rec : dict[str, Any]
            追記する評価レコード。

        Returns
        -------
        None
        """
        if not self.cfg.log_history:
            return

        self._history.append(rec)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.history_path)

    def _autocast(self):
        """現在の設定に応じた autocast context を返す。

        Returns
        -------
        context manager
            CUDA AMP が有効な場合は `torch.amp.autocast`。
            それ以外の場合は `nullcontext`。
        """
        if not self.cfg.amp or self.device.type != "cuda":
            from contextlib import nullcontext

            return nullcontext()

        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    def _to_x(self, batch: object) -> torch.Tensor:
        """DataLoader の batch から入力スペクトル `x` を取り出す。

        Parameters
        ----------
        batch : object
            DataLoader が返す batch。
            Tensor, `(x,)`, `(x, y)`, `(x, meta)` などを想定する。

        Returns
        -------
        x : torch.Tensor
            device 上へ転送済みの入力スペクトル。shape は (B, L)。

        Raises
        ------
        TypeError
            batch 内の `x` が torch.Tensor でない場合、または浮動小数 Tensor でない場合。
        ValueError
            `x` が 2D Tensor でない場合。
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"batch must contain a torch.Tensor, got {type(x)}.")
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, L), got shape={tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating tensor.")

        return x.to(self.device, non_blocking=True)

    def _prepare_fixed_visible(
        self,
        fixed_visible: torch.Tensor,
        *,
        batch_size: int,
        num_features: int,
    ) -> torch.Tensor:
        """固定可視マスクを現在 batch の shape に合わせる。

        Parameters
        ----------
        fixed_visible : torch.Tensor
            固定可視マスク。True が visible を表す。
            shape は (L,), (1, L), または (B, L)。
        batch_size : int
            現在 batch のサンプル数。
        num_features : int
            入力スペクトル長 L。

        Returns
        -------
        visible_mask : torch.Tensor
            shape (B, L) の bool Tensor。

        Raises
        ------
        TypeError
            `fixed_visible` が bool Tensor でない場合。
        ValueError
            shape が (L,) または (B, L) として解釈できない場合。
        """
        visible_mask = fixed_visible.to(self.device)

        if visible_mask.dtype != torch.bool:
            raise TypeError(
                "fixed_visible must be a bool tensor where True means visible, "
                f"got dtype={visible_mask.dtype}."
            )

        if visible_mask.ndim == 1:
            if visible_mask.shape[0] != num_features:
                raise ValueError(
                    "1D fixed_visible must have shape (L,), "
                    f"got {tuple(visible_mask.shape)} for L={num_features}."
                )
            visible_mask = visible_mask.unsqueeze(0).expand(batch_size, num_features)

        elif visible_mask.ndim == 2:
            if visible_mask.shape[1] != num_features:
                raise ValueError(
                    "2D fixed_visible must have shape (B, L), "
                    f"got {tuple(visible_mask.shape)} for L={num_features}."
                )

            if visible_mask.shape[0] == 1:
                visible_mask = visible_mask.expand(batch_size, num_features)
            elif visible_mask.shape[0] != batch_size:
                raise ValueError(
                    "2D fixed_visible batch dimension must match current batch size "
                    "or be 1 for broadcasting, "
                    f"got {visible_mask.shape[0]} and batch_size={batch_size}."
                )

        else:
            raise ValueError(
                "fixed_visible must have shape (L,) or (B, L), "
                f"got shape={tuple(visible_mask.shape)}."
            )

        return visible_mask

    def _compute_loss(
        self,
        x_recon: torch.Tensor,
        target: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        """masked reconstruction loss を計算する。

        Parameters
        ----------
        x_recon : torch.Tensor
            再構成スペクトル。shape は (B, L)。
        target : torch.Tensor
            再構成ターゲット。shape は (B, L)。
        masked : torch.Tensor
            損失を計算する位置を示す bool mask。True が masked 位置を表す。

        Returns
        -------
        loss : torch.Tensor
            設定された loss/reduction に基づく scalar loss。

        Raises
        ------
        ValueError
            `cfg.loss_type` が未知の場合。
        """
        if self.cfg.loss_type == "sse":
            return masked_sse(
                x_recon,
                target,
                masked,
                reduction=self.cfg.reduction,
            )
        if self.cfg.loss_type == "mse":
            return masked_mse(
                x_recon,
                target,
                masked,
                reduction=self.cfg.reduction,
            )

        raise ValueError(f"unknown loss_type: {self.cfg.loss_type}")

    def __call__(self, data_loader: Iterable) -> float:
        """DataLoader 全体を評価し、sample-weighted average loss を返す。

        Parameters
        ----------
        data_loader : Iterable
            評価対象の DataLoader または Iterable。
            各 batch は Tensor または Tensor を第1要素に持つ tuple/list を想定する。

        Returns
        -------
        avg : float
            サンプル数で重み付けした平均 test loss。
        """
        self.model.eval()

        augmenter_was_training: bool | None = None
        if self.augmenter is not None:
            self.augmenter.to(self.device)
            augmenter_was_training = self.augmenter.training
            self.augmenter.train()

        total = torch.zeros((), device=self.device)
        count = 0
        fixed_visible = self.cfg.fixed_visible

        try:
            with torch.inference_mode():
                for batch in tqdm(data_loader, desc="Testing", unit="batch"):
                    x = self._to_x(batch)
                    x_input = self.augmenter(x) if self.augmenter is not None else x
                    batch_size, num_features = x_input.shape

                    if fixed_visible is None:
                        with self._autocast():
                            x_recon, _, visible_mask = self.model(x_input)
                    else:
                        visible_mask = self._prepare_fixed_visible(
                            fixed_visible,
                            batch_size=batch_size,
                            num_features=num_features,
                        )
                        with self._autocast():
                            z = self.model.encoder(x_input, visible_mask)
                            x_recon = self.model.decoder(z)

                    masked = ~visible_mask
                    loss = self._compute_loss(x_recon, x, masked)

                    total += loss.detach() * batch_size
                    count += batch_size

        finally:
            if self.augmenter is not None and augmenter_was_training is not None:
                self.augmenter.train(augmenter_was_training)

        avg = (total / max(1, count)).item()
        self._append_history(
            {
                "phase": "test",
                "test_loss": float(avg),
                "loss_type": self.cfg.loss_type,
                "reduction": self.cfg.reduction,
                "augmented": self.augmenter is not None,
            }
        )
        return float(avg)