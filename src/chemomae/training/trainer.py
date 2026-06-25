from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..models.losses import masked_mse, masked_sse
from .augmenter import SpectraAugmenter
from .callbacks import EMACallback


@dataclass
class TrainerConfig:
    r"""
    Configuration container for the `Trainer`.

    概要
    ----
    ChemoMAE の自己教師あり事前学習に使う固定 epoch / 固定 step 型の
    training loop 設定をまとめた dataclass。

    Notes
    -----
    - validation loss による best checkpoint selection は行わない。
    - early stopping は行わない。
    - 最終モデルは fixed budget の終了時点で保存する。
    - EMA は validation 用ではなく、任意の weight stabilization として扱う。

    Attributes
    ----------
    out_dir : str | Path, default="runs"
        すべての出力（checkpoint, history, exported weights など）を保存するディレクトリ。
    device : {"cuda", "mps", "cpu"} | None, default=None
        使用デバイス。None の場合は自動判定（cuda → mps → cpu の優先順）。
    amp : bool, default=True
        AMP (Automatic Mixed Precision) を使用するかどうか。
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        AMP の精度種別。
    enable_tf32 : bool, default=False
        TensorFloat-32 を有効化するか。Ampere 以降の CUDA GPU で効果あり。
    grad_clip : float | None, default=1.0
        勾配クリッピングの最大ノルム。None の場合は無効。
    use_ema : bool, default=True
        EMA (Exponential Moving Average) によるモデルパラメータ追跡を有効化するか。
    ema_decay : float, default=0.999
        EMA の減衰率。大きいほど過去 step の影響が長く残る。
    loss_type : {"sse", "mse"}, default="mse"
        損失関数の種類。
    reduction : {"sum", "mean", "batch_mean"}, default="mean"
        `masked_sse` / `masked_mse` に渡す損失の集約方法。
    resume_from : str | Path | None, default="auto"
        学習再開用 checkpoint。
        - "auto" = `out_dir/checkpoints/last.pt` を自動検出。
        - str/Path = 指定 checkpoint から復元。
        - None = 常に新規学習。
    """

    out_dir: str | Path = "runs"
    device: Optional[str] = None
    amp: bool = True
    amp_dtype: str = "bf16"
    enable_tf32: bool = False
    grad_clip: Optional[float] = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    loss_type: str = "mse"
    reduction: str = "mean"
    resume_from: Optional[str | Path] = "auto"


class Trainer:
    r"""
    Trainer for ChemoMAE-style masked reconstruction with AMP/EMA/checkpointing.

    概要
    ----
    ChemoMAE 系の masked reconstruction pretraining を固定 epoch / 固定 step で実行する。
    validation loss に基づく model selection や early stopping は行わない。

    主な機能
    --------
    - AMP（bf16/fp16）
    - fp16 CUDA 時のみ GradScaler
    - optional TF32（CUDA）
    - optional EMA（最終重みの安定化用）
    - optional SpectraAugmenter（学習入力にのみ適用）
    - gradient clipping
    - batch-wise scheduler.step()
    - resume 用 full checkpoint
    - raw last / EMA last の weights export
    - training history JSON の atomic write

    モデルの入出力契約
    ------------------
    学習対象 `model` は `model(x)` が **(x_recon, z, visible_mask)** を返すこと。

    - `x_recon`: shape (B, L)
    - `z`: 任意（本 Trainer では損失計算に使用しない）
    - `visible_mask`: bool, shape (B, L), **True=visible**
    - masked 領域は `~visible_mask` として損失計算に渡される。

    Augmenter の扱い
    ----------------
    `augmenter` が指定された場合、学習時のみ

    - `x_input = augmenter(x)`

    をモデルへ入力する。一方で reconstruction target は常に元の `x` とする。
    したがって損失は

    - `loss(x_recon, x, ~visible_mask)`

    で計算される。これは denoising 的な正則化として機能する。

    保存物
    ------
    - `out_dir/training_history.json`
      各 epoch の `epoch`, `train_loss`, `lr`, `time_sec` を保存する。
    - `out_dir/checkpoints/last.pt`
      resume 用 full checkpoint。
    - `out_dir/last_model.pt`
      fixed budget 終了時点の raw model weights。
    - `out_dir/ema_last_model.pt`
      fixed budget 終了時点の EMA model weights（EMA 有効時のみ）。

    Notes
    -----
    - validation loop は持たない。
    - best checkpoint は保存しない。
    - `last.pt` は resume 用の full checkpoint、`last_model.pt` / `ema_last_model.pt` は
      推論・抽出用の weights export として分離している。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: Iterable,
        *,
        scheduler: Optional[LambdaLR] = None,
        augmenter: SpectraAugmenter | None = None,
        cfg: TrainerConfig | None = None,
    ) -> None:
        if cfg is None:
            cfg = TrainerConfig()

        if cfg.device is None:
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
            cfg.device = resolved_device

        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.augmenter = augmenter.to(self.device) if augmenter is not None else None
        self.cfg = cfg

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / "training_history.json"
        self.history: list[dict] = []

        self.amp = bool(cfg.amp)
        self.amp_dtype = cfg.amp_dtype.lower()
        if self.amp_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"amp_dtype must be 'bf16' or 'fp16', got {cfg.amp_dtype!r}")

        if self.device.type == "cuda" and cfg.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        use_scaler = self.amp and self.amp_dtype == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # type: ignore[arg-type]
        self.ema = EMACallback(self.model, cfg.ema_decay) if cfg.use_ema else None

        try:
            if self.history_path.exists():
                loaded_history = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(loaded_history, list):
                    self.history = loaded_history
        except Exception:
            self.history = []

    # ------------------------------ utils ------------------------------
    def _to_x(self, batch: object) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"batch must contain a torch.Tensor, got {type(x)}")
        return x.to(self.device, non_blocking=True)

    def _autocast_ctx(self):
        if not self.amp or self.device.type != "cuda":
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)  # type: ignore[arg-type]

    def _atomic_torch_save(self, obj: object, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp.as_posix())
        tmp.replace(path)

    def _save_history(self, rec: Dict) -> None:
        self.history.append(rec)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.history_path)

    def _save_ema_weights_only(self, filename: str = "ema_last_model.pt") -> None:
        """
        Save EMA weights only.

        Notes
        -----
        - 現在の model weights を backup する。
        - EMA weights を一時的に model へ apply して保存する。
        - 保存後、元の raw weights に戻す。
        - EMA が無効な場合は何もしない。
        """
        if self.ema is None:
            return

        backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            self.ema.apply_to(self.model)
            self._atomic_torch_save(self.model.state_dict(), self.out_dir / filename)
        finally:
            self.model.load_state_dict(backup, strict=True)

    # ------------------------------ loss -------------------------------
    def _compute_loss(self, x_recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_type == "sse":
            return masked_sse(x_recon, x, mask, reduction=self.cfg.reduction)
        if self.cfg.loss_type == "mse":
            return masked_mse(x_recon, x, mask, reduction=self.cfg.reduction)
        raise ValueError(f"unknown loss_type: {self.cfg.loss_type}")

    # ----------------------------- checkpoint --------------------------
    def _checkpoint_state(self, epoch: int) -> Dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "ema_decay": self.ema.decay if self.ema is not None else None,
            "amp": {"enabled": self.amp, "dtype": self.amp_dtype},
            "history": list(self.history),
            "device": self.device.type,
            "selection_rule": "ema_last" if self.ema is not None else "raw_last",
        }

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save the latest full training checkpoint for resume.

        Parameters
        ----------
        epoch : int
            保存対象 epoch。
        """
        state = self._checkpoint_state(epoch)
        self._atomic_torch_save(state, self.ckpt_dir / "last.pt")

    def save_weights_only(self, filename: str = "last_model.pt") -> None:
        """
        Save current raw model weights only.
        """
        self._atomic_torch_save(self.model.state_dict(), self.out_dir / filename)

    def load_checkpoint(self, path: str | Path) -> int:
        """
        Load a full training checkpoint and restore trainer state.

        Returns
        -------
        int
            次に開始すべき epoch。たとえば checkpoint が epoch=10 なら 11 を返す。
        """
        state = torch.load(Path(path).as_posix(), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])

        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])

        if state.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(state["scaler"])
            except Exception:
                pass

        if state.get("ema") is not None:
            if self.ema is None:
                self.ema = EMACallback(self.model, state.get("ema_decay", 0.999))
            self.ema.load_state_dict(state["ema"])

        if "history" in state and isinstance(state["history"], list):
            self.history = list(state["history"])

        last_epoch = int(state.get("epoch", 0))
        return last_epoch + 1

    def _latest_checkpoint(self) -> Optional[Path]:
        p = self.ckpt_dir / "last.pt"
        return p if p.exists() else None

    # ------------------------------ loops ------------------------------
    def train_one_epoch(self) -> float:
        self.model.train()
        if self.augmenter is not None:
            self.augmenter.train()

        meter_sum, meter_cnt = 0.0, 0

        for batch in tqdm(self.train_loader, desc="Training", unit="batch"):
            x = self._to_x(batch)
            x_input = self.augmenter(x) if self.augmenter is not None else x

            with self._autocast_ctx():
                x_recon, _, visible_mask = self.model(x_input)
                loss = self._compute_loss(x_recon, x, ~visible_mask)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            batch_size = x.size(0)
            meter_sum += float(loss.item()) * batch_size
            meter_cnt += batch_size

        return meter_sum / max(1, meter_cnt)

    # ------------------------------ fit -------------------------------
    def fit(self, epochs: int) -> Dict:
        """
        Fit the model for a fixed number of epochs.

        Parameters
        ----------
        epochs : int
            学習を終了する epoch 数。resume 時は checkpoint の次 epoch から再開し、
            `epochs` まで到達したら終了する。

        Returns
        -------
        dict
            実行結果の概要。
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        start_epoch = 1
        if self.cfg.resume_from is not None:
            if str(self.cfg.resume_from).lower() == "auto":
                p = self._latest_checkpoint()
                if p is not None:
                    start_epoch = self.load_checkpoint(p)
            else:
                start_epoch = self.load_checkpoint(self.cfg.resume_from)

        if start_epoch > epochs:
            self.save_weights_only("last_model.pt")
            if self.ema is not None:
                self._save_ema_weights_only("ema_last_model.pt")
            return {
                "epochs": start_epoch - 1,
                "completed": True,
                "final_model": "ema_last_model.pt" if self.ema is not None else "last_model.pt",
            }

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs + 1):
            ep_t0 = time.time()
            train_loss = self.train_one_epoch()
            time_sec = time.time() - ep_t0
            lr = float(self.optimizer.param_groups[0]["lr"])

            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": lr,
                "time_sec": time_sec,
            }
            self._save_history(rec)

            print(
                f"[Epoch {epoch:03d}] "
                f"train={train_loss:.4f}  "
                f"lr={lr:.2e}  "
                f"time={time_sec:.1f}s"
            )

            self.save_checkpoint(epoch)
            last_epoch = epoch

        self.save_weights_only("last_model.pt")
        if self.ema is not None:
            self._save_ema_weights_only("ema_last_model.pt")

        return {
            "epochs": last_epoch,
            "completed": last_epoch >= epochs,
            "final_model": "ema_last_model.pt" if self.ema is not None else "last_model.pt",
        }