from __future__ import annotations
import json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Iterable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from ..models.losses import masked_sse, masked_mse
from .callbacks import EarlyStopping, EMACallback


@dataclass
class TrainerConfig:
    r"""
    Configuration container for the `Trainer`.

    概要
    ----
    学習ループでよく使う設定をひとまとめにしたクラス。
    実体はただの名前付き属性（dataclass 風）なので、インスタンス化後に上書き可能。

    Attributes
    ----------
    out_dir : str | Path, default="runs"
        すべての出力（チェックポイント, 履歴, 可視化画像など）を保存するディレクトリ。
    device : {"cuda","mps","cpu"} | None, default=None
        使用デバイス。None の場合は自動判定（cuda → mps → cpu の優先順）。
    amp : bool, default=True
        AMP (Automatic Mixed Precision) を使用するかどうか。
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        AMP の精度種別。
        - "bf16" は近年の GPU (A100/H100 など) で安定。
        - "fp16" はより古い GPU でも動作。
    enable_tf32 : bool, default=False
        TensorFloat-32 を有効化するか。Ampere 以降の GPU で効果あり。
    grad_clip : float | None, default=1.0
        勾配クリッピングの最大ノルム。None の場合は無効。
    use_ema : bool, default=True
        EMA (Exponential Moving Average) によるモデルパラメータ追跡を有効化するか。
    ema_decay : float, default=0.999
        EMA の減衰率。大きいほど履歴の影響が長く残る。
    loss_type : {"sse", "mse"}, default="mse"
        損失関数の種類。
        - "sse" = masked_sse
        - "mse" = masked_mse
    reduction : {"sum", "mean", "batch_mean"}, default="mean"
        損失の集約方法。`masked_sse`/`masked_mse` に渡される。
    early_stop_patience : int | None, default=20
        EarlyStopping を使う場合の patience。
        None の場合は無効。
    early_stop_start_ratio: float = 0.5
        EarlyStopping の監視を開始する時期を総エポック数に対する割合で指定。
        例: 0.5 → 学習の半分を経過してから監視を開始。
    early_stop_min_delta : float, default=0.0
        EarlyStopping 判定での改善幅の閾値。
    resume_from : str | Path | None, default="auto"
        学習再開のチェックポイントパス。
        - "auto" = `out_dir` 内の最新スナップショットを自動検出。
        - str/Path を指定するとそのファイルから復元。
        - None なら常に新規学習。
    """
    out_dir: str | Path = "runs"
    device: Optional[str] = None
    amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" | "fp16"
    enable_tf32: bool = False
    grad_clip: Optional[float] = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    loss_type: str = "mse"   # "sse" | "mse"
    reduction: str = "mean"  # for sse/mse
    early_stop_patience: Optional[int] = 20
    early_stop_start_ratio: float = 0.5
    early_stop_min_delta: float = 0.0
    resume_from: Optional[str | Path] = "auto"


class Trainer:
    r"""
    Generic trainer for ChemoMAE reconstruction tasks.

    概要
    ----
    - ChemoMAE 系モデルの学習ループを管理するクラス。
    - AMP (bf16/fp16), TF32, EMA, 勾配クリッピング, スケジューラに対応。
    - チェックポイントや履歴ログの保存機構を備え、resume/early stopping も可能。

    主な機能
    --------
    - **train_one_epoch**: 1エポック分の学習。AMP/grad_clip/EMA を適用。
    - **validate**: 検証ループ。EMA を一時適用して val_loss を測定。
    - **fit**: 学習全体の制御。履歴保存、checkpointing、early stopping を実行。
    - **save_checkpoint** / **load_checkpoint**:
      モデル・optimizer・scheduler・scaler・EMA・履歴を含む完全な状態を保存/復元。
    - **save_weights_only**:
      モデルの重みのみ保存（推論用の `best_model.pt` など）。

    ログと保存
    ----------
    - out_dir/training_history.json : エポックごとの train/val loss と学習率を追記保存。
    - out_dir/checkpoints/last.pt   : 最新の完全 checkpoint。
    - out_dir/checkpoints/best.pt   : 検証損失が改善した時点での完全 checkpoint。
    - out_dir/best_model.pt         : 検証損失最良時のモデル重みのみ。

    損失の扱い
    ----------
    - モデルは `(x_recon, z, visible_mask)` を返す前提。
    - 内部で `loss = masked_sse/mse(x_recon, x, ~visible_mask)` を計算。

    Parameters
    ----------
    model : nn.Module
        学習対象の ChemoMAE モデル。
    optimizer : torch.optim.Optimizer
        最適化手法。
    train_loader : Iterable
        学習データローダー。
    val_loader : Iterable, optional
        検証データローダー。
    scheduler : torch.optim.lr_scheduler.LambdaLR, optional
        学習率スケジューラ。
    cfg : TrainerConfig, default=TrainerConfig()
        学習設定（デバイス、AMP, EMA, early stop, 出力先など）。

    戻り値
    ------
    fit() : dict
        - "best": {"epoch": int, "val_loss": float}
        - "epochs": int （実際に走ったエポック数）

    使用例
    ------
    >>> cfg = TrainerConfig(device=None)  # None → 自動判定（cuda/mps/cpu）
    >>> trainer = Trainer(model, optimizer, train_loader, val_loader, scheduler=scheduler, cfg=cfg)
    >>> history = trainer.fit(epochs=100)
    >>> print(history["best"])
    {'epoch': 42, 'val_loss': 0.0123}
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        *,
        scheduler: Optional[LambdaLR] = None,
        cfg: TrainerConfig = TrainerConfig(),
    ):
        # ---- device 決定（cfg.device が None の場合は自動判定）----
        if cfg.device is None:
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS Metal
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
            cfg.device = resolved_device  # cfg にも反映（ログ/保存のため）
        self.device = torch.device(cfg.device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.cfg = cfg

        # I/O
        self.out_dir = Path(cfg.out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.out_dir / "checkpoints"; self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / "training_history.json"
        self.history: list[dict] = []

        # AMP/TF32 設定
        self.amp = bool(cfg.amp)
        self.amp_dtype = cfg.amp_dtype.lower()
        if self.device.type == "cuda" and cfg.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        use_scaler = self.amp and (self.amp_dtype == "fp16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        # EMA
        self.ema = EMACallback(self.model, cfg.ema_decay) if cfg.use_ema else None

        # best
        self.best = {"epoch": -1, "val_loss": float("inf")}

        # 既存 history の読み込み（再実行で追記される）
        try:
            if self.history_path.exists():
                self.history = json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            self.history = []

    # ------------------------------ utils ------------------------------
    def _to_x(self, batch):
        return (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device, non_blocking=True)

    def _autocast_ctx(self):
        if not self.amp or self.device.type != "cuda":
            from contextlib import nullcontext
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)

    def _save_history(self, rec: Dict):
        self.history.append(rec)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.history_path)

    # ------------------------------ loss -------------------------------
    def _compute_loss(self, x_recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_type == "sse":
            return masked_sse(x_recon, x, mask, reduction=self.cfg.reduction)
        elif self.cfg.loss_type == "mse":
            return masked_mse(x_recon, x, mask, reduction=self.cfg.reduction)
        raise ValueError(f"unknown loss_type: {self.cfg.loss_type}")

    # ----------------------------- checkpoint --------------------------
    def _checkpoint_state(self, epoch: int) -> Dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
            "scaler": (self.scaler.state_dict() if self.scaler.is_enabled() else None),
            "ema": (self.ema.state_dict() if self.ema is not None else None),
            "ema_decay": (self.ema.decay if self.ema is not None else None),
            "amp": {"enabled": self.amp, "dtype": self.amp_dtype},
            "best": dict(self.best),
            "history": list(self.history),
            "device": self.device.type, 
        }

    def save_checkpoint(self, epoch: int, *, is_best: bool):
        state = self._checkpoint_state(epoch)
        last = self.ckpt_dir / "last.pt"
        tmp_last = last.with_suffix(".pt.tmp")
        torch.save(state, tmp_last.as_posix())
        tmp_last.replace(last)

        if is_best:
            best = self.ckpt_dir / "best.pt"
            tmp_best = best.with_suffix(".pt.tmp")
            torch.save(state, tmp_best.as_posix())
            tmp_best.replace(best)

    def save_weights_only(self, filename: str = "best_model.pt"):
        torch.save(self.model.state_dict(), (self.out_dir / filename).as_posix())

    def load_checkpoint(self, path: str | Path) -> int:
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
        if "history" in state:
            self.history = list(state["history"])
        self.best = dict(state.get("best", self.best))
        last_epoch = int(state.get("epoch", 0))
        return last_epoch + 1

    def _latest_checkpoint(self) -> Optional[Path]:
        p = self.ckpt_dir / "last.pt"
        return p if p.exists() else None

    # ------------------------------ loops ------------------------------
    def train_one_epoch(self) -> float:
        self.model.train()
        meter_sum, meter_cnt = 0.0, 0
        
        # tqdmで進捗表示
        for batch in tqdm(self.train_loader, desc="Training  ", unit="batch"):
            x = self._to_x(batch)

            with self._autocast_ctx():
                # モデルは (x_recon, z, visible_mask) を返す
                x_recon, _, visible_mask = self.model(x)
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

            meter_sum += float(loss.item()) * x.size(0)
            meter_cnt += x.size(0)
        return meter_sum / max(1, meter_cnt)

    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()

        backup = None
        if self.ema is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)

        meter_sum, meter_cnt = 0.0, 0

        with torch.inference_mode():
            for batch in tqdm(self.val_loader, desc="Validating", unit="batch"):
                x = self._to_x(batch)
                with self._autocast_ctx():
                    x_recon, _, visible_mask = self.model(x)
                    loss = self._compute_loss(x_recon, x, ~visible_mask)
                meter_sum += float(loss.item()) * x.size(0)
                meter_cnt += x.size(0)

        if backup is not None:
            self.model.load_state_dict(backup, strict=True)
        return meter_sum / max(1, meter_cnt)

    # ------------------------------ fit -------------------------------
    def fit(self, epochs: int) -> Dict:
        es = EarlyStopping(
            patience=self.cfg.early_stop_patience if self.cfg.early_stop_patience is not None else 10**9,
            min_delta=self.cfg.early_stop_min_delta,
            start_epoch_ratio=self.cfg.early_stop_start_ratio,
        )
        es.setup(epochs)

        start_epoch = 1
        if self.cfg.resume_from is not None:
            if str(self.cfg.resume_from).lower() == "auto":
                p = self._latest_checkpoint()
                if p is not None:
                    start_epoch = self.load_checkpoint(p)
            else:
                start_epoch = self.load_checkpoint(self.cfg.resume_from)

        for epoch in range(start_epoch, epochs + 1):
            ep_t0 = time.time()
            tr = self.train_one_epoch()
            vl = self.validate()
            rec = {"epoch": epoch, "train_loss": tr, "val_loss": vl, "lr": self.optimizer.param_groups[0]["lr"]}
            self._save_history(rec)

            improved = (not math.isnan(vl)) and (self.best["val_loss"] - vl) > self.cfg.early_stop_min_delta
            if improved:
                self.best = {"epoch": epoch, "val_loss": vl}
                self.save_weights_only("best_model.pt")

            # print
            took = time.time() - ep_t0
            tag = "  <-- BEST" if improved else ""
            vl_str = f"{vl:.4f}" if not math.isnan(vl) else "nan"
            print(f"[Epoch {epoch:03d}] train={tr:.4f}  val={vl_str}  lr={self.optimizer.param_groups[0]['lr']:.2e}  time={took:.1f}s{tag}")

            # checkpoint
            self.save_checkpoint(epoch, is_best=improved)

            # early stop
            if (not math.isnan(vl)) and es.step(epoch, vl):
                print(f"[EarlyStop] best@{es.best_epoch:03d} val={es.best:.4f} (start={es._start_epoch}, patience={es.patience})")
                # 直前エポックの状態を last として確実に保存
                self.save_checkpoint(epoch, is_best=False)
                break

        return {"best": self.best, "epochs": epoch}
