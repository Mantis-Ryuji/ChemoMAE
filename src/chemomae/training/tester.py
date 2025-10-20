from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from tqdm import tqdm

import json
import torch
import torch.nn as nn

from ..models.losses import masked_sse, masked_mse


@dataclass
class TesterConfig:
    """
    訓練済み ChemoMAE モデルを評価するための設定クラス。

    Attributes
    ----------
    out_dir : str | Path, default="runs"
        評価履歴を保存するディレクトリ。
    device : str | torch.device, default="cuda"
        評価に使用するデバイス（"cuda" または "cpu"）。
    amp : bool, default=True
        自動混合精度 (Automatic Mixed Precision, AMP) を有効にするかどうか。
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        AMPで使用するデータ型。bf16推奨（Ampere以降）。
    loss_type : {"sse", "mse"}, default="mse"
        再構成誤差の指標。sse=二乗和誤差、mse=平均二乗誤差。
    reduction : {"sum", "mean", "batch_mean"}, default="mean"
        損失の集約方法。
        - "sum": 全サンプルの誤差を単純合計
        - "mean": 要素平均
        - "batch_mean": 各バッチ平均をサンプル数で重み付き平均
    fixed_visible : torch.Tensor[bool] | None, default=None
        固定の可視マスクを与える場合は (L,) または (B,L) の bool テンソル。
        None の場合はモデル内部でマスクを生成。
    log_history : bool, default=True
        True の場合、評価履歴を JSON ファイルに追記。
    history_filename : str, default="training_history.json"
        履歴を保存するファイル名。
    """
    out_dir: str | Path = "runs"
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    # loss settings
    loss_type: Literal["sse", "mse"] = "mse"
    reduction: Literal["sum", "mean", "batch_mean"] = "mean"
    fixed_visible: Optional[torch.Tensor] = None  # True=visible のブールマスク
    # logging
    log_history: bool = True
    history_filename: str = "training_history.json"


class Tester:
    r"""
    Tester for ChemoMAE models.

    指定した DataLoader 全体を走査し、Masked Autoencoder の
    再構成誤差（SSE または MSE）を算出する。

    Parameters
    ----------
    model : nn.Module
        訓練済み ChemoMAE モデル。
    cfg : TesterConfig, optional
        評価設定。省略時はデフォルト設定が使用される。

    Notes
    -----
    - AMP (bf16/fp16) に対応。速度とメモリ効率を改善できる。
    - 評価履歴は JSON ファイルとして保存され、追記される。

    Examples
    --------
    >>> tester_cfg = TesterConfig(device="cuda", amp=True, amp_dtype="bf16")
    >>> tester = Tester(model, tester_cfg)
    >>> import torch
    >>> avg_loss = tester(test_loader)
    >>> print(avg_loss)
    0.1342
    """
    __test__ = False

    def __init__(self, model: nn.Module, cfg: TesterConfig = TesterConfig()):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device).eval()

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / cfg.history_filename

        # 既存履歴をロード（壊れていたら無視）
        self._history: list[Dict[str, Any]] = []
        if self.cfg.log_history and self.history_path.exists():
            try:
                data = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._history = data
            except Exception:
                pass

    def _append_history(self, rec: Dict[str, Any]) -> None:
        if not self.cfg.log_history:
            return
        self._history.append(rec)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._history, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.history_path)

    def _autocast(self):
        if not self.cfg.amp or self.device.type != "cuda":
            from contextlib import nullcontext
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    def __call__(self, data_loader) -> float:
        """
        DataLoader 全体を評価して平均損失を返す。
        """
        self.model.eval()

        total = torch.zeros((), device=self.device)
        count = 0

        crit = self.cfg.loss_type
        red = self.cfg.reduction
        fixed_visible = self.cfg.fixed_visible

        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="Testing", unit="batch"):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device, non_blocking=True)  # (B, L)
                B, L = x.shape

                if fixed_visible is None:
                    with self._autocast():
                        x_recon, _, visible_mask = self.model(x)
                else:
                    visible_mask = fixed_visible.to(self.device)
                    if visible_mask.dim() == 1:
                        visible_mask = visible_mask.expand(B, L)
                    with self._autocast():
                        z = self.model.encoder(x, visible_mask)
                        x_recon = self.model.decoder(z)

                masked = ~visible_mask
                if crit == "sse":
                    loss = masked_sse(x_recon, x, masked, reduction=red)
                elif crit == "mse":
                    loss = masked_mse(x_recon, x, masked, reduction=red)
                else:
                    raise ValueError(f"unknown loss_type: {crit}")

                total += loss.detach() * B
                count += B

        avg = (total / max(1, count)).item()
        self._append_history({
            "phase": "test",
            "test_loss": float(avg),
            "loss_type": crit,
            "reduction": red,
        })
        return avg
