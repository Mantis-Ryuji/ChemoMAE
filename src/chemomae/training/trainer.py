"""Reconstruction training with public batch, lifecycle, and checkpoint hooks."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from ..models.losses import masked_mse, masked_sse
from .augmenter import SpectraAugmenter
from .callbacks import EMACallback

__all__ = ["PreparedBatch", "TrainerConfig", "Trainer"]


@dataclass(frozen=True)
class PreparedBatch:
    """Explicit reconstruction inputs, clean targets, and optional visible mask.

    Parameters
    ----------
    model_input : torch.Tensor
        Floating spectra with shape ``(B, L)``, possibly augmented.
    target : torch.Tensor
        Clean floating spectra with the same shape as ``model_input``.
    visible_mask : torch.Tensor or None, default=None
        Boolean mask of the same shape; true means visible. None delegates mask
        generation to the model. Prepared inputs are not augmented again.
    """

    model_input: torch.Tensor
    target: torch.Tensor
    visible_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in (("model_input", self.model_input), ("target", self.target)):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating Torch tensor.")
            if value.layout != torch.strided or value.device.type == "meta":
                raise ValueError(f"{name} must be a dense tensor with stored values.")
            if value.ndim != 2 or any(size == 0 for size in value.shape):
                raise ValueError(f"{name} must have nonempty shape (B, L), got {tuple(value.shape)}.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or infinity.")
        if self.target.shape != self.model_input.shape:
            raise ValueError("model_input and target must have the same shape.")
        if self.visible_mask is not None:
            if not isinstance(self.visible_mask, torch.Tensor) or self.visible_mask.dtype != torch.bool:
                raise TypeError("visible_mask must be a boolean Torch tensor.")
            if self.visible_mask.layout != torch.strided or self.visible_mask.device.type == "meta":
                raise ValueError("visible_mask must be a dense tensor with stored values.")
            if self.visible_mask.shape != self.target.shape:
                raise ValueError("visible_mask must have the same shape as target.")

    def to(self, device: torch.device | str) -> PreparedBatch:
        """Move inputs, targets, and mask together without changing dtype."""
        return PreparedBatch(
            self.model_input.to(device, non_blocking=True),
            self.target.to(device, non_blocking=True),
            None if self.visible_mask is None else self.visible_mask.to(device, non_blocking=True),
        )


@dataclass
class TrainerConfig:
    """Configuration for fixed-epoch reconstruction pretraining.

    Parameters
    ----------
    out_dir : str or pathlib.Path, default="runs"
        History, epoch checkpoints, and final weight export directory.
    device : str, torch.device, or None, default=None
        Computation device; None follows model parameters, or CPU when absent.
    amp : bool, default=False
        Enable explicit CUDA autocast. CPU/MPS AMP is rejected.
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        Autocast dtype. CUDA fp16 uses GradScaler; bf16 requires device support.
    enable_tf32 : bool, default=False
        Enable process-wide TF32 settings on CUDA.
    grad_clip : float or None, default=1.0
        Maximum gradient norm, or None to disable clipping.
    use_ema : bool, default=True
        Track EMA weights after successful optimizer updates.
    ema_decay : float, default=0.999
        EMA decay, between zero and one inclusive.
    loss_type : {"sse", "mse"}, default="mse"
        Reconstruction loss implementation.
    loss_region : {"masked", "all"}, default="masked"
        Nonvisible elements or the full spectrum.
    reduction : {"sum", "mean", "batch_mean"}, default="mean"
        Reduction passed to the loss implementation.
    resume_from : str, pathlib.Path, or None, default="auto"
        Auto selects ``checkpoints/last.pt``; None starts a fresh run and rejects
        existing standard artifacts. An explicit path resumes that checkpoint.

    Notes
    -----
    The scheduler and EMA advance only after successful optimizer updates.
    There is no validation-based selection or early stopping. ``fit`` takes an
    epoch budget, not a successful-update budget.
    """

    out_dir: str | Path = "runs"
    device: str | torch.device | None = None
    amp: bool = False
    amp_dtype: str = "bf16"
    enable_tf32: bool = False
    grad_clip: float | None = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    loss_type: str = "mse"
    loss_region: Literal["masked", "all"] = "masked"
    reduction: str = "mean"
    resume_from: str | Path | None = "auto"

    def __post_init__(self) -> None:
        if self.loss_region not in {"masked", "all"}:
            raise ValueError(f"loss_region must be 'masked' or 'all', got {self.loss_region!r}")
        if self.loss_type not in {"mse", "sse"}:
            raise ValueError(f"loss_type must be 'mse' or 'sse', got {self.loss_type!r}")
        if self.reduction not in {"sum", "mean", "batch_mean"}:
            raise ValueError(f"unknown reduction: {self.reduction!r}")
        if self.amp_dtype.lower() not in {"bf16", "fp16"}:
            raise ValueError(f"amp_dtype must be 'bf16' or 'fp16', got {self.amp_dtype!r}")
        if self.grad_clip is not None and (not math.isfinite(self.grad_clip) or self.grad_clip < 0):
            raise ValueError("grad_clip must be finite and nonnegative, or None.")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay <= 1:
            raise ValueError("ema_decay must be finite and between zero and one.")


class Trainer:
    """Fixed-epoch reconstruction trainer with narrowly scoped override points.

    The default model returns ``(reconstruction, latent, visible_mask)``. Public
    hooks customize batch preparation, forward/loss computation, epoch ordering,
    lifecycle events, and namespaced checkpoint state without replacing the
    AMP/backward/clipping/optimizer loop.

    Parameters
    ----------
    model : torch.nn.Module
        Reconstruction model. Move it to the device before constructing the
        optimizer; the constructor also ensures its device matches the config.
    optimizer : torch.optim.Optimizer
        Optimizer for the supplied model parameters.
    train_loader : Iterable
        Tensor/tuple/list batches, or PreparedBatch objects. ``train_batches``
        can supply a different iterable for each one-based epoch.
    scheduler : torch.optim.lr_scheduler.LRScheduler or None, default=None
        Scheduler supporting ``step()`` without arguments. Successful updates
        advance it after the optimizer; metric-based schedulers are not supported.
    augmenter : SpectraAugmenter or None, default=None
        Applied only by the default preparation of ordinary spectrum batches.
    cfg : TrainerConfig or None, default=None
        Training configuration.

    Notes
    -----
    Resume is at completed epoch boundaries. Core checkpoints do not capture
    arbitrary loader/RNG/application state; use the extension-state hooks for
    caller-owned state and define its validation explicitly.
    """

    STEP_POLICY = "successful_optimizer_update"

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: Iterable,
        *,
        scheduler: LRScheduler | None = None,
        augmenter: SpectraAugmenter | None = None,
        cfg: TrainerConfig | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else TrainerConfig()
        reference = next(model.parameters(), None)
        self.device = torch.device(cfg.device) if cfg.device is not None else (
            reference.device if reference is not None else torch.device("cpu")
        )
        if self.device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("Trainer supports CPU, CUDA, and MPS devices.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        if cfg.amp and self.device.type != "cuda":
            raise ValueError("AMP training requires a CUDA device.")
        if cfg.amp and cfg.amp_dtype.lower() == "fp16" and any(
            parameter.dtype == torch.float16 for parameter in model.parameters()
        ):
            raise ValueError("fp16 AMP cannot use FP16 model parameters; keep FP32 master parameters.")
        if cfg.amp and cfg.amp_dtype.lower() == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("bf16 AMP is unsupported on the selected CUDA device.")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.augmenter = augmenter.to(self.device) if augmenter is not None else None
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.history_path = self.out_dir / "training_history.json"
        standard_artifacts = (
            self.history_path, self.ckpt_dir / "last.pt",
            self.out_dir / "last_model.pt", self.out_dir / "ema_last_model.pt",
        )
        if cfg.resume_from is None and any(path.exists() for path in standard_artifacts):
            raise FileExistsError(
                "Training artifacts already exist; resume explicitly or choose a new out_dir."
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # The checkpoint is authoritative; do not load an independently newer JSON history.
        self.history: list[dict[str, object]] = []
        self.current_epoch = 0
        self.attempted_steps = 0
        self.optimizer_updates = 0
        self.amp_skips = 0
        self._fit_started = False
        self.amp = bool(cfg.amp)
        self.amp_dtype = cfg.amp_dtype.lower()
        if self.device.type == "cuda" and cfg.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        use_scaler = self.amp and self.amp_dtype == "fp16" and self.device.type == "cuda"
        scaler_type = getattr(torch.amp, "GradScaler", None)
        if scaler_type is not None:
            self.scaler = scaler_type("cuda", enabled=use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        self.ema = EMACallback(self.model, cfg.ema_decay) if cfg.use_ema else None

    def train_batches(self, epoch: int) -> Iterable[object]:
        """Return batches for the one-based epoch; override to control ordering."""
        return self.train_loader

    def prepare_batch(self, batch: object) -> PreparedBatch:
        """Prepare input/target/mask; ordinary batches use clean reconstruction targets.

        PreparedBatch bypasses augmentation. For Tensor/tuple/list batches, the
        first Tensor is the clean target and optional augmentation creates the
        model input. The loop moves returned PreparedBatch fields to the device.
        """
        if isinstance(batch, PreparedBatch):
            return batch
        x = self._to_x(batch)
        return PreparedBatch(self.augmenter(x) if self.augmenter is not None else x, x)

    def forward_batch(self, batch: PreparedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction and the actual visible mask, without computing loss."""
        if batch.visible_mask is None:
            reconstructed, _, visible = self.model(batch.model_input)
        else:
            reconstructed, _, visible = self.model(batch.model_input, visible_mask=batch.visible_mask)
        return reconstructed, visible

    def compute_loss(
        self, reconstructed: torch.Tensor, target: torch.Tensor, visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return scalar reconstruction loss; true mask values mean visible."""
        if not all(isinstance(value, torch.Tensor) for value in (reconstructed, target, visible_mask)):
            raise TypeError("Reconstruction, target, and visible_mask must be Torch tensors.")
        if target.ndim != 2 or not reconstructed.is_floating_point() or not target.is_floating_point():
            raise TypeError("Reconstruction and target must be floating tensors with shape (B, L).")
        if reconstructed.shape != target.shape or visible_mask.shape != target.shape:
            raise ValueError("Reconstruction, target, and visible_mask must have the same shape.")
        if visible_mask.dtype != torch.bool:
            raise TypeError("visible_mask must be a boolean tensor.")
        if reconstructed.device != target.device or visible_mask.device != target.device:
            raise ValueError("Reconstruction, target, and visible_mask must be on the same device.")
        for name, value in (("Reconstruction", reconstructed), ("target", target)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"{name} contains NaN or infinity, including positions outside the loss mask."
                )
        if self.cfg.loss_region == "masked":
            loss_mask = ~visible_mask
            if not bool(loss_mask.any()):
                raise ValueError("loss_region='masked' requires at least one masked element.")
        else:
            loss_mask = torch.ones_like(visible_mask)
        loss_fn = masked_sse if self.cfg.loss_type == "sse" else masked_mse
        return loss_fn(reconstructed, target, loss_mask, reduction=self.cfg.reduction)

    def before_epoch(self, epoch: int) -> None:
        """Run before an epoch in fit, after current_epoch has been set."""

    def after_epoch(self, epoch: int, record: dict[str, object]) -> None:
        """Run before history/checkpoint save; add JSON-serializable record fields here."""

    def before_step(self, epoch: int, batch_index: int, batch: PreparedBatch) -> None:
        """Run before forward/backward; batch_index is zero-based. Set a custom LR here."""

    def after_step(
        self, epoch: int, batch_index: int, batch: PreparedBatch,
        *, loss: float, optimizer_updated: bool,
    ) -> None:
        """Run after counters/scheduler/EMA; skipped AMP attempts are reported too."""

    def checkpoint_extra_state(self) -> dict[str, object]:
        """Return caller-owned state stored only under checkpoint['extension_state']."""
        return {}

    def load_checkpoint_extra_state(self, state: dict[str, object]) -> None:
        """Validate/restore extension state; override when checkpoint_extra_state is nonempty."""
        if state:
            raise ValueError(
                "Checkpoint has extension state; override load_checkpoint_extra_state to restore it."
            )

    def _to_x(self, batch: object) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("A training batch cannot be an empty tuple/list.")
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            raise TypeError("A training batch must contain a Torch tensor.")
        if not batch.is_floating_point():
            raise TypeError("Training spectra must have a floating dtype.")
        if batch.layout != torch.strided or batch.device.type == "meta":
            raise ValueError("Training spectra must be dense tensors with stored values.")
        if batch.ndim != 2 or any(size == 0 for size in batch.shape):
            raise ValueError("Training spectra must have nonempty shape (B, L).")
        return batch.to(self.device, non_blocking=True)

    @contextmanager
    def _autocast_ctx(self) -> Iterator[None]:
        """Apply configured precision, overriding ambient CPU/CUDA autocast."""
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        with torch.autocast("cpu", enabled=False), torch.autocast(
            "cuda", dtype=dtype, enabled=self.amp,
        ):
            yield

    def _atomic_torch_save(self, obj: object, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def _save_history(self, record: dict[str, object]) -> None:
        self.history.append(record)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.history, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(self.history_path)

    def _save_ema_weights_only(self, filename: str = "ema_last_model.pt") -> None:
        if self.ema is None:
            return
        backup = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        try:
            self.ema.apply_to(self.model)
            self._atomic_torch_save(self.model.state_dict(), self.out_dir / filename)
        finally:
            self.model.load_state_dict(backup, strict=True)

    def _checkpoint_state(self, epoch: int) -> dict[str, object]:
        extension = self.checkpoint_extra_state()
        if not isinstance(extension, dict) or not all(isinstance(key, str) for key in extension):
            raise TypeError("checkpoint_extra_state must return a dictionary with string keys.")
        return {
            "epoch": epoch, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "ema_decay": self.ema.decay if self.ema is not None else None,
            "amp": {"enabled": self.amp, "dtype": self.amp_dtype},
            "loss_region": self.cfg.loss_region, "loss_type": self.cfg.loss_type,
            "reduction": self.cfg.reduction, "history": list(self.history),
            "device": self.device.type,
            "selection_rule": "ema_last" if self.ema is not None else "raw_last",
            "step_policy": self.STEP_POLICY,
            "progress": {
                "attempted_steps": self.attempted_steps,
                "optimizer_updates": self.optimizer_updates, "amp_skips": self.amp_skips,
            },
            "extension_state": extension,
        }

    def save_checkpoint(self, epoch: int) -> None:
        """Save last.pt; callers outside fit must save only completed epoch boundaries."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Checkpoint epoch must be a nonnegative integer.")
        self._atomic_torch_save(self._checkpoint_state(epoch), self.ckpt_dir / "last.pt")

    def save_weights_only(self, filename: str = "last_model.pt") -> None:
        """Save current raw weights. Full model configuration artifacts are separate work."""
        self._atomic_torch_save(self.model.state_dict(), self.out_dir / filename)

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore a trusted epoch checkpoint and return the next one-based epoch.

        Extension tensors are loaded on CPU; the extension hook owns any device
        transfer. Scaler failures propagate. Broader portable/versioned artifact
        and arbitrary loader/RNG restoration are not supplied by this increment.
        """
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError("Training checkpoint must be a dictionary.")
        for key, expected in (
            ("loss_region", self.cfg.loss_region), ("loss_type", self.cfg.loss_type),
            ("reduction", self.cfg.reduction), ("step_policy", self.STEP_POLICY),
        ):
            if state.get(key) != expected:
                raise ValueError(f"checkpoint {key} mismatch: saved={state.get(key)!r}, current={expected!r}")
        if state.get("amp") != {"enabled": self.amp, "dtype": self.amp_dtype}:
            raise ValueError("Checkpoint AMP configuration mismatch.")
        for key, enabled in (
            ("scheduler", self.scheduler is not None),
            ("scaler", self.scaler.is_enabled()), ("ema", self.ema is not None),
        ):
            if (state.get(key) is not None) != enabled:
                raise ValueError(f"Checkpoint {key} configuration mismatch.")
        epoch = state.get("epoch")
        history = state.get("history")
        progress = state.get("progress")
        extension = state.get("extension_state")
        if type(epoch) is not int or epoch < 0 or not isinstance(history, list):
            raise ValueError("Checkpoint epoch/history is invalid.")
        if not isinstance(progress, dict) or not isinstance(extension, dict):
            raise ValueError("Checkpoint progress/extension_state is missing or invalid.")
        counts = [progress.get(key) for key in ("attempted_steps", "optimizer_updates", "amp_skips")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Checkpoint progress counters must be nonnegative integers.")
        attempted, updates, skips = counts
        if updates > attempted or skips != attempted - updates:
            raise ValueError("Checkpoint progress counters are inconsistent.")
        if not all(isinstance(record, dict) for record in history):
            raise ValueError("Checkpoint history records must be dictionaries.")
        if not all(isinstance(key, str) for key in extension):
            raise ValueError("Checkpoint extension-state keys must be strings.")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        if self.ema is not None:
            ema_state = state["ema"]
            shadow = {key: value.to(self.device) for key, value in ema_state["shadow"].items()}
            self.ema.load_state_dict({**ema_state, "shadow": shadow})
        self.history = [dict(record) for record in history]
        self.attempted_steps, self.optimizer_updates, self.amp_skips = attempted, updates, skips
        self.current_epoch = epoch
        self.load_checkpoint_extra_state(extension)
        return epoch + 1

    def _latest_checkpoint(self) -> Path | None:
        path = self.ckpt_dir / "last.pt"
        return path if path.exists() else None

    def train_one_epoch(self) -> float:
        """Run batches for current_epoch and return a sample-weighted batch loss mean.

        A direct call defaults to epoch one and does not invoke fit's epoch
        events, history, or checkpoint save. Empty iterables and nonfinite losses
        fail instead of reporting a successful zero-loss epoch.
        """
        if self.current_epoch == 0:
            self.current_epoch = 1
        epoch = self.current_epoch
        self.model.train()
        if self.augmenter is not None:
            self.augmenter.train()
        meter_sum, meter_count = 0.0, 0
        batches = tqdm(self.train_batches(epoch), desc="Training", unit="batch")
        for batch_index, raw_batch in enumerate(batches):
            with self._autocast_ctx():
                batch = self.prepare_batch(raw_batch)
                if not isinstance(batch, PreparedBatch):
                    raise TypeError("prepare_batch must return PreparedBatch.")
                batch = batch.to(self.device)
                self.before_step(epoch, batch_index, batch)
                self.optimizer.zero_grad(set_to_none=True)
                reconstructed, visible_mask = self.forward_batch(batch)
                loss = self.compute_loss(reconstructed, batch.target, visible_mask)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.is_floating_point():
                raise TypeError("compute_loss must return a scalar floating tensor.")
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}, batch {batch_index}.")
            if self.scaler.is_enabled():
                previous_scale = self.scaler.get_scale()
                if not math.isfinite(previous_scale) or previous_scale <= 0:
                    raise ValueError("GradScaler scale must remain finite and positive.")
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Standard GradScaler decreases its scale on overflow, including
                # fused optimizers that handle the skipped update internally.
                optimizer_updated = self.scaler.get_scale() >= previous_scale
            else:
                loss.backward()
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                optimizer_updated = True
            self.attempted_steps += 1
            self.optimizer_updates += int(optimizer_updated)
            self.amp_skips += int(not optimizer_updated)
            if optimizer_updated:
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)
            scalar_loss = float(loss.detach())
            self.after_step(
                epoch, batch_index, batch, loss=scalar_loss, optimizer_updated=optimizer_updated
            )
            batch_size = batch.target.shape[0]
            meter_sum += scalar_loss * batch_size
            meter_count += batch_size
        if meter_count == 0:
            raise ValueError(f"No training samples were provided for epoch {epoch}.")
        return meter_sum / meter_count

    def fit(self, epochs: int) -> dict[str, object]:
        """Train through the final one-based epoch, checkpointing completed epochs.

        Resume uses the checkpoint as the history/progress source. Recreate the
        Trainer to resume rather than fitting one instance multiple times.
        """
        if type(epochs) is not int or epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
        if self._fit_started:
            raise RuntimeError("Create a new Trainer to resume; fit can run once per instance.")
        self._fit_started = True
        start_epoch = 1
        if self.cfg.resume_from is not None:
            if str(self.cfg.resume_from).lower() == "auto":
                checkpoint = self._latest_checkpoint()
                if checkpoint is not None:
                    start_epoch = self.load_checkpoint(checkpoint)
                elif any(path.exists() for path in (
                    self.history_path, self.out_dir / "last_model.pt", self.out_dir / "ema_last_model.pt",
                )):
                    raise FileExistsError(
                        "Training artifacts exist without a resume checkpoint; choose a new out_dir."
                    )
            else:
                start_epoch = self.load_checkpoint(self.cfg.resume_from)
        model_n_mask = getattr(self.model, "n_mask", None)
        n_mask = int(model_n_mask) if isinstance(model_n_mask, int) else None
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            self.before_epoch(epoch)
            starting = (self.attempted_steps, self.optimizer_updates, self.amp_skips)
            started = time.perf_counter()
            train_loss = self.train_one_epoch()
            record: dict[str, object] = {
                "epoch": epoch, "train_loss": train_loss,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "time_sec": time.perf_counter() - started,
                "loss_region": self.cfg.loss_region, "n_mask": n_mask,
                "attempted_steps": self.attempted_steps - starting[0],
                "optimizer_updates": self.optimizer_updates - starting[1],
                "amp_skips": self.amp_skips - starting[2],
                "cumulative_attempted_steps": self.attempted_steps,
                "cumulative_optimizer_updates": self.optimizer_updates,
                "cumulative_amp_skips": self.amp_skips,
            }
            core_record = dict(record)
            self.after_epoch(epoch, record)
            if any(record.get(key) != value for key, value in core_record.items()):
                raise ValueError(
                    "after_epoch may add fields but must not change the standard history fields."
                )
            self._save_history(record)
            self.save_checkpoint(epoch)
            print(
                f"[Epoch {epoch:03d}] train={train_loss:.4f} "
                f"updates={record['optimizer_updates']} skips={record['amp_skips']}"
            )
            last_epoch = epoch
        self.save_weights_only()
        if self.ema is not None:
            self._save_ema_weights_only()
        return {
            "epochs": last_epoch, "completed": last_epoch >= epochs,
            "final_model": "ema_last_model.pt" if self.ema is not None else "last_model.pt",
            "attempted_steps": self.attempted_steps, "optimizer_updates": self.optimizer_updates,
            "amp_skips": self.amp_skips,
        }
