# Trainer — ChemoMAE Training Loop

> Module: `chemomae.training.trainer`

This document describes the `Trainer` and its configuration (`TrainerConfig`), covering AMP/TF32 support, EMA, gradient clipping, masked loss computation, checkpointing/resume, and full training–validation management.

---

## Overview

The `Trainer` implements a robust training routine for **masked reconstruction** using ChemoMAE.
It integrates precision management, exponential moving averages, and fully resumable state tracking.

### Key features

* **Automatic mixed precision (AMP)** — `torch.amp.autocast` (bf16/fp16)
* **TF32 acceleration** on Ampere+ GPUs
* **EMA (Exponential Moving Average)** of model parameters
* **Gradient clipping** (global-norm based)
* **Masked losses** (`masked_mse`, `masked_sse`) consistent with MAE principle
* **Checkpointing and resume** — full training state (model, optimizer, scheduler, scaler, EMA, RNG, history)
* **JSON-based training history** for reproducibility and visualization

The model must return `(x_recon, z, visible_mask)`, and the Trainer computes loss only on the **masked** tokens (`mask = ~visible_mask`).

---

## Configuration — `TrainerConfig`

```python
@dataclass
class TrainerConfig:
    out_dir: str | Path = "runs"
    device: Optional[str] = None   # {"cuda","mps","cpu"} or None (= auto-detect)
    amp: bool = True
    amp_dtype: str = "bf16"        # {"bf16","fp16"}
    enable_tf32: bool = False
    grad_clip: float | None = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    loss_type: str = "mse"         # {"mse","sse"}
    reduction: str = "mean"        # {"sum","mean","batch_mean"}
    early_stop_patience: int | None = 20
    early_stop_start_ratio: float = 0.5
    early_stop_min_delta: float = 0.0
    resume_from: str | Path | None = "auto"
```

#### Parameters

| Name                     | Type                      | Default  | Description                                                                  |
| ------------------------ | ------------------------- | -------- | ---------------------------------------------------------------------------- |
| `out_dir`                | `str` or `Path`           | `"runs"` | Output directory for checkpoints and logs.                                   |
| `device`                 | `str` or `None`           | `None`   | `"cuda"`, `"mps"`, or `"cpu"`; if `None`, auto-detect in order CUDA→MPS→CPU. |
| `amp`                    | `bool`                    | `True`   | Enable PyTorch autocast for mixed precision.                                 |
| `amp_dtype`              | `str`                     | `"bf16"` | Precision type (`"bf16"` stable, `"fp16"` faster).                           |
| `enable_tf32`            | `bool`                    | `False`  | Allow TF32 matmul/convolution acceleration (Ampere+).                        |
| `grad_clip`              | `float` or `None`         | `1.0`    | Gradient norm clipping threshold; `None` disables.                           |
| `use_ema`                | `bool`                    | `True`   | Track an exponential moving average of model weights.                        |
| `ema_decay`              | `float`                   | `0.999`  | EMA decay rate.                                                              |
| `loss_type`              | `str`                     | `"mse"`  | Loss type (`"mse"` or `"sse"`).                                              |
| `reduction`              | `str`                     | `"mean"` | Reduction mode (`"sum"`, `"mean"`, `"batch_mean"`).                          |
| `early_stop_patience`    | `int` or `None`           | `20`     | Stop if no improvement within this many epochs.                              |
| `early_stop_start_ratio` | `float`                   | `0.5`    | Start monitoring after this fraction of total epochs.                        |
| `early_stop_min_delta`   | `float`                   | `0.0`    | Minimum improvement threshold to reset patience.                             |
| `resume_from`            | `str` or `Path` or `None` | `"auto"` | `"auto"` resumes from `out_dir/checkpoints/last.pt` if available.            |

---

## API — `Trainer`

```python
trainer = Trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    *,
    scheduler: Optional[LambdaLR] = None,
    cfg: TrainerConfig = TrainerConfig(),
)
```

### Methods

* **`fit(epochs)` → `dict`**
  Executes the full training loop with validation, checkpointing, and early stopping.
  Returns: `{"best": {"epoch": int, "val_loss": float}, "epochs": int}`.

* **`train_one_epoch()` → `float`**
  One training epoch under `model.train()`.
  Applies AMP, gradient clipping, scheduler step, and EMA update.
  Returns mean training loss.

* **`validate()` → `float`**
  Runs evaluation under `model.eval()`.
  If EMA is enabled, temporarily applies EMA weights, evaluates, and restores.

* **Checkpoint I/O**

  * `save_checkpoint(epoch, is_best)` — Saves `last.pt` and optionally `best.pt`.
  * `save_weights_only(filename="best_model.pt")` — Weights-only export for inference.
  * `load_checkpoint(path)` → `int` — Loads state and returns next epoch index.

---

## Directory Layout & History

| File                              | Description                             |
| --------------------------------- | --------------------------------------- |
| `{out_dir}/training_history.json` | Per-epoch JSON records (loss, lr, etc.) |
| `{out_dir}/checkpoints/last.pt`   | Full checkpoint (latest)                |
| `{out_dir}/checkpoints/best.pt`   | Full checkpoint (best validation loss)  |
| `{out_dir}/best_model.pt`         | Weights only                            |

Example record:

```json
{"epoch": 12, "train_loss": 0.0231, "val_loss": 0.0219, "lr": 2.0e-4}
```

Updates are atomic via a temp file and replace to avoid corruption.

---

## Masked Loss Handling

The Trainer computes losses **only on masked tokens**, using the inverted visibility mask:

```python
mask = ~visible
if cfg.loss_type == "mse":
    loss = masked_mse(x_recon, x, mask, reduction=cfg.reduction)
elif cfg.loss_type == "sse":
    loss = masked_sse(x_recon, x, mask, reduction=cfg.reduction)
```

| Reduction      | Meaning                                            |
| -------------- | -------------------------------------------------- |
| `"mean"`       | Average over all masked elements.                  |
| `"sum"`        | Total SSE.                                         |
| `"batch_mean"` | `(SSE / B)`                                        |

---

## Precision & Performance

* **Autocast:** Uses `torch.amp.autocast("cuda", dtype=bf16|fp16)` when `amp=True`.
* **GradScaler:** Enabled automatically for `fp16` (not needed for `bf16`).
* **TF32:** Activates TF32 matmul/cuDNN acceleration via `torch.backends.cuda.matmul.allow_tf32=True`.
* **Gradient clipping:** `clip_grad_norm_` applied after unscaling when GradScaler is active.
* **EMA:** Maintains shadow parameters with decay `ema_decay`; applied during validation.

---

## Usage Examples

### Minimal training loop

```python
from chemomae.models import ChemoMAE
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.training.trainer import Trainer, TrainerConfig

model = ChemoMAE(seq_len=256, latent_dim=64, n_blocks=16, n_mask=12)
opt = build_optimizer(model, lr=2e-4, weight_decay=0.05)
sched = build_scheduler(opt, steps_per_epoch=len(train_loader), epochs=100, warmup_epochs=5)
cfg = TrainerConfig(out_dir="runs", amp=True, amp_dtype="bf16", use_ema=True, loss_type="mse", reduction="mean")

trainer = Trainer(model, opt, train_loader, val_loader, device="cuda", scheduler=sched, cfg=cfg)
history = trainer.fit(epochs=100)
print("Best:", history["best"])  # {'epoch': ..., 'val_loss': ...}
```

### Resume training automatically

```python
cfg = TrainerConfig(out_dir="runs", resume_from="auto")
trainer = Trainer(model, opt, train_loader, val_loader, device="cuda", scheduler=sched, cfg=cfg)
trainer.fit(epochs=100)
```

---

## Notes & Gotchas

* **Data loader output:**
  If the loader yields `(x, meta)`, the Trainer automatically uses only `x`.
* **Shape consistency:**
  `x.shape == visible.shape == (B, L)` must hold; `L` = `seq_len`.
* **History persistence:**
  Existing `training_history.json` is read and appended to when continuing runs.
* **Checkpoints:**

  * `best_model.pt`: weights only (inference)
  * `checkpoints/best.pt`: full training state
* **Reduction tip:**
  Prefer `"batch_mean"` when the number of masked tokens per sample may vary.

---

## Version

* Introduced in `chemomae.training.trainer` — initial public draft.