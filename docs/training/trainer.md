# Trainer — ChemoMAE Training Loop

> Module: `chemomae.training.trainer`

This document describes the `Trainer` and its configuration (`TrainerConfig`), covering AMP/TF32 support, EMA, gradient clipping, optional spectral augmentation, masked loss computation, checkpointing/resume, and full training–validation management.

---

## Overview

The `Trainer` implements a robust training routine for **masked reconstruction** using ChemoMAE.  
It integrates precision management, exponential moving averages, optional spectral augmentation, and fully resumable state tracking.

### Key features

* **Automatic mixed precision (AMP)** — `torch.amp.autocast` (bf16/fp16)
* **TF32 acceleration** on Ampere+ GPUs
* **EMA (Exponential Moving Average)** of model parameters
* **Optional `SpectraAugmenter`** applied only during training
* **Gradient clipping** (global-norm based)
* **Masked losses** (`masked_mse`, `masked_sse`) consistent with the MAE principle
* **Checkpointing and resume** — full training state (model, optimizer, scheduler, scaler, EMA, history)
* **Weights-only export** for best and final model variants
* **JSON-based training history** for reproducibility and visualization

The model must return `(x_recon, z, visible_mask)`, and the Trainer computes loss only on the **masked** tokens (`mask = ~visible_mask`).  
If an augmenter is provided, the model input is augmented, but the reconstruction target remains the **original** input spectrum.

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
| `amp_dtype`              | `str`                     | `"bf16"` | Precision type (`"bf16"` stable, `"fp16"` supported).                        |
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
    augmenter: SpectraAugmenter | None = None,
    cfg: TrainerConfig | None = None,
)
```

### Constructor behavior

* `cfg=None` creates a fresh `TrainerConfig` instance internally.

* If `cfg.device is None`, device is auto-resolved in the order:

  ```python
  "cuda" -> "mps" -> "cpu"
  ```

* `augmenter` is optional and, if provided, is moved onto the same device as the model.

---

## Methods

### `fit(epochs)` → `dict`

Executes the full training loop with validation, checkpointing, early stopping, and final export.

Returns:

```python
{"best": {"epoch": int, "val_loss": float}, "epochs": int}
```

Behavior:

* **With validation**

  * best selection is based on **EMA-applied validation loss**
  * `best_model_ema.pt` is saved using **EMA weights** when EMA is enabled
  * `best_model.pt` is saved using **raw weights** only when EMA is disabled
  * `last_model.pt` is exported at the end
  * `last_model_ema.pt` is exported at the end if EMA is enabled

* **Without validation**

  * no best model is selected
  * `last_model.pt` is exported at the end
  * `last_model_ema.pt` is exported at the end if EMA is enabled

### `train_one_epoch()` → `float`

Runs one training epoch under `model.train()`.

* Applies optional augmentation to the input
* Computes masked reconstruction loss against the **original** input
* Uses AMP, gradient clipping, scheduler stepping, and EMA update

Returns mean training loss.

### `validate()` → `float`

Runs evaluation under `model.eval()`.

If EMA is enabled:

1. the current model weights are backed up,
2. EMA weights are temporarily applied,
3. validation loss is computed,
4. original training weights are restored.

Returns mean validation loss, or `nan` if `val_loader is None`.

### Checkpoint / weight I/O

* `save_checkpoint(epoch, is_best)`
  Saves `last.pt` and optionally `best.pt`.

* `save_weights_only(filename="last_model.pt")`
  Saves current raw model weights only.

* `_save_ema_weights_only(filename)`
  Applies EMA weights temporarily, exports them, then restores the original model weights.

* `load_checkpoint(path)` → `int`
  Loads a full checkpoint and returns the next epoch index.

---

## Augmentation Handling

If `augmenter` is provided, the training loop uses:

```python
x = self._to_x(batch)
x_input = self.augmenter(x)
x_recon, _, visible_mask = self.model(x_input)
loss = self._compute_loss(x_recon, x, ~visible_mask)
```

Thus:

* **model input** = augmented spectrum
* **reconstruction target** = original spectrum

This makes the MAE objective behave as a **denoising-style regularizer** rather than reconstructing the augmented input itself. Augmentation is applied **only during training** and is disabled during validation.

---

## Directory Layout & History

| File                              | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| `{out_dir}/training_history.json` | Per-epoch JSON records (loss, lr, etc.)                                  |
| `{out_dir}/checkpoints/last.pt`   | Full checkpoint (latest, resume target)                                  |
| `{out_dir}/checkpoints/best.pt`   | Full checkpoint at best validation epoch                                 |
| `{out_dir}/last_model.pt`         | Final raw model weights at the end of training                           |
| `{out_dir}/last_model_ema.pt`     | Final EMA weights at the end of training (if EMA enabled)                |
| `{out_dir}/best_model_ema.pt`     | Best-validation EMA weights (validation-enabled runs, if EMA enabled)    |
| `{out_dir}/best_model.pt`         | Best-validation raw weights (validation-enabled runs, EMA disabled only) |

Example record:

```json
{"epoch": 12, "train_loss": 0.0231, "val_loss": 0.0219, "lr": 2.0e-4}
```

History updates use atomic temp-file replacement to reduce the risk of corruption.

---

## What Each Saved Artifact Means

### `checkpoints/last.pt`

This is the **resume checkpoint**. It contains:

* raw model weights
* optimizer state
* scheduler state
* scaler state
* EMA state
* history
* best metadata

This file is intended for **continuing training**, not for direct inference export.

### `checkpoints/best.pt`

This is the **full checkpoint** at the best validation epoch.
It includes the same full state as `last.pt`, but frozen at the best validation step.

### `last_model.pt`

This always stores the **final raw model weights** at the end of training.

Use this when you want the exact last-step model without reconstructing it from `last.pt`.

### `last_model_ema.pt`

When EMA is enabled:

* the final EMA weights are exported as `last_model_ema.pt`

This is the canonical final EMA export, regardless of whether validation was used.

### `best_model_ema.pt`

When validation is available and EMA is enabled:

* best epoch is selected based on **EMA validation**
* `best_model_ema.pt` stores the **EMA weights** corresponding to that selection

This is the recommended inference / extraction / downstream-export artifact for validation-based training under EMA.

### `best_model.pt`

When validation is available and EMA is disabled:

* best epoch is selected using the raw validation model
* `best_model.pt` stores the **raw weights** corresponding to that selection

---

## Masked Loss Handling

The Trainer computes losses **only on masked tokens**, using the inverted visibility mask:

```python
mask = ~visible_mask
if cfg.loss_type == "mse":
    loss = masked_mse(x_recon, x, mask, reduction=cfg.reduction)
elif cfg.loss_type == "sse":
    loss = masked_sse(x_recon, x, mask, reduction=cfg.reduction)
```

| Reduction      | Meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| `"mean"`       | Average over all masked elements.                                |
| `"sum"`        | Total squared error over masked elements.                        |
| `"batch_mean"` | Batch-weighted mean style reduction defined by the loss utility. |

---

## Precision & Performance

* **Autocast:** Uses `torch.amp.autocast("cuda", dtype=bf16|fp16)` when `amp=True`
* **GradScaler:** Enabled automatically for `fp16` on CUDA
* **TF32:** Activates TF32 matmul/cuDNN acceleration via `torch.backends.cuda.matmul.allow_tf32=True`
* **Gradient clipping:** `clip_grad_norm_` applied after unscaling when GradScaler is active
* **EMA:** Shadow weights are updated after every optimizer step and used only for evaluation/export, not for training forward/backward itself

---

## Usage Examples

### Minimal training loop with validation

```python
from chemomae.models import ChemoMAE
from chemomae.training.trainer import Trainer, TrainerConfig
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

model = ChemoMAE(seq_len=256, latent_dim=64, n_patches=16, n_mask=12)

aug_cfg = SpectraAugmenterConfig(
    noise_prob=0.5,
    noise_cos_range=(0.995, 0.9995),
    tilt_prob=0.3,
    tilt_cos_range=(0.997, 0.9998),
)
augmenter = SpectraAugmenter(aug_cfg)

cfg = TrainerConfig(
    out_dir="runs",
    amp=True,
    amp_dtype="bf16",
    use_ema=True,
    loss_type="mse",
    reduction="mean",
)

trainer = Trainer(
    model=model,
    optimizer=opt,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=sched,
    augmenter=augmenter,
    cfg=cfg,
)

history = trainer.fit(epochs=100)
print("Best:", history["best"])

# Outputs:
# - runs/checkpoints/last.pt
# - runs/checkpoints/best.pt
# - runs/last_model.pt
# - runs/last_model_ema.pt   (if EMA enabled)
# - runs/best_model_ema.pt   (if EMA enabled)
# - runs/best_model.pt       (if EMA disabled)
```

### Validation-free SSL pretraining

```python
cfg = TrainerConfig(
    out_dir="runs_ssl",
    amp=True,
    amp_dtype="bf16",
    use_ema=True,
    resume_from="auto",
)

trainer = Trainer(
    model=model,
    optimizer=opt,
    train_loader=train_loader,
    val_loader=None,
    scheduler=sched,
    augmenter=augmenter,
    cfg=cfg,
)

trainer.fit(epochs=100)

# Outputs:
# - runs_ssl/checkpoints/last.pt
# - runs_ssl/last_model.pt
# - runs_ssl/last_model_ema.pt   (if EMA enabled)
```

### Resume training automatically

```python
cfg = TrainerConfig(out_dir="runs", resume_from="auto")
trainer = Trainer(
    model=model,
    optimizer=opt,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=sched,
    augmenter=augmenter,
    cfg=cfg,
)
trainer.fit(epochs=100)
```

---

## Notes & Gotchas

* **Data loader output**
  If the loader yields `(x, meta)`, the Trainer automatically uses only `x`.

* **Shape consistency**
  `x.shape == visible_mask.shape == (B, L)` must hold.

* **History persistence**
  Existing `training_history.json` is read and appended to when continuing runs.

* **EMA semantics**
  EMA is **not** used for training-time forward/backward. It is used for:

  * validation-time evaluation
  * best-model EMA export
  * final EMA export

* **Checkpoint vs export**

  * `last.pt` / `best.pt` are full checkpoints
  * `last_model.pt` / `last_model_ema.pt` / `best_model_ema.pt` / `best_model.pt` are weights-only export files

* **Reduction tip**
  Prefer `"batch_mean"` when the number of masked tokens per sample may vary.

---

## Version

* Updated for the Augmenter-enabled ChemoMAE training pipeline, with unified final export naming:
  `last_model.pt`, `last_model_ema.pt`, and `best_model_ema.pt` / `best_model.pt`.