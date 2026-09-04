# Trainer — ChemoMAE Training Loop

> Module: `chemomae.training.trainer`

This document describes the `Trainer` and its configuration (`TrainerConfig`) for ChemoMAE-style reconstruction pretraining.
The current Trainer is designed for **fixed epoch / fixed step self-supervised pretraining** and does **not** perform validation-loss-based model selection or early stopping.

---

## Overview

The `Trainer` implements a fixed-budget training routine using ChemoMAE. The reconstruction loss can target masked positions or the full spectrum.

It integrates precision management, exponential moving averages, optional spectral augmentation, checkpoint/resume, and final weights export.

### Design principle

ChemoMAE is primarily a self-supervised pretraining method.  
In this setting, validation reconstruction loss is not necessarily a reliable proxy for:

* downstream representation quality,
* clustering quality,
* segmentation usefulness,
* out-of-domain generalization,
* or downstream task performance.

Therefore, the Trainer deliberately avoids:

* `val_loader`,
* validation loops,
* early stopping,
* validation-loss-based best checkpoint selection,
* and artifact names such as `best_model.pt`.

Instead, training is controlled by a fixed budget, such as a predefined number of epochs or optimizer update steps.  
The final model is selected by an explicit rule:

* use `ema_last_model.pt` when EMA is enabled,
* use `last_model.pt` when EMA is disabled.

### Key features

* **Automatic mixed precision (AMP)** — `torch.amp.autocast` with bf16/fp16 on CUDA
* **TF32 acceleration** on Ampere+ CUDA GPUs
* **EMA (Exponential Moving Average)** of model parameters
* **Optional `SpectraAugmenter`** applied during training
* **Gradient clipping** using global norm
* **Selectable loss region** — masked positions (default) or all spectral positions
* **SSE/MSE losses** with `sum`, `mean`, or `batch_mean` reduction
* **Batch-wise scheduler stepping**
* **Checkpointing and resume** — model, optimizer, scheduler, scaler, EMA, and history
* **Weights-only final export**
* **JSON-based training history** for reproducibility and visualization

The model must return `(x_recon, z, visible_mask)`. `TrainerConfig.loss_region` selects the loss mask:

```python
loss_mask = ~visible_mask  # loss_region="masked"
loss_mask = torch.ones_like(visible_mask, dtype=torch.bool)  # loss_region="all"
```

The default is `loss_region="masked"`, preserving the v0.2.0 behavior. `loss_region="all"` must be explicit and allows `n_mask=0`. If an augmenter is provided, the model input is augmented, but the reconstruction target remains the **original** input spectrum in both modes.

---

## Configuration — `TrainerConfig`

```python
@dataclass
class TrainerConfig:
    out_dir: str | Path = "runs"
    device: Optional[str] = None   # {"cuda", "mps", "cpu"} or None (= auto-detect)
    amp: bool = True
    amp_dtype: str = "bf16"        # {"bf16", "fp16"}
    enable_tf32: bool = False
    grad_clip: float | None = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    loss_type: str = "mse"         # {"mse", "sse"}
    loss_region: Literal["masked", "all"] = "masked"
    reduction: str = "mean"        # {"sum", "mean", "batch_mean"}
    resume_from: str | Path | None = "auto"
```

### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `out_dir` | `str` or `Path` | `"runs"` | Output directory for checkpoints, history, and exported weights. |
| `device` | `str` or `None` | `None` | `"cuda"`, `"mps"`, or `"cpu"`; if `None`, auto-detects in the order CUDA → MPS → CPU. |
| `amp` | `bool` | `True` | Enables PyTorch autocast for mixed precision. |
| `amp_dtype` | `str` | `"bf16"` | AMP dtype. Must be `"bf16"` or `"fp16"`. |
| `enable_tf32` | `bool` | `False` | Allows TF32 matmul/convolution acceleration on supported CUDA GPUs. |
| `grad_clip` | `float` or `None` | `1.0` | Gradient norm clipping threshold. `None` disables clipping. |
| `use_ema` | `bool` | `True` | Tracks an exponential moving average of model weights. |
| `ema_decay` | `float` | `0.999` | EMA decay rate. Larger values preserve longer history. |
| `loss_type` | `str` | `"mse"` | Reconstruction loss type. Must be `"mse"` or `"sse"`. |
| `loss_region` | `Literal["masked", "all"]` | `"masked"` | Positions included in reconstruction loss. `"masked"` uses `~visible_mask`; `"all"` uses every element. |
| `reduction` | `str` | `"mean"` | Reduction passed to `masked_mse` / `masked_sse`. |
| `resume_from` | `str`, `Path`, or `None` | `"auto"` | `"auto"` resumes from `{out_dir}/checkpoints/last.pt` if available. `None` always starts a fresh run. |

---

## API — `Trainer`

```python
trainer = Trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: Iterable,
    *,
    scheduler: Optional[LambdaLR] = None,
    augmenter: SpectraAugmenter | None = None,
    cfg: TrainerConfig | None = None,
)
```

### Constructor behavior

* `cfg=None` creates a fresh `TrainerConfig` instance internally.
* If `cfg.device is None`, device is auto-resolved in this order:

  ```python
  "cuda" -> "mps" -> "cpu"
  ```

* `model` is moved to the resolved device.
* `augmenter` is optional. If provided, it is moved to the same device as the model.
* An unsupported `loss_region` raises `ValueError` during `TrainerConfig` construction.
* `val_loader` is not accepted.

---

## Methods

### `fit(epochs)` → `dict`

Executes training for a fixed number of epochs.

```python
result = trainer.fit(epochs=100)
```

Returns:

```python
{
    "epochs": int,
    "completed": bool,
    "final_model": str,
}
```

where:

* `epochs` is the last completed epoch,
* `completed` indicates whether the requested budget was reached,
* `final_model` is:
  * `"ema_last_model.pt"` if EMA is enabled,
  * `"last_model.pt"` if EMA is disabled.

Behavior:

* resumes from `checkpoints/last.pt` when `resume_from="auto"` and the file exists,
* trains until the requested final epoch,
* saves a full resume checkpoint after every epoch,
* saves final raw weights to `last_model.pt`,
* saves final EMA weights to `ema_last_model.pt` when EMA is enabled.

The Trainer does not compute validation loss and does not return `best` metadata.

### `train_one_epoch()` → `float`

Runs one training epoch under `model.train()`.

For each batch:

1. extracts `x` from the batch,
2. applies optional augmentation to obtain `x_input`,
3. forwards `x_input` through the model,
4. computes reconstruction loss over the configured region against the original `x`,
5. runs backward,
6. applies gradient clipping if configured,
7. performs `optimizer.step()`,
8. performs `scheduler.step()` if a scheduler is provided,
9. updates EMA if enabled.

Returns the sample-weighted mean training loss over the epoch.

### Checkpoint / weight I/O

#### `save_checkpoint(epoch)` → `None`

Saves the latest full training checkpoint:

```text
{out_dir}/checkpoints/last.pt
```

This checkpoint is intended for resuming training.

#### `save_weights_only(filename="last_model.pt")` → `None`

Saves the current raw model weights only.

#### `_save_ema_weights_only(filename="ema_last_model.pt")` → `None`

Temporarily applies EMA weights to the model, saves them, then restores the raw model weights.

#### `load_checkpoint(path)` → `int`

Loads a full checkpoint and returns the next epoch index.

For example, if the checkpoint stores `epoch=10`, this method returns `11`.

---

## Augmentation Handling

If `augmenter` is provided, the training loop uses:

```python
x = self._to_x(batch)
x_input = self.augmenter(x)
x_recon, _, visible_mask = self.model(x_input)
loss = self._compute_loss(x_recon, x, visible_mask)
```

Thus:

* **model input** = augmented spectrum,
* **reconstruction target** = original spectrum.

This makes the MAE objective behave as a denoising-style regularizer.

If `augmenter` is not provided, the behavior is unchanged:

```python
x_input = x
```

---

## Scheduler Handling

The Trainer calls:

```python
scheduler.step()
```

after each optimizer update.

Therefore, the scheduler should be designed in terms of **total optimizer update steps**, not epoch-level calls.

Recommended pattern:

```python
from chemomae.training.optim import build_optimizer, build_scheduler

epochs = 500

optimizer = build_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.05,
    betas=(0.9, 0.95),
)

scheduler = build_scheduler(
    optimizer,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    warmup_epochs=10,
    min_lr_scale=0.1,
)
```

This means:

```python
total_steps = len(train_loader) * epochs
warmup_steps = len(train_loader) * warmup_epochs
```

So even though `warmup_epochs` is specified in epoch units, it is internally converted into update steps.

### Incorrect usage

Do not pass an epoch-level scheduler that expects `scheduler.step()` to be called once per epoch unless it is explicitly adapted.

For example, a scheduler designed as:

```python
CosineAnnealingLR(optimizer, T_max=epochs)
```

would decay too quickly if stepped once per batch.

---

## Directory Layout & History

| File | Description |
| --- | --- |
| `{out_dir}/training_history.json` | Per-epoch JSON records. |
| `{out_dir}/checkpoints/last.pt` | Full latest checkpoint for resume. |
| `{out_dir}/last_model.pt` | Final raw model weights. |
| `{out_dir}/ema_last_model.pt` | Final EMA model weights, saved only when EMA is enabled. |

The following artifacts are no longer produced:

| Removed file | Reason |
| --- | --- |
| `{out_dir}/checkpoints/best.pt` | No validation-based best checkpoint selection. |
| `{out_dir}/best_model.pt` | No validation-based best raw export. |
| `{out_dir}/best_model_ema.pt` | No validation-based best EMA export. |
| `{out_dir}/last_model_ema.pt` | Replaced by explicit `ema_last_model.pt`. |

Example history record:

```json
{
  "epoch": 12,
  "train_loss": 0.0231,
  "lr": 2.0e-4,
  "time_sec": 18.7,
  "loss_region": "masked",
  "n_mask": 16
}
```

History updates use atomic temp-file replacement to reduce the risk of corruption.
At the start of `fit`, the Trainer also prints `loss_region` and the model's `n_mask`, so an `n_mask=0`, `loss_region="all"` run is identifiable in the console log.

---

## What Each Saved Artifact Means

### `checkpoints/last.pt`

This is the full **resume checkpoint**.

It contains:

* raw model weights,
* optimizer state,
* scheduler state,
* scaler state,
* EMA state,
* AMP metadata,
* loss region,
* training history,
* device metadata,
* final selection rule.

This file is intended for continuing training, not for direct inference/export.

The checkpoint includes:

```python
"selection_rule": "ema_last"  # if EMA is enabled
```

or:

```python
"selection_rule": "raw_last"  # if EMA is disabled
```

### `last_model.pt`

This stores the final **raw model weights** at the end of training.

Use this when:

* EMA is disabled,
* or you explicitly want the raw last-step model.

### `ema_last_model.pt`

When EMA is enabled, this stores the final **EMA model weights** at the end of training.

This is the canonical final export for downstream feature extraction / testing when EMA is used.

---

## Reconstruction Loss Region Handling

The Trainer builds the boolean mask passed to `masked_mse` or `masked_sse` from `loss_region`:

```python
if cfg.loss_region == "masked":
    loss_mask = ~visible_mask
elif cfg.loss_region == "all":
    loss_mask = torch.ones_like(visible_mask, dtype=torch.bool)

if cfg.loss_type == "mse":
    loss = masked_mse(x_recon, x, loss_mask, reduction=cfg.reduction)
elif cfg.loss_type == "sse":
    loss = masked_sse(x_recon, x, loss_mask, reduction=cfg.reduction)
```

With `reduction="mean"`, the selected elements are averaged. Consequently, `loss_region="all"` with `reduction="mean"` is ordinary full-spectrum MSE over `B * L` elements. `sum` and `batch_mean` keep their existing definitions and only change which elements are selected.

`loss_region="masked"` requires at least one masked element in every processed batch. An all-visible mask, including the mask generated by `n_mask=0`, raises `ValueError` rather than returning a silent zero loss. `loss_region="all"` is valid for any mask count.

| Reduction | Meaning |
| --- | --- |
| `"mean"` | Average over masked elements. |
| `"sum"` | Total squared error over masked elements. |
| `"batch_mean"` | Batch-weighted mean style reduction defined by the loss utility. |

---

## Precision & Performance

* **Autocast**: uses `torch.amp.autocast("cuda", dtype=bf16|fp16)` when `amp=True` and device is CUDA.
* **GradScaler**: enabled automatically for fp16 on CUDA.
* **TF32**: activates TF32 matmul/cuDNN acceleration when `enable_tf32=True` on CUDA.
* **Gradient clipping**: `clip_grad_norm_` is applied after unscaling when GradScaler is active.
* **EMA**: shadow weights are updated after every optimizer step. EMA is used for final export, not for training-time forward/backward.

---

## Usage Examples

### Fixed-budget SSL pretraining with EMA and augmentation

```python
from chemomae.models import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.training.trainer import Trainer, TrainerConfig

epochs = 500

model = ChemoMAE(
    seq_len=256,
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    latent_dim=16,
    latent_normalize=True,
    decoder_num_layers=2,
    n_patches=32,
    n_mask=16,
)

optimizer = build_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.05,
    betas=(0.9, 0.95),
)

scheduler = build_scheduler(
    optimizer,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    warmup_epochs=10,
    min_lr_scale=0.1,
)

aug_cfg = SpectraAugmenterConfig(
    noise_prob=0.5,
    noise_cos_range=(0.995, 0.9995),
    tilt_prob=0.3,
    tilt_cos_range=(0.997, 0.9998),
)
augmenter = SpectraAugmenter(aug_cfg)

cfg = TrainerConfig(
    out_dir="runs_ssl",
    amp=True,
    amp_dtype="bf16",
    use_ema=True,
    ema_decay=0.999,
    loss_type="mse",
    loss_region="masked",
    reduction="mean",
    resume_from="auto",
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    scheduler=scheduler,
    augmenter=augmenter,
    cfg=cfg,
)

result = trainer.fit(epochs=epochs)
print(result["final_model"])

# Outputs:
# - runs_ssl/checkpoints/last.pt
# - runs_ssl/training_history.json
# - runs_ssl/last_model.pt
# - runs_ssl/ema_last_model.pt
```

### Training without EMA

```python
cfg = TrainerConfig(
    out_dir="runs_raw",
    amp=True,
    amp_dtype="bf16",
    use_ema=False,
    resume_from=None,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    scheduler=scheduler,
    augmenter=None,
    cfg=cfg,
)

result = trainer.fit(epochs=500)
print(result["final_model"])  # "last_model.pt"

# Outputs:
# - runs_raw/checkpoints/last.pt
# - runs_raw/training_history.json
# - runs_raw/last_model.pt
```

### Full-spectrum autoencoder training

```python
model = ChemoMAE(
    seq_len=256,
    latent_dim=16,
    n_patches=32,
    n_mask=0,
)

cfg = TrainerConfig(
    out_dir="runs_autoencoder",
    loss_type="mse",
    loss_region="all",
    reduction="mean",
    resume_from=None,
)
```

This keeps every patch visible to the encoder and computes reconstruction loss over the full clean spectrum. The Trainer does not infer `loss_region="all"` from `n_mask=0`.

### Resume training automatically

```python
cfg = TrainerConfig(
    out_dir="runs_ssl",
    resume_from="auto",
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    scheduler=scheduler,
    augmenter=augmenter,
    cfg=cfg,
)

trainer.fit(epochs=500)
```

If `{out_dir}/checkpoints/last.pt` exists, training resumes from the next epoch.  
If it does not exist, training starts from epoch 1.

The checkpoint stores `loss_region`. Resume raises `ValueError` if the saved value differs from the current configuration. A v0.2.0 checkpoint without this field is interpreted as `loss_region="masked"` for backward compatibility.

---

## Version v0.2.1

* added explicit `loss_region="masked" | "all"` selection,
* added full-spectrum loss support for `n_mask=0`,
* added fail-fast handling for an empty masked loss region,
* persisted and validated the loss region across history and checkpoint/resume.

### v0.2.0

Updated for the validation-free ChemoMAE Trainer:

* removed `val_loader`,
* removed early stopping,
* removed validation-loss-based best checkpoint selection,
* switched final EMA export from `last_model_ema.pt` to `ema_last_model.pt`,
* clarified batch-wise scheduler stepping,
* documented fixed-budget SSL pretraining semantics.
