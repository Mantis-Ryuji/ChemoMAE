# Trainer — ChemoMAE Training Loop

> Module: `chemomae.training.trainer`

This document explains the **Trainer** and its **TrainerConfig**, covering AMP/TF32, EMA, gradient clipping, masked‑loss handling, checkpointing/resume, and training/validation loops.

---

## Overview

The `Trainer` encapsulates a robust training routine for **masked reconstruction** with ChemoMAE:

* Mixed precision (bf16/fp16) via `torch.amp.autocast` and optional GradScaler
* Optional **TF32** acceleration (Ampere+
  GPUs)
* **EMA** (Exponential Moving Average) of model parameters
* Gradient clipping by global‑norm
* Flexible **masked losses** (`masked_mse` / `masked_sse`)
* Full‑state checkpointing (model/optimizer/scheduler/scaler/EMA/history) and **resume**
* JSON history logging per epoch

The model is assumed to return `(x_recon, z, visible_mask)`, and the Trainer computes loss **only on masked positions** (`mask = ~visible_mask`).

---

## Configuration — `TrainerConfig`

```python
@dataclass
class TrainerConfig:
    out_dir: str | Path = "runs"
    device: Optional[str] = None   # {"cuda","mps","cpu"} or None (= auto-detect)
    amp: bool = True
    amp_dtype: str = "bf16"    # {"bf16","fp16"}
    enable_tf32: bool = False
    grad_clip: float | None = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    loss_type: str = "mse"      # {"mse","sse"}
    reduction: str = "mean"     # {"sum","mean","batch_mean"}
    early_stop_patience: int | None = 20
    early_stop_start_ratio: float = 0.5
    early_stop_min_delta: float = 0.0
    resume_from: str | Path | None = "auto"
```

**Key fields**

* **Device:** `device` can be `"cuda"`, `"mps"`, or `"cpu"`. If `None`, the trainer **auto-detects** availability in the order `cuda → mps → cpu`.
* **Precision:** `amp=True` enables autocast; `amp_dtype` selects `bf16` (stable default) or `fp16`.
* **TF32:** Enable on Ampere+ for faster GEMMs/conv: `enable_tf32=True`.
* **EMA:** Turn on with `use_ema`; `ema_decay≈0.999` is a robust starting point.
* **Loss & reduction:** `loss_type∈{mse,sse}`, `reduction∈{"sum","mean","batch_mean"}` (see *Masked Loss* below).
* **Early stopping:** Start monitoring after a fraction of total epochs (`early_stop_start_ratio`), require improvement of at least `early_stop_min_delta` within `patience` epochs.
* **Resume:** `resume_from="auto"` restores from `out_dir/checkpoints/last.pt` if present; set a path or `None` for fresh training.

---

## Public API — `Trainer`

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

* **`fit(epochs) -> dict`**
  Runs the full training loop with validation, checkpointing, and early stopping.
  Returns `{"best": {"epoch": int, "val_loss": float}, "epochs": int}`.

* **`train_one_epoch() -> float`**
  Trains for one epoch (model in `train()`), applies AMP, grad‑clip, scheduler stepping, EMA updates. Returns mean train loss.

* **`validate() -> float`**
  Evaluates on `val_loader` (model in `eval()`). If EMA is enabled, temporarily applies EMA parameters for evaluation and then restores the original ones.

* **Checkpoint I/O**

  * `save_checkpoint(epoch, is_best)` — Saves `last.pt` every epoch and `best.pt` when improved.
  * `save_weights_only(filename="best_model.pt")` — Saves model weights only (for inference/deployment).
  * `load_checkpoint(path) -> int` — Loads a full state and returns the **next** epoch index to continue from.

---

## Directory Layout & History

* `{out_dir}/training_history.json` — list of JSON records per epoch:

  ```json
  {"epoch": 12, "train_loss": 0.0231, "val_loss": 0.0219, "lr": 2.0e-4}
  ```
* `{out_dir}/checkpoints/last.pt` — last full checkpoint
* `{out_dir}/checkpoints/best.pt` — best full checkpoint (lowest val loss)
* `{out_dir}/best_model.pt` — weights only at best validation

The trainer appends to `training_history.json` safely via a temp file and atomic replace.

---

## Masked Loss Handling

For input `x` and reconstruction `x_recon` with visibility mask `visible` (`True=visible`), the **error is computed on masked tokens only**:

```python
mask = ~visible
if cfg.loss_type == "mse":
    loss = masked_mse(x_recon, x, mask, reduction=cfg.reduction)
elif cfg.loss_type == "sse":
    loss = masked_sse(x_recon, x, mask, reduction=cfg.reduction)
```

**Reductions**

* `"mean"` — average over masked elements (standard MSE)
* `"sum"` — raw SSE
* `"batch_mean"` — `(SSE / B)`; stable w.r.t. varying masked counts per sample

---

## Precision & Performance

* **Autocast**: Uses `torch.amp.autocast("cuda", dtype=bf16|fp16)` when `amp=True` on CUDA.
* **GradScaler**: Enabled automatically for `fp16` on CUDA; not used for `bf16`.
* **TF32**: Optionally set `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` (and, if available, `torch.set_float32_matmul_precision("high")`).
* **Gradient clipping**: Applies `clip_grad_norm_` after unscaling when scaler is enabled.
* **EMA**: Maintains a shadow copy with decay `ema_decay`; `validate()` evaluates under EMA then restores.

---

## Usage Examples

### Minimal training

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

### Resuming automatically

```python
cfg = TrainerConfig(out_dir="runs", resume_from="auto")
trainer = Trainer(model, opt, train_loader, val_loader, device="cuda", scheduler=sched, cfg=cfg)
trainer.fit(epochs=100)
```

---

## Notes & Gotchas

* **Data loader output**: If the loader yields `(x, meta)`, the trainer automatically uses only `x`.
* **Shape checks**: The model is expected to enforce `x.shape == visible.shape == (B, L)` and `L == seq_len`.
* **History persistence**: Existing JSON history is read and appended to on subsequent runs in the same `out_dir`.
* **Best model files**: `best_model.pt` contains **weights only**. `checkpoints/best.pt` includes the full training state.
* **`batch_mean` vs `mean`**: Prefer `batch_mean` if the number of masked tokens per sample varies significantly across batches.

---

## Version

* Introduced in `chemomae.training.trainer` — initial public draft.