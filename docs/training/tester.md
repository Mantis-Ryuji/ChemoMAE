# Tester — ChemoMAE Evaluation Utility

> Module: `chemomae.training.tester`

This document describes the `Tester` and its configuration (`TesterConfig`), a lightweight evaluation utility for trained **ChemoMAE** models.

`Tester` computes **masked reconstruction losses** (SSE/MSE) with AMP support, optional fixed masks, optional spectral augmentation, and optional JSON logging.

---

## Overview

The `Tester` evaluates a trained ChemoMAE model over an entire dataset.

Its behavior mirrors the MAE training principle:

* reconstruction loss is computed only on **masked positions**,
* visible positions are excluded from the loss,
* if an optional `SpectraAugmenter` is provided, the model input is augmented while the reconstruction target remains the original input spectrum.

This makes the augmented testing path consistent with the Trainer's denoising-style objective.

### Key features

* **Masked-only loss** — evaluates reconstruction error only on masked tokens.
* **Loss types** — supports SSE (`masked_sse`) and MSE (`masked_mse`).
* **Mixed precision** — supports AMP with `bf16` or `fp16` on CUDA.
* **Optional `SpectraAugmenter` support** — applies spectral augmentation before reconstruction testing.
* **Fixed masks** — can evaluate with a predefined visible mask (`fixed_visible`).
* **JSON logging** — optionally records test results for reproducibility.

---

## Configuration — `TesterConfig`

```python
@dataclass
class TesterConfig:
    out_dir: str | Path = "runs"
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    loss_type: Literal["sse", "mse"] = "mse"
    reduction: Literal["sum", "mean", "batch_mean"] = "mean"
    fixed_visible: Optional[torch.Tensor] = None

    log_history: bool = True
    history_filename: str = "test_history.json"
```

### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `out_dir` | `str` or `Path` | `"runs"` | Directory where evaluation history is saved. |
| `device` | `str` or `torch.device` | `"cuda"` | Evaluation device (`"cuda"`, `"cpu"`, etc.). |
| `amp` | `bool` | `True` | Enables automatic mixed precision during CUDA evaluation. |
| `amp_dtype` | `"bf16"` or `"fp16"` | `"bf16"` | Precision type for autocast. `bf16` is recommended on recent CUDA GPUs. |
| `loss_type` | `"sse"` or `"mse"` | `"mse"` | Reconstruction loss type. |
| `reduction` | `"sum"`, `"mean"`, `"batch_mean"` | `"mean"` | Loss aggregation mode passed to `masked_sse` / `masked_mse`. |
| `fixed_visible` | `torch.Tensor` or `None` | `None` | Optional fixed visible mask. True means visible. |
| `log_history` | `bool` | `True` | If True, appends evaluation results to a JSON file. |
| `history_filename` | `str` | `"test_history.json"` | JSON filename for result logging. |

---

## Class: `Tester`

### Initialization

```python
tester = Tester(
    model,
    cfg=TesterConfig(),
    augmenter=None,
)
```

| Argument | Description |
| --- | --- |
| `model` | Trained ChemoMAE model. `model(x)` must return `(x_recon, z, visible_mask)`. |
| `cfg` | Optional `TesterConfig` controlling device, AMP, loss, mask, and logging behavior. |
| `augmenter` | Optional `SpectraAugmenter` applied before reconstruction testing. |

The model is moved to `cfg.device` and set to `eval()`.

If `augmenter` is provided, it is also moved to `cfg.device`.

---

## Call Interface

```python
avg_loss = tester(data_loader)
```

The tester iterates through `data_loader`, reconstructs inputs, computes masked loss, and returns a sample-weighted average loss as `float`.

For each batch:

1. Extracts `x` from the batch.
2. Moves `x` to `cfg.device`.
3. Applies optional augmentation:

   ```python
   x_input = augmenter(x) if augmenter is not None else x
   ```

4. Obtains reconstruction and visible mask:

   If `fixed_visible is None`, the tester calls:

   ```python
   x_recon, _, visible_mask = model(x_input)
   ```

   If `fixed_visible` is provided, the tester bypasses model-side random masking:

   ```python
   z = model.encoder(x_input, visible_mask)
   x_recon = model.decoder(z)
   ```

5. Computes loss against the original input `x`:

   ```python
   masked = ~visible_mask
   loss = loss_fn(x_recon, x, masked)
   ```

6. Accumulates the sample-weighted loss.

---

## Augmenter Handling

`Tester` supports optional `SpectraAugmenter`:

```python
from chemomae.training import (
    Tester,
    TesterConfig,
    SpectraAugmenter,
    SpectraAugmenterConfig,
)

aug_cfg = SpectraAugmenterConfig(
    shift_prob=0.5,
    shift_delta_range=(-2.0, 2.0),
    noise_prob=0.5,
    noise_angle_deg_range=(0.5, 3.0),
)

augmenter = SpectraAugmenter(aug_cfg)

tester = Tester(
    model,
    TesterConfig(device="cuda"),
    augmenter=augmenter,
)
```

### Important mode behavior

`SpectraAugmenter` is implemented as an `nn.Module`.

It returns the input unchanged in `eval()` mode.  
Therefore, when an augmenter is provided, `Tester` temporarily sets only the augmenter to `train()` during testing:

```python
model.eval()
augmenter.train()
```

After testing, the original train/eval state of the augmenter is restored.

This means:

* the ChemoMAE model remains in evaluation mode,
* the augmenter is active during testing,
* the augmenter state is restored after testing.

### Target semantics

When an augmenter is provided:

```python
x_input = augmenter(x)
x_recon = model(x_input)
loss = loss_fn(x_recon, x, masked)
```

That is:

* **model input** = augmented spectrum,
* **reconstruction target** = original spectrum.

This is intentionally aligned with the Trainer's denoising-style reconstruction objective.

---

## Fixed Visible Mask

`fixed_visible` provides deterministic mask control.

Accepted shapes:

| Shape | Behavior |
| --- | --- |
| `(L,)` | Broadcast to every batch as `(B, L)`. |
| `(1, L)` | Broadcast to current batch size. |
| `(B, L)` | Used directly when batch size matches current batch. |

`fixed_visible` must be a boolean tensor.

```python
visible = torch.zeros(seq_len, dtype=torch.bool)
visible[: seq_len // 2] = True  # True = visible

cfg = TesterConfig(
    fixed_visible=visible,
    loss_type="mse",
    reduction="batch_mean",
)
tester = Tester(model, cfg)
avg_loss = tester(test_loader)
```

If `fixed_visible=None`, the model uses its normal mask generation behavior.

---

## Loss Handling

| Setting | Behavior |
| --- | --- |
| `loss_type="sse"` | Uses `masked_sse`. |
| `loss_type="mse"` | Uses `masked_mse`. |

### Reduction modes

| Mode | Description |
| --- | --- |
| `"sum"` | Total squared error over masked elements. |
| `"mean"` | Mean squared error over masked elements. |
| `"batch_mean"` | Batch-weighted reduction defined by the loss utility. |

---

## Logging

If `cfg.log_history=True`, results are appended to:

```text
{out_dir}/{history_filename}
```

Default:

```text
runs/test_history.json
```

Example record without augmenter:

```json
{
  "phase": "test",
  "test_loss": 0.1342,
  "loss_type": "mse",
  "reduction": "mean",
  "augmented": false
}
```

Example record with augmenter:

```json
{
  "phase": "test",
  "test_loss": 0.1487,
  "loss_type": "mse",
  "reduction": "mean",
  "augmented": true
}
```

Multiple test runs accumulate sequentially.

Writes use temporary files and atomic replacement to reduce the risk of corruption.

---

## Usage Examples

### Basic evaluation

```python
from chemomae.training import Tester, TesterConfig

cfg = TesterConfig(
    device="cuda",
    loss_type="mse",
    reduction="mean",
)

tester = Tester(model, cfg)
avg_loss = tester(test_loader)

print("Test MSE:", avg_loss)
```

### Fixed visible mask evaluation

```python
import torch
from chemomae.training import Tester, TesterConfig

visible = torch.zeros(seq_len, dtype=torch.bool)
visible[: seq_len // 2] = True

cfg = TesterConfig(
    device="cuda",
    fixed_visible=visible,
    loss_type="sse",
    reduction="batch_mean",
)

tester = Tester(model, cfg)
avg_loss = tester(test_loader)
```

### Augmented reconstruction testing

```python
from chemomae.training import (
    Tester,
    TesterConfig,
    SpectraAugmenter,
    SpectraAugmenterConfig,
)

aug_cfg = SpectraAugmenterConfig(
    shift_prob=0.5,
    shift_delta_range=(-2.0, 2.0),
    noise_prob=0.5,
    noise_angle_deg_range=(0.5, 3.0),
    recenter_after_each_op=True,
    renorm_to_input_norm=True,
)

augmenter = SpectraAugmenter(aug_cfg)

cfg = TesterConfig(
    device="cuda",
    amp=True,
    amp_dtype="bf16",
    loss_type="mse",
    reduction="batch_mean",
    history_filename="test_history_aug.json",
)

tester = Tester(
    model,
    cfg,
    augmenter=augmenter,
)

avg_loss = tester(test_loader)
```

This evaluates how well the model reconstructs the original spectrum from augmented input.

---

## Design Notes

### AMP

Uses:

```python
torch.amp.autocast(device_type="cuda", dtype=bf16|fp16)
```

only when:

* `cfg.amp=True`, and
* `cfg.device` resolves to CUDA.

### JSON logging

Records loss settings and whether augmentation was applied.

Default logging is separated from Trainer's `training_history.json`.

### Fixed vs. random masks

* `fixed_visible=None`: uses the model's normal masking behavior.
* `fixed_visible` provided: bypasses model-side random mask generation.

This is useful when comparing reconstruction losses under a controlled mask.

### Device management

The tester transfers input tensors, model, and optional augmenter to the configured device.

### Model/evaluator state

The model is always evaluated under:

```python
model.eval()
```

If augmenter is provided, only the augmenter is temporarily placed in train mode:

```python
augmenter.train()
```

This is required because `SpectraAugmenter.eval()` is an identity mapping.

---

## Minimal Tests

### Basic test path

```python
from chemomae.training import Tester, TesterConfig

cfg = TesterConfig(
    device="cpu",
    loss_type="mse",
    reduction="mean",
    log_history=False,
)

tester = Tester(model, cfg)
loss = tester(test_loader)

assert isinstance(loss, float)
assert loss >= 0
```

### Augmenter path

```python
from chemomae.training import (
    Tester,
    TesterConfig,
    SpectraAugmenter,
    SpectraAugmenterConfig,
)

augmenter = SpectraAugmenter(SpectraAugmenterConfig())

cfg = TesterConfig(
    device="cpu",
    loss_type="mse",
    reduction="batch_mean",
    log_history=False,
)

tester = Tester(
    model,
    cfg,
    augmenter=augmenter,
)

loss = tester(test_loader)

assert isinstance(loss, float)
assert loss >= 0
```

---

## Version v0.2.0

Updated for the optional augmenter-enabled ChemoMAE testing pipeline.

Changes:

* added optional `SpectraAugmenter` support,
* documented augmenter train/eval mode handling,
* clarified denoising-style target semantics,
* changed default history filename to `test_history.json`,
* documented the `"augmented"` history field,
* clarified fixed-visible shape handling.