# Tester — ChemoMAE Evaluation Utility

> Module: `chemomae.training.tester`

This document describes the `Tester` and its configuration (`TesterConfig`), a lightweight evaluation utility for trained **ChemoMAE** models.
It computes **masked reconstruction losses** (SSE/MSE) with AMP support and optional JSON logging.

---

## Overview

The `Tester` evaluates a trained ChemoMAE model over an entire dataset.
Its behavior mirrors the MAE training principle — loss is computed **only on masked positions**, not visible ones.

### Key features

* **Masked-only loss:** Evaluates reconstruction error only on masked tokens.
* **Loss types:** SSE (sum of squared errors) or MSE (mean squared error).
* **Mixed precision:** Supports AMP with `bf16` or `fp16`.
* **Logging:** Optionally records results to JSON for reproducibility.
* **Fixed masks:** Can evaluate with a predefined visible mask (`fixed_visible`).

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
    history_filename: str = "training_history.json"
```

#### Parameters

| Name               | Type                              | Default                   | Description                                           |
| ------------------ | --------------------------------- | ------------------------- | ----------------------------------------------------- |
| `out_dir`          | `str` or `Path`                   | `"runs"`                  | Directory to save evaluation results.                 |
| `device`           | `str` or `torch.device`           | `"cuda"`                  | Execution device (`"cuda"`, `"cpu"`, etc.).           |
| `amp`              | `bool`                            | `True`                    | Enable automatic mixed precision during evaluation.   |
| `amp_dtype`        | `"bf16"` or `"fp16"`              | `"bf16"`                  | Precision type for autocast (`bf16` preferred).       |
| `loss_type`        | `"sse"` or `"mse"`                | `"mse"`                   | Loss formulation (sum or mean squared error).         |
| `reduction`        | `"sum"`, `"mean"`, `"batch_mean"` | `"mean"`                  | Loss aggregation mode (see below).                    |
| `fixed_visible`    | `torch.Tensor` or `None`          | `None`                    | Optional fixed visible mask; overrides model masking. |
| `log_history`      | `bool`                            | `True`                    | Append evaluation results to a JSON file.             |
| `history_filename` | `str`                             | `"training_history.json"` | JSON file name for result logging.                    |

---

## Class: `Tester`

### Initialization

```python
tester = Tester(model, cfg=TesterConfig())
```

| Argument | Description                                                           |
| -------- | --------------------------------------------------------------------- |
| `model`  | Trained ChemoMAE model (`forward(x)` → `(x_recon, z, visible_mask)`). |
| `cfg`    | Optional `TesterConfig` with device, AMP, and logging options.        |

The model is automatically moved to `cfg.device` and set to `eval()`.

---

### Call Interface

```python
avg_loss = tester(data_loader)
```

* Iterates through `data_loader`, reconstructs inputs, and computes masked loss.
* Returns mean loss across all batches.
* If `cfg.fixed_visible` is provided, the mask is applied directly (model’s own masking is bypassed).

---

## Loss Handling

| Setting           | Behavior           |
| ----------------- | ------------------ |
| `loss_type="sse"` | Uses `masked_sse`. |
| `loss_type="mse"` | Uses `masked_mse`. |

### Reduction modes

| Mode           | Description                                                      |
| -------------- | ---------------------------------------------------------------- |
| `"sum"`        | Total SSE over all masked elements.                              |
| `"mean"`       | Mean over masked elements (standard MSE).                        |
| `"batch_mean"` | SSE normalized by batch size.                                    |

### Edge cases

If the mask is empty (no masked tokens), the loss safely returns `0.0`.

---

## Logging

Results are appended to `{out_dir}/{history_filename}` as JSON records:

```json
{
  "phase": "test",
  "test_loss": 0.1342,
  "loss_type": "mse",
  "reduction": "mean"
}
```

* Multiple test runs accumulate sequentially.
* Writes use temporary files and atomic replacement to avoid corruption.

---

## Usage Examples

### Basic evaluation

```python
from chemomae.training import Tester, TesterConfig

tester = Tester(model, TesterConfig(device="cuda", loss_type="mse", reduction="mean"))
avg_loss = tester(test_loader)
print("Test MSE:", avg_loss)
```

### Fixed visible mask evaluation

```python
import torch
visible = torch.ones(seq_len, dtype=torch.bool)
cfg = TesterConfig(fixed_visible=visible, loss_type="sse", reduction="batch_mean")
tester = Tester(model, cfg)
avg_loss = tester(test_loader)
```

---

## Design Notes

* **AMP:**
  Uses `torch.amp.autocast(device_type="cuda", dtype=bf16|fp16)` for faster evaluation.
  AMP is recommended for large datasets.

* **JSON logging:**
  Records loss and configuration parameters for downstream analysis.

* **Fixed vs. random masks:**
  `fixed_visible` allows deterministic evaluation (e.g., same mask across runs).

* **Device management:**
  Automatically transfers input tensors and model to the correct device.

---

## Minimal Tests

```python
from chemomae.training import Tester, TesterConfig

cfg = TesterConfig(device="cpu", loss_type="mse", reduction="mean", log_history=False)
tester = Tester(model, cfg)
loss = tester(test_loader)
assert isinstance(loss, float)
assert loss >= 0
```

---

## Version

* Introduced in `chemomae.training.tester` — initial public draft.