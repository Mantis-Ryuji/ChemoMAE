# Tester — WaveMAE Evaluation Utility

> Module: `wavemae.training.tester`

This document describes the **Tester** and its configuration, designed for evaluating trained WaveMAE models on a dataset. It provides masked reconstruction error (SSE/MSE) with AMP support and JSON logging.

---

## Overview

The `Tester` computes reconstruction loss over an entire DataLoader:

* **Masked-only loss**: Error is restricted to masked positions (as in MAE training).
* **Criterion**: Supports SSE (sum of squared errors) or MSE (mean squared error).
* **Precision**: Supports Automatic Mixed Precision (AMP) with `bf16` or `fp16`.
* **Logging**: Optionally appends results to a JSON file under `out_dir`.
* **Fixed visible mask**: Can use a pre-specified mask instead of random masking.

---

## Configuration — `TesterConfig`

```python
@dataclass
class TesterConfig:
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    out_dir: str | Path = "runs"
    criterion: Literal["sse", "mse"] = "mse"
    reduction: Literal["sum", "mean", "batch_mean"] = "mean"
    fixed_visible: Optional[torch.Tensor] = None
    log_history: bool = True
    history_filename: str = "training_history.json"
```

**Fields**

* `device`: Which device to run on (`"cuda"` or `"cpu"`).
* `amp`: Whether to enable mixed precision evaluation.
* `amp_dtype`: Data type for AMP (`bf16` recommended on Ampere+).
* `out_dir`: Directory to save evaluation history.
* `criterion`: Error type (`"sse"` = squared sum, `"mse"` = mean squared error).
* `reduction`: Aggregation mode (see *Loss Handling* below).
* `fixed_visible`: Optional fixed visible mask (1D or 2D bool tensor).
* `log_history`: Append evaluation results to JSON file if `True`.
* `history_filename`: File to store history.

---

## Class: `Tester`

### Initialization

```python
tester = Tester(model, cfg=TesterConfig())
```

* `model`: Trained WaveMAE model (must implement `forward(x)` → `(x_recon, z, visible_mask)`).
* `cfg`: Optional TesterConfig.

The model is moved to `cfg.device` and set to `eval()`.

### Call

```python
avg_loss = tester(data_loader)
```

* Iterates over batches, computes reconstruction and masked loss.
* Returns mean loss across dataset.
* If `cfg.fixed_visible` is provided, bypasses model masking and enforces the given mask.

### Logging

* Results are appended as a JSON record under `{out_dir}/{history_filename}`:

```json
{
  "phase": "test",
  "test_loss": 0.1342,
  "criterion": "mse",
  "reduction": "mean"
}
```

* History is accumulated across multiple runs.

---

## Loss Handling

### Criterion

* **SSE (`"sse"`)**: Uses `masked_sse`.
* **MSE (`"mse"`)**: Uses `masked_mse`.

### Reduction

* `"sum"`: Total sum over all masked tokens.
* `"mean"`: Mean over masked tokens (standard MSE).
* `"batch_mean"`: SSE normalized by batch size (independent of number of masked tokens).

### Edge cases

* Empty mask (no masked tokens): returns 0.0 safely.

---

## Usage Examples

### Basic evaluation

```python
from wavemae.training import Tester, TesterConfig

tester = Tester(model, TesterConfig(device="cuda", criterion="mse", reduction="mean"))
with torch.inference_mode():
    avg_loss = tester(test_loader)
print("Test MSE:", avg_loss)
```

### Fixed visible mask evaluation

```python
visible = torch.ones(seq_len, dtype=torch.bool)
cfg = TesterConfig(fixed_visible=visible, criterion="sse", reduction="batch_mean")
tester = Tester(model, cfg)

with torch.inference_mode():
    avg_loss = tester(test_loader)
```

---

## Design Notes

* **AMP**: Uses `torch.amp.autocast` for GPU acceleration. Dtype = `bf16` (preferred) or `fp16`.
* **History safety**: Writes via temp file and atomic replace to avoid corruption.
* **Flexibility**: Works with either model-internal masking or fixed masks.

---

## Minimal Tests

```python
cfg = TesterConfig(device="cpu", criterion="mse", reduction="mean", log_history=False)
tester = Tester(model, cfg)
loss = tester(test_loader)
assert isinstance(loss, float)
```

---

## Version

* Introduced in `wavemae.training.tester` — initial public draft.
