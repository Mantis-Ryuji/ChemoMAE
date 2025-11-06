# Masked Loss Functions — Reconstruction Losses for ChemoMAE

> Module: `chemomae.models.losses`

This document describes **masked reconstruction losses** implemented in `losses.py`.
These functions compute squared errors **only on masked (hidden) positions**, consistent with the **Masked Autoencoder (MAE)** training principle.

---

## Overview

During MAE training, a large portion of each spectral sequence is randomly masked.
The model should be penalized **only for reconstructing the hidden (masked) regions**, not for simply copying visible inputs.<br>

Two loss functions are provided:

| Function     | Description                                        |
| ------------ | -------------------------------------------------- |
| `masked_sse` | Sum of squared errors (SSE) over masked positions. |
| `masked_mse` | Mean squared error (MSE) over masked positions.    |

Both functions support multiple reduction modes to control how losses are aggregated across the batch.

---

## API

### `masked_sse(x_recon, x, mask, *, reduction="batch_mean")`

**Masked Sum of Squared Errors**

| Parameter   | Type                                 | Description                                                                                                                                                                                                                    |
| ----------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `x_recon`   | `torch.Tensor`, shape `(B, L)`       | Reconstructed sequence.                                                                                                                                                                                                        |
| `x`         | `torch.Tensor`, shape `(B, L)`       | Ground-truth input sequence.                                                                                                                                                                                                   |
| `mask`      | `torch.Tensor`, shape `(B, L)`, bool | `True` = masked positions (loss applied); `False` = visible positions (ignored).                                                                                                                                               |
| `reduction` | `{"sum", "mean", "batch_mean"}`      | Aggregation mode:<br>• `"sum"` — total sum of masked errors.<br>• `"mean"` — average over all masked elements.<br>• `"batch_mean"` — sum over masked elements divided by batch size `B` (independent of mask count). |

**Returns:**
Scalar tensor (loss value).

**Edge Cases:**
If `mask.sum() == 0`, returns `0.0` to avoid NaN.

---

### `masked_mse(x_recon, x, mask, *, reduction="mean")`

**Masked Mean Squared Error**

| Parameter   | Type                                 | Description                                                                                                                                                                       |
| ----------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x_recon`   | `torch.Tensor`, shape `(B, L)`       | Reconstructed sequence.                                                                                                                                                           |
| `x`         | `torch.Tensor`, shape `(B, L)`       | Ground-truth input sequence.                                                                                                                                                      |
| `mask`      | `torch.Tensor`, shape `(B, L)`, bool | `True` = masked positions (loss applied).                                                                                                                                         |
| `reduction` | `{"mean", "sum", "batch_mean"}`      | Aggregation mode:<br>• `"mean"` — average over all masked elements.<br>• `"sum"` — total SSE over masked elements.<br>• `"batch_mean"` — sum divided by batch size `B`. |

**Returns:**
Scalar tensor (loss value).

**Edge Cases:**
If `mask.sum() == 0`, returns `0.0` for all reductions.

---

## Usage Examples

### Basic usage with ChemoMAE visible mask

```python
import torch
from chemomae.models.losses import masked_sse, masked_mse

x = torch.randn(2, 4)
x_recon = torch.randn(2, 4)

# visible=True → seen tokens
visible = torch.tensor([[1,1,0,0],
                        [1,0,1,0]], dtype=torch.bool)
mask = ~visible   # masked = unseen tokens

loss_sse = masked_sse(x_recon, x, mask, reduction="batch_mean")
loss_mse = masked_mse(x_recon, x, mask, reduction="mean")
```

---

## Design Notes

* **Masked-only error:**
  Enforces MAE behavior — model learns to infer missing content, not to replicate visible regions.

* **Safe gradients:**
  Gradients propagate correctly through `x_recon`; typically `x` is treated as constant (`requires_grad=False`).

* **Numerical safety:**
  All reductions return finite results even when `mask.sum() == 0`.

* **Consistency:**
  Compatible with ChemoMAE’s `visible` mask convention (`True=visible`, `False=masked` → invert before use).

---

## Minimal Tests

```python
import torch
from chemomae.models.losses import masked_sse, masked_mse

x = torch.randn(2, 4)
x_recon = x + 0.1
visible = torch.tensor([[1,1,0,0],
                        [1,0,1,0]], dtype=torch.bool)
mask = ~visible

assert masked_sse(x_recon, x, mask, reduction="sum").item() >= 0
assert masked_mse(x_recon, x, mask, reduction="mean").item() >= 0
```

---

## Version

* Introduced in `chemomae.models.losses` — initial public draft.