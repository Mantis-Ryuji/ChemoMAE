# Masked Loss Functions for WaveMAE

> Module: `wavemae.models.losses`

This document describes the masked reconstruction losses provided in `losses.py`. These functions compute squared error **only on the masked (hidden) positions**, consistent with the MAE training principle.

---

## Overview

During training, WaveMAE masks a large fraction of the spectral sequence. The loss should therefore be restricted to **masked positions only**, so the model is penalized for reconstructing what it did not see. Visible tokens are excluded from the loss.

Two functions are provided:

* **`masked_sse`**: Sum of squared errors.
* **`masked_mse`**: Mean squared error.

Both functions accept a reduction mode that determines how losses are aggregated across the batch.

---

## API

### `masked_sse(x_recon, x, mask, *, reduction="batch_mean")`

**Masked Sum of Squared Errors.**

* `x_recon` (`torch.Tensor`, shape `(B, L)`): Reconstructed sequence.
* `x` (`torch.Tensor`, shape `(B, L)`): Ground truth input sequence.
* `mask` (`torch.Tensor`, shape `(B, L)`, bool): `True` = masked positions (loss applied), `False` = visible positions (ignored).
* `reduction` (`{"sum", "mean", "batch_mean"}`):

  * `"sum"`: Total sum over all masked positions.
  * `"mean"`: Average over number of masked elements.
  * `"batch_mean"` (default): Sum over masked elements divided by batch size `B`. This ensures scaling is independent of the number of masked tokens.

**Returns:** Scalar tensor (loss).

**Edge cases:**

* If `mask.sum() == 0`, returns `0.0` for all reductions (avoids NaN).

---

### `masked_mse(x_recon, x, mask, *, reduction="mean")`

**Masked Mean Squared Error.**

* `x_recon` (`torch.Tensor`, shape `(B, L)`): Reconstructed sequence.
* `x` (`torch.Tensor`, shape `(B, L)`): Ground truth input sequence.
* `mask` (`torch.Tensor`, shape `(B, L)`, bool): `True` = masked positions.
* `reduction` (`{"mean", "sum", "batch_mean"}`):

  * `"mean"` (default): Average over number of masked elements.
  * `"sum"`: Total sum over masked elements (equivalent to SSE).
  * `"batch_mean"`: Sum over masked elements divided by batch size `B`.

**Returns:** Scalar tensor (loss).

**Edge cases:**

* If `mask.sum() == 0`, returns `0.0` for all reductions.

---

## Usage Examples

### With visible mask from WaveMAE

```python
from wavemae.models.losses import masked_sse, masked_mse

x = torch.randn(2, 4)
x_recon = torch.randn(2, 4)

# Suppose visible tokens are marked True
visible = torch.tensor([[1,1,0,0],[1,0,1,0]], dtype=torch.bool)
mask = ~visible   # masked positions = False → invert

# Compute losses
loss_sse = masked_sse(x_recon, x, mask, reduction="batch_mean")
loss_mse = masked_mse(x_recon, x, mask, reduction="mean")
```

### Choice of reduction

* Use **`batch_mean`** for stability across varying mask counts.
* Use **`mean`** if you want strict MSE scaling by number of masked tokens.
* Use **`sum`** if you need raw SSE accumulation.

---

## Design Notes

* **Masked-only error:** Aligns with MAE principle; avoids trivial copying of visible inputs.
* **No gradient issues:** Gradients propagate through both `x_recon` and `x` (if `requires_grad=True`). Typically, `x` does not require gradients.
* **Numerical safety:** All reductions return finite values even if no masked elements exist.

---

## Minimal Tests

```python
x = torch.randn(2, 4)
x_recon = x + 0.1
visible = torch.tensor([[1,1,0,0],[1,0,1,0]], dtype=torch.bool)
mask = ~visible

assert masked_sse(x_recon, x, mask, reduction="sum").item() >= 0
assert masked_mse(x_recon, x, mask, reduction="mean").item() >= 0
```

---

## Version

* Introduced in `wavemae.models.losses` — initial public draft.