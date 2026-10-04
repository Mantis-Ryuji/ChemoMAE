# Selected reconstruction losses

Module: `chemomae.models.losses`.

`masked_sse` and `masked_mse` aggregate squared errors wherever a boolean
selection mask is `True`. They can be used with ChemoMAE, another reconstruction
model, or a custom PyTorch training loop. They do not generate masks or choose
the target, preprocessing, or augmentation policy.

The functions have identical behavior for the same `reduction`; only their
default reduction differs.

## Quick start

This CPU example selects four errors, checks each reduction, and backpropagates
through the selected mean.

```python
import torch
from chemomae.models.losses import masked_mse, masked_sse

target = torch.zeros(2, 4)
prediction = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]],
    requires_grad=True,
)
selection = torch.tensor(
    [[True, False, True, False], [False, True, False, True]],
)

total = masked_sse(prediction, target, selection, reduction="sum")
per_batch = masked_sse(prediction, target, selection)
per_element = masked_mse(prediction, target, selection)
torch.testing.assert_close(total, torch.tensor(12.0))
torch.testing.assert_close(per_batch, torch.tensor(6.0))
torch.testing.assert_close(per_element, torch.tensor(3.0))

per_element.backward()
assert prediction.grad is not None
assert torch.count_nonzero(prediction.grad[~selection]) == 0
assert prediction.grad[selection].abs().sum() > 0
```

## API contract

```text
masked_sse(x_recon, x, mask, *, reduction="batch_mean")
masked_mse(x_recon, x, mask, *, reduction="mean")
```

| Argument | Contract |
| --- | --- |
| `x_recon` | Floating Torch reconstruction tensor, shape `(B, L)`. |
| `x` | Floating Torch target tensor with the same shape and device. |
| `mask` | Boolean Torch tensor with the same shape and device. `True` includes a position in the loss; `False` excludes it. |
| `reduction` | `"sum"`, `"mean"`, or `"batch_mean"`. An unknown value raises `ValueError`. |

Each call returns one scalar Torch tensor on the computation device. Arithmetic
follows the input dtypes and ordinary Torch operations; these helpers do not
add a higher-precision accumulator or move data between devices. Supply finite
values over the **entire** reconstruction and target, including excluded
positions: subtraction and squaring happen before boolean selection. Finite
inputs can still overflow when squared in a limited-precision dtype.

### Reductions

Let $m_{bj}=1$ for selected positions and $m_{bj}=0$ otherwise. The selected sum
of squared errors $E$ and selected element count $M$ are

$$
E = \sum_{b=1}^{B}\sum_{j=1}^{L}
    m_{bj}(\hat{x}_{bj}-x_{bj})^2,
\qquad
M = \sum_{b=1}^{B}\sum_{j=1}^{L}m_{bj}.
$$

For a nonempty batch and selection:

| Reduction | Value | Interpretation |
| --- | --- | --- |
| `"sum"` | $E$ | Total selected SSE. |
| `"mean"` | $E/M$ | Mean over all selected elements, rather than a mean of per-spectrum means. |
| `"batch_mean"` | $E/B$ | Selected SSE per batch member. Its value still depends on the number of selected channels. |

If spectra have different selected counts, `"mean"` weights each selected element
equally. Changing a mask ratio or sequence length changes the scale of
`"batch_mean"` even when the per-element error is unchanged. Choose a reduction
that matches the quantity you intend to optimize or compare.

### Empty selections and gradients

For an empty selection, every reduction returns a numeric zero. Their autograd
behavior differs:

| Reduction | Empty-selection behavior |
| --- | --- |
| `"sum"` | Empty tensor sum; retains a gradient connection when the inputs require gradients. |
| `"batch_mean"` | Empty sum divided by `max(B, 1)`; retains that connection. |
| `"mean"` | New constant zero tensor without a gradient connection. Calling `.backward()` on this result alone fails. |

For nonempty selections, gradients can reach both `x_recon` and `x`. Treat the
target as constant explicitly when that is the intended objective. A custom
loop must decide how to handle empty selections before backward. The
[Trainer](../training/trainer.md) rejects an empty selected region when
`loss_region="masked"`.

## Using model masks and targets

ChemoMAE's `visible_mask` uses the opposite convention: `True` means an input
channel belongs to a visible patch. In the ordinary masked case, select hidden
channels with `mask = ~visible_mask`. See the model's
[mask contract](chemo_mae.md#mask-contract) for patch alignment and the legacy
all-hidden-batch exception.

Trainer supplies `~visible_mask` for `loss_region="masked"`, or an all-`True`
selection for `loss_region="all"`. The decoder returns all channels in either
case. For a full-spectrum autoencoder, combine `n_mask=0` with
`loss_region="all"`.

In denoising training, an application can perturb the encoder input while using
the input from before that added perturbation as the target. That target is an
observed spectrum; it is not necessarily noise-free. These loss functions also
accept other targets that satisfy the tensor contract above.

## Related references

- [ChemoMAE model and masking](chemo_mae.md)
- [Trainer and custom loops](../training/trainer.md)
- [Spectral augmentation](../training/augmenter.md)
- Implementation checks: `tests/models/test_losses.py`
