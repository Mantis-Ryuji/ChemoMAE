# Tester — Reconstruction Evaluation

> API reference for ChemoMAE v0.2.4.

Module: `chemomae.training.tester`.

Tester evaluates reconstruction against the input from before any added
augmentation, optionally using perturbed inputs. It supports both masked and
full-spectrum loss and performs no training or checkpoint selection.

The returned scalar summarizes the selected squared errors across the complete
iterable. Evaluation temporarily controls model mode and precision, then restores
the original training flags. Load the intended model weights before evaluation.

## Quick start

This small example uses all-visible reconstruction, so its evaluation region is
explicitly the complete spectrum:

```python
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from chemomae.models import ChemoMAE
from chemomae.training import Tester, TesterConfig

model = ChemoMAE(
    seq_len=12, n_patches=3, n_mask=0, d_model=16,
    nhead=4, num_layers=1, dim_feedforward=32, latent_dim=4,
)
spectra = torch.randn(6, 12)  # Illustrative synthetic data.
loader = DataLoader(TensorDataset(spectra), batch_size=2, shuffle=False)
tester = Tester(model, TesterConfig(
    device="cpu", loss_region="all", reduction="mean",
    amp=False, log_history=False, progress=False,
))
mse = tester(loader)
assert isinstance(mse, float) and math.isfinite(mse) and mse >= 0.0
assert model.training  # The mode from before evaluation was restored.
```

For masked reconstruction, set a nonzero model mask count and
loss_region="masked". The loss region is never inferred from the model mask count.
An all-visible mask with masked evaluation raises ValueError; an empty loader
also raises rather than returning zero. Failed evaluations are not logged.

## Configuration and defaults

| Setting | Default | Contract |
| --- | --- | --- |
| device | None | Follow model parameters; CPU for parameter-free models. Explicit CPU/CUDA override moves the model. |
| amp | False | Explicit CUDA autocast; CPU AMP is rejected. Disabled AMP also disables surrounding autocast for model inference. |
| amp_dtype | "bf16" | "bf16" or "fp16"; requested bf16 requires support on the chosen CUDA device. |
| loss_region | "masked" | "masked" evaluates hidden positions; "all" evaluates every feature. |
| loss_type | "mse" | "mse" or "sse"; both use squared errors and the selected reduction determines scaling. |
| reduction | "mean" | Dataset-wide reduction described below. |
| fixed_visible | None | Optional boolean visible mask of shape (L,), (1, L), or the current (B, L). |
| log_history | True | Write successful evaluations to JSON. |
| out_dir | "runs" | Created only when logging is enabled. |
| history_filename | "test_history.json" | Separate from training history. |
| progress | True | Display evaluation progress. |

## Input, output, and precision

Batches may be floating Torch tensors of shape (B, L), or tuples/lists whose
first item is that tensor. Empty batches are skipped. Spectra are moved to the
evaluation device and cast to the model parameter dtype resolved at Tester
construction. Device moves persist; create a new Tester if the model device/dtype
or precision configuration changes. Dataset iteration order is unchanged.
Changing precision can change reconstruction outputs; error subtraction, squaring,
and aggregation use float64 without claiming bitwise equality across devices.
Nonfinite loader inputs or reconstruction outputs and mismatched reconstruction/
augmentation shapes fail explicitly. The selected autocast scope covers optional
augmentation as well.

## Dataset-wide reductions

Let $E$ be the selected squared-error sum, $M$ the selected element count, and $N$
the spectrum count across all evaluated batches:

$$
E = \sum_{(i,j)\in\mathcal{S}} (\hat{x}_{ij}-x_{ij})^2.
$$

| Reduction | Result |
| --- | --- |
| "sum" | $E$ |
| "mean" | $E/M$ |
| "batch_mean" | $E/N$ |

The same reduction definitions apply for both loss_type names, consistent with
the underlying squared-error utilities. These are dataset reductions, not the
average of separately normalized batch losses. A final short batch or unequal
numbers of hidden features therefore do not change the aggregation definition.
Model predictions can still change when stochastic masks/augmentation or batch-
dependent models change; hold those fixed for comparisons.

## Fixed masks and augmentation

`True` requests visibility. A fixed mask must match spectral patch boundaries.
`(L,)` and `(1, L)` masks broadcast to each current batch; a `(B, L)` mask requires `B`
to match that batch, including a final shorter batch. Fixed-mask evaluation calls
the encoder/decoder directly and consumes no model masking RNG.
ChemoMAE retains a legacy fallback when every patch in the entire batch is hidden;
see the [model mask contract](../models/chemo_mae.md#mask-contract) before using all-false masks.

An explicitly supplied SpectraAugmenter is temporarily active in training mode.
The model itself runs in evaluation mode. Inputs are augmented, and targets remain
the spectra supplied by the loader before that added perturbation:

```python
from chemomae.preprocessing import snv
from chemomae.training import SpectraAugmenter, SpectraAugmenterConfig

augmented_loader = DataLoader(TensorDataset(snv(spectra)), batch_size=2, shuffle=False)
augmenter = SpectraAugmenter(SpectraAugmenterConfig(
    shift_prob=0.5, noise_prob=0.5,
))
tester = Tester(
    model, TesterConfig(device="cpu", loss_region="all", log_history=False),
    augmenter=augmenter,
)
augmented_input_mse = tester(augmented_loader)
```

This example standardizes the synthetic spectra before augmentation. It defines
a stochastic prediction task rather than a fixed experimental protocol.
The caller must specify perturbation strength, random streams, repetitions, and
data separation for a reproducible comparison. Use the [augmenter's default
generator](augmenter.md#random-streams-and-continuation) to own its random stream;
Tester does not seed it. Random model masks use the model's normal RNG stream.

## State and history

Construction preserves model mode. Evaluation restores every original
model/augmenter submodule training flag, including mixed modes, on success and
failure. Device movement persists. The helper uses inference mode and records
no gradients.

Successful history records include phase, test_loss, loss_type, loss_region,
reduction, augmented, samples, and selected_elements. Existing malformed JSON or
a history that is not a list of records raises an error; it is not silently
discarded. Writes use temporary-file replacement. No history directory is created
with log_history=False.

Reconstruction error evaluates the configured prediction task. Evaluate any
downstream classifier, clustering, or other application separately; improvement
in reconstruction alone does not establish improvement in those objectives.
