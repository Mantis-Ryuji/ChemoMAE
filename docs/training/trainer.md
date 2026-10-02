# Reconstruction Trainer and public customization hooks

Module: `chemomae.training.trainer`.

`Trainer` supplies a fixed-epoch reconstruction loop, CUDA AMP, optional EMA,
gradient clipping, scheduler stepping, history, epoch checkpoints, and final
raw/EMA weight exports. Customization uses public methods instead of copying the
backward/optimizer loop or overriding private checkpoint helpers.

It has no validation loop, early stopping, or validation-based model selection.
`fit(epochs=N)` means the **final absolute epoch**, including completed epochs
when resuming. It is not a budget of successful optimizer updates.

## Configuration and the simple path

| Field | Default | Contract |
| --- | --- | --- |
| `out_dir` | `"runs"` | Standard history/checkpoint/weight directory |
| `device` | `None` | Follow model parameters, or CPU for a parameter-free model |
| `amp` | `False` | Explicit CUDA opt-in; CPU/MPS AMP is rejected |
| `amp_dtype` | `"bf16"` | `"bf16"` requires CUDA device support; fp16 uses GradScaler |
| `enable_tf32` | `False` | Enable process-wide CUDA TF32 settings |
| `grad_clip` | `1.0` | Finite nonnegative max norm, or `None` |
| `use_ema` | `True` | Update EMA after successful optimizer steps |
| `ema_decay` | `0.999` | Finite value from zero to one inclusive |
| `loss_type` | `"mse"` | `"mse"` or `"sse"` |
| `loss_region` | `"masked"` | `"masked"` or `"all"` |
| `reduction` | `"mean"` | `"mean"`, `"sum"`, or `"batch_mean"` |
| `resume_from` | `"auto"` | Auto, explicit epoch-checkpoint path, or fresh `None` |

Move the model before constructing its optimizer. The following uses small
synthetic CPU data for demonstrating the API, rather than an experimental recipe.
Device selection does not discover or select a GPU implicitly. An explicit
`device="cuda"` requires available CUDA; `amp=True` requires CUDA and a supported
dtype. For fp16 AMP, retain FP32 master parameters and let autocast choose the
operation dtype; FP16 model parameters such as `model.half()` are rejected because
GradScaler cannot unscale ordinary FP16 gradients. Explicit MPS remains available
for ordinary training. Configured precision
overrides ambient CPU/CUDA autocast during batch preparation and forward/loss;
the default follows the supplied tensors' dtypes without an implicit cast.

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from chemomae.models import ChemoMAE
from chemomae.training import Trainer, TrainerConfig, build_optimizer, build_scheduler

device = torch.device("cpu")
generator = torch.Generator().manual_seed(7)
spectra = torch.randn(12, 16, generator=generator)
loader = DataLoader(TensorDataset(spectra), batch_size=4, shuffle=False)
model = ChemoMAE(
    seq_len=16, n_patches=4, n_mask=1, d_model=16,
    nhead=4, num_layers=1, latent_dim=8,
).to(device)
optimizer = build_optimizer(model, lr=1e-3)
scheduler = build_scheduler(optimizer, steps_per_epoch=len(loader), epochs=2)
trainer = Trainer(
    model, optimizer, loader, scheduler=scheduler,
    cfg=TrainerConfig(
        out_dir=Path("runs/synthetic"), device=str(device),
        amp=False, use_ema=False, resume_from=None,
    ),
)
result = trainer.fit(epochs=2)
selected = trainer.out_dir / str(result["final_model"])
model.load_state_dict(torch.load(selected, map_location=device, weights_only=True))
model.eval()
```

Choose a new `out_dir` for a fresh run. `resume_from=None` rejects existing
standard artifacts; it does not silently append a new run to an old history.
Auto resume also rejects stranded history or weight exports with no checkpoint.
The checkpoint is the authoritative source for history and progress, even if a
JSON history file was written just before an interrupted checkpoint save.

## Prepared inputs, targets, and masks

`PreparedBatch(model_input, target, visible_mask=None)` holds nonempty dense floating
spectra of shape `(B, L)`, with matching input/target shapes and finite stored values. Its optional mask
has the same shape and bool dtype: **true means visible**.

The ordinary loader path takes a Tensor or the first Tensor in a tuple/list.
Its clean target stays unchanged; the optional augmenter creates only the model
input. A `PreparedBatch` bypasses augmentation because its input is already
prepared. The loop moves all returned fields to `trainer.device` together,
without changing dtype or detaching caller-provided tensors.

The default model returns `(reconstruction, latent, actual_visible_mask)`.
`forward_batch` returns just `(reconstruction, actual_visible_mask)`; an explicit
prepared mask is passed through `model(..., visible_mask=...)`.
`compute_loss` uses the actual returned mask and the clean target.

For `loss_region="masked"`, at least one nonvisible element is required.
For full-spectrum AE reconstruction, use `n_mask=0` and `loss_region="all"`
explicitly. Empty loaders and nonfinite/scalar-invalid loss outputs fail.
Prepared inputs, targets, and standard-loss reconstructions must be finite over
the entire spectrum, including positions outside the loss mask: nonfinite values
there can still contaminate gradients. CUDA validity checks may synchronize.

## Public override points

| Method | Called when / expected return |
| --- | --- |
| `train_batches(epoch)` | Returns the iterable for a one-based epoch |
| `prepare_batch(batch)` | Returns `PreparedBatch`; owns application preparation |
| `forward_batch(prepared)` | Returns reconstruction and bool visible mask |
| `compute_loss(reconstructed, target, visible_mask)` | Returns a scalar floating loss Tensor |
| `before_epoch(epoch)` | After `current_epoch` is set, before training |
| `before_step(epoch, batch_index, prepared)` | After preparation/device transfer, before forward/backward |
| `after_step(epoch, batch_index, prepared, *, loss, optimizer_updated)` | After counters, scheduler, and EMA; also called for AMP skips |
| `after_epoch(epoch, record)` | Before history/checkpoint save; can add JSON fields but cannot change standard fields |
| `checkpoint_extra_state()` | Returns a dict stored under `extension_state` |
| `load_checkpoint_extra_state(state)` | Validates/restores that extension dict |

`batch_index` is zero-based within the epoch. A custom LR can be set in
`before_step`. `after_step` receives a Python float and the successful-update
flag; the public cumulative counters have already advanced. Do not call an
optimizer, scheduler, or EMA step from a hook in addition to the core step.
Use either the built-in scheduler or your explicit LR updates, with a clear
application convention for their timing.

`train_one_epoch()` remains available for low-level use. It uses `current_epoch`
(one initially), but does not call fit's epoch events or save history/checkpoints.
Prefer `fit` for the complete lifecycle. One Trainer instance runs `fit` once;
create a new instance to resume after completion or an exception.

## Custom ordering, masks, and caller state

This compact adapter illustrates a caller-owned CPU Generator, clean targets,
prepared masks, and namespaced checkpoint state. Data/split/annotation policy
belongs to the application; this example does not define a research protocol.

```python
from typing import Iterator
import torch
from chemomae.training.trainer import PreparedBatch, Trainer
from chemomae.training import TrainerConfig

class OrderedTrainer(Trainer):
    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        spectra: torch.Tensor, *, cfg: TrainerConfig,
    ) -> None:
        self.spectra = spectra
        self.order_generator = torch.Generator().manual_seed(17)
        super().__init__(model, optimizer, (), cfg=cfg)

    def train_batches(self, epoch: int) -> Iterator[torch.Tensor]:
        order = torch.randperm(len(self.spectra), generator=self.order_generator)
        for rows in order.split(4):
            yield self.spectra[rows]

    def prepare_batch(self, batch: object) -> PreparedBatch:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("Expected a spectra Tensor.")
        clean = batch
        visible = torch.ones_like(clean, dtype=torch.bool)
        # One complete patch hidden for the 16-channel / four-patch example.
        visible[:, -4:] = False
        return PreparedBatch(clean + 0.01, clean, visible)

    def after_epoch(self, epoch: int, record: dict[str, object]) -> None:
        record["application_note"] = "explicit masks and caller-owned ordering"

    def checkpoint_extra_state(self) -> dict[str, object]:
        return {"order_generator": self.order_generator.get_state()}

    def load_checkpoint_extra_state(self, state: dict[str, object]) -> None:
        generator_state = state["order_generator"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("Expected a Generator-state Tensor.")
        self.order_generator.set_state(generator_state)
```

The extension payload cannot replace core checkpoint keys such as `model` or
`progress`. Use string keys and Tensor/primitive containers suitable for Torch
serialization. Extension tensors load on CPU; the restore hook owns any required
device transfer and validation. Override both save/restore hooks when returning
nonempty extension state. A base Trainer rejects unrecognized nonempty state.

## A plain PyTorch loop

Use the model and augmentation primitives directly when your application owns
the complete lifecycle. This small CPU example is independent of Trainer; its
synthetic data and perturbation strengths are illustrative.

```python
import torch
from chemomae.models import ChemoMAE, masked_mse
from chemomae.preprocessing import snv
from chemomae.training import SpectraAugmenter, SpectraAugmenterConfig, build_optimizer

device = torch.device("cpu")
data_stream = torch.Generator().manual_seed(3)
mask_stream = torch.Generator(device=device).manual_seed(4)
augmentation_stream = torch.Generator(device=device).manual_seed(5)
spectra = snv(torch.randn(8, 16, generator=data_stream)).to(device)
model = ChemoMAE(
    seq_len=16, n_patches=4, n_mask=1, d_model=16,
    nhead=4, num_layers=1, latent_dim=8, dropout=0.0,
).to(device)
optimizer = build_optimizer(model, lr=1e-3)
augmenter = SpectraAugmenter(
    SpectraAugmenterConfig(shift_prob=0.5, noise_prob=0.5),
    generator=augmentation_stream,
).to(device).train()
model.train()
for clean in spectra.split(4):
    visible = model.make_visible(len(clean), device=device, generator=mask_stream)
    optimizer.zero_grad(set_to_none=True)
    reconstructed, _, visible = model(augmenter(clean), visible_mask=visible)
    loss = masked_mse(reconstructed, clean, ~visible)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite reconstruction loss.")
    loss.backward()
    optimizer.step()
```

This loop supplies no checkpoint, scheduler, AMP, EMA, or metrics automatically.
Own their update timing and state explicitly if you add them. Supervised
fine-tuning is separate: feed `model.encode(..., representation="raw_latent")`
or `"cls"` into a task head under your own training/grad scopes. Extractor yields
detached features and is for frozen-feature use.

## Successful updates and AMP skips

For each prepared batch the loop performs forward/loss/backward, optional
unscaled gradient clipping, and an optimizer step attempt. With standard
GradScaler, an overflow decreases the scale and skips the optimizer update;
this also covers fused optimizers that process a skip internally.

Only successful optimizer updates advance `scheduler.step()` and EMA.
Skip attempts still contribute their finite forward loss to epoch reporting and
invoke `after_step(..., optimizer_updated=False)`. A successful call is not proof
that parameter values changed: zero LR or zero gradients can still be counted.

`attempted_steps`, `optimizer_updates`, and `amp_skips` are cumulative public
attributes and fit-result fields. Each history record also contains epoch-local
versions plus `cumulative_attempted_steps`, `cumulative_optimizer_updates`, and
`cumulative_amp_skips`. The invariant is

$$
\text{attempted steps}=\text{optimizer updates}+\text{AMP skips}.
$$

The checkpoint stores these under `progress`, with
`step_policy="successful_optimizer_update"`. CUDA overflow behavior needs
validation in a supported CUDA environment; a CPU scaler test double checks the
lifecycle/counter policy without requiring GPU execution.

The reported epoch loss is a sample-weighted average of the scalar batch losses:

$$
\text{train loss} = \frac{\sum_b B_b\ell_b}{\sum_b B_b}.
$$

Its units depend on `reduction`. With `sum` it is not dataset-total SSE, and with
varying masked-element counts it is not necessarily a global element-weighted
MSE. Applications needing a different summary should add a clearly named metric.

## Epoch checkpoints, export selection, and resume

After each completed epoch, fit saves `training_history.json` and
`checkpoints/last.pt`. At completion it exports `last_model.pt` and, when enabled,
`ema_last_model.pt`. `result["final_model"]` selects EMA-last if EMA is enabled,
otherwise raw-last. The in-memory model remains raw; load the chosen export
explicitly before evaluation/extraction.

To resume, build a new model, optimizer, optional scheduler, and Trainer with
matching settings, specify `resume_from=out_dir / "checkpoints/last.pt"`, and call
`fit(epochs=final_epoch)`. A checkpoint after epoch one resumes at epoch two;
an interrupted epoch is replayed from the last completed boundary.

Loss settings, AMP/scheduler/scaler/EMA presence, successful-update policy, and
progress consistency are checked. Scaler restoration errors propagate rather
than being ignored. Configuration/model/data identity beyond these checks remains
the caller's responsibility; full versioned model/training artifacts are separate
work. Load only checkpoints you trust: the current full checkpoint uses Torch
serialization with `weights_only=False`.

Neither the core loader nor the default Trainer captures arbitrary loader worker,
global RNG, or external application state. Use the public state hooks for owned
streams and validate matching conditions. Do not claim exact continuation from
weights and epoch counters alone.

## Focused verification

```powershell
pytest tests/training/test_trainer_smoke.py -q
```

The tests cover the ordinary path, public customization hooks, generator-state
resume for a small deterministic adapter, progress persistence, prepared-input
augmentation bypass, and simulated AMP skips. Tests and CUDA checks must be run
explicitly; documentation and source review alone do not establish validation.
