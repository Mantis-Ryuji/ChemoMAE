# Model artifacts and training checkpoints

ChemoMAE separates inference configuration/weights from resumable training state.
Use a **model artifact** to reconstruct a ChemoMAE for inference. Use a
**Trainer checkpoint** to continue training at a completed epoch boundary.
Raw weights remain useful when the caller constructs the model explicitly.

A raw `state_dict` cannot recover attention heads, dropout, normalization, or the
masking recipe. A model artifact records those settings alongside its weights.

## Quick start

This CPU example saves a small model, reloads its configuration and weights, and
checks all-visible features. It creates `runs/persistence/model.artifact.pt`.

```python
from pathlib import Path
import torch
from chemomae.models import ChemoMAE, ChemoMAEConfig

model = ChemoMAE(
    seq_len=16, n_patches=4, n_mask=2,
    d_model=8, nhead=2, num_layers=1, latent_dim=3,
).eval()
spectra = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 31
configuration = model.get_config()
assert ChemoMAEConfig.from_dict(configuration).to_dict() == configuration

with torch.inference_mode():
    expected = model.encode(spectra)

artifact_path = Path("runs/persistence/model.artifact.pt")
model.save(artifact_path)
restored = ChemoMAE.load(artifact_path, device="cpu")
with torch.inference_mode():
    actual = restored.encode(spectra)

assert artifact_path.is_file()
assert not restored.training
assert restored.get_config() == configuration
torch.testing.assert_close(actual, expected)
```

## Choosing a format

| Format | Contains | Loading responsibility |
| --- | --- | --- |
| Model artifact | Full ChemoMAE configuration, dtype, and selected weights. | `ChemoMAE.load(path, device=...)` constructs the model and returns eval mode. |
| Raw weight file | Model `state_dict`. | Caller constructs the matching model, then loads the state dict. |
| Trainer checkpoint | Model, optimizer, scheduler, scaler, EMA, progress, and captured RNG/extension state. | Caller reconstructs the training components and supplies `resume_from` to Trainer. |

For filenames produced by Trainer, use the [output-file guide](#raw-and-ema-exports).
The shared `.pt` suffix does not make these three formats interchangeable.

Model artifacts are specific to `ChemoMAE`. Custom model classes, including
subclasses with their own architecture, own their configuration-and-weights
format. Trainer can still export raw weights for caller-created models.

## Model artifact API

```text
model.get_config()
model.save(path, *, state_dict=None)
ChemoMAE.load(path, *, device="cpu")
```

`ChemoMAEConfig` validates constructor fields. `get_config()` includes all fields
and the current `n_mask` and `encoder.latent_normalize`; the remaining architecture
must not be replaced in place after construction. `ChemoMAEConfig.from_dict()`
accepts a complete saved configuration and rejects unknown or missing keys.

### Save behavior

`save` validates keys, shapes, dtype, and finite floating tensors before writing.
It creates parent directories and replaces the destination via a temporary file.
The live weights, model mode, and gradients are not changed. Supply
`state_dict=compatible_snapshot` to export an explicit selected weight snapshot,
such as EMA weights. The snapshot must match the live model's keys, shapes, and
dtypes. Concurrent writers to the same path are not supported.

### Load, dtype, and device behavior

`load` uses `weights_only=True`, reconstructs on CPU, validates tensors before
copying, preserves the saved dtype, moves to the requested device, and returns
eval mode. Model construction consumes the normal CPU initialization RNG stream.
The caller still controls `torch.inference_mode()` and the input device/dtype.

Loading an inference artifact does not restore optimizer, RNG, or progress state.
Unsupported schemas, raw weight files, and training checkpoints fail rather than
inferring configuration from weight shapes. A supported saved dtype does not
guarantee that every operation supports that dtype on every computation device.

### Model artifact schema

Format 1 stores:

| Field | Contract |
| --- | --- |
| `artifact` | `"chemomae.model"` |
| `format_version` | Integer `1` |
| `package_version` | Creating ChemoMAE version. |
| `config` | Complete ChemoMAE constructor configuration. |
| `dtype` | Homogeneous float16/bfloat16/float32/float64 model dtype. |
| `state_dict` | CPU copies of every model state tensor. |

## Trainer checkpoints

`Trainer.save_checkpoint(epoch)` writes `last.pt` under the configured checkpoint
directory, relative to `out_dir` unless the directory is absolute. An explicit
`path=` overrides that destination and is used as supplied. Save only completed
epoch boundaries. Format 1 uses `artifact="chemomae.training"` and records:

- Model weights, full ChemoMAE configuration when applicable, and model class.
- Optimizer, scheduler, GradScaler, EMA weights/decay and component types.
- Loss/reduction, precision, clipping/TF32 settings, and successful-update policy.
- Epoch, history, attempted/update/skip counters, and namespaced extension state.
- Python, NumPy global MT19937, Torch CPU, and already initialized CUDA RNG streams.

Recreate the same model/optimizer/scheduler/augmentation recipe, then use
`resume_from=checkpoint_path` with a new Trainer. Output locations, display flags,
and final epoch budget may change. Model/training configuration and saved tensor
shape/dtype must match. The scheduler's callable recipe, data identities, loader
ordering, and custom model settings remain caller-owned.

The loader validates the format, semantic configuration, required state,
progress, model/EMA tensors, and global RNG snapshot. Scaler/component loader
errors propagate. Validating core state does not make arbitrary extension hooks
atomic; a failure during component/extension restoration requires recreating the
Trainer. Load only checkpoints you trust: full training checkpoints use
`weights_only=False` to support caller-owned extension state.

### RNG restoration boundaries

With `restore_rng=True` (default), global RNG is restored after component and
extension loading. A CPU Trainer restores CPU global streams and leaves CUDA
streams untouched. CUDA restoration requires the saved number of devices.
Independent NumPy/Torch generators, DataLoader worker state, MPS RNG, and external
state are not automatically captured; use public extension hooks for these.
Cross-device/version trajectory equivalence is not promised.

`capture_rng_state()` and `restore_rng_state()` are also public in
`chemomae.utils`. The snapshot uses primitives/tensors and does not initialize
CUDA just to capture state. Validation uses isolated generators before changing
global streams. `restore_cuda=False` explicitly leaves CUDA streams unchanged.

## Raw and EMA exports

With `model_artifacts=True`, Trainer exports for ChemoMAE receive a sibling
bundle: `last_model.pt` / `last_model.artifact.pt`, and
`ema_last_model.pt` / `ema_last_model.artifact.pt`. Custom filenames use the same
`.artifact` insertion. The EMA export uses a selected snapshot and leaves live
raw weights unchanged. `result["final_model"]` selects an enabled EMA export,
otherwise an enabled raw export, otherwise `None`.

With the default filenames and enabled outputs, choose a file by purpose:

| Purpose | File relative to `out_dir` | How to use it |
| --- | --- | --- |
| Inference with raw final weights | `last_model.artifact.pt` | `ChemoMAE.load(path, device=...)` |
| Inference with EMA weights | `ema_last_model.artifact.pt` | `ChemoMAE.load(path, device=...)`; requires `use_ema=True` |
| Load weights into a matching model you constructed | `last_model.pt` or `ema_last_model.pt` | `model.load_state_dict(torch.load(path, map_location=device, weights_only=True))` |
| Resume training | `checkpoints/last.pt` | Recreate training components and pass the path as `TrainerConfig(resume_from=path)` |

**`result["final_model"]` names the weight file, not the inference artifact.**
With default exports it is `"ema_last_model.pt"` when `use_ema=True` (the default),
or `"last_model.pt"` when `use_ema=False`. Resolve a relative name below
`trainer.out_dir`; an absolute configured path remains absolute. Pass the
corresponding `.artifact.pt` sibling to `ChemoMAE.load`, or use `load_state_dict`
as shown in the table. `model_artifacts=False` disables artifact siblings, and
disabling both weight exports makes `final_model` equal to `None`.

Both filenames and each output's enablement are configurable in TrainerConfig.
See [Trainer](../training/trainer.md) for configuration and extension hooks, or
the [workflow tutorial](../tutorials/workflow.md) for a resume and selected-artifact
example.

Implementation checks are in `tests/models/test_chemo_mae_persistence.py` and
`tests/training/test_trainer_smoke.py`.
