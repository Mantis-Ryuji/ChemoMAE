# Model artifacts and training checkpoints

ChemoMAE v0.2.3 separates inference configuration/weights from resumable training
state. A raw `state_dict` cannot recover attention heads, dropout, normalization,
or the masking recipe. The library records these settings explicitly.

## Config-and-weights model artifact

```python
from chemomae.models import ChemoMAE, ChemoMAEConfig

model = ChemoMAE(seq_len=64, n_patches=8, n_mask=4, d_model=16, nhead=4, num_layers=1)
configuration = model.get_config()
assert ChemoMAEConfig.from_dict(configuration).to_dict() == configuration
model.save("runs/model.artifact.pt")
restored = ChemoMAE.load("runs/model.artifact.pt", device="cpu")
```

`ChemoMAEConfig` validates constructor fields. `get_config` includes all fields
and the current `n_mask` and `encoder.latent_normalize`; the remaining architecture
must not be replaced in place after construction. `from_dict` accepts a complete
saved configuration and rejects unknown or missing keys.

Model artifact format 1 stores:

| Field | Contract |
| --- | --- |
| `artifact` | `"chemomae.model"` |
| `format_version` | Integer `1` |
| `package_version` | Creating ChemoMAE version |
| `config` | Complete ChemoMAE constructor configuration |
| `dtype` | Homogeneous float16/bfloat16/float32/float64 model dtype |
| `state_dict` | CPU copies of every model state tensor |

`save` validates keys, shapes, dtype, and finite floating tensors before writing.
It creates parent directories and replaces the destination via a temporary file.
The live weights, model mode, and gradients are not changed. Supply
`state_dict=compatible_snapshot` to export an explicit selected weight snapshot.
Concurrent writers to the same path are not supported.

`load` uses `weights_only=True`, reconstructs on CPU, validates tensors before
copying, preserves the saved dtype, moves to the requested device, and returns
eval mode. Model construction consumes the normal CPU initialization RNG stream.
It does not restore optimizer/RNG/progress state. Unsupported schemas, raw weight
files, and training checkpoints fail rather than inferring configuration from
weight shapes. Custom model classes own their architecture artifact format.

## Trainer checkpoints

`Trainer.save_checkpoint(epoch)` writes the configured `checkpoint_dir/last.pt`;
an explicit `path=` overrides that destination. Save only completed epoch
boundaries. Format 1 uses `artifact="chemomae.training"` and records:

- Model weights and full ChemoMAE configuration, plus the model's class.
- Optimizer, scheduler, GradScaler, EMA weights/decay and component types.
- Loss/reduction, precision, clipping/TF32 settings, and successful-update policy.
- Epoch, history, attempted/update/skip counters, and namespaced extension state.
- Python, NumPy global MT19937, Torch CPU, and already initialized CUDA RNG streams.

Recreate the same model/optimizer/scheduler/augmentation recipe, then use
`resume_from=checkpoint_path` with a new Trainer. Output locations, display flags,
and final epoch budget may change. Model/training configuration and saved tensor
shape/dtype must match. The scheduler's callable recipe, data identities, loader
ordering, and custom model settings remain caller-owned.

The loader validates the format, semantic configuration, required state, progress,
model/EMA tensors, and global RNG snapshot. Scaler/component loader errors
propagate. Validating core state does not make arbitrary extension hooks atomic;
a failure during component/extension restoration requires recreating the Trainer.
Load only checkpoints you trust: full training checkpoints use
`weights_only=False` to support caller-owned extension state.

With `restore_rng=True` (default), global RNG is restored after component and
extension loading. CPU transfer restores CPU global streams and leaves CUDA
streams untouched. CUDA restoration requires the saved number of devices.
Independent NumPy/Torch generators, DataLoader worker state, MPS RNG, and external
state are not automatically captured; use public extension hooks for these.
Cross-device/version trajectory equivalence is not promised.

`capture_rng_state()` and `restore_rng_state()` are also public in
`chemomae.utils`. The snapshot uses primitives/tensors and does not initialize
CUDA just to capture state. Validation uses isolated generators before changing
global streams. `restore_cuda=False` explicitly leaves CUDA streams unchanged.

## Raw and EMA exports

Trainer exports ordinary raw/EMA weight files for caller-created models.
With `model_artifacts=True`, ChemoMAE exports also receive a sibling bundle:
`last_model.pt` / `last_model.artifact.pt`, and
`ema_last_model.pt` / `ema_last_model.artifact.pt`. Custom filenames use the same
`.artifact` insertion. The EMA export uses a selected snapshot and leaves live
raw weights unchanged. `result["final_model"]` selects an enabled EMA export,
otherwise an enabled raw export, otherwise `None`.

Both filenames and each output's enablement are configurable in TrainerConfig.
Read the [workflow tutorial](../tutorials/workflow.md) for a complete resume and
selected-artifact example.
