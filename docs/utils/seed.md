# Seed and RNG utilities

Module: `chemomae.utils.seed`.

These helpers seed Python, NumPy, and PyTorch global random streams, capture
and restore their states, and configure cuDNN flags. Independent generators
remain caller-owned. Common seeds aid repeatability under matching execution
conditions; they do not guarantee identical results across devices or versions.

## Quick start

This CPU example restores the next draws from all three standard global streams.
It leaves cuDNN flags unchanged and does not restore CUDA RNG state.

```python
import random
import numpy as np
import torch
from chemomae.utils import set_global_seed, capture_rng_state, restore_rng_state

set_global_seed(1234, fix_cudnn=False)
snapshot = capture_rng_state()
expected = (random.random(), np.random.randn(4), torch.randn(4))
restore_rng_state(snapshot, restore_cuda=False)
actual = (random.random(), np.random.randn(4), torch.randn(4))
assert expected[0] == actual[0]
np.testing.assert_array_equal(expected[1], actual[1])
torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
```

## API and defaults

| Function | Defaults | Effect |
| --- | --- | --- |
| `set_global_seed(seed=42, *, fix_cudnn=True)` | Seed `42`; configure cuDNN | Seed standard global streams; returns `None` |
| `enable_deterministic(enable=True)` | `True` | Toggle cuDNN deterministic/benchmark flags without reseeding; returns `None` |
| `capture_rng_state()` | No arguments | Return a format-1 snapshot of standard global streams |
| `restore_rng_state(state, *, restore_cuda=True)` | Restore saved CUDA streams | Validate and restore a snapshot; returns `None` |

### Global seeding

`set_global_seed` converts `seed` to `int`, then calls `random.seed`,
`np.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`.
The accepted seed range is constrained by the underlying generators, including
NumPy's legacy global generator. Invalid values are not silently wrapped.

It also sets `os.environ["PYTHONHASHSEED"]` for subsequently started Python
interpreters. This does not change hash randomization in the current interpreter.
Independent `numpy.random.Generator` and `torch.Generator` objects are not seeded.

With `fix_cudnn=True`, it sets `torch.backends.cudnn.deterministic=True` and
`torch.backends.cudnn.benchmark=False`. `fix_cudnn=False` leaves those flags
unchanged; it does not reset them to their defaults.

### cuDNN flags

`enable_deterministic(True)` applies the same two cuDNN settings without changing
seeds. `enable_deterministic(False)` sets `deterministic=False` and
`benchmark=True`, allowing cuDNN autotuning. These process-wide flags can affect
performance. They do not call `torch.use_deterministic_algorithms` or control
all sources of nondeterminism.

## Snapshot format and restoration

Format 1 contains primitive values and tensors:

| Field | Stored state |
| --- | --- |
| `format_version` | Integer `1` |
| `python` | Python's global `random` state |
| `numpy` | NumPy's legacy global MT19937 algorithm, keys, position, and Gaussian cache |
| `torch_cpu` | Torch CPU global generator byte state |
| `torch_cuda` | Already initialized CUDA global generator states, or an empty list |

Capturing does not initialize CUDA. Restoration validates the schema and checks
Python/Torch states with isolated generators before changing global streams.
CUDA restoration requires compatible software and the saved device count when
CUDA states are present. `restore_cuda=False` leaves CUDA streams unchanged.

The snapshot does not capture independent NumPy/Torch generators, DataLoader
worker state, MPS RNG, cuDNN/precision flags, or external application state.
Persist those explicitly when the application needs them. A restored RNG state
is not a substitute for matching inputs, data order, modes, operation order,
batch partitioning, dtype, device, and software.

[Trainer checkpoints](../models/persistence.md) capture these global snapshots
automatically and restore them by default at completed epoch boundaries.
Use [Trainer extension hooks](../training/trainer.md#custom-ordering-masks-and-caller-state)
for caller-owned state, including a [SpectraAugmenter generator](../training/augmenter.md#random-streams-and-continuation).

For changes from earlier releases, see the [changelog](../../CHANGELOG.md).
