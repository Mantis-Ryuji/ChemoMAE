# Seed Utilities — Reproducibility Helpers

> Module: `chemomae.utils.seed`

This document describes global seeding, RNG snapshots, and cuDNN controls across
Python, NumPy, and PyTorch, with explicit reproducibility limits.

---

## Overview

Experiments involving stochastic operations — such as **random masking** in ChemoMAE — require reproducible results for debugging and benchmarking.
This module provides unified helpers for:

* **Global seeding** across Python, NumPy, and PyTorch.
* **cuDNN flags** for deterministic algorithms and disabled benchmarking.
* **Versioned RNG snapshots** for standard global streams.

---

## API

### `set_global_seed(seed: int = 42, *, fix_cudnn: bool = True) -> None`

Set the same seed across all major random number generators.

**Parameters**

| Name        | Type   | Default | Description                                                        |
| ----------- | ------ | ------- | ------------------------------------------------------------------ |
| `seed`      | `int`  | `42`    | Global seed value applied to Python, NumPy, and PyTorch.           |
| `fix_cudnn` | `bool` | `True`  | If `True`, enables deterministic CuDNN mode (disables autotuning). |

**Behavior**

* Python: `random.seed(seed)`
* NumPy: `np.random.seed(seed)`
* OS: `os.environ["PYTHONHASHSEED"] = str(seed)` for future interpreters; the
  current interpreter's hash randomization is not changed.
* PyTorch (if available):

  * `torch.manual_seed(seed)`
  * `torch.cuda.manual_seed_all(seed)`
  * If `fix_cudnn=True`:

    ```python
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ```

**Usage**

```python
from chemomae.utils.seed import set_global_seed

set_global_seed(1234)
```

---

### `enable_deterministic(enable: bool = True) -> None`

Toggle CuDNN deterministic mode **without resetting seeds**.

**Parameters**

| Name     | Type   | Default | Description                                |
| -------- | ------ | ------- | ------------------------------------------ |
| `enable` | `bool` | `True`  | Whether to enforce deterministic behavior. |

**Behavior**

* If PyTorch is unavailable, the function is a no-op.
* Otherwise:

  ```python
  torch.backends.cudnn.deterministic = enable
  torch.backends.cudnn.benchmark = not enable
  ```

**Usage**

```python
from chemomae.utils.seed import enable_deterministic

enable_deterministic(True)   # enforce reproducibility
enable_deterministic(False)  # allow kernel autotuning for speed
```

---

## Design Notes

* `set_global_seed()` provides common starting seeds. Independent generators
  need their own seeds; identical seeds do not guarantee all GPU operations or
  cross-device/version results are identical.
* `enable_deterministic()` is a lightweight control to toggle performance–reproducibility trade-offs after seeding.
* ChemoMAE declares PyTorch as a runtime dependency. cuDNN flags do not call
  `torch.use_deterministic_algorithms` or control every nondeterministic operation.

## Capturing and restoring RNG state

```python
import random
import torch
from chemomae.utils import capture_rng_state, restore_rng_state

snapshot = capture_rng_state()
expected = (random.random(), torch.randn(4))
restore_rng_state(snapshot, restore_cuda=False)
actual = (random.random(), torch.randn(4))
assert expected[0] == actual[0]
torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
```

Format 1 stores Python state, NumPy's global MT19937 state, Torch CPU state,
and already initialized CUDA states using primitive values and tensors. Capturing
does not initialize CUDA. Validation uses isolated generators before altering
global streams. CUDA restoration requires compatible software and matching
device count; restore_cuda=False explicitly leaves CUDA streams unchanged.
Independent generators, worker RNG, MPS RNG, and external state remain caller-owned.

Trainer checkpoints use these snapshots automatically. Read
[model/training persistence](../models/persistence.md) for resume boundaries and
extension hooks. Snapshot and resumed-trajectory tests are written but unrun.

---

## Minimal Tests

```python
import numpy as np
from chemomae.utils import set_global_seed, enable_deterministic

set_global_seed(1)
expected = np.random.randint(0, 100, size=4)
set_global_seed(1)
np.testing.assert_array_equal(expected, np.random.randint(0, 100, size=4))

enable_deterministic(True)
```

---

## Version

* Introduced in `chemomae.utils.seed` — initial public draft.
