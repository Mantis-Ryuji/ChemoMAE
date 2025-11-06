# Seed Utilities — Reproducibility Helpers

> Module: `chemomae.utils.seed`

This document describes utility functions for global seeding and deterministic control across Python, NumPy, and PyTorch, ensuring reproducible experiments in ChemoMAE.

---

## Overview

Experiments involving stochastic operations — such as **random masking** in ChemoMAE — require reproducible results for debugging and benchmarking.
This module provides unified helpers for:

* **Global seeding** across Python, NumPy, and PyTorch.
* **Deterministic CuDNN settings** to ensure identical GPU results.

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
* OS: `os.environ["PYTHONHASHSEED"] = str(seed)`
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

* `set_global_seed()` ensures reproducibility across Python, NumPy, and PyTorch.
* `enable_deterministic()` is a lightweight control to toggle performance–reproducibility trade-offs after seeding.
* Both functions **silently degrade** if PyTorch is not installed (no errors raised).

---

## Minimal Tests

```python
set_global_seed(1)
import numpy as np
assert np.random.randint(0, 100) == np.random.randint(0, 100)

enable_deterministic(True)
```

---

## Version

* Introduced in `chemomae.utils.seed` — initial public draft.