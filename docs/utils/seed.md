# Seed Utilities — Reproducibility Helpers

> Module: `chemomae.utils.seed`

This document describes utility functions for controlling random seeds and deterministic behavior across Python, NumPy, and PyTorch.

---

## Overview

Experiments involving stochastic training (e.g., ChemoMAE with random masking) often require reproducibility. This module provides:

* **Global seeding** across Python, NumPy, and PyTorch (if available).
* **Deterministic flags** for CuDNN backends.

---

## API

### `set_global_seed(seed: int = 42, *, fix_cudnn: bool = True) -> None`

Sets the seed globally for multiple libraries.

**Parameters**

* `seed` (`int`, default=42): Seed value applied across libraries.
* `fix_cudnn` (`bool`, default=True): If `True`, forces CuDNN deterministic mode.

  * `torch.backends.cudnn.deterministic = True`
  * `torch.backends.cudnn.benchmark = False`

**Behavior**

* Python `random.seed(seed)`
* NumPy `np.random.seed(seed)`
* `os.environ["PYTHONHASHSEED"] = str(seed)`
* PyTorch (if installed):

  * `torch.manual_seed(seed)`
  * `torch.cuda.manual_seed_all(seed)`
  * CuDNN flags set if `fix_cudnn=True`

**Usage**

```python
from chemomae.utils.seed import set_global_seed

set_global_seed(1234)
```

---

### `enable_deterministic(enable: bool = True) -> None`

Toggles CuDNN deterministic mode **without resetting seeds**.

**Parameters**

* `enable` (`bool`, default=True): Whether to enforce deterministic behavior.

**Behavior**

* If PyTorch is unavailable, no-op.
* Otherwise:

  * `torch.backends.cudnn.deterministic = enable`
  * `torch.backends.cudnn.benchmark = not enable`

**Usage**

```python
from chemomae.utils.seed import enable_deterministic

enable_deterministic(True)   # enforce reproducibility
enable_deterministic(False)  # allow autotuned kernels
```

---

## Design Notes

* `set_global_seed` is the main entry point to unify random states across libraries.
* `enable_deterministic` is lighter-weight and can be used after seeding to toggle performance/reproducibility tradeoffs.
* If PyTorch is not installed, both functions silently degrade (no effect on Torch).

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
