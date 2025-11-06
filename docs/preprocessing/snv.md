# SNVScaler — Standard Normal Variate for 1D Spectra

> Module: `chemomae.preprocessing.snv`

This document describes the **SNVScaler**, an implementation of the *Standard Normal Variate (SNV)* transform commonly used in chemometrics for per-spectrum normalization.

---

## Overview

**Standard Normal Variate (SNV)** standardizes each spectrum independently by subtracting its mean and dividing by its (unbiased) standard deviation.
For `X ∈ ℝ^{N×L}` (N spectra, each of length L):

```math
\mu_i = \frac{1}{L}\sum_{j=1}^{L} X_{ij},\qquad
s_i = \sqrt{\frac{1}{L-1}\sum_{j=1}^{L} (X_{ij}-\mu_i)^2},\quad (L≥2)
```

```math
Y_{ij} = \frac{X_{ij}-\mu_i}{\,s_i + \varepsilon\,}
```

#### Key characteristics:

* **Stateless:** No `fit()` step; each call computes per-row mean and standard deviation on demand.
* **Framework-agnostic I/O:** Accepts **NumPy** arrays or **PyTorch** tensors and returns the same type. For tensors, both `device` and `dtype` are preserved.
* **Unbiased standard deviation:** Uses `ddof=1` when `L ≥ 2`, and falls back to `ddof=0` when `L = 1` to prevent `NaN`.
* **Numerical stability:** Adds a small constant `eps` (default `1e-12`) to the denominator.
* **Optional statistics:** If `transform_stats=True`, `transform()` returns `(Y, mu, sd)` so that you can later call `inverse_transform(Y, mu, sd)` to reconstruct the original values.

> **Note:**<br>
> After SNV, each spectrum has **zero mean** and **unit variance**, implying that the **L2 norm of each spectrum is constant**:
> $$
> |Y_i|_2 = \sqrt{L-1}
> $$
> Thus, all spectra are mapped onto a hypersphere of radius $`\sqrt{L-1}`$.
> This property benefits cosine-similarity–based methods since inter-spectral angles are preserved.
> If you prefer vectors on a *unit* hypersphere, apply an additional L2 normalization after SNV.

---

## API

### Class: `SNVScaler(eps=1e-12, copy=True, transform_stats=False)`

A lightweight, stateless transformer that optionally returns per-sample statistics for inverse transformation.

#### Parameters

| Parameter         | Type    | Description                                                         |
| ----------------- | ------- | ------------------------------------------------------------------- |
| `eps`             | `float` | Small constant added to the denominator for numerical stability.    |
| `copy`            | `bool`  | If `True`, the input is copied before transformation.               |
| `transform_stats` | `bool`  | If `True`, `transform()` returns `(Y, mu, sd)` instead of only `Y`. |

#### Methods

* `transform(X)` → `Y` **or** `(Y, mu, sd)`

  * **Input:** 1D `(L,)` or 2D `(N, L)`; supports NumPy or PyTorch.
  * **Output:** Same type as input, `float32`.
    If `transform_stats=True`, also returns:

    * `mu`: Row-wise means (`float32`, shape `(N,1)` or scalar for 1D input)
    * `sd`: Row-wise **unbiased** standard deviations with `eps` **already added** (`sd + eps`)

* `inverse_transform(Y, *, mu, sd)` → `X_rec`

  * Reconstructs `X` from `Y` and the given `mu`, `sd`.
    For 2D input, broadcasting with `(N,1)` is expected.

**Supported shapes**

* 1D vectors `(L,)` → per-vector SNV
* 2D matrices `(N, L)` → per-row SNV

**Not supported:** Higher-rank tensors (e.g. `(H, W, L)`).
Reshape or flatten them to `(N, L)` before use.

---

## Usage Examples

### NumPy — basic transform

```python
import numpy as np
from chemomae.preprocessing.snv import SNVScaler

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)

scaler = SNVScaler()
Y = scaler.transform(X)  # -> np.ndarray, dtype float32, same shape
```

### NumPy — round-trip with inverse transform

```python
scaler = SNVScaler(transform_stats=True)
Y, mu, sd = scaler.transform(X)
X_rec = scaler.inverse_transform(Y, mu=mu, sd=sd)
```

### PyTorch — device/dtype preservation

```python
import torch
from chemomae.preprocessing.snv import SNVScaler

Xt = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]], dtype=torch.float32, device="cuda")

scaler = SNVScaler(transform_stats=True)
Yt, mu, sd = scaler.transform(Xt)
Xt_rec = scaler.inverse_transform(Yt, mu=mu, sd=sd)
```

---

## Design Notes

* **Unbiased std (`ddof=1`):** Preferable for sample-level normalization. Automatically switches to `ddof=0` when `L=1`.
* **`eps` handling:** The returned `sd` already includes `eps` (`sd + eps`), enabling direct reuse in `inverse_transform()`.
* **Precision:** All computations are internally performed in `float64` for numerical stability and cast back to `float32` for output.
* **Torch compatibility:** Internally converts to NumPy, performs computation, and then maps back to Torch while preserving `device` and `dtype`.
* **Complexity:** Time complexity `O(N·L)`; memory overhead `O(1)` besides the output and optional statistics.

---

## When to Use SNV in ChemoMAE Pipelines

* **Spectral pre-processing:**
  SNV is a robust baseline for mitigating per-spectrum intensity offsets and scaling effects prior to self-supervised ChemoMAE training or downstream clustering (e.g., cosine K-Means).
* **Cosine-based metrics:**
  Since SNV outputs lie on a hypersphere of radius $`\sqrt{L-1}`$, angular relationships are preserved.
  Apply L2 normalization if unit-norm vectors are desired.

---

## Common Pitfalls

* Passing 3D HSI cubes directly → **Solution:** reshape to `(N, L)`.
* Expecting unit-length vectors after SNV → **Solution:** apply L2 normalization.
* Forgetting that returned `sd` already includes `eps` → **Solution:** pass `sd` directly to `inverse_transform()`.

---

## Minimal Test Snippets

```python
# 1D vector
x = np.array([1., 2., 3.], dtype=np.float32)
sc = SNVScaler()
y = sc.transform(x)
assert y.shape == x.shape
assert abs(y.mean()) < 1e-6

# 2D rows
X = np.stack([x, x + 1.0])
Y = sc.transform(X)
assert Y.shape == X.shape

# Inverse round-trip (approx)
sc = SNVScaler(transform_stats=True)
Y, mu, sd = sc.transform(X)
X_rec = sc.inverse_transform(Y, mu=mu, sd=sd)
np.testing.assert_allclose(X_rec, X, rtol=1e-5, atol=1e-5)
```

---

## Version

* Introduced in `chemomae.preprocessing.snv` — initial public draft.