# SNVScaler — Standard Normal Variate for 1D Spectra

> Module: `chemomae.preprocessing.snv`

This document describes the **SNVScaler**, which implements the Standard Normal Variate (SNV) transform commonly used in chemometrics for per-spectrum standardization.

---

## Overview

**Standard Normal Variate (SNV)** rescales each spectrum independently by subtracting its mean and dividing by its (unbiased) standard deviation. In matrix form for `X ∈ ℝ^{N×L}` (N spectra, length L):

```math
\mu_i = \frac{1}{L}\sum_{j=1}^{L} X_{ij},\qquad
s_i = \sqrt{\frac{1}{L-1}\sum_{j=1}^{L} (X_{ij}-\mu_i)^2},\quad (L≥2)
```
```math
Y_{ij} = \frac{X_{ij}-\mu_i}{\,s_i + \varepsilon\,}
```

* **Stateless:** No `fit()`. Each call computes per-row mean/std on the fly.
* **Framework-agnostic I/O:** Accepts **NumPy** arrays or **PyTorch** tensors and returns the same type; device/dtype are preserved for tensors.
* **Unbiased standard deviation:** Uses `ddof=1` when the row length `L≥2` (falls back to `ddof=0` when `L=1`) to avoid `NaN`.
* **Numerical stability:** Adds a small `eps` (default `1e-12`) to the standard deviation in the denominator.
* **Optional stats round-trip:** If `transform_stats=True`, `transform()` returns `(Y, mu, sd)`; you can then call `inverse_transform(Y, mu, sd)` to recover the original scale.

> **Note:** After SNV, each spectrum has **zero mean** and **unit variance**. This also implies that the **L2 norm of each spectrum is constant** across samples:
>
> $$
> \|Y_i\|_2 = \sqrt{L-1}
> $$
>
> Therefore, spectra are mapped onto a hypersphere of radius $`\sqrt{L-1}`$. This is useful for cosine similarity–based methods, since the relative angles between spectra are preserved. If you prefer a unit sphere (radius 1), apply an additional L2 normalization step after SNV.

---

## API

### Class: `SNVScaler(eps=1e-12, copy=True, transform_stats=False)`

A light-weight, stateless transformer with optional return of per-sample statistics for inverse transforms.

**Parameters**

* `eps` (`float`): Small constant added to the standard deviation in the denominator.
* `copy` (`bool`): If `True`, the input is copied before transformation.
* `transform_stats` (`bool`): If `True`, `transform()` returns `(Y, mu, sd)`; otherwise it returns just `Y`.

**Methods**

* `transform(X)` → `Y` **or** `(Y, mu, sd)`

  * **Input:** 1D `(L,)` or 2D `(N, L)`; NumPy or Torch.
  * **Output:** Same type as input, `float32`. If `transform_stats=True`, also returns:

    * `mu`: Row-wise means (NumPy `float32`, shape `(N,1)` or scalar for 1D input)
    * `sd`: Row-wise **(unbiased)** standard deviations with `eps` **already added**: `sd + eps` (NumPy `float32`)
* `inverse_transform(Y, *, mu, sd)` → `X_rec`

  * Reconstructs `X` from `Y` and the provided `mu`, `sd` (as returned by `transform`). For 2D input, broadcasting with `(N,1)` is expected.

**Supported shapes**

* 1D vectors `(L,)` → per-vector SNV
* 2D matrices `(N, L)` → per-row SNV

**Not supported:** Higher-rank tensors/arrays (e.g., `(H, W, L)`). Flatten or reshape to `(N, L)` first.

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

### NumPy — round-trip with inverse_transform

```python
scaler = SNVScaler(transform_stats=True)
Y, mu, sd = scaler.transform(X)  # sd already includes eps
X_rec = scaler.inverse_transform(Y, mu=mu, sd=sd)
```

### PyTorch — device/dtype preserved

```python
import torch
from chemomae.preprocessing.snv import SNVScaler

Xt = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]], dtype=torch.float32, device="cuda")

scaler = SNVScaler(transform_stats=True)

# With stats (mu, sd returned as NumPy float32 for lightness)
Yt, mu, sd = scaler.transform(Xt)
Xt_rec = scaler.inverse_transform(Yt, mu=mu, sd=sd)
```

---

## Design Notes & Edge Cases

* **Unbiased std (ddof=1):** More appropriate for per-row sample statistics than `ddof=0`. If a row has only one element (`L=1`), the implementation automatically falls back to `ddof=0` to avoid `NaN`.
* **`eps` handling:** `eps` is added in the denominator during the forward transform. When `transform_stats=True`, the returned `sd` is **already** `sd + eps`, so you can directly pass it to `inverse_transform()`.
* **Dtypes & precision:** Computations are performed in `float64` internally for numerical robustness and then cast to `float32` for output.
* **Torch I/O:** Inputs are converted to NumPy for computation and then mapped back to Torch with the original `device`/`dtype`. This keeps the API simple while preserving runtime context.
* **Complexity:** `O(N·L)` time and `O(1)` extra memory aside from the output and optional `(mu, sd)`.

---

## When to Use SNV in ChemoMAE Pipelines

* **Spectral pre-processing:** SNV is a strong baseline for removing per-spectrum intensity offsets and scaling effects prior to self-supervised training with ChemoMAE or downstream clustering (e.g., cosine K-Means).
* **Downstream cosine metrics:** Since SNV outputs lie on a hypersphere of radius $`\sqrt{L}`$, relative angles are preserved. For unit vectors, apply L2 normalization.

---

## Common Pitfalls

* Passing 3D HSI cubes directly. **Fix:** reshape to `(N, L)` first.
* Expecting unit-length vectors after SNV. **Fix:** add L2 normalization if needed.
* Forgetting that `sd` returned with `transform_stats=True` already includes `eps`.

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
X = np.stack([x, x+1.0])
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
