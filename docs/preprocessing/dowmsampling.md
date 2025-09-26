# FPS Downsampling — Farthest-Point Sampling on the unit sphere

> Module: `chemomae.preprocessing.downsampling`

This document describes **`fps_downsample`**, a diversity-first subsampling method that selects spectra that are maximally spread in direction under cosine geometry. The implementation auto-uses **CUDA** when available and returns data in the **original scale** (normalization is internal to selection only). 

---

## Overview

Given spectra $`X=\{\mathbf{x}_1,\dots,\mathbf{x}_N\}\subset\mathbb{R}^C`$, define L2-normalized directions

```math
\tilde{\mathbf{x}}_i=\frac{\mathbf{x}_i}{\lVert \mathbf{x}_i\rVert_2+\varepsilon},\quad \lVert\tilde{\mathbf{x}}_i\rVert_2\approx 1.
```

Define cosine distance (monotone in angle):

```math
d(\tilde{\mathbf{x}}_i,\tilde{\mathbf{x}}_j)=1-\tilde{\mathbf{x}}_i^\top\tilde{\mathbf{x}}_j\in[0,2].
```

**FPS** builds a subset $`\mathcal{S}_k=\{s_1,\dots,s_k\}`$ of size $`k=\max(1,\mathrm{round}(\rho N))`$ by the greedy rule:

```math
s_1\ \text{(random or fixed)},\qquad
s_t=\arg\max_{i\notin\mathcal{S}_{t-1}}\ \min_{j\in\mathcal{S}_{t-1}} d(\tilde{\mathbf{x}}_i,\tilde{\mathbf{x}}_j).
```

Vectorized update maintains $`\mathbf{d}_{\min}\in\mathbb{R}^N`$:

```math
\mathbf{d}_{\min}\leftarrow \min\!\Big(\mathbf{d}_{\min},\ \mathbf{1}-X_{\text{unit}}\tilde{\mathbf{x}}_{s}\Big),
```

so each iteration is a single **matrix–vector product** plus an elementwise `min`. This is implemented with torch `matmul` and in-place updates. 

> **Note:** With `ensure_unit_sphere=True`, rows are L2-normalized **internally** to align with cosine geometry, but the function **returns rows from the original `X`** (not the normalized copy). 

---

## API

### Function: `fps_downsample(...)`

```python
fps_downsample(
    X: np.ndarray | torch.Tensor, *,
    ratio: float = 0.1,
    ensure_unit_sphere: bool = True,
    seed: Optional[int] = None,
    init_index: Optional[int] = None,
    return_numpy: bool = True,
    eps: float = 1e-12,
) -> (np.ndarray | torch.Tensor)  # shape: (k, C), k = max(1, round(N*ratio))
```

**Parameters**

* `X`: `(N, C)` spectra. NumPy or Torch.
* `ratio`: target fraction $\rho$; selects $k=\max(1,\mathrm{round}(\rho N))$.
* `ensure_unit_sphere`: if `True`, L2-normalizes rows internally before FPS.
* `seed`: RNG seed for the initial point (used if `init_index` is `None`).
* `init_index`: fix the initial index deterministically.
* `return_numpy`: `True` → return NumPy; `False` → return Torch tensor.

**Behavior & Types**

* **Device:** runs on CUDA automatically when available; falls back to CPU. 
* **Return type:** NumPy in → NumPy out by default; Torch in → stays Torch if `return_numpy=False` (device preserved). 
* **Complexity:** $O(Nk)$ inner products; memory $O(N)$. Implementation reuses temporaries to reduce GPU churn. 

---

## Usage Examples

### NumPy — basic

```python
import numpy as np
from chemomae.preprocessing import fps_downsample

# X: (N, C) NumPy array (e.g., after SNV)
X_sub = fps_downsample(X, ratio=0.10, ensure_unit_sphere=True, seed=42)
# -> NumPy array, shape (ceil(0.1*N), C)
```

### Torch — return tensor, same device

```python
import torch
from chemomae.preprocessing import fps_downsample

Xt = torch.randn(5000, 128, device="cuda", dtype=torch.float32)
Xt_sub = fps_downsample(Xt, ratio=0.1, return_numpy=False)  # -> torch.Tensor on CUDA
```

### With SNV (recommended before cosine geometry)

```python
from chemomae.preprocessing import SNVScaler, fps_downsample

X_snv = SNVScaler().transform(X)           # per-spectrum standardization
X_sub = fps_downsample(X_snv, ratio=0.1)   # diversity-first subset
```

(Use L2 normalization internally via `ensure_unit_sphere=True`.) 

---

## Design Notes & Edge Cases

* **Internal normalization:** When `ensure_unit_sphere=True`, the function computes row L2 norms and normalizes once; the returned subset comes from the **original** `X`. 
* **CUDA auto-selection:** uses `torch.cuda.is_available()` to decide device; tensors are moved once and kept there. 
* **In-place / buffer reuse:** distance updates avoid new allocations (`addmv_`, `minimum(..., out=...)`), reducing “reserved memory creep”. 
* **Empty input:** if `N=0`, returns an empty array/tensor with shape `(0, C)`.
* **Seed & determinism:** set `seed` or provide `init_index` for reproducible starts. Subsequent choices are deterministic given the start. 

---

## When to Use `fps_downsample` in ChemoMAE Pipelines

* **Diversity over density:** prefer FPS when you want broad directional coverage and to avoid redundant micro-clusters (common with large-K k-means).
* **HSI/NIR preprocessing:** after SNV, FPS provides a compact, diverse subset prior to self-supervised training. (If you need density-proportional sampling instead, use cosine k-means + stratified allocation.) 

---

## Common Pitfalls

* **Expecting density weighting:** FPS is diversity-first; it won’t oversample dense regions by design.
* **Forgetting internal normalization:** if you disable `ensure_unit_sphere`, cosine geometry assumptions may break unless your input already lies on the unit sphere. 
* **Reading GPU memory graphs:** PyTorch **reserves** CUDA memory; “reserved” can grow and plateau even when “allocated” goes up/down—this is normal, not a leak.

---

## Minimal Test Snippets

```python
import numpy as np
from chemomae.preprocessing import fps_downsample

# Shapes and types
X = np.random.randn(123, 7).astype(np.float32)
Y = fps_downsample(X, ratio=0.1)                    # -> (max(1, round(12)), 7)
assert Y.shape[1] == X.shape[1]

# Reproducibility with seed
A = fps_downsample(X, ratio=0.2, seed=111)
B = fps_downsample(X, ratio=0.2, seed=111)
np.testing.assert_allclose(A, B)

# Invariance to row scaling when ensure_unit_sphere=True
scales = np.exp(np.random.randn(X.shape[0], 1).astype(np.float32))
X2 = X * scales
U  = fps_downsample(X,  ratio=0.1, ensure_unit_sphere=True)
U2 = fps_downsample(X2, ratio=0.1, ensure_unit_sphere=True)
def unit(Z): return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
np.testing.assert_allclose(unit(U), unit(U2), atol=1e-5)
```

---

## Version

* Introduced in `chemomae.preprocessing.downsampling` — initial public draft. 
