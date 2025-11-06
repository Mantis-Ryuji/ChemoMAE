# FPS Downsampling — Farthest-Point Sampling on the Unit Hypersphere

> Module: `chemomae.preprocessing.downsampling`

This document describes **`cosine_fps_downsample`**, a diversity-first subsampling method that selects spectra maximally spread in *direction* under cosine geometry.
The implementation automatically utilizes **CUDA** when available and returns data in the **original scale** (normalization is internal to the selection phase only).

<p align="center">
<img src="../../images/cosine_fps_sampling_3d.gif" width="500">
</p>

---

## Overview

Consider a collection of spectra:

```math
X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \subset \mathbb{R}^L
```

Each spectrum is **internally** projected onto the unit hypersphere via L2 normalization (for selection only):

```math
\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i}{\lVert \mathbf{x}_i \rVert_2 + \varepsilon},
\quad \lVert \tilde{\mathbf{x}}_i \rVert_2 \approx 1
```

The dissimilarity measure used is the **cosine distance**:

```math
d(\tilde{\mathbf{x}}_i,\tilde{\mathbf{x}}_j)
= 1 - \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j
\in [0,2]
```

Interpretation:

* $`d \approx 0`$ — spectra point in nearly the same direction (high similarity)
* $`d \approx 2`$ — spectra point in opposite directions (maximally dissimilar)

---

### Farthest Point Sampling (FPS)

The objective of FPS is to select a diverse subset of size

```math
k = \min\!\bigl(N,\ \max(1,\ \mathrm{round}(\rho N))\bigr)
```

Let the selected indices be

```math
\mathcal{S}_k = \{s_1, \dots, s_k\}
```

The greedy selection proceeds as follows:

```math
s_1 \text{ chosen randomly (or fixed)}, \qquad
s_{k+1} = \arg\max_{i \notin \mathcal{S}_k} \
        \min_{j \in \mathcal{S}_k} 
        d(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_j)
```

Intuitively:

* For each candidate $i$, compute its distance to every already selected point.
* Record the **minimum** distance (its nearest neighbor in the subset).
* Add the candidate whose nearest distance is **largest overall**.

Thus, FPS iteratively adds the sample **farthest from all selected points**, ensuring the chosen subset covers the hypersphere as uniformly as possible.

---

### Vectorized Implementation

For efficiency, the algorithm maintains a vector of current nearest distances

```math
\mathbf{d}_{\min} \in \mathbb{R}^N,
```

where $d_{\min}[i]$ is the distance between candidate $i$ and its closest selected point.

When a new point $\tilde{\mathbf{x}}_s$ is selected, the update rule is:

```math
\mathbf{d}_{\min} \leftarrow
\min \Bigl( \mathbf{d}_{\min},\ \mathbf{1} - X_{\text{unit}} \tilde{\mathbf{x}}_{s} \Bigr),
```

where $X_{\text{unit}}$ is the row-normalized version of `X` (computed internally).
This update involves:

* Computing cosine distances (`1 - X_unit @ x_s`) between all points and the new sample
* Updating $d_{\min}$ in-place using an elementwise minimum

Each iteration thus requires only **one matrix–vector multiplication** and a `min` operation, efficiently implemented with `torch.matmul` and in-place updates.

---

## API

### Function: `cosine_fps_downsample(...)`

```python
cosine_fps_downsample(
    X: np.ndarray | torch.Tensor, *,
    ratio: float = 0.1,
    seed: Optional[int] = None,
    init_index: Optional[int] = None,
    return_numpy: bool = True,
    return_indices: bool = False,
    eps: float = 1e-12,
) -> (np.ndarray | torch.Tensor)
```

#### Parameters

| Name             | Type                  | Description                                                                      |
| ---------------- | --------------------- | -------------------------------------------------------------------------------- |
| `X`              | `(N, L)` array/tensor | Input spectra (NumPy or Torch).                                                  |
| `ratio`          | `float`               | Target fraction $\rho$ → selects $k = \min(N, \max(1, \mathrm{round}(\rho N)))$. |
| `seed`           | `int`, optional       | RNG seed for reproducible initialization (ignored if `init_index` is provided).  |
| `init_index`     | `int`, optional       | Deterministically fix the first selected index.                                  |
| `return_numpy`   | `bool`                | If `True`, returns NumPy array; otherwise keeps Torch tensor type.               |
| `return_indices` | `bool`                | If `True`, also returns the selected indices.                                    |
| `eps`            | `float`               | Small constant for L2 normalization stability.                                   |

#### Behavior & Types

* **Device:** Automatically runs on CUDA if available; otherwise CPU.
* **Return type:**

  * NumPy in → NumPy out (default)
  * Torch in → Torch out if `return_numpy=False` (device preserved)
* **Normalization:** Always performed internally (selection only). The output spectra remain in the **original scale**.
* **Complexity:** $O(Nk)$ inner products; memory $O(N)$. Intermediate buffers are reused for GPU efficiency.

---

## Usage Examples

### NumPy — basic

```python
import numpy as np
from chemomae.preprocessing import cosine_fps_downsample

X = np.random.randn(5000, 128).astype(np.float32)
X_sub = cosine_fps_downsample(X, ratio=0.1, seed=42)
# -> NumPy array, shape (ceil(0.1*N), 128)
```

### Torch — return tensor (same device)

```python
import torch
from chemomae.preprocessing import cosine_fps_downsample

Xt = torch.randn(5000, 128, device="cuda", dtype=torch.float32)
Xt_sub = cosine_fps_downsample(Xt, ratio=0.1, return_numpy=False)
# -> torch.Tensor on CUDA
```

### Combined with SNV (recommended before cosine geometry)

```python
from chemomae.preprocessing import SNVScaler, cosine_fps_downsample

X_snv = SNVScaler().transform(X)
X_sub = cosine_fps_downsample(X_snv, ratio=0.1)
```

(*Per-row L2 normalization is applied internally during selection.*)

---

## Design Notes

* **Internal normalization:** Always L2-normalized internally (cosine geometry); returned subset uses the original scale.
* **CUDA handling:** Uses `torch.cuda.is_available()`; moves data once and reuses on device.
* **Memory efficiency:** In-place updates (`addmv_`, `minimum(out=...)`) reduce memory churn and CUDA “reserved memory” inflation.
* **Empty input:** For `N=0`, returns `(0, L)` array/tensor.
* **Reproducibility:** Specify `seed` or `init_index` for deterministic runs.

---

## When to Use `cosine_fps_downsample` in ChemoMAE Pipelines

* **Goal = maximize diversity, not density**
  FPS excels in *directional diversity*. It avoids redundancy in datasets like NIR-HSI, where many spectra are nearly identical. This makes it ideal for **efficient self-supervised training**.
  However, it is *not* suited for preserving sample *density* distributions.

* **Typical placement in preprocessing:**
  Apply **after SNV or L2 normalization**, i.e., once spectra are mapped onto the hypersphere.
  FPS then produces a compact, diversity-balanced subset for training or visualization.

* **Granularity:**
  Recommended at the **per-sample or per-tile level** (e.g., within each image or batch).
  This ensures consistent angular coverage and prevents overrepresentation of similar spectra.

---

## Common Pitfalls

* **Assuming density preservation:** FPS intentionally ignores local density—it seeks maximal spread.
* **GPU memory readings:** PyTorch may show large “reserved” memory even with stable “allocated” usage; this is expected behavior, not a leak.

---

## Minimal Test Snippets

```python
import numpy as np
from chemomae.preprocessing import cosine_fps_downsample

# Shapes and types
X = np.random.randn(123, 7).astype(np.float32)
Y = cosine_fps_downsample(X, ratio=0.1)
assert Y.shape[1] == X.shape[1]

# Reproducibility
A = cosine_fps_downsample(X, ratio=0.2, seed=111)
B = cosine_fps_downsample(X, ratio=0.2, seed=111)
np.testing.assert_allclose(A, B)

# Invariance to row scaling
scales = np.exp(np.random.randn(X.shape[0], 1).astype(np.float32))
X2 = X * scales
U1 = cosine_fps_downsample(X,  ratio=0.1)
U2 = cosine_fps_downsample(X2, ratio=0.1)
def unit(Z): return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
np.testing.assert_allclose(unit(U1), unit(U2), atol=1e-5)
```

---

## Version

* Introduced in `chemomae.preprocessing.downsampling` — initial public draft.