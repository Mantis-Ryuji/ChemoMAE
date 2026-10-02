# FPS Downsampling — Farthest-Point Sampling on the Unit Hypersphere

> Module: `chemomae.preprocessing.downsampling`

This document describes **`cosine_fps_downsample`**, a diversity-first subsampling method that selects spectra maximally spread in *direction* under cosine geometry.
The implementation uses an explicit CPU/CUDA computation device and returns data
in the original scale. With `device=None`, Torch input uses its current device and
NumPy input uses CPU. CUDA is never chosen merely because it is available.

<p align="center">
<img src="../../images/cosine_fps_sampling_3d.gif" width="500">
</p>

---

## Overview

Consider a collection of spectra:

$$
X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \subset \mathbb{R}^L
$$

Each spectrum is **internally** projected onto the unit hypersphere via L2 normalization (for selection only):

$$
\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i}{\lVert \mathbf{x}_i \rVert_2 + \varepsilon},
\quad \lVert \tilde{\mathbf{x}}_i \rVert_2 \approx 1
$$

The dissimilarity measure used is the **cosine distance**:

$$
d(\tilde{\mathbf{x}}_i,\tilde{\mathbf{x}}_j)
= 1 - \tilde{\mathbf{x}}_i^\top \tilde{\mathbf{x}}_j
\in [0,2]
$$

Interpretation:

* $d \approx 0$ — spectra point in nearly the same direction (high similarity)
* $d \approx 2$ — spectra point in opposite directions (maximally dissimilar)

---

### Farthest Point Sampling (FPS)

The objective of FPS is to select a diverse subset of size

$$
k = \min\!\bigl(N,\ \max(1,\ \mathrm{round}(\rho N))\bigr)
$$

Let the selected indices be

$$
\mathcal{S}_k = \{s_1, \dots, s_k\}
$$

The greedy selection proceeds as follows:

$$
s_1 \text{ chosen randomly (or fixed)}, \qquad
s_{k+1} = \arg\max_{i \notin \mathcal{S}_k}
        \min_{j \in \mathcal{S}_k} 
        d(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_j)
$$

Intuitively:

* For each candidate $i$, compute its distance to every already selected point.
* Record the **minimum** distance (its nearest neighbor in the subset).
* Add the candidate whose nearest distance is **largest overall**.

Thus, FPS iteratively adds the sample **farthest from all selected points**, ensuring the chosen subset covers the hypersphere as uniformly as possible.

---

### Vectorized Implementation

For efficiency, the algorithm maintains a vector of current nearest distances

$$
\mathbf{d}_{\min} \in \mathbb{R}^N,
$$

where $d_{\min}[i]$ is the distance between candidate $i$ and its closest selected point.

When a new point $\tilde{\mathbf{x}}_s$ is selected, the update rule is:

$$
\mathbf{d}_{\min} \leftarrow
\min \Bigl( \mathbf{d}_{\min},\ \mathbf{1} - X_{\text{unit}} \tilde{\mathbf{x}}_{s} \Bigr),
$$

where $X_{\text{unit}}$ is the row-normalized version of `X` (computed internally).
This update involves:

* Computing cosine distances (`1 - X_unit @ x_s`) between all points and the new sample
* Replacing $d_{\min}$ with the elementwise minimum

Each iteration requires **one matrix–vector multiplication** and a `min`
operation. Selecting the next index also synchronizes a scalar on CUDA; FPS
does not provide a synchronization-free GPU sampling loop.

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
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
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
| `device` | `str` or `torch.device`, optional | Explicit CPU/CUDA computation; default follows Torch input or CPU for NumPy. |
| `generator` | `torch.Generator`, optional | Caller-owned initial-point stream on the computation device; mutually exclusive with seed. |

#### Behavior & Types

* **Device:** Follows the explicit request; the default follows Torch input or CPU for NumPy.
* **Return type:**

  * NumPy in → NumPy out (default)
  * Torch in → Torch out if `return_numpy=False` (device preserved)
* **Normalization:** Always performed internally (selection only). The output spectra remain in the **original scale**.
* **Complexity:** $O(Nk)$ inner products, $O(NC)$ resident working data, and $O(N)$ distances.
  Arithmetic is float64 for float64 input and otherwise float32. No autograd is recorded.
* **Validation:** ratio and eps must be positive and finite; init_index must be an
  integer in range. Ratios >= 1 select all rows. Empty row inputs return the requested
  output type; a zero feature dimension is invalid.
* **Randomness:** With neither seed nor generator, one draw uses the computation
  device's global stream without reseeding it. An explicit init_index draws no random
  numbers. Persist caller-owned generator state outside FPS.
* **Zero rows:** Stay zero, with dissimilarity one to every direction. Ties choose
  the first remaining row. Selected rows cannot repeat. Torch-to-NumPy bfloat16
  output is promoted to float32 because NumPy does not represent bfloat16.

---

## Usage Examples

### NumPy — basic

```python
import numpy as np
from chemomae.preprocessing import cosine_fps_downsample

X = np.random.randn(5000, 128).astype(np.float32)
X_sub = cosine_fps_downsample(X, ratio=0.1, seed=42)
# -> NumPy array, shape (round(0.1*N), 128), clipped to [1, N].
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
* **CUDA handling:** Honors the explicit computation device and rejects unavailable CUDA.
* **Precision:** Selection disables surrounding CPU/CUDA autocast. Float64 input
  uses float64 arithmetic; other supported inputs use float32.
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
U1 = cosine_fps_downsample(X,  ratio=0.1, seed=42)
U2 = cosine_fps_downsample(X2, ratio=0.1, seed=42)
def unit(Z): return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
np.testing.assert_allclose(unit(U1), unit(U2), atol=1e-5)
```

---

## Version

* Introduced in `chemomae.preprocessing.downsampling` — initial public draft.
