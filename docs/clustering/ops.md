# Clustering Ops — Utility Functions

> Module: `chemomae.clustering.ops`

This document describes utility functions supporting **hyperspherical k-means** clustering, including **row normalization**, **cosine similarity/dissimilarity**, **elbow curvature detection**, and **plotting utilities**.

---

## Overview

These operations provide the numerical foundation for `CosineKMeans` and `elbow_ckmeans`, ensuring stable and geometry-aware computation on the unit hypersphere.

---

## API

### Function: `l2_normalize_rows(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor`

Row-wise L2 normalization.

* Each row vector is divided by its L2 norm.
* Guarantees unit-length rows, suitable for cosine-based clustering.

**Formula**

For a row vector $`x`$:

```math
\tilde{x} = \frac{x}{\lVert x \rVert_2 + \varepsilon}
```

---

### Function: `cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Compute pairwise cosine similarity between row-normalized tensors.

* Assumes both `A` and `B` are already L2-normalized.
* Returns an `(N, M)` matrix of $`\cos(x_i, y_j)`$ values.

---

### Function: `cosine_dissimilarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Compute cosine dissimilarity ($1 - \cos$).

* Used as the inertia metric in `CosineKMeans`.
* Returns `(N, M)` matrix of $`1 - \cos(x_i, y_j)`$.

---

### Function:

`find_elbow_curvature(k_list: List[int], inertia_list: List[float], *, smooth: bool = True, window_length: int = 5, polyorder: int = 2) -> Tuple[int, int, np.ndarray]`

Estimate the optimal cluster count via **curvature-based elbow detection** using the **Savitzky–Golay derivative method**.

#### Steps

1. **Monotonicity Enforcement**
   Enforce non-increasing inertia:

   ```math
   y \leftarrow \mathrm{cummin}(y) = \min_{i \le j} y_i
   ```

2. **Normalization**
   Scale $(x, y)$ to $[0, 1]$ for numerical stability:

   ```math
   x_n = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \varepsilon}, \quad
   y_n = \frac{y - y_{\min}}{y_{\max} - y_{\min} + \varepsilon}
   ```

3. **Savitzky–Golay Derivatives**
   If `smooth=True` and $n \ge 5$, compute analytic derivatives on $y_n$. <br>

   Let $`\Delta x = \mathrm{median}(\mathrm{diff}(x_n))`$, then:

   ```math
   \tilde{y} = \mathrm{SG}(y_n;\ 0), \quad
   y' = \mathrm{SG}(y_n;\ 1,\ \Delta x), \quad
   y'' = \mathrm{SG}(y_n;\ 2,\ \Delta x)
   ```

   *Safety adjustment:*
   `window_length` is clipped to the largest odd number ≤ `n`,
   and `polyorder < window_length`.

4. **Curvature Calculation**

   ```math
   \kappa = \frac{|y''|}{(1 + (y')^2)^{3/2}}
   ```

5. **Endpoint Handling**
   ```math
   \text{Set} \quad \kappa_0 = \kappa_{n-1} = -\infty
   ```
6. **Elbow Selection**

   ```math
   \text{optimal\_k} = k_{\arg\max \kappa}, \quad
   \text{elbow\_idx} = \arg\max \kappa
   ```

#### Parameters

| Name            | Type          | Default | Description                                 |
| --------------- | ------------- | ------- | ------------------------------------------- |
| `k_list`        | `List[int]`   | —       | List of tested cluster counts.              |
| `inertia_list`  | `List[float]` | —       | Mean inertia per K (e.g., mean $`1-\cos`$). |
| `smooth`        | `bool`        | `True`  | Enable S–G derivative smoothing.            |
| `window_length` | `int`         | `5`     | Window size for S–G filter (auto-adjusted). |
| `polyorder`     | `int`         | `2`     | Polynomial order for S–G filter.            |

#### Returns

| Name        | Type         | Description                                  |
| ----------- | ------------ | -------------------------------------------- |
| `optimal_k` | `int`        | Selected cluster count at maximum curvature. |
| `elbow_idx` | `int`        | Index of the elbow in `k_list`.              |
| `kappa`     | `np.ndarray` | Curvature array (endpoints = −∞).            |

#### Notes

* The Savitzky–Golay derivatives yield **analytic curvature estimates** that are smooth yet responsive to the global elbow shape.
* For small `n` or `smooth=False`, finite-difference derivatives can be used as a fallback.

---

### Function: `plot_elbow(k_list, inertias, optimal_k, elbow_idx)`

Visualize the inertia curve and elbow location.

* Plots `k_list` vs `inertias` as a line graph.
* Highlights the selected elbow point (`optimal_k`) with a marker and vertical line.
* Labels y-axis as **“Mean Cosine Inertia”**.
* Does not call `plt.show()` — suitable for both notebooks and scripts.

---

## Example Usage

```python
from chemomae.clustering.ops import (
    l2_normalize_rows,
    find_elbow_curvature,
    plot_elbow,
)
import matplotlib.pyplot as plt
import torch

# Normalize feature matrix
X = l2_normalize_rows(torch.randn(100, 64))

# Example elbow detection
k_list = [2, 3, 4, 5, 6]
inertias = [0.7, 0.5, 0.42, 0.39, 0.38]
K, idx, kappa = find_elbow_curvature(k_list, inertias)

plot_elbow(k_list, inertias, K, idx)
plt.show()
```

---

## Design Notes

* Functions assume **cosine-based** clustering context (inputs typically pre-normalized).
* `find_elbow_curvature` ensures monotonic inertia for numerical robustness.
* `plot_elbow` uses Matplotlib with minimal dependencies, designed for flexible integration.

---

## Minimal Tests

```python
X = torch.randn(10, 5)
Xn = l2_normalize_rows(X)
assert torch.allclose(Xn.norm(dim=1), torch.ones(10), atol=1e-6)

opt_k, idx, kappa = find_elbow_curvature([2, 3, 4], [0.7, 0.6, 0.55])
assert isinstance(opt_k, int)
```

---

## Version

* Introduced in `chemomae.clustering.ops` — initial public draft.