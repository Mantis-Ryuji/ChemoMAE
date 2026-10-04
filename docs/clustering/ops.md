# Clustering Ops — Utility Functions

> Module: `chemomae.clustering.ops`

Utilities for row normalization, pairwise directional comparisons, and
objective-curve inspection. They can be used independently of a ChemoMAE model.

---

## Overview

`CosineKMeans` and `elbow_ckmeans` use these operations internally. On nonzero
unit-norm vectors, cosine similarity is the dot product. Applying it to two
different feature spaces does not imply that their similarity values agree.

## Quick start

This CPU example checks normalization and pairwise similarities, then inspects
a supplied objective curve. No clustering fit is required.

```python
import torch
from chemomae.clustering.ops import (
    l2_normalize_rows,
    cosine_similarity,
    cosine_dissimilarity,
    find_elbow_curvature,
)

X = torch.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
normalized = l2_normalize_rows(X)
similarities = cosine_similarity(normalized, normalized)
distances = cosine_dissimilarity(normalized, normalized)
assert torch.equal(normalized, torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
assert torch.equal(similarities, torch.diag(torch.tensor([1.0, 1.0, 0.0])))
assert torch.equal(distances, 1 - similarities)

k_list = [1, 2, 3, 4, 5]
inertias = [0.7, 0.5, 0.42, 0.39, 0.38]
optimal_k, elbow_idx, curvature = find_elbow_curvature(k_list, inertias)
assert optimal_k == k_list[elbow_idx] and 0 < elbow_idx < len(k_list) - 1
assert curvature >= 0
```

---

## API

### Function: `l2_normalize_rows(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor`

Row-wise L2 normalization.

* Each row vector is divided by its L2 norm clamped below by `eps`.
* Nonzero rows with norm at least `eps` become unit length. Zero rows stay zero;
  smaller rows remain below unit length.
* Supply a floating-point tensor of shape `(N, D)` and a positive epsilon. The
  function delegates to Torch normalization without separate validation;
  output has the input shape and device and retains autograd.

**Formula**

For a row vector $x$:

$$
\tilde{x} = \frac{x}{\max(\lVert x \rVert_2, \varepsilon)}
$$

---

### Function: `cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Compute pairwise cosine similarity between row-normalized tensors.

* Assumes `A (N, D)` and `B (M, D)` are already L2-normalized on compatible
  devices and with compatible dtypes. It computes `A @ B.T` without validation
  or clipping; a zero row gives zero similarity, including to itself.
* Returns an `(N, M)` matrix of dot products, equal to $\cos(x_i, y_j)$ for unit rows.

---

### Function: `cosine_dissimilarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Compute cosine dissimilarity ($1 - \cos$).

* Used as the inertia metric in `CosineKMeans`.
* Returns `(N, M)` matrix of $1 - \cos(x_i, y_j)$.
* Uses the same normalized-input assumptions as `cosine_similarity`; neither
  function normalizes inputs. Dtype and autocast behavior follow Torch arithmetic.

---

### Function:

`find_elbow_curvature(k_list: List[int], inertia_list: List[float], smooth: bool = True, window_length: int = 5, polyorder: int = 2) -> Tuple[int, int, float]`

Find an interior elbow in a supplied objective curve via **curvature-based elbow
detection**, using **Savitzky–Golay derivatives** when the curve supports them.
The returned `optimal_k` is a heuristic candidate at the chosen curvature
maximum. This heuristic describes a supplied curve; it does not establish a
uniquely correct cluster count or the semantic meaning of its clusters.

#### Steps

1. **Monotonicity Enforcement**
   Enforce non-increasing inertia:

$$
   y_j \leftarrow \min_{i \le j} y_i
$$

2. **Normalization**
   Scale $(x, y)$ to $[0, 1]$ for numerical stability:

$$
   x_n = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \varepsilon}, \quad
   y_n = \frac{y - y_{\min}}{y_{\max} - y_{\min} + \varepsilon}
$$

3. **Savitzky–Golay Derivatives**
   If `smooth=True`, $n \ge 5$, and K values are uniformly spaced, compute
   analytic derivatives on $y_n$. Shorter or nonuniform curves use
   coordinate-aware numerical gradients. <br>

   Let $\Delta x = \mathrm{median}(\mathrm{diff}(x_n))$, then:

$$
   y' = \mathrm{SG}(y_n;\ 1,\ \Delta x), \quad
   y'' = \mathrm{SG}(y_n;\ 2,\ \Delta x)
$$

   *Safety adjustment:*
   Require an odd `window_length` at least three, with
   `2 <= polyorder < window_length`. The effective window is reduced to the
   largest allowed odd length that fits the curve, including even-length inputs;
   the effective polynomial degree is bounded by that window.

4. **Curvature Calculation**

$$
   \kappa = \frac{|y''|}{(1 + (y')^2)^{3/2}}
$$

5. **Endpoint Handling**

$$
   \text{Set} \quad \kappa_0 = \kappa_{n-1} = -\infty
$$

6. **Elbow Selection**

$$
   k_{\mathrm{opt}} = k_{\arg\max \kappa}, \quad
   i_{\mathrm{elbow}} = \arg\max \kappa
$$

Here $k_{\mathrm{opt}}$ and $i_{\mathrm{elbow}}$ correspond to the returned
`optimal_k` and `elbow_idx`, respectively.

#### Parameters

| Name            | Type          | Default | Description                                 |
| --------------- | ------------- | ------- | ------------------------------------------- |
| `k_list`        | `List[int]`   | —       | List of tested cluster counts.              |
| `inertia_list`  | `List[float]` | —       | Mean inertia per K (e.g., mean $1-\cos$). |
| `smooth`        | `bool`        | `True`  | Enable S–G derivative smoothing.            |
| `window_length` | `int`         | `5`     | Window size for S–G filter (auto-adjusted). |
| `polyorder`     | `int`         | `2`     | Polynomial order for S–G filter.            |

#### Returns

| Name        | Type         | Description                                  |
| ----------- | ------------ | -------------------------------------------- |
| `optimal_k` | `int`        | Selected cluster count at maximum curvature. |
| `elbow_idx` | `int`        | Index of the elbow in `k_list`.              |
| `kappa`     | `float`      | Curvature at the selected elbow index.      |

Inputs must be matching finite one-dimensional curves of length at least three.
K values must be strictly increasing positive integers. Duplicate/reversed K,
mismatched lengths, nonfinite values, and invalid smoothing parameters raise
ValueError. A flat curve selects the first interior point with zero curvature;
this is a heuristic, not evidence that the chosen K has scientific meaning.

#### Notes

* The Savitzky–Golay derivatives yield **analytic curvature estimates** that are smooth yet responsive to the global elbow shape.
* For small `n` or `smooth=False`, finite-difference derivatives can be used as a fallback.

---

### Function: `plot_elbow_ckm(k_list, inertias, optimal_k, elbow_idx)`

Visualize the inertia curve and elbow location.

* Plots `k_list` vs `inertias` as a line graph.
* Highlights the selected elbow point (`optimal_k`) with a marker and vertical line.
* Labels y-axis as **“Mean Cosine Inertia”**.
* Does not call `plt.show()` — suitable for both notebooks and scripts.
* Creates a Matplotlib figure and returns `None`; display, saving, and closing
  the figure are caller-owned. `plot_elbow_vmf` is also defined in this module;
  see its [reference and current sweep limitation](vmf_mixture.md#plot_elbow_vmf).

---

## Plotting example

This snippet continues the quick start.

```python
from chemomae.clustering import plot_elbow_ckm
import matplotlib.pyplot as plt

plot_elbow_ckm(k_list, inertias, optimal_k, elbow_idx)
plt.show()
```

---

## Design Notes

* Directional comparison functions assume normalized inputs; the curve helper
  operates on supplied numeric curves and does not inspect feature vectors.
* `find_elbow_curvature` replaces increases with the cumulative minimum before
  computing curvature. This can hide nonmonotonic behavior; inspect the original
  curve as well as the selected point.
* `plot_elbow_ckm` uses Matplotlib with minimal dependencies, designed for flexible integration.

---

See [the changelog](../../CHANGELOG.md) for release history.
