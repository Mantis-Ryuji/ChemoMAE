# Clustering Ops — Utility Functions

> Module: `chemomae.clustering.ops`

This document describes helper functions for hyperspherical k-means clustering: **row normalization**, **cosine similarity/dissimilarity**, **elbow detection**, and **plotting**.

---

## API

### `l2_normalize_rows(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor`

Row-wise L2 normalization.

* Each row is divided by its L2 norm.
* Ensures unit vectors, suitable for cosine-based clustering.

**Formula**
For row vector $x$:

```math
\tilde{x} = \frac{x}{\lVert x \rVert_2 + \varepsilon}
```

---

### `cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Cosine similarity matrix between row-normalized A and B.

* Assumes both A and B are already L2-normalized.
* Returns `(N,M)` matrix of $`\cos(x_i, y_j)`$.

---

### `cosine_dissimilarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor`

Cosine dissimilarity (1 - similarity).

* Useful as inertia metric in `CosineKMeans`.
* Returns `(N,M)` matrix of $`1 - \cos(x_i, y_j)`$.

---

### `find_elbow_curvature(k_list: List[int], inertia_list: List[float]) -> Tuple[int,int,np.ndarray]`

Estimate optimal cluster count via curvature.

**Steps**

1. Enforce monotone non-increasing inertia: `y = np.minimum.accumulate(y)`.
2. Normalize $`x`$ and $`y`$ to [0,1].
3. Compute gradients $`y'`$, $`y''`$.
4. Curvature:

```math
\kappa = \frac{|y''|}{(1 + (y')^2)^{3/2}}
```
5. Ignore endpoints (set to -inf).
6. Choose `optimal_k = k_list[argmax(κ)]`.

**Returns**

* `optimal_k`: Chosen cluster count.
* `elbow_idx`: Index in `k_list`.
* `kappa`: Array of curvature values.

---

### `plot_elbow(k_list, inertias, optimal_k, elbow_idx)`

Plot inertia curve with elbow annotation.

* Line plot of `k_list` vs `inertias`.
* Highlights chosen elbow with marker + vertical line.
* Labels y-axis as "Mean Cosine Inertia".
* Does not call `plt.show()`, leaving display/saving to caller.

---

## Example Usage

```python
from chemomae.clustering.ops import l2_normalize_rows, find_elbow_curvature, plot_elbow
import matplotlib.pyplot as plt

# Normalize features
X = l2_normalize_rows(torch.randn(100, 64))

# Example elbow detection
k_list = [2,3,4,5,6]
inertias = [0.7,0.5,0.42,0.39,0.38]
K, idx, kappa = find_elbow_curvature(k_list, inertias)
plot_elbow(k_list, inertias, K, idx)
plt.show()
```

---

## Design Notes

* Functions assume cosine clustering context (inputs often pre-normalized).
* `find_elbow_curvature` ensures monotonicity before curvature computation for robustness.
* `plot_elbow` uses Matplotlib; suitable for notebooks and scripts.

---

## Minimal Tests

```python
X = torch.randn(10,5)
Xn = l2_normalize_rows(X)
assert torch.allclose(Xn.norm(dim=1), torch.ones(10), atol=1e-6)

opt_k, idx, kappa = find_elbow_curvature([2,3,4], [0.7,0.6,0.55])
assert isinstance(opt_k, int)
```

---

## Version

* Introduced in `chemomae.clustering.ops` — initial public draft.
