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

### `find_elbow_curvature(k_list: List[int], inertia_list: List[float], *, smooth: bool = True, window_length: int = 5, polyorder: int = 2) -> Tuple[int, int, np.ndarray]`

Estimate optimal cluster count via curvature **with Savitzky–Golay–based derivatives**.

**Steps**

1. **Monotonicity**: Enforce non-increasing inertia

```math
y \leftarrow \mathrm{cummin}(y) = \min_{i\le j} y_i
```

2. **Normalization**: Scale (x,y) to ([0,1]).

```math
x_n=\dfrac{x-x_{\min}}{x_{\max}-x_{\min}+\varepsilon},\quad
y_n=\dfrac{y-y_{\min}}{y_{\max}-y_{\min}+\varepsilon}
```

3. **Savitzky–Golay (S-G)**: If `smooth=True` and $`n\ge 5`$, compute **analytic derivatives** on $`y_n`$.
   Let $`\Delta x`$ be the median grid spacing of $`x_n`$:

```math
\Delta x \;=\; \mathrm{median}\!\left(\,\mathrm{diff}(x_n)\,\right)
```

Then obtain

```math
\tilde{y}=\mathrm{SG}(y_n;\ \texttt{window\_length},\ \texttt{polyorder},\,\texttt{deriv}=0),\quad
y'=\mathrm{SG}(y_n;\ \cdots,\,\texttt{deriv}=1,\,\Delta x),\quad
y''=\mathrm{SG}(y_n;\ \cdots,\,\texttt{deriv}=2,\,\Delta x)
```

*Small-n safety*: $`\texttt{window\_length}`$ is clipped to the largest odd $`\le \lfloor n/2\rfloor\times2+1`$, and $`\texttt{polyorder} < \texttt{window\_length}`$.

4. **Curvature**:

```math
\kappa = \frac{|y''|}{\left(1 + (y')^{2}\right)^{3/2}}
```

5. **Endpoint handling**: Set $`\kappa_0=\kappa_{n-1}=-\infty`$.

6. **Selection**:  $`\texttt{optimal\_k} = k_{\argmax \kappa}`$ , and `elbow_idx = argmax κ`.

**Arguments**

* `k_list`: evaluated cluster counts.
* `inertia_list`: inertia per K (e.g., mean ($`1-\cos`$)).
* `smooth` (default True): enable S-G–based derivative computation.
* `window_length`, `polyorder`: S-G parameters; auto-adjusted for short series.

**Returns**

* `optimal_k`: chosen cluster count.
* `elbow_idx`: index in `k_list` where curvature peaks.
* `kappa`: curvature array (endpoints are $`-\infty`$).

**Notes**

* Using S-G with `deriv=1,2` yields **analytic derivatives of the local polynomial fit**, stabilizing curvature against small wiggles while preserving the elbow shape.
* If `smooth=False` or $`n<5`$, you may fallback to finite differences for $`y'`$, $`y''`$ on $`(x_n,y_n)`$.

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
