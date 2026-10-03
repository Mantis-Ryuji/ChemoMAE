# CosineKMeans — Hyperspherical K-Means Clustering

> Module: `chemomae.clustering.cosine_kmeans`

**CosineKMeans** implements spherical k-means: it partitions feature directions
using cosine similarity and normalizes each nonzero centroid after updating it.
Ordinary Euclidean k-means uses arithmetic-mean centroids whose lengths can differ
even when the input rows have unit norm. Normalizing the centroids keeps the
assignment rule based on direction.

In the ChemoMAE workflow, the encoder first learns a low-dimensional spectral
representation. With the learned model fixed, extract features without spectral
augmentation and with all patches visible, then fit the clustering model. The
unit-norm latent representation concentrates information in direction, matching
the use of cosine similarity for SNV-transformed spectra. It has no zero-mean
constraint, and the encoder need not preserve input cosine similarities.

Clustering quantizes this representation at an observation granularity $K$ so
that the spatial distributions of spectral differences can be examined. Cluster
IDs are assignment identifiers; they do not encode known chemical states or an
ordered degree of degradation. Spatial coordinates can be used to place labels
back into an image after prediction; `CosineKMeans` does not use coordinates or
neighbor relations in fitting. The optional **`elbow_ckmeans`** helper explores
the objective curve over several values of $K$.

---

## Overview

* **Objective** — minimize mean cosine dissimilarity:

  $$
  J = \mathrm{mean}(1 - \cos(x, c))
  $$

* **E-step:** Assign each sample to the centroid with the highest cosine similarity.

* **M-step:** Update centroids as the L2-normalized mean of assigned samples.

* **k-means++ Initialization:** Uses cosine dissimilarity for sampling (not squared), ensuring reproducible seeding with `random_state`.

* **Streaming support:** Large datasets can be processed in CPU→GPU chunks.

* **Precision:** Internally computed in `float32` (even for half/bf16 inputs).

* **Post-condition:** Nonzero centroids are L2-normalized; zero means remain zero.

---

## API

### Class: `CosineKMeans`

```python
ckm = CosineKMeans(
    n_components=8,
    tol=1e-4,
    max_iter=500,
    device="cuda",
    random_state=42
)
```

#### Parameters

| Name               | Type    | Default       | Description                                                  |
| ------------------ | ------- | ------------- | ------------------------------------------------------------ |
| `n_components`     | `int`   | —             | Number of clusters  $K$.                                   |
| `tol`              | `float` | `1e-4`        | Convergence tolerance on inertia.                            |
| `max_iter`         | `int`   | `500`         | Maximum number of centroid updates.                          |
| `device`           | `str` or `torch.device` | `"cuda"` | Device for computation.                           |
| `random_state`     | `int` or `None`         | `42`   | Initialization stream seed.                     |



#### Attributes

| Name         | Type                  | Description                                        |
| ------------ | --------------------- | -------------------------------------------------- |
| `centroids`  | `torch.Tensor (K, D)` | Learned cluster centroids (L2-normalized).         |
| `latent_dim` | `int`                 | Feature dimension $D$.                           |
| `inertia_`   | `float`               | Final objective value $\mathrm{mean}(1 - \cos)$. |
| `n_iter_` | `int` | Number of completed centroid updates; zero before fit. |
| `converged_` | `bool` | Whether the existing objective-tolerance criterion stopped fitting. |
| `stop_reason_` | `str` or `None` | `"tolerance"`, `"max_iter"`, or `None` before fit. |

---

### Methods

| Method                                      | Description                                                                      |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| `fit(X, chunk=None)`                        | Train centroids on data `X (N, D)`. If `chunk > 0`, enables streaming (CPU→GPU). |
| `fit_predict(X, chunk=None)`                | Fit and return cluster assignments.                                              |
| `predict(X, return_dist=False, chunk=None)` | Predict labels for `X`. Returns `(labels, dist)` if `return_dist=True`.          |
| `save_centroids(path)`                      | Save versioned fitted prediction state with exact centers, config, and diagnostics. |
| `load_centroids(path, strict_k=True)`       | Validate and restore fitted state; check K if strict, otherwise adopt saved K. |

---

## Usage Examples

### Training and prediction

```python
from chemomae.clustering.cosine_kmeans import CosineKMeans

X = torch.randn(1000, 64)
ckm = CosineKMeans(n_components=10)
ckm.fit(X)
labels = ckm.predict(X)
```

### Saving and reloading

```python
ckm.save_centroids("centroids.pt")
ckm2 = CosineKMeans(n_components=10).load_centroids("centroids.pt")
labels2 = ckm2.predict(X)
```

### Streaming large datasets

```python
ckm = CosineKMeans(n_components=50, device="cuda")
ckm.fit(X, chunk=1_000_000)  # process in CPU→GPU batches
```

### Returning distances

```python
labels, dist = ckm.predict(X, return_dist=True)
```

---

## Exploring Cluster Count — `elbow_ckmeans`

The helper fits a separate model for each tested $K$ and returns a curvature-based
elbow of the mean cosine-inertia curve. The API name `optimal_k` denotes that
heuristic candidate. It does not estimate the true number of chemical states.
For comparisons at a prescribed observation granularity, fit `CosineKMeans`
directly with the same `n_components` across conditions. A sweep can then describe
how the partition depends on that granularity.

```python
from chemomae.clustering.cosine_kmeans import elbow_ckmeans

# Find a heuristic elbow candidate from the inertia curve
k_list, inertias, optimal_k, elbow_idx, kappa = elbow_ckmeans(CosineKMeans, X, k_max=30, chunk=1_000_000)
```

#### Parameters

| Name             | Type                   | Description                                   |
| ---------------- | ---------------------  | --------------------------------------------- |
| `cluster_module` | callable               | Constructor (compatible with `CosineKMeans`). |
| `X`              | `torch.Tensor (N, D)`  | Input dataset.                                |
| `device`         | `str` or `torch.device`| Target device.                                |
| `k_max`          | `int`                  | Maximum number of clusters to test.           |
| `chunk`          | `int` or `None`        | Optional streaming batch size.                |
| `verbose`        | `bool`                 | Print per-K inertia if `True`.                |
| `random_state`   | `int` or `None`        | RNG seed for reproducibility.                 |

#### Returns

| Name        | Type          | Description                                |
| ----------- | ------------- | ------------------------------------------ |
| `k_list`    | `list[int]`   | Values of K tested.                        |
| `inertias`  | `list[float]` | Corresponding mean inertia values.         |
| `optimal_k` | `int`         | Heuristic cluster-count candidate at the curvature-based elbow. |
| `elbow_idx` | `int`         | Index of that candidate in `k_list`.        |
| `kappa`     | `float`       | Curvature score at the elbow.              |

---

## Design Notes

* **Normalization:** Each input vector is internally L2-normalized before similarity computation.
* **Empty clusters:** If a cluster receives no assignments, it is reinitialized with the farthest sample.
* **Inertia metric:** Uses $\mathrm{mean} (1 - \cos)$, not Euclidean SSE.
* **Memory safety:** Frees GPU cache per iteration during streaming/elbow search.
* **Numerical stability:** Small epsilon added in normalization to avoid division by zero.

---

## Minimal Tests

```python
X = torch.randn(200, 16)
ckm = CosineKMeans(n_components=5)
ckm.fit(X)
labels = ckm.predict(X)
assert labels.shape == (200,)

# Save / reload
ckm.save_centroids("tmp.pt")
ckm2 = CosineKMeans(n_components=5).load_centroids("tmp.pt")
assert torch.allclose(ckm.centroids, ckm2.centroids)
```

---

## Version

### v0.2.3

`inertia_` is recomputed against the exact final prediction buffer after the last
centroid update. Stopping still uses the existing objective-tolerance test; the
extra final assignment pass provides accurate diagnostics without changing that
criterion. This pass adds one assignment cost to fitting.

Fitted-state files use `format_version=1` and include constructor configuration,
feature dimension, FP32 centers, final objective, and convergence diagnostics.
Centers are saved and restored without renormalization or dtype conversion;
this preserves the fitted predictor's stored values. Loading validates the
payload before changing state. Unsupported/old unversioned files fail clearly.
The caller's selected device is retained, and advanced initialization RNG state
is not saved. This is prediction reuse, not continuation of an interrupted fit.

Streaming bounds transfer/assignment intermediates but still allocates full
label and maximum-similarity outputs. `return_dist=True` additionally allocates
the full `(N, K)` matrix. It does not make the complete input/output resident
memory independent of dataset size.

* Introduced in `chemomae.clustering.cosine_kmeans` — initial public draft.
