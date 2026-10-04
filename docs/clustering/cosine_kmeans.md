# CosineKMeans — Hyperspherical K-Means Clustering

> API reference for ChemoMAE v0.2.4.

> Module: `chemomae.clustering.cosine_kmeans`

**CosineKMeans** implements spherical k-means: it partitions feature directions
using cosine similarity and applies row normalization after centroid updates.
Ordinary Euclidean k-means uses arithmetic-mean centroids whose lengths can differ
even when the input rows have unit norm. Normalizing the centroids keeps the
assignment rule based on direction.

Use it with finite feature tensors when direction, rather than magnitude, is the
intended comparison. Features may come from an encoder, a preprocessing pipeline,
or another source; no ChemoMAE model or spatial coordinates are required. Cluster
IDs are unordered assignment identifiers. For probabilistic assignments with
component-specific concentrations, see [VMFMixture](vmf_mixture.md).

The optional **`elbow_ckmeans`** helper explores the objective curve over several
values of $K$. A [workflow example](../tutorials/workflow.md) shows how to combine
feature extraction, clustering, and spatial evaluation.

## Quick start

This synthetic CPU example fits two directional groups and checks the returned
distances against the reported objective.

Pass `device` explicitly when composing a workflow: the constructor defaults to
`"cuda"` and does not infer the computation device from the input features.
Use `device="cpu"` on a CPU-only machine; omitting it raises a CUDA availability
error. Trainer/Extractor have a different default: they follow their model's device.

```python
import torch
from chemomae.clustering import CosineKMeans

X = torch.tensor([
    [1.0, 0.0], [1.0, 0.1], [1.0, -0.1],
    [-1.0, 0.0], [-1.0, 0.1], [-1.0, -0.1],
])
ckm = CosineKMeans(n_components=2, device="cpu", random_state=42, max_iter=10)
ckm.fit(X)
labels, distances = ckm.predict(X, return_dist=True)

assert labels.shape == (6,) and labels.dtype == torch.int64
assert labels.device.type == "cpu" and distances.shape == (6, 2)
assert torch.equal(labels, distances.argmin(dim=1))
assert torch.isclose(torch.tensor(ckm.inertia_), distances.min(dim=1).values.mean())
```

---

## Overview

* **Objective** — minimize mean cosine dissimilarity:

  $$
  J = \mathrm{mean}(1 - \cos(x, c))
  $$

* **E-step:** Assign each sample to the centroid with the highest cosine similarity.

* **M-step:** Update centroids as the L2-normalized mean of assigned samples.

* **k-means++ Initialization:** Samples using cosine dissimilarity, without squaring it, through an instance-owned CPU generator.

* **Streaming support:** Large datasets can be processed in CPU→GPU chunks.

* **Precision:** Inputs, centers, and accumulators use `float32`, including for half/bf16 inputs. Calls do not disable ambient autocast; run outside autocast when FP32 matrix arithmetic is required.

* **Normalization convention:** Norms are clamped below by `1e-6` before division. Zero vectors stay zero; sufficiently small nonzero vectors can remain below unit length.

---

## API

### Class: `CosineKMeans`

```python
from chemomae.clustering import CosineKMeans

configured_ckm = CosineKMeans(
    n_components=8,
    tol=1e-4,
    max_iter=500,
    device="cpu",  # the constructor default is "cuda"
    random_state=42
)
```

#### Parameters

| Name               | Type    | Default       | Description                                                  |
| ------------------ | ------- | ------------- | ------------------------------------------------------------ |
| `n_components`     | `int`   | `8`           | Positive number of clusters $K$, at most the number of fitting rows. |
| `tol`              | `float` | `1e-4`        | Finite nonnegative tolerance: stop when successive pre-update objectives differ relatively by less than `tol` or absolutely by less than `tol * 1e-3`. Zero disables this stopping condition. |
| `max_iter`         | `int`   | `500`         | Maximum number of centroid updates.                          |
| `device`           | `str` or `torch.device` | `"cuda"` | Device for computation.                           |
| `random_state`     | `int` or `None`         | `42`   | CPU initialization stream seed. The stream advances across fits; `None` retains the generator's default seed. |



#### Attributes

| Name         | Type                  | Description                                        |
| ------------ | --------------------- | -------------------------------------------------- |
| `centroids`  | `torch.Tensor (K, D)` | FP32 centers on the computation device, using the normalization convention above; empty before fit. |
| `latent_dim` | `int` or `None`       | Feature dimension $D$; `None` before fit. |
| `inertia_`   | `float`               | Final mean nearest-center dissimilarity; infinity before fit. |
| `n_iter_` | `int` | Number of completed centroid updates; zero before fit. |
| `converged_` | `bool` | Whether the existing objective-tolerance criterion stopped fitting. |
| `stop_reason_` | `str` or `None` | `"tolerance"`, `"max_iter"`, or `None` before fit. |

---

### Methods

| Method                                      | Description                                                                      |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| `fit(X, chunk=None)`                        | Fit centers and return `self`. A positive chunk enables CPU→GPU streaming on CUDA; CPU fitting uses the full matrix. |
| `fit_predict(X, chunk=None)`                | Fit and return cluster assignments.                                              |
| `predict(X, return_dist=False, chunk=None)` | Predict labels for `X`. Returns `(labels, dist)` if `return_dist=True`.          |
| `save_centroids(path)`                      | Save versioned fitted prediction state with exact centers, config, and diagnostics. |
| `load_centroids(path, *, strict_k=True)`    | Validate and restore fitted state; check K if strict, otherwise adopt saved K. Return `self`. |

`X` must be a nonempty dense real Torch tensor of shape `(N, D)` with finite
values representable in FP32. NumPy inputs must be converted by the caller.
Prediction requires fitted centers and the same feature dimension. Invalid
features, `K > N`, and nonpositive/noninteger chunks raise errors. Zero rows are
accepted and have similarity zero to every center; ties select the first center.

Labels have shape `(N,)`, dtype `torch.int64`, and the centers' device.
`return_dist=True` also returns the full `(N, K)` dissimilarity matrix on that
device. Outside autocast, distances are FP32. Chunking does not reduce this
output allocation. Constructor device defaults to CUDA; choose `device="cpu"`
when CUDA is unavailable.

---

## Additional examples

These snippets continue the quick start. The save/load example writes to the
chosen output path.

### Saving and reloading

```python
ckm.save_centroids("centroids.pt")
ckm2 = CosineKMeans(n_components=2, device="cpu").load_centroids("centroids.pt")
labels2 = ckm2.predict(X)
assert torch.equal(labels, labels2)
```

### Streaming large datasets

```python
if torch.cuda.is_available():
    ckm_gpu = CosineKMeans(n_components=2, device="cuda")
    ckm_gpu.fit(X, chunk=3)  # keep X on CPU and transfer batches
```

### Returning distances

```python
labels, dist = ckm.predict(X, return_dist=True)
```

---

## Exploring Cluster Count — `elbow_ckmeans`

The helper fits a separate model for each tested $K$ and returns a curvature-based
elbow of the mean cosine-inertia curve. The API name `optimal_k` denotes that
heuristic candidate; it does not establish a uniquely correct cluster count.
If the application specifies $K$, fit `CosineKMeans` directly. A sweep can
describe how the partition depends on $K$.

```python
from chemomae.clustering.cosine_kmeans import elbow_ckmeans

# Find a heuristic elbow candidate from the inertia curve
k_list, inertias, optimal_k, elbow_idx, kappa = elbow_ckmeans(
    CosineKMeans, X, device="cpu", k_max=4, verbose=False,
)
```

#### Parameters

| Name             | Type                   | Description                                   |
| ---------------- | ---------------------  | --------------------------------------------- |
| `cluster_module` | callable               | Constructor (compatible with `CosineKMeans`). |
| `X`              | `torch.Tensor (N, D)`  | Input dataset.                                |
| `device`         | `str` or `torch.device`| Target device; default `"cuda"`. |
| `k_max`          | `int`                  | Sweep `1..k_max`; default `50`. Use `3 <= k_max <= N` so the curve has an interior point and each fit is valid. |
| `chunk`          | `int` or `None`        | CUDA streaming batch size; default `None`. |
| `verbose`        | `bool`                 | Print per-K inertia; default `True`. |
| `random_state`   | `int` or `None`        | Initialization seed supplied to each model; default `42`. |

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

* **Normalization:** Uses [row normalization](ops.md) with a norm floor of `1e-6`; dot products equal cosine similarity for unit rows. Zero and sub-threshold rows follow the stated numerical convention.
* **Empty clusters:** If a cluster receives no assignments, it is reinitialized with the farthest sample.
* **Inertia metric:** Uses $\mathrm{mean} (1 - \cos)$, not Euclidean SSE.
* **Memory:** Streaming bounds assignment/transfer temporaries on CUDA, while the full CPU feature matrix and device labels/maximum similarities remain resident. Cache cleanup occurs after streaming fit and after each K in an elbow sweep, not after every update. It does not free live tensors or guarantee a memory bound.
* **Reproducibility:** A fresh instance with the same seed uses the same initial RNG stream. Arithmetic and sampling probabilities may differ across devices; the seed does not promise identical CPU/CUDA fits.

---

## Fitted results and persistence

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

See the [v0.2.4 release notes](../../CHANGELOG.md#024).
