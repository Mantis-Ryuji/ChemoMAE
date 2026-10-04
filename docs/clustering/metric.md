# Cosine Silhouette — CPU and CUDA

> API reference for ChemoMAE v0.2.4.

> Module: `chemomae.clustering.metric`

The **cosine-based silhouette score** describes within-cluster compactness
relative to separation from other clusters in a given feature space. These
functions evaluate that diagnostic on CPU or CUDA using cluster sums rather
than a full pairwise distance matrix.

## Quick start

This small CPU example evaluates two directional groups. The scalar score is
the mean of the per-row coefficients.

```python
import numpy as np
from chemomae.clustering import (
    silhouette_samples_cosine_gpu,
    silhouette_score_cosine_gpu,
)

X = np.array([[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0], [-1.0, -0.1]], dtype=np.float32)
labels = np.array([0, 0, 7, 7], dtype=np.int64)
scores = silhouette_samples_cosine_gpu(X, labels, device="cpu", chunk=2)
score = silhouette_score_cosine_gpu(X, labels, device="cpu", chunk=2)

assert scores.shape == (4,) and scores.dtype == np.float32
assert np.isfinite(scores).all() and np.all((scores > 0) & (scores <= 1))
np.testing.assert_allclose(score, scores.mean(), rtol=1e-6, atol=1e-6)
```

---

## Overview

The **cosine-based silhouette coefficient** quantifies clustering compactness and separation for each sample $i$:

$$
d(x,y) = 1 - \cos(x,y)
$$

$$
a_i = \frac{1}{|C_{c(i)}|-1} \sum_{j\in C_{c(i)}, j\neq i} d(x_i, x_j)
$$

$$
b_i = \min_{k \neq c(i)} \frac{1}{|C_k|} \sum_{j\in C_k} d(x_i, x_j)
$$

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1,1].
$$

* **Cosine distance:** $d(x,y) = 1 - \cos(x,y)$.
  Rows are normalized with a norm floor of `eps`; zero vectors remain zeros
  (`cos=0 → distance=1`). Sub-threshold nonzero rows remain below unit length,
  so their dot products follow this numerical convention rather than exact cosine.

* **CPU/CUDA implementation:** Vectorized cluster-sum arithmetic with time **O(NKD)**.

* **Chunked evaluation:** Supports block-wise computation of $b_i$ to reduce memory usage.

* **Reference definition:** Computes cosine silhouette under the stated normalization
  and precision conventions. It is not a general replacement for scikit-learn's
  metric selection, string labels, or sampling options.

---

## API

### Function: `silhouette_samples_cosine_gpu(X, labels, *, device="cuda", chunk=1_000_000, dtype=torch.float32, return_numpy=True, eps=1e-12)`

Compute the silhouette coefficient for each sample.

#### Parameters

| Name           | Type                                       | Default         | Description                                                                       |
| -------------- | ------------------------------------------ | --------------- | --------------------------------------------------------------------------------- |
| `X`            | `(N,D)` `np.ndarray` or `torch.Tensor`     | —               | Finite real input features. Row norms are floored by `eps` during normalization; zero rows remain zero. |
| `labels`       | `(N,)` `np.ndarray` or `torch.Tensor[int]` | —               | Cluster assignments. Non-consecutive labels are remapped internally to `0..K-1`.  |
| `device`       | `str` or `torch.device`                    | `"cuda"`        | CPU or CUDA computation device; it does not automatically follow a tensor input. |
| `chunk`        | `int` or `None`                            | `1_000_000`     | Block size for inter-cluster distance computation (smaller → lower memory).       |
| `dtype`        | `torch.dtype`                              | `torch.float32` | Input/output precision: float16, bfloat16, float32, or float64. Half intermediates use float32. |
| `return_numpy` | `bool`                                     | `True`          | Return type (`np.ndarray` if True, else `torch.Tensor`).                          |
| `eps`          | `float`                                    | `1e-12`         | Finite positive floor for row norms. The silhouette denominator uses a separate fixed floor of `1e-12`. |

#### Returns

| Name | Type                           | Description                                                                         |
| ---- | ------------------------------ | ----------------------------------------------------------------------------------- |
| `s`  | `(N,)` same type as input flag | Silhouette coefficients in `[-1,1]`. Singleton clusters (`n=1`) are assigned 0. |

#### Notes

* Time: $O(NKD)$; no full pairwise sample-distance matrix is constructed.
* Chunking bounds the `(B, K)` similarity temporary. Full input features, labels,
  class sums, within-class work arrays, and output remain resident.
* Features must be finite real values of shape `(N,D)`. Integer labels must have
  shape `(N,)`; floating labels are rejected before any truncation.
* Require `2 <= K < N`, positive integer chunks (or None), and finite positive eps.
  One-class/all-singleton assignments have undefined silhouette and raise ValueError.
* A singleton's score and a zero-distance denominator produce zero. Positive
  denominators below `1e-12` are floored, which can differ from the exact formula.
* Requested arithmetic overrides ambient AMP. Tensor output retains requested
  dtype on `device`; NumPy output is always float32 on CPU.

---

### Function: `silhouette_score_cosine_gpu(X, labels, *, return_numpy=True, **kwargs)`

Convenience function returning the **mean silhouette coefficient** (scalar).

* Reduces per-sample values in float32. Returns a Python float by default, or a
  scalar float32 tensor when `return_numpy=False`.
* Same arguments as above (`**kwargs` are forwarded).

---

## Additional examples

### Torch tensor output (CPU)

This snippet continues the NumPy quick start and requests a tensor result.

```python
import torch

tensor_scores = silhouette_samples_cosine_gpu(
    torch.as_tensor(X), torch.as_tensor(labels),
    device="cpu", dtype=torch.float64, return_numpy=False,
)
assert tensor_scores.dtype == torch.float64 and tensor_scores.device.type == "cpu"
```

---

### PyTorch (GPU)

```python
import torch
from chemomae.clustering.metric import silhouette_samples_cosine_gpu

if torch.cuda.is_available():
    gpu_X = torch.randn(200, 32, device="cuda", dtype=torch.float32)
    gpu_labels = torch.arange(200, device="cuda") % 5
    gpu_scores = silhouette_samples_cosine_gpu(
        gpu_X, gpu_labels, device="cuda", return_numpy=False,
    )
    assert gpu_scores.device.type == "cuda"
```

---

### Chunked computation (large N)

```python
# For N > 100,000, this reduces the B-by-K temporary relative to the default.
# Here it demonstrates the call on the quick start's small CPU inputs.
s = silhouette_samples_cosine_gpu(X, labels, device="cpu", chunk=100_000)
```

---

## Design Notes

* **Zero vectors:** Treated as cosine=0 vs. any vector → distance=1.
* **Singleton clusters:** Assigned silhouette=0, consistent with sklearn.
* **Non-consecutive labels:** Automatically remapped; results unaffected.
* **Precision:** Supports float16/bfloat16 input/output with float32 intermediates,
  plus float32/float64 arithmetic. Quantized inputs can differ from full precision.
* **Memory:** Similarity chunking reduces one temporary; it does not bound full
  input/output or every work array. Runtime and GPU memory are unmeasured.

---

## Interpretation and aggregation

Use silhouette to describe separation within the supplied feature space.
Different encoders or preprocessing choices produce different distance
structures, so a larger score alone does not establish a more useful
representation for an application. A partition can also be useful without
maximizing this diagnostic.

Spatial coherence is a separate property of the resulting label map, evaluated
by [Local Label Agreement](spatial.md). Inspect both diagnostics with cluster
occupancy and the map when interpreting spatial data. Neither metric establishes
agreement with external ground truth.

The sample function returns one coefficient per supplied row. Its distance
averages use all supplied members of each cluster. If the application requires
equal-weight groups (for example, images, subjects, or batches), retain the row
mapping, average coefficients within each group, then aggregate those means.
The scalar score function instead averages all rows, giving larger groups more
weight. Equal-weight final aggregation does not remove row-count influence from
the distances used to compute the coefficients. Group selection and aggregation
policy belong to the caller.

---

## Common Pitfalls

* Works **only with cosine distance** — other metrics are unsupported.
* Input normalization is handled internally. Pre-normalizing very small rows can
  change results because it changes which values fall below the norm floor.
* Very small clusters may yield unstable values (same behavior as sklearn).
* Empty/mismatched arrays, invalid class counts, nonfinite values, unsupported
  precision, and nonpositive chunks fail before computation.

---

## Minimal Test Snippets

The reference snippet additionally requires scikit-learn from the development extras.

```python
import numpy as np
import torch
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from chemomae.clustering.metric import silhouette_samples_cosine_gpu

X = np.random.randn(50, 8).astype(np.float32)
labels = np.random.randint(0, 3, size=50)

ours = silhouette_samples_cosine_gpu(
    X, labels, device="cpu", return_numpy=True, dtype=torch.float32
)
ref = sk_silhouette_samples(X, labels, metric="cosine")

np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)
```

---

See the [v0.2.4 release notes](../../CHANGELOG.md#024).
