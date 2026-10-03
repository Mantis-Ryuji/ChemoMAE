# Cosine Silhouette — GPU Implementation

> Module: `chemomae.clustering.metric`

The **cosine-based silhouette score** describes within-cluster compactness
relative to separation from other clusters in a given feature space. These
functions evaluate that diagnostic on CPU or CUDA using cluster sums rather
than a full pairwise distance matrix.

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
  Internally, all rows are L2-normalized; zero vectors remain zeros (`cos=0 → distance=1`).

* **CPU/CUDA implementation:** Vectorized cluster-sum arithmetic with time **O(NKD)**.

* **Chunked evaluation:** Supports block-wise computation of $b_i$ to reduce memory usage.

* **API parity:** Equivalent to `sklearn.metrics.silhouette_samples` / `silhouette_score`, but specialized for cosine distance.

---

## API

### Function: `silhouette_samples_cosine_gpu(X, labels, *, device="cuda", chunk=1_000_000, dtype=torch.float32, return_numpy=True, eps=1e-12)`

Compute the silhouette coefficient for each sample.

#### Parameters

| Name           | Type                                       | Default         | Description                                                                       |
| -------------- | ------------------------------------------ | --------------- | --------------------------------------------------------------------------------- |
| `X`            | `(N,D)` `np.ndarray` or `torch.Tensor`     | —               | Input feature matrix. Rows are L2-normalized internally (zero rows remain zeros). |
| `labels`       | `(N,)` `np.ndarray` or `torch.Tensor[int]` | —               | Cluster assignments. Non-consecutive labels are remapped internally to `0..K-1`.  |
| `device`       | `str`                                      | `"cuda"`        | Target device for computation (`"cuda"` or `"cpu"`).                              |
| `chunk`        | `int` or `None`                            | `1_000_000`     | Block size for inter-cluster distance computation (smaller → lower memory).       |
| `dtype`        | `torch.dtype`                              | `torch.float32` | Input/output precision: float16, bfloat16, float32, or float64. Half intermediates use float32. |
| `return_numpy` | `bool`                                     | `True`          | Return type (`np.ndarray` if True, else `torch.Tensor`).                          |
| `eps`          | `float`                                    | `1e-12`         | Small constant for numerical stability.                                           |

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
* A singleton's score and a zero-distance denominator are assigned zero.
* Requested arithmetic overrides ambient AMP. Tensor output retains requested
  dtype; NumPy output is always float32.

---

### Function: `silhouette_score_cosine_gpu(X, labels, **kwargs)`

Convenience function returning the **mean silhouette coefficient** (scalar).

* Reduces per-sample values in float32. Returns a Python float by default, or a
  scalar float32 tensor when `return_numpy=False`.
* Same arguments as above (`**kwargs` are forwarded).

---

## Usage Examples

### NumPy (CPU)

```python
import numpy as np
from chemomae.clustering.metric import (
    silhouette_samples_cosine_gpu,
    silhouette_score_cosine_gpu,
)

# Data: 100 samples, 16-dim
X = np.random.randn(100, 16).astype(np.float32)
labels = np.random.randint(0, 4, size=100)

s = silhouette_samples_cosine_gpu(X, labels, device="cpu")
print(s.shape)        # (100,)
print(s.min(), s.max())

score = silhouette_score_cosine_gpu(X, labels, device="cpu")
print("Mean silhouette:", score)
```

---

### PyTorch (GPU)

```python
import torch
from chemomae.clustering.metric import silhouette_samples_cosine_gpu

X = torch.randn(200, 32, device="cuda", dtype=torch.float32)
labels = torch.randint(0, 5, (200,), device="cuda")

s = silhouette_samples_cosine_gpu(X, labels, device="cuda", return_numpy=False)
print(s[:5])  # torch.Tensor on CUDA
```

---

### Chunked computation (large N)

```python
# For very large datasets, use chunking to save GPU memory
s = silhouette_samples_cosine_gpu(X, labels, device="cuda", chunk=5_000_000)
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

## Interpretation in ChemoMAE

ChemoMAE learns a spectral representation that can be quantized at a chosen
granularity $K$. This use of clustering does not require clearly separated
populations or maximization of silhouette. Use cosine silhouette as a diagnostic
of separation within each representation. Different encoders or preprocessing
choices produce different distance structures, so a larger score alone does not
establish a better representation of chemical-state differences.

Spatial coherence is a separate property of the resulting label map, evaluated
by [Local Label Agreement](spatial.md). Inspect both diagnostics with cluster
occupancy and the map when interpreting a partition. Neither metric measures
agreement with chemical ground truth.

The sample function returns one coefficient per supplied row. Its distance
averages use all supplied members of each cluster. For specimen-level analysis,
retain the row-to-specimen mapping and average the returned coefficients within
each specimen before any equal-weight specimen aggregation. The scalar score
function instead averages all rows, giving specimens with more rows more weight.
Equal-weight final aggregation does not remove that row-count influence from
the distances used to compute the coefficients.

---

## Common Pitfalls

* Works **only with cosine distance** — other metrics are unsupported.
* Input normalization is handled internally; external L2 normalization is optional.
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

## Version

* Introduced in `chemomae.clustering.metric` — initial public draft.
