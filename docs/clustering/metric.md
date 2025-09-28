# Cosine Silhouette — GPU implementation

> Module: `chemomae.clustering.metric`

This document describes the **cosine-based silhouette score** functions implemented with GPU acceleration for clustering evaluation.

---

## Overview

The **silhouette coefficient** is a clustering quality measure defined for each sample (i):

```math
d(x,y) = 1 - \cos(x,y),\qquad
a_i = \frac{1}{|C_{c(i)}|-1} \sum_{j\in C_{c(i)}, j\neq i} d(x_i, x_j),\qquad
b_i = \min_{k \neq c(i)} \frac{1}{|C_k|} \sum_{j\in C_k} d(x_i, x_j)
```

```math
s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1,1].
```

* **Cosine distance:** ($`d(x,y) = 1 - \cos(x,y)`$). Internally, all rows are L2-normalized; zero vectors remain zeros (cos=0 → distance=1).
* **GPU accelerated:** Vectorized with PyTorch, complexity **O(NK)** (vs. O(N²) naïve).
* **Chunked evaluation:** Supports block-wise computation of ($`b_i`$) to reduce memory.
* **API parity:** Equivalent to `sklearn.metrics.silhouette_samples` / `silhouette_score`, but specialized for cosine distance.

---

## API

### Function: `silhouette_samples_cosine_gpu(X, labels, *, device="cuda", chunk=1_000_000, dtype=torch.float32, return_numpy=True, eps=1e-12)`

Compute the silhouette coefficient for each sample.

**Parameters**

* `X` (`(N,D)` NumPy or Torch): Feature matrix.

  * Rows are L2-normalized internally.
  * Zero rows remain zeros.
* `labels` (`(N,)` NumPy or Torch, int): Cluster assignments. Non-consecutive labels are remapped internally to `0..K-1`.
* `device` (`str`): `"cuda"`, `"cpu"`, etc. for computation.
* `chunk` (`int | None`): Block size for computing inter-cluster distances. Smaller → less memory.
* `dtype` (`torch.dtype`): Computation precision (`float16`, `bfloat16`, `float32`).
* `return_numpy` (`bool`): If `True`, return `np.ndarray`; else `torch.Tensor`.
* `eps` (`float`): Small constant for numerical stability.

**Returns**

* `s` (`(N,)` same type as input flag): Silhouette coefficients per sample in `[-1,1]`.
  Singleton clusters (`n=1`) are assigned **0**.

**Notes**

* Complexity: ($`O(ND + KD + N K_{\text{chunk}})`$). Dominated by `X @ Mᵀ`.
* Memory: depends on `chunk` size; small `chunk` lowers peak memory.

---

### Function: `silhouette_score_cosine_gpu(X, labels, **kwargs)`

Convenience function returning the **mean silhouette coefficient** (scalar).

* Calls `silhouette_samples_cosine_gpu` and returns `float(s.mean())`.
* Same arguments as above (`**kwargs` forwarded).

---

## Usage Examples

### NumPy (CPU)

```python
import numpy as np
from chemomae.clustering.metric import silhouette_samples_cosine_gpu, silhouette_score_cosine_gpu

# Data: 100 samples, 16-dim
X = np.random.randn(100, 16).astype(np.float32)
labels = np.random.randint(0, 4, size=100)

s = silhouette_samples_cosine_gpu(X, labels, device="cpu")
print(s.shape)        # (100,)
print(s.min(), s.max())

score = silhouette_score_cosine_gpu(X, labels, device="cpu")
print("Mean silhouette:", score)
```

### PyTorch (GPU)

```python
import torch
from chemomae.clustering.metric import silhouette_samples_cosine_gpu

X = torch.randn(200, 32, device="cuda", dtype=torch.float32)
labels = torch.randint(0, 5, (200,), device="cuda")

s = silhouette_samples_cosine_gpu(X, labels, device="cuda", return_numpy=False)
print(s[:5])  # torch.Tensor on CUDA
```

### Chunked computation

```python
# For large N, use chunking to save memory
s = silhouette_samples_cosine_gpu(X, labels, device="cuda", chunk=5000)
```

---

## Design Notes & Edge Cases

* **Zero vectors:** Treated as cosine=0 vs. any vector → distance=1.
* **Singleton clusters:** Assigned silhouette=0, consistent with sklearn.
* **Non-consecutive labels:** Remapped internally, results unaffected.
* **Precision:** Supports `float16` / `bfloat16` for speed on GPU (with small numerical drift).
* **Performance:** On GPU, `O(NK)` evaluation is scalable; chunking avoids OOM for large N.

---

## When to Use in ChemoMAE

* **Cluster validation:** After applying `CosineKMeans` or other cosine-based clustering.
* **Model selection:** Compare silhouette scores across numbers of clusters.
* **Unsupervised evaluation:** Provides a heuristic for structure quality in latent spaces.

---

## Common Pitfalls

* Expecting it to work with arbitrary metrics — **only cosine distance** is supported.
* Passing pre-normalized data incorrectly: input is always normalized inside, so external normalization is optional.
* Very small clusters → silhouette values may be unstable (as with sklearn).

---

## Minimal Test Snippets

```python
import numpy as np
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from chemomae.clustering.metric import silhouette_samples_cosine_gpu

X = np.random.randn(50, 8).astype(np.float32)
labels = np.random.randint(0, 3, size=50)

ours = silhouette_samples_cosine_gpu(X, labels, device="cpu", return_numpy=True, dtype=torch.float64)
ref = sk_silhouette_samples(X, labels, metric="cosine")

np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-7)
```

---

## Version

* Introduced in `chemomae.clustering.metric`.
