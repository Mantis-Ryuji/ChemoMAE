# Cosine Silhouette — GPU Implementation

> Module: `chemomae.clustering.metric`

This document describes the **cosine-based silhouette score** functions implemented with GPU acceleration for clustering evaluation.

---

## Overview

The **cosine-based silhouette coefficient** quantifies clustering compactness and separation for each sample $`i`$:

```math
d(x,y) = 1 - \cos(x,y)
```

```math
a_i = \frac{1}{|C_{c(i)}|-1} \sum_{j\in C_{c(i)}, j\neq i} d(x_i, x_j)
```

```math
b_i = \min_{k \neq c(i)} \frac{1}{|C_k|} \sum_{j\in C_k} d(x_i, x_j)
```

```math
s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1,1].
```

* **Cosine distance:** $`d(x,y) = 1 - \cos(x,y)`$.
  Internally, all rows are L2-normalized; zero vectors remain zeros (`cos=0 → distance=1`).

* **GPU accelerated:** Vectorized with PyTorch, complexity **O(NK)**.

* **Chunked evaluation:** Supports block-wise computation of $`b_i`$ to reduce memory usage.

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
| `dtype`        | `torch.dtype`                              | `torch.float32` | Precision (`float16`, `bfloat16`, or `float32`).                                  |
| `return_numpy` | `bool`                                     | `True`          | Return type (`np.ndarray` if True, else `torch.Tensor`).                          |
| `eps`          | `float`                                    | `1e-12`         | Small constant for numerical stability.                                           |

#### Returns

| Name | Type                           | Description                                                                         |
| ---- | ------------------------------ | ----------------------------------------------------------------------------------- |
| `s`  | `(N,)` same type as input flag | Silhouette coefficients in `[-1,1]`. Singleton clusters (`n=1`) are assigned 0. |

#### Notes

* Complexity: $O(ND + KD + N K_{\text{chunk}})$ (approximately linear in both $N$ and $K$).
* Memory: Depends on `chunk` size; smaller `chunk` lowers peak memory usage.

---

### Function: `silhouette_score_cosine_gpu(X, labels, **kwargs)`

Convenience function returning the **mean silhouette coefficient** (scalar).

* Calls `silhouette_samples_cosine_gpu` and returns `float(s.mean())`.
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
* **Precision:** Supports `float16` / `bfloat16` for speed (minor numerical drift).
* **Performance:** On GPU, `O(NK)` evaluation is scalable; chunking prevents OOM.

---

## When to Use in ChemoMAE

* **Cluster validation:** Evaluate compactness and separation after `CosineKMeans` or any cosine-based clustering in latent embeddings.
* **Model selection:** Compare silhouette scores across different cluster counts.
* **Unsupervised evaluation:** Quantify structure quality in latent hyperspherical spaces.

---

## Common Pitfalls

* Works **only with cosine distance** — other metrics are unsupported.
* Input normalization is handled internally; external L2 normalization is optional.
* Very small clusters may yield unstable values (same behavior as sklearn).

---

## Minimal Test Snippets

```python
import numpy as np
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