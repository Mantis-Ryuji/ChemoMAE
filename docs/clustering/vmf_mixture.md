# VMFMixture — von Mises–Fisher Mixture on the Unit Hypersphere

> Module: `chemomae.clustering.vmf_mixture`
> Purpose: Probabilistic clustering of L2-normalized features on ($`S^{d-1}`$) via an EM algorithm; model selection with `elbow_vmf`; visualization via `plot_elbow_vmf`.

---

## Overview

* **Likelihood (unit vectors, $|x_i|=1$)**

```math
\max_{\pi_k,\mu_k,\kappa_k}\ \sum_{i=1}^N \log\!\Big(\sum_{k=1}^K 
\pi_k\,C_d(\kappa_k)\,e^{\kappa_k\,\mu_k^\top x_i}\Big),\quad
C_d(\kappa)=\frac{\kappa^\nu}{(2\pi)^{\nu+1}I_\nu(\kappa)},\ 
\nu=\tfrac{d}{2}-1
```

where ($`\mu_k`$) are **unit directions**, ($`\kappa_k>0`$) are **concentrations**, and ($`\pi_k`$) are **mixture weights**.

* **E-step (responsibilities)**

```math
\gamma_{ik}\propto \pi_k\,C_d(\kappa_k)\,e^{\kappa_k\,\mu_k^\top x_i}, 
\qquad \sum_k \gamma_{ik}=1
```

* **M-step**

Let $`N_k=\sum_i\gamma_{ik}`$. Then:

```math
\tilde\mu_k=\frac{\sum_i\gamma_{ik}x_i}{N_k}, \quad
\mu_k = \frac{\tilde\mu_k}{\|\tilde\mu_k\|_2}
```

The resultant length $`\bar R_k=\|\sum_i\gamma_{ik}x_i\|/N_k`$ gives a **closed-form update** for $`\kappa_k`$:

```math
\kappa_k \approx \frac{\bar R_k\,(d-\bar R_k^2)}{1-\bar R_k^2},\qquad
\pi_k = N_k / N.
```

* **Initialization:** cosine (hyperspherical) k-means++ seeding.
* **Special functions:** stable, torch-native approximations for $`\log I_\nu(\kappa)`$ and $`I_{\nu+1}(\kappa)/I_\nu(\kappa)`$ blending small-/large-$`\kappa`$ regimes.
* **Scalability:** chunked E-step (`chunk`) streams data CPU→GPU.
* **Normalization:** Inputs are L2-normalized row-wise internally.

---

## API

### Class: `VMFMixture`

```python
from chemomae.clustering.vmf_mixture import VMFMixture

vmf = VMFMixture(
    n_components=8,     # K
    d=None,             # inferred on first fit(X) if None
    device="cuda",
    random_state=42,
    tol=1e-4,
    max_iter=200,
    init="kmeans++",    # or "random"
)
```

#### Parameters

| Name           | Type                       | Default      | Description                                      |
| -------------- | -------------------------- | ------------ | ------------------------------------------------ |
| `n_components` | `int`                      | —            | Number of mixture components (K).                |
| `d`            | `Optional[int]`            | `None`       | Feature dimension; inferred on first `fit`.      |
| `device`       | `str or torch.device`      | `"cuda"`     | Target computation device.                       |
| `random_state` | `int or None`              | `42`         | RNG seed for deterministic initialization.       |
| `tol`          | `float`                    | `1e-4`       | EM convergence tolerance (relative or absolute). |
| `max_iter`     | `int`                      | `200`        | Maximum EM iterations.                           |
| `init`         | `{ "kmeans++", "random" }` | `"kmeans++"` | Initialization strategy.                         |

#### Attributes

| Name           | Type                  | Description                              |
| -------------- | --------------------- | ---------------------------------------- |
| `mus`          | `torch.Tensor (K, D)` | Unit mean directions.                    |
| `kappas`       | `torch.Tensor (K,)`   | Concentration parameters ($\kappa_k>0$). |
| `logpi`        | `torch.Tensor (K,)`   | Logits of mixture weights.               |
| `n_iter_`      | `int`                 | Number of iterations performed.          |
| `lower_bound_` | `float`               | Final (approximate) log-likelihood.      |
| `_fitted`      | `bool`                | Whether the model has been trained.      |

---

### Methods

| Method                                         | Description                                                                    |
| ---------------------------------------------- | ------------------------------------------------------------------------------ |
| `fit(X, *, chunk=None)`                        | Train mixture parameters via EM. If `chunk>0`, enables streaming (CPU→GPU).    |
| `predict_proba(X, *, chunk=None)`              | Compute soft assignments $`\gamma_{ik}`$ (rows sum to 1).                      |
| `predict(X, *, chunk=None)`                    | Hard cluster assignment via `argmax`.                                          |
| `loglik(X, *, chunk=None, average=False)`      | Evaluate total or per-sample log-likelihood.                                   |
| `num_params()`                                 | Return total parameter count for BIC ($p = Kd + (K-1)$).                       |
| `bic(X, *, chunk=None)`                        | Compute $`\mathrm{BIC} = -2\log L + p\log N`$.                                 |
| `sample(n)`                                    | Generate samples from fitted mixture (Wood’s method + Householder reflection). |
| `save(path)` / `load(path, map_location=None)` | Save/load model state (like `state_dict`, including RNG).                      |

---

## Functions

### `elbow_vmf`

```python
from chemomae.clustering.vmf_mixture import elbow_vmf

k_list, scores, optimal_k, elbow_idx, kappa = elbow_vmf(
    VMFMixture, X, device="cuda", k_max=30, chunk=8192,
    criterion="bic",   # or "nll"
    random_state=42, verbose=True
)
```

* `criterion="bic"` → use **BIC** (lower = better).
* `criterion="nll"` → use **mean NLL** (= − mean log-lik; lower = better).
* Converts to a decreasing series (−score) and applies `find_elbow_curvature`.

**Returns:**
`k_list`, `scores`, `optimal_k`, `elbow_idx`, `kappa` (curvature array).

---

### `plot_elbow_vmf`

```python
from chemomae.clustering.vmf_mixture import plot_elbow_vmf
plot_elbow_vmf(k_list, scores, optimal_k, elbow_idx, criterion="bic")
```

Plots the score curve with an annotated elbow.
y-axis label automatically switches between **BIC** and **Mean NLL**.
(Use `plt.show()` or `plt.savefig(...)` externally.)

---

## Usage Examples

### Fit and infer

```python
X = torch.randn(10000, 64, device="cuda")
vmf = VMFMixture(n_components=32, device="cuda", random_state=0)
vmf.fit(X, chunk=1000000)
labels = vmf.predict(X, chunk=1000000)
resp = vmf.predict_proba(X, chunk=1000000)
```

### Model selection (BIC elbow)

```python
ks, scores, K, idx, curv = elbow_vmf(
    VMFMixture, X, device="cuda", k_max=30, chunk=1000000, criterion="bic"
)
print("Optimal K:", K)
```

### Save & load

```python
vmf.save("vmf_k32.pt")
vmf2 = VMFMixture.load("vmf_k32.pt", map_location="cuda")
assert torch.allclose(vmf.mus, vmf2.mus, atol=1e-6)
```

---

## Design Notes & Tips

* **Normalization:** Inputs are automatically L2-normalized (safe to pre-normalize).
* **Bessel approximations:** Stable GPU implementations of $`\log I_\nu(\kappa)`$ and $`I_{\nu+1}(\kappa)/I_\nu(\kappa)`$.
* **Closed-form κ update:** From resultant length $`\bar R_k`$; can switch to Newton refinement if needed.
* **Chunked E-step:** Enables massive-scale mixtures without GPU OOM.
* **Robustness:** Soft responsibilities down-weight outliers; if you see “spiky” clusters, cap $`\kappa_k`$ or add a uniform background.

---

## Minimal Checks

```python
X = torch.randn(200, 16)
vmf = VMFMixture(n_components=5, device="cpu").fit(X)
assert vmf.mus.shape == (5, 16) and vmf.kappas.min() > 0

labels = vmf.predict(X)
resp = vmf.predict_proba(X)
assert labels.shape == (200,) and resp.shape == (200, 5)

bic = vmf.bic(X)
ll = vmf.loglik(X, average=True)

vmf.save("tmp_vmf.pt")
vmf2 = VMFMixture.load("tmp_vmf.pt")
assert torch.allclose(vmf.mus, vmf2.mus, atol=1e-6)
```

---

## Version

* Introduced in `chemomae.clustering.vmf_mixture` — initial public draft.