# VMFMixture — von Mises–Fisher Mixture on the Unit Hypersphere

> Module: `chemomae.clustering.vmf_mixture` <br>
> Purpose: probabilistic clustering of L2-normalized features on ($`S^{d-1}`$) with an EM algorithm; model selection via `elbow_vmf`; visualization via `plot_elbow_vmf`.

---

## Overview

* **Likelihood (unit vectors ($`|x_i|=1`$))**
```math
  \max_{{\pi_k,\mu_k,\kappa_k}}\ \sum_{i=1}^N \log\Big(\sum_{k=1}^K \pi_k, C_d(\kappa_k), e^{\kappa_k,\mu_k^\top x_i}\Big),
  \quad C_d(\kappa)=\frac{\kappa^\nu}{(2\pi)^{\nu+1} I_\nu(\kappa)},\ \nu=\tfrac{d}{2}-1
```
  ($`\mu_k`$) are **unit directions**, ($`\kappa_k>0`$) are concentrations, ($`\pi_k`$) are mixture weights.

* **E-step (responsibilities)**
```math
\gamma_{ik}\propto \pi_k,C_d(\kappa_k),e^{\kappa_k,\mu_k^\top x_i}), \text{with} \sum_k\gamma_{ik}=1
```
* **M-step**
$`N_k=\sum_i\gamma_{ik}`$; <br>
$`\tilde\mu_k=\big(\sum_i \gamma_{ik} x_i\big)/N_k), \text{then L2-normalize to} (\mu_k);
`$ <br>
resultant length ($`\bar R_k=|\sum_i\gamma_{ik}x_i|/N_k`$) gives a fast **closed-form ($`\kappa`$) update**

```math
  \kappa_k \approx \frac{\bar R_k,(d-\bar R_k^2)}{1-\bar R_k^2};
  \qquad \pi_k=N_k/N.
```

* **Initialization**: cosine (hyperspherical) k-means++ seeding.

* **Special functions**: stable, torch-only approximations for ($`\log I_\nu(\kappa)`$) and ($`I_{\nu+1}/I_\nu`$) by blending small/large-($`\kappa`$) series—as `torch.special.iv` may be unavailable.

* **Scalability**: chunked E-step (`chunk`) streams large (N) from CPU→GPU.

* **Normalization**: inputs are internally L2-normalized row-wise.

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

**Parameters**

* `n_components: int` — number of components (K).
* `d: Optional[int]` — feature dim. If `None`, inferred from `X.size(1)` on `fit`.
* `device: str | torch.device` — where to compute.
* `random_state: int | None` — RNG seed.
* `tol: float` — EM stopping threshold (relative/absolute LL change).
* `max_iter: int` — EM iteration cap.
* `init: {"kmeans++","random"}` — direction initialization.

**Attributes**

* `mus: (K, D)` — unit mean directions.
* `kappas: (K,)` — concentrations ($`\kappa_k>0`$).
* `logpi: (K,)` — logits for mixture weights.
* `n_iter_: int` — iterations run.
* `lower_bound_: float` — final objective (approx log-lik).
* `_fitted: bool` — fitted flag.

---

### Methods

* `fit(X, *, chunk=None) -> self`
  Train via EM. Set `chunk>0` to stream large datasets.

* `predict_proba(X, *, chunk=None) -> Tensor (N, K)`
  Responsibilities ($`\gamma_{ik}`$) (rows sum to 1).

* `predict(X, *, chunk=None) -> LongTensor (N,)`
  Hard assignments via `argmax` over responsibilities.

* `loglik(X, *, chunk=None, average=False) -> float`
  Total or per-sample log-likelihood under current parameters.

* `num_params() -> int`
  DoF for BIC: ($`p = K,d + (K-1)`$).

* `bic(X, *, chunk=None) -> float`
  ( $`\mathrm{BIC} = -2\log L + p\log N`$ ) (lower is better).

* `sample(n) -> Tensor (n, D)`
  Draw samples from the learned mixture (Wood’s method + Householder).

* `save(path)` / `load(path, map_location=None) -> VMFMixture`
  Lightweight `state_dict`-style persistence (incl. RNG state).

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

* `criterion="bic"`: use **BIC** (lower is better).
* `criterion="nll"`: use **mean NLL** (= − mean log-lik; lower is better).
* Internally converts to a **decreasing** series (`-score`) and calls `find_elbow_curvature` to pick the elbow.

**Returns**:
`k_list`, `scores`, `optimal_k`, `elbow_idx`, `kappa` (curvature magnitude).

### `plot_elbow_vmf`

```python
from chemomae.clustering.ops import plot_elbow_vmf
plot_elbow_vmf(k_list, scores, optimal_k, elbow_idx, criterion="bic")
```

Plots the curve and highlights the chosen elbow; ylabel switches between **BIC** and **Mean NLL**.
Call `plt.savefig(...)`/`plt.show()` on the caller side.

---

## Usage Examples

### Fit & infer

```python
X = torch.randn(10000, 64, device="cuda")
vmf = VMFMixture(n_components=32, d=None, device="cuda", random_state=0)
vmf.fit(X, chunk=1000000)
labels = vmf.predict(X, chunk=1000000)
resp   = vmf.predict_proba(X, chunk=1000000)
```

### Model selection (BIC elbow)

```python
ks, scores, K, idx, curv = elbow_vmf(
    VMFMixture, X, device="cuda", k_max=30, chunk=1000000, criterion="bic"
)
print("Elbow K:", K)
```

### Save & load

```python
vmf.save("vmf_k32.pt")
vmf2 = VMFMixture.load("vmf_k32.pt", map_location="cuda")
assert torch.allclose(vmf.mus, vmf2.mus, atol=1e-6)
```

---

## Design Notes & Tips

* **Normalization**: Inputs are L2-normalized internally; you can pre-normalize too.
* **Bessel approximations**: ($`\log I_\nu`$) and ($`I_{\nu+1}/I_\nu`$) use blended small/large-($`\kappa`$) expansions for stability on GPU.
* **($`\kappa`$) update**: closed-form from resultant length; swap to Newton solves with $`I_{\nu+1}/I_\nu`$ if you need higher accuracy.
* **Chunked E-step**: set `chunk` to fit very large (N) with limited VRAM.
* **Robustness**: soft responsibilities usually down-weight outliers; if you observe “spiky outlier components,” consider ($`\kappa`$) caps/priors or adding a uniform background component.

---

## Minimal Checks

```python
# basic fit
X = torch.randn(200, 16)
vmf = VMFMixture(n_components=5, d=None, device="cpu").fit(X)
assert vmf.mus.shape == (5, 16) and vmf.kappas.min() > 0

# predict / proba
labels = vmf.predict(X); resp = vmf.predict_proba(X)
assert labels.shape == (200,) and resp.shape == (200, 5)

# BIC / loglik
bic = vmf.bic(X); ll = vmf.loglik(X, average=True)

# persistence
vmf.save("tmp_vmf.pt"); vmf2 = VMFMixture.load("tmp_vmf.pt")
assert torch.allclose(vmf.mus, vmf2.mus, atol=1e-6)
```

---

## Version

* Public draft: `chemomae.clustering.vmf_mixture` **0.1.0**.
