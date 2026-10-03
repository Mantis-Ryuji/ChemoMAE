# VMFMixture — von Mises–Fisher Mixture on the Unit Hypersphere

> Module: `chemomae.clustering.vmf_mixture`
> Purpose: Probabilistic clustering of L2-normalized features on $S^{d-1}$ via an EM algorithm.
> API reference for ChemoMAE v0.2.3.

**VMFMixture** fits a **von Mises–Fisher mixture model** to feature directions on
the unit hypersphere. Unlike hard nearest-direction assignments, it estimates
component weights and concentrations and returns soft responsibilities under
the fitted mixture. Responsibilities express membership under that model;
they are not probabilities of known chemical states.

The accompanying ChemoMAE study uses [CosineKMeans](cosine_kmeans.md) for its main
analysis. `VMFMixture` is an additional library option for directional analyses
that call for a probabilistic mixture. It provides EM-based parameter estimation,
objective-curve exploration through `elbow_vmf`, and visualization through
`plot_elbow_vmf`.

---

## Overview

* **Likelihood (unit vectors, $\lVert x_i\rVert_2=1$)**

$$
\max_{\pi_k,\mu_k,\kappa_k}\ \sum_{i=1}^N \log\Big(\sum_{k=1}^K
\pi_k\thinspace C_d(\kappa_k)\thinspace e^{\kappa_k\thinspace\mu_k^\top x_i}\Big),\quad
C_d(\kappa)=\frac{\kappa^\nu}{(2\pi)^{\nu+1}I_\nu(\kappa)},\ 
\nu=\tfrac{d}{2}-1
$$

where $\mu_k$ are **unit directions**, $\kappa_k>0$ are **concentrations**, and $\pi_k$ are **mixture weights**.

* **E-step (responsibilities)**

$$
\gamma_{ik}\propto \pi_k\thinspace C_d(\kappa_k)\thinspace e^{\kappa_k\thinspace\mu_k^\top x_i},
\qquad \sum_k \gamma_{ik}=1
$$

* **M-step**

Let $N_k=\sum_i\gamma_{ik}$. For a nonzero resultant:

$$
\tilde\mu_k=\frac{\sum_i\gamma_{ik}x_i}{N_k}, \quad
\mu_k = \frac{\tilde\mu_k}{\lVert\tilde\mu_k\rVert_2}
$$

The resultant length $\bar R_k=\lVert\sum_i\gamma_{ik}x_i\rVert_2/N_k$ gives a **closed-form approximation** for $\kappa_k$:

$$
\kappa_k \approx \frac{\bar R_k\thinspace(d-\bar R_k^2)}{1-\bar R_k^2},\qquad
\pi_k = N_k / N.
$$

* **Initialization:** cosine (hyperspherical) k-means++ seeding; all draws use a CPU Generator, including CUDA fits.
* **Special functions:** CPU float64 scaled Bessel evaluation with a convergent-series fallback; no interpolation at concentrations 2 or 12.
* **Scalability:** chunked E-step (`chunk`) streams data CPU→GPU.
* **Normalization:** finite, nonzero inputs are L2-normalized row-wise internally. Empty data, zero rows, and dimensions below 2 are rejected.

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
| `d`            | `Optional[int]`            | `None`       | Feature dimension >= 2; inferred on first `fit`. |
| `device`       | `str or torch.device`      | `"cuda"`     | Target computation device.                       |
| `random_state` | `int or None`              | `42`         | CPU RNG seed; does not guarantee identical CPU/CUDA arithmetic or fits. |
| `tol`          | `float`                    | `1e-4`       | Finite, nonnegative relative improvement tolerance. |
| `max_iter`     | `int`                      | `200`        | Positive maximum number of M-step updates.       |
| `init`         | `{ "kmeans++", "random" }` | `"kmeans++"` | Initialization strategy.                         |
| `kappa_init`   | `float`                    | `10.0`       | Finite initial concentration, at least `kappa_min`, representable in `dtype`. |
| `kappa_min`    | `float`                    | `1e-6`       | Finite positive concentration floor, representable as positive in `dtype`. |
| `dtype`        | `torch.dtype`              | `torch.float32` | `float32` or `float64` for parameters, directions, responsibilities and sufficient statistics. |

#### Attributes

| Name           | Type                  | Description                              |
| -------------- | --------------------- | ---------------------------------------- |
| `mus`          | `torch.Tensor (K, D)` | Unit mean directions.                    |
| `kappas`       | `torch.Tensor (K,)`   | Concentration parameters ($\kappa_k>0$). |
| `logpi`        | `torch.Tensor (K,)`   | Logits of mixture weights.               |
| `n_iter_`      | `int`                 | Number of iterations performed.          |
| `_logC`        | `torch.Tensor (K,)`   | Float64 log-normalizer cache on the computation device. |
| `lower_bound_` | `float`               | Total log-likelihood of the final parameters on the fitting data; matches `loglik(X, chunk=...)` with the same chunk. |
| `converged_`   | `bool`                | Whether a nonnegative improvement met the tolerance. |
| `stop_reason_` | `str or None`         | `"tol"`, `"likelihood_decreased"`, `"max_iter"`; `None` before fit or after legacy load. |
| `_fitted`      | `bool`                | Whether the model has been trained.      |

---

### Methods

| Method                                         | Description                                                                    |
| ---------------------------------------------- | ------------------------------------------------------------------------------ |
| `fit(X, *, chunk=None)`                        | Train mixture parameters via EM. If `chunk>0`, enables streaming (CPU→GPU).    |
| `predict_proba(X, *, chunk=None)`              | Compute soft assignments $\gamma_{ik}$ (rows sum to 1).                      |
| `predict(X, *, chunk=None)`                    | Hard cluster assignment via `argmax`.                                          |
| `loglik(X, *, chunk=None, average=False)`      | Evaluate total or per-sample log-likelihood.                                   |
| `num_params()`                                 | Return total parameter count for BIC ($p = Kd + (K-1)$).                       |
| `bic(X, *, chunk=None)`                        | Compute $\mathrm{BIC} = -2\log L + p\log N$.                                 |
| `save(path)` / `load(path, map_location=None)` | Save/load model state including CPU RNG; explicit `map_location` sets the restored computation device. |

---

## Functions

### `vmf_logC` and `vmf_bessel_ratio`

```python
from chemomae.clustering.vmf_mixture import vmf_logC, vmf_bessel_ratio

kappa = torch.tensor([0.0, 2.0, 12.0, 1000.0], dtype=torch.float64)
logc = vmf_logC(16, kappa)
ratio = vmf_bessel_ratio(torch.tensor(7.0), kappa)
```

`vmf_logC(d, kappa)` requires integer `d >= 2`. `vmf_bessel_ratio(nu, k)` accepts
broadcastable finite orders `nu >= 0`. Concentrations must be real floating-point,
finite and nonnegative. Results retain the concentration tensor's dtype and device;
the ratio uses the broadcast shape. At zero, the exact limits are

$$
\log C_d(0)=\log\Gamma(d/2)-\log 2-\frac d2\log\pi,
\qquad \frac{I_{\nu+1}(0)}{I_\nu(0)}\equiv 0.
$$

The ratio at zero denotes its continuous limit. The helper does not floor a small
positive ratio. Its small-concentration expansion has the cubic term

$$
\frac{I_{\nu+1}(\kappa)}{I_\nu(\kappa)}
=\frac{\kappa}{a}\left(1-\frac{\kappa^2}{ab}+O(\kappa^4)\right),
\qquad a=2\nu+2,\quad b=2\nu+4.
$$

Both helpers evaluate on CPU in float64 using the existing SciPy dependency and
return detached tensors; they do not provide autograd. For float64 outputs, supply
float64 concentrations. The mixture always requests a float64 log normalizer.

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
* Both criteria pass the negated score curve to `find_elbow_curvature` and return its elbow K.
* Minimum-score K is reported separately when `verbose=True`; it can differ from the returned elbow K.

**Returns:**
`k_list`, `scores`, `optimal_k`, `elbow_idx`, `kappa` (curvature at the selected index).

Calling `fit` does not select K. Fixed-K comparisons should construct a model with
the prescribed `n_components` directly.

The return name `optimal_k` refers to the curvature-based elbow, not the minimum
BIC or NLL and not an identified number of chemical states. These criteria
describe fit under the mixture model; they do not evaluate spatial coherence or
chemical interpretation. Use the score curve as an exploratory diagnostic when
deciding how finely to partition a representation.

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
print(vmf.stop_reason_, vmf.converged_, vmf.lower_bound_)
```

### Exploring component count (elbow of the BIC curve)

```python
ks, scores, K, idx, curv = elbow_vmf(
    VMFMixture, X, device="cuda", k_max=30, chunk=1000000, criterion="bic"
)
print("Elbow candidate K:", K)
```

### Save & load

```python
vmf.save("vmf_k32.pt")
vmf2 = VMFMixture.load("vmf_k32.pt", map_location="cpu")
assert vmf2.device == torch.device("cpu")
assert torch.allclose(vmf.mus.cpu(), vmf2.mus, atol=1e-6)
```

---

## Design Notes & Tips

* **Normalization and invalid data:** Inputs are converted to `dtype`, scaled by the largest absolute coordinate, then normalized. This avoids norm overflow or clipping a small nonzero norm. Nonfinite values and rows that are zero after conversion raise `ValueError` in fit and inference; rows are never silently dropped. Validation follows the chosen chunks.
* **Normalizers:** [SciPy `ive`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html) removes the exponential growth of the Bessel function. For concentration <= 1 or scaled-Bessel underflow, a log-space sum of the [defining positive series](https://dlmf.nist.gov/10.25.E2) avoids underflow. A decreasing-term tail bound controls truncation at float64 epsilon; failure to converge within 10,000 terms raises `FloatingPointError`.
* **Precision and cost:** Each normalizer refresh transfers K concentrations to CPU and K float64 results back to the computation device. Log densities, softmax and log-likelihood reductions use float64; dot products and sufficient statistics use the configured dtype. The M-step evaluates ratios from those statistics in float64. Float32 dot products can still lose angular detail at extreme concentration; use `dtype=torch.float64` when that matters. This introduces CPU synchronization and additional float64 device work.
* **Concentration update:** The existing closed-form approximation remains. Positive-mass resultant lengths are clamped to $[0,1-10^{-6}]$, the denominator includes $10^{-8}$, and concentrations are floored at `kappa_min`. There is no new concentration cap or Newton solve. The ratio helper is a diagnostic and is not called by fit.
* **Degenerate components:** With positive mass and an exactly zero resultant, retain the previous unit direction and set concentration to `kappa_min`. With exactly zero mass, retain both direction and concentration. Mixture weights retain the existing `1e-20` floor before log normalization. Tiny positive masses are used directly; there is no deletion threshold or automatic reinitialization. Since `kappa_min > 0`, a zero resultant is represented by a near-uniform component, not an exactly uniform component.
* **Stopping:** An initial E-step is followed by M-step/E-step pairs. `lower_bound_` always describes the final parameters. A negative likelihood change stops with `stop_reason_="likelihood_decreased"`, `converged_=False`; the updated parameters are retained. A nonnegative change stops with `"tol"` when relative improvement is below `tol` or absolute improvement is below `1e-6`. Otherwise the model stops at `"max_iter"` with `converged_=False`. Approximate concentration updates do not guarantee monotonic likelihood. Nonfinite sufficient statistics or updated likelihood raise `FloatingPointError`.
* **Chunking and initialization:** Chunking bounds intermediate E-step allocations, but `predict_proba` still allocates the full N-by-K result on the computation device. Float64 log-density blocks add to memory use. If N exceeds `chunk`, initialization randomly samples up to `min(N, max(10000, 50*K))` rows, using a CPU permutation of N indices. Chunked and unchunked fits can therefore start differently; compare E-step paths with fixed parameters.
* **Reproducibility:** CPU RNG state is saved and restored, and k-means++ probability vectors are transferred to CPU for draws. Same-device repeated initialization with the same seed is tested. Device arithmetic can change probabilities, so the CPU generator policy does not promise identical fits across CPU and CUDA.
* **Persistence:** An explicit string or `torch.device` `map_location` overrides the saved device. With `None`, the saved device is retained. RNG state is always restored on CPU. `_logC` is recomputed with current numerics. v0.2.2 stores the additive `numerics_version=2`, `converged_` and `stop_reason_` fields. Older checkpoints with valid parameters remain loadable, but old likelihoods are invalidated to `NaN` and convergence metadata is unknown; call `loglik` with the original fitting data to reevaluate. Invalid saved directions, concentrations or log weights raise `ValueError`; zero directions from a legacy fit require refitting. Corrected numerics can change predictions and BIC for old parameters, and existing fits are not retroactively corrected.

---

## Minimal Checks

The regression suite uses synthetic CPU data and optional small CUDA checks. It
does not run research training or evaluate real datasets. Run from the repository
root with the test environment activated and existing development dependencies
installed. Use an editable installation of the `v0.2.3` Git tag so that package
metadata and source match:

```powershell
python -m pip install --no-deps -e .
python -m pip show chemomae
python -m pytest -q -ra tests/clustering/test_vmf_mixture.py tests/test_import.py -k "not cuda"
```

For a separate short CUDA initialization/fit/restore check on an available GPU:

```powershell
python -m pytest -q -ra tests/clustering/test_vmf_mixture.py -k cuda
```

The full clustering regression command is `python -m pytest -q -ra tests/clustering`.
The CUDA cases skip when CUDA is unavailable. Test definitions cover:

| Check | Reference or acceptance condition |
| --- | --- |
| `logC` and Bessel ratio | Independent 80-digit Decimal series for d=2, 3, 16, 256; concentration 0 through 1000, including neighborhoods of 1, 2 and 12 and scaled-Bessel underflow. Float64 `logC`: `rtol=2e-12`, `atol=2e-10`; ratio: `rtol=2e-12`, `atol=1e-14`. Float32 `logC`: `rtol=2e-7`, `atol=3e-5`; ratio: `rtol=2e-7`, `atol=1e-14`. |
| High concentration | Finite scaled SciPy references for d=16, 256 at 1e4, 1e6 and 2e8, plus the independent closed form in d=3. |
| Density and responsibilities | Numerical sphere integration in d=3, 16, 256 at concentrations 0, 2, 12 and 100 within `2e-9` of unit mass; unequal-concentration responsibilities against Decimal normalizers; permutation and cosine-assignment invariants. |
| Concentration approximation | Reference Bessel-ratio residual <= `5e-3` for d=16, 256 and resultant lengths 0.01, 0.1, 0.5, 0.9, 0.99, 0.999. This checks the retained approximation, not an exact M-step solution. |
| EM behavior | Final likelihood after one or more updates, separate stopping reasons, antipodal inputs, empty components, and tiny positive component mass. |
| Chunk and persistence | Fixed-parameter sufficient statistics, probabilities and likelihood; parameter/RNG round trips; CPU restoration of CUDA device metadata; legacy cache invalidation; optional actual CUDA round trip. |

These tolerances define regression acceptance at the listed points, not a global
error guarantee. Float32 parameters/statistics and approximate concentration updates
remain sources of numerical error.

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

* **v0.2.2:** fixes CUDA k-means++ generator placement, replaces the piecewise Bessel approximation and incorrect small-ratio coefficient, records the final likelihood and stopping reason, respects CPU load targets, handles degenerate components without invalid directions, and adds numerical/behavioral regression tests. No new dependencies or model-selection policy are introduced.
