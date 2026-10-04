# VMFMixture — von Mises–Fisher Mixture on the Unit Hypersphere

> Module: `chemomae.clustering.vmf_mixture`
> Purpose: Probabilistic clustering of L2-normalized features on $S^{d-1}$ via an EM algorithm.
> API reference for ChemoMAE v0.2.4 (release preparation).

**VMFMixture** fits a **von Mises–Fisher mixture model** to feature directions on
the unit hypersphere. Unlike hard nearest-direction assignments, it estimates
component weights and concentrations and returns soft responsibilities under
the fitted mixture. Responsibilities express membership under the assumed model;
they do not establish the semantic meaning of a component.

Use it with finite, nonzero feature directions when soft membership and
component-specific concentration are useful. No encoder or spatial information
is required. [CosineKMeans](cosine_kmeans.md) provides hard nearest-direction
assignments without mixture weights or concentrations.

## Quick start

This small CPU example fits a two-component mixture and checks probabilities
and the final fitting likelihood.

Pass `device` explicitly when composing a workflow: the constructor defaults to
`"cuda"` and does not infer the computation device from the input features.
Use `device="cpu"` on a CPU-only machine; the CUDA default does not fall back to
CPU. Trainer/Extractor have a different default: they follow their model's device.

```python
import math
import torch
from chemomae.clustering import VMFMixture

X = torch.tensor([
    [1.0, 0.0], [1.0, 0.2], [1.0, -0.2], [1.0, 0.1],
    [-1.0, 0.0], [-1.0, 0.2], [-1.0, -0.2], [-1.0, -0.1],
])
vmf = VMFMixture(n_components=2, device="cpu", random_state=42, max_iter=5)
vmf.fit(X)
responsibilities = vmf.predict_proba(X)
labels = vmf.predict(X)

assert responsibilities.shape == (8, 2) and labels.shape == (8,)
assert responsibilities.device.type == "cpu"
assert torch.allclose(responsibilities.sum(dim=1), torch.ones(8), atol=1e-6)
assert torch.equal(labels, responsibilities.argmax(dim=1))
assert math.isfinite(vmf.lower_bound_) and vmf.lower_bound_ == vmf.loglik(X)
```

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
* **Special functions:** CPU float64 scaled Bessel evaluation with a convergent-series fallback.
* **Chunking:** `chunk` bounds E-step blocks on CPU or CUDA. Keep input on CPU to stream blocks to CUDA.
* **Normalization:** finite, nonzero inputs are L2-normalized row-wise internally. Empty data, zero rows, and dimensions below 2 are rejected.

---

## API

### Class: `VMFMixture`

```python
from chemomae.clustering.vmf_mixture import VMFMixture

configured_vmf = VMFMixture(
    n_components=8,     # K
    d=None,             # inferred on first fit(X) if None
    device="cpu",       # the constructor default is "cuda"
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
| `n_iter_`      | `int`                 | Number of completed M-step updates; zero before fit. |
| `lower_bound_` | `float`               | Total log-likelihood of the final parameters on the fitting data; matches `loglik(X, chunk=...)` with the same chunk. |
| `converged_`   | `bool`                | Whether a nonnegative improvement met the tolerance. |
| `stop_reason_` | `str or None`         | `"tol"`, `"likelihood_decreased"`, `"max_iter"`; `None` before fit or after legacy load. |

Parameter tensors are empty before fitting. `lower_bound_` starts at negative
infinity and `converged_` starts as `False`.

---

### Methods

| Method                                         | Description                                                                    |
| ---------------------------------------------- | ------------------------------------------------------------------------------ |
| `fit(X, *, chunk=None)`                        | Fit mixture parameters via EM and return `self`; a positive chunk bounds E-step blocks. |
| `predict_proba(X, *, chunk=None)`              | Compute soft assignments $\gamma_{ik}$ (rows sum to 1).                      |
| `predict(X, *, chunk=None)`                    | Hard cluster assignment via `argmax`.                                          |
| `loglik(X, *, chunk=None, average=False)`      | Evaluate total or per-sample log-likelihood.                                   |
| `num_params()`                                 | Return total parameter count for BIC ($p = Kd + (K-1)$).                       |
| `bic(X, *, chunk=None)`                        | Compute $\mathrm{BIC} = -2\log L + p\log N$.                                 |
| `save(path)` / `load(path, map_location=None)` | Save/load model state including CPU RNG; explicit `map_location` sets the restored computation device. |

`X` is a Torch tensor of shape `(N, D)` with `N > 0` and `D >= 2`. NumPy inputs
must be converted by the caller. Each row must remain finite and nonzero after
conversion to the model dtype. `d`, once inferred or provided, remains fixed.
Prediction and likelihood methods require a fitted model. A chunk must be a
positive integer or `None`; it does not change the output shape.

`predict_proba` returns `(N, K)` on the model device in its configured dtype.
`predict` returns `(N,)` integer labels on that device. Both allocate the full
responsibility matrix, even with chunking. `loglik` and `bic` return Python
floats; `num_params` returns a Python integer. `load` is a class method returning
a new instance. The internal `_logC` cache and `_fitted` flag are implementation
details, not additional public configuration.

A label-only prediction path that avoids the full responsibility matrix is under
consideration as a low-priority improvement. The current need is limited, and it
is not a v0.2.4 release requirement. For the existing caller-side slicing option,
see [memory planning](../tutorials/first_experiment.md#estimate-memory-by-operation).

---

## Functions

### `vmf_logC` and `vmf_bessel_ratio`

```python
import torch
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

The sweep passes lower-is-better BIC/NLL scores directly to
[`find_elbow_curvature`](ops.md) and returns an interior curvature
candidate. The returned raw `scores` are unchanged. In v0.2.3, a score-direction
mismatch could flatten a decreasing curve and return `K=2` with zero curvature;
v0.2.4 removes that sign reversal. Fixed-K `fit`, `loglik`, and `bic` are unchanged.

This example continues the quick start. The sweep fits models for `K=1..k_max`
with `tol=1e-4` and `max_iter=200`. Defaults are `device="cuda"`, `k_max=50`,
`chunk=None`, `verbose=True`, `random_state=42`, and `criterion="bic"`. At least
three K values are needed by the curvature helper; a noninteger `k_max` or one
below 3 raises `ValueError` before transferring data or fitting any models.

```python
from chemomae.clustering.vmf_mixture import elbow_vmf

k_list, scores, optimal_k, elbow_idx, kappa = elbow_vmf(
    VMFMixture, X, device="cpu", k_max=4,
    criterion="bic",   # or "nll"
    random_state=42, verbose=True
)
```

* `criterion="bic"` → use **BIC** (lower = better).
* `criterion="nll"` → use **mean NLL** (= − mean log-lik; lower = better).
* Both criteria pass the original score curve to `find_elbow_curvature`, which
  replaces increases with the cumulative minimum before computing curvature.
  This can conceal nonmonotonic behavior, including a BIC increase after its
  minimum. Inspect the raw curve alongside the selected candidate.
* A flat curve, including one made flat by the cumulative minimum, selects the
  first interior point (`K=2`) with zero curvature. This is a tie-breaking
  convention and does not supply evidence for a preferred K.
* Minimum-score K is reported separately when `verbose=True`; it can differ from the returned elbow K.

**Returns:**
`k_list`, `scores`, `optimal_k`, `elbow_idx`, `kappa` (curvature at the selected index).

Calling `fit` does not select K. Fixed-K comparisons should construct a model with
the prescribed `n_components` directly.

The return name `optimal_k` does not mean minimum BIC or NLL. The score criteria
describe fit under the mixture model; they do not establish the semantic meaning
of components or a universally correct K. Inspect raw scores and convergence
diagnostics under the comparison protocol chosen for the application.

---

### `plot_elbow_vmf`

```python
from chemomae.clustering import plot_elbow_vmf
plot_elbow_vmf(k_list, scores, optimal_k, elbow_idx, criterion="bic")
```

Plots the raw score curve with the supplied candidate annotated. It does not
validate that candidate or establish that a meaningful elbow exists.
y-axis label automatically switches between **BIC** and **Mean NLL**.
(Use `plt.show()` or `plt.savefig(...)` externally.)

---

## Additional examples

These snippets continue the quick start. Save/load writes to the chosen path.

### CUDA E-step chunks

```python
if torch.cuda.is_available():
    vmf_gpu = VMFMixture(n_components=2, device="cuda", max_iter=5)
    vmf_gpu.fit(X, chunk=4)  # X stays on CPU
    gpu_labels = vmf_gpu.predict(X, chunk=4)  # still allocates all N-by-K responsibilities
```

### Save & load

```python
vmf.save("vmf.pt")
vmf2 = VMFMixture.load("vmf.pt", map_location="cpu")
assert vmf2.device == torch.device("cpu")
assert torch.allclose(vmf.mus.cpu(), vmf2.mus, atol=1e-6)
```

---

## Design Notes & Tips

* **Normalization and invalid data:** Inputs are converted to `dtype`, scaled by the largest absolute coordinate, then normalized. This avoids norm overflow or clipping a small nonzero norm. Nonfinite values and rows that are zero after conversion raise `ValueError` in fit and inference; rows are never silently dropped. Validation follows the chosen chunks.
* **Normalizers:** [SciPy `ive`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html) removes the exponential growth of the Bessel function. For concentration <= 1 or scaled-Bessel underflow, a log-space sum of the [defining positive series](https://dlmf.nist.gov/10.25.E2) avoids underflow. A decreasing-term tail bound controls truncation at float64 epsilon; failure to converge within 10,000 terms raises `FloatingPointError`.
* **Precision and cost:** Each normalizer refresh transfers K concentrations to CPU and K float64 results back to the computation device. Outside autocast, log densities, softmax and log-likelihood reductions use float64; dot products and sufficient statistics use the configured dtype. Calls do not disable ambient autocast, so run outside it to retain this matrix-arithmetic precision. The M-step evaluates ratios from those statistics in float64. Float32 dot products can still lose angular detail at extreme concentration; use `dtype=torch.float64` when that matters. This introduces CPU synchronization and additional float64 device work.
* **Concentration update:** Positive-mass resultant lengths are clamped to $[0,1-10^{-6}]$, the denominator includes $10^{-8}$, and concentrations are floored at `kappa_min`. Concentrations use the stated closed-form approximation without a Newton solve. The ratio helper is a diagnostic and is not called by fit.
* **Degenerate components:** With positive mass and an exactly zero resultant, retain the previous unit direction and set concentration to `kappa_min`. With exactly zero mass, retain both direction and concentration. Mixture weights retain the existing `1e-20` floor before log normalization. Tiny positive masses are used directly; there is no deletion threshold or automatic reinitialization. Since `kappa_min > 0`, a zero resultant is represented by a near-uniform component, not an exactly uniform component.
* **Stopping:** An initial E-step is followed by M-step/E-step pairs. `lower_bound_` always describes the final parameters. A negative likelihood change stops with `stop_reason_="likelihood_decreased"`, `converged_=False`; the updated parameters are retained. A nonnegative change stops with `"tol"` when relative improvement is below `tol` or absolute improvement is below `1e-6`. Otherwise the model stops at `"max_iter"` with `converged_=False`. Approximate concentration updates do not guarantee monotonic likelihood. Nonfinite sufficient statistics or updated likelihood raise `FloatingPointError`.
* **Chunking and initialization:** Chunking bounds intermediate E-step allocations, but both `predict_proba` and `predict` allocate the full N-by-K responsibility matrix on the computation device. Float64 log-density blocks add to memory use. If N exceeds `chunk`, initialization randomly samples up to `min(N, max(10000, 50*K))` rows, using a CPU permutation of N indices. This initialization subset can exceed the chunk size. Chunked and unchunked fits can therefore start differently; compare E-step paths with fixed parameters.
* **Reproducibility:** CPU RNG state is saved and restored, and k-means++ probability vectors are transferred to CPU for draws. Same-device repeated initialization with the same seed is tested. Device arithmetic can change probabilities, so the CPU generator policy does not promise identical fits across CPU and CUDA.
* **Persistence:** An explicit string or `torch.device` `map_location` overrides the saved device. With `None`, the saved device is retained. RNG state is always restored on CPU. `_logC` is recomputed with current numerics. Saved files include `numerics_version=2`, `converged_` and `stop_reason_`. Older files with valid parameters remain loadable, but old likelihoods are invalidated to `NaN` and convergence metadata is unknown; call `loglik` with the original fitting data to reevaluate. Invalid saved directions, concentrations or log weights raise `ValueError`; zero directions from a legacy fit require refitting. Updated numerics can change predictions and BIC for old parameters. Loading restores fitted parameters and the initialization RNG stream; a subsequent `fit` starts a new fit, not an interrupted EM run.

---

## Minimal Checks

The regression suite uses synthetic CPU data and optional small CUDA checks.
Run from the repository root with the test environment activated and existing
development dependencies installed. For v0.2.4 release preparation, use an
editable installation of the current checkout so that package metadata and
source match:

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
| Elbow score direction | CPU score stubs for BIC and mean NLL, positive and negative score ranges, known quadratic curves with independently specified K and curvature, short-curve gradients, raw-score preservation, cumulative-minimum/flat behavior, and invalid sweep rejection before fitting. |

These tolerances define regression acceptance at the listed points, not a global
error guarantee. Float32 parameters/statistics and approximate concentration updates
remain sources of numerical error.

---

See [the changelog](../../CHANGELOG.md) for release history.
