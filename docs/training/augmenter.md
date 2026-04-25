# Spectral Augmentation on the Hypersphere — Fractional Shift and Tangent Gaussian Noise

> Module: `chemomae.training.augmenter`

This document describes **`SpectraAugmenter`** , a hypersphere-aware spectral augmentation module for **SNV-normalized spectra** in ChemoMAE training.

The implementation provides two lightweight training-time augmentations:

- **Fractional shift**
- **Tangent Gaussian noise**

Both transformations are designed for spectra that have already been standardized by SNV. After each augmentation, the spectrum can be re-centered and re-normalized so that the augmented sample remains compatible with the SNV-induced geometry.

The intended role of this module is **auxiliary regularization** for masked reconstruction. It is not designed as a strong multi-view augmentation pipeline for contrastive learning.

---

## Overview

Consider a batch of SNV-normalized spectra

```math
X = \{\mathbf{x}_1, \dots, \mathbf{x}_B\} \subset \mathbb{R}^L
```

where each spectrum satisfies approximately

```math
\frac{1}{L}\sum_{\ell=1}^{L} x_{i,\ell} \approx 0 \quad\text{and}\quad\lVert \mathbf{x}_i \rVert_2 \approx r
```

for some nearly constant radius $`r > 0`$ .

Under exact SNV with population standard deviation, each spectrum lies on the intersection of:

1. the zero-mean hyperplane, and
2. a fixed-radius hypersphere.

That is,

```math
\mathbf{x}_i \in \mathcal{M}=\left\{\mathbf{x} \in \mathbb{R}^L\;\middle|\;\mathbf{1}^{\top}\mathbf{x}=0,\;\lVert \mathbf{x} \rVert_2=r\right\}.
```

A naive Euclidean perturbation,

```math
\mathbf{x}_i' = \mathbf{x}_i + \boldsymbol{\varepsilon}_i,
```

generally violates this structure because it may change both the sample mean and the L2 norm.

`SpectraAugmenter` instead applies weak spectral perturbations and then, when enabled, projects the result back to the SNV-compatible space by:

1. re-centering each spectrum to mean zero,
2. re-normalizing it to the original per-sample L2 norm.

---

## Design Goal

The main learning signal in ChemoMAE is **masked reconstruction** .

The model receives a partially visible spectrum and learns to reconstruct masked wavelength regions. Augmentation is used only as a secondary regularizer:

```math
A(\mathbf{x})_{\Omega_v}\longrightarrow\mathbf{x}_{\Omega_m},
```

where:

* $`A`$ is the augmentation operator,
* $`\Omega_v`$ is the visible wavelength region,
* $`\Omega_m`$ is the masked wavelength region.

Thus, the module should perturb spectra enough to improve robustness, but not so strongly that it destroys chemically or physically meaningful degradation-related variation.

For this reason, the recommended augmentation set is intentionally compact:

```math
\text{fractional shift} + \text{tangent Gaussian noise}.
```

Structured low-frequency augmentations such as tilt or quadratic baseline are intentionally excluded from this version because they may interfere with degradation-related low-frequency spectral changes.

---

## Strength Control via Angle

This implementation controls augmentation strength by **geodesic angle**, not by cosine similarity.

For an input spectrum $`\mathbf{x}`$ and augmented spectrum $`\mathbf{x}_{\mathrm{aug}}`$ , the angle is defined through

```math
\cos(\theta)=\frac{\mathbf{x}^{\top}\mathbf{x}_{\mathrm{aug}}}{\lVert \mathbf{x} \rVert_2 \lVert \mathbf{x}_{\mathrm{aug}} \rVert_2}.
```

The API specifies angle ranges in **degrees** :

```python
noise_angle_deg_range: tuple[float, float]
shift_angle_deg_range: tuple[float, float]
```

Internally, sampled angles are converted to radians.

Angle-based control is often easier to reason about than cosine-based control because the perturbation magnitude is specified directly as a movement angle on the sphere.

---

## SNV-Compatible Reprojection

After each augmentation, the module can apply re-centering:

```math
\mathbf{x}_{\mathrm{cand}}\leftarrow\mathbf{x}_{\mathrm{cand}}-\frac{1}{L}\left(\mathbf{1}^{\top}\mathbf{x}_{\mathrm{cand}}\right)\mathbf{1},
```

followed by re-normalization:

```math
\mathbf{x}_{\mathrm{cand}}\leftarrow\lVert \mathbf{x} \rVert_2\frac{\mathbf{x}_{\mathrm{cand}}}{\lVert \mathbf{x}_{\mathrm{cand}} \rVert_2}.
```

Here:

* $`\mathbf{x}`$ is the input to the current augmentation operation,
* $`\mathbf{x}_{\mathrm{cand}}`$ is the intermediate augmented candidate.

This operation preserves the original per-sample norm while enforcing zero mean.

The corresponding configuration flags are:

```python
recenter_after_each_op: bool = True
renorm_to_input_norm: bool = True
```

When both are enabled, the output after each augmentation remains compatible with the SNV geometry.

---

## Tangent-Space Construction

For tangent Gaussian noise, the implementation constructs a perturbation direction in the tangent space of the sphere.

At a spectrum $`\mathbf{x}`$ , the tangent space of the sphere is

```math
T_{\mathbf{x}}\mathbb{S}^{L-1}(r)=\left\{\mathbf{v} \in \mathbb{R}^L\;\middle|\;\mathbf{v}^{\top}\mathbf{x}=0\right\}.
```

Given an arbitrary direction $`d`$ , the projection onto this tangent space is

```math
\mathbf{v}=\mathbf{d}-\frac{\mathbf{d}^{\top}\mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}.
```

In this implementation, the random direction is first centered before tangent projection. This makes the perturbation more compatible with the zero-mean SNV hyperplane.

---

## Geodesic Rotation

Given a unit tangent direction $`\mathbf{u}`$ , the spectrum is rotated along the sphere by angle $`\theta`$ :

```math
\mathbf{x}_{\mathrm{aug}}=r\left(\cos(\theta)\frac{\mathbf{x}}{r}+\sin(\theta)\mathbf{u}\right),\qquad r = \lVert \mathbf{x} \rVert_2.
```

This operation preserves the L2 norm before re-centering. Since re-centering may slightly change the norm, the implementation can re-normalize the result to the input norm afterward.

---

## Augmentation 1 — Fractional Shift

### Idea

Fractional shift models small wavelength-axis misalignment.

This is useful for spectra because small peak-position or wavelength-grid deviations can occur due to measurement conditions, interpolation, calibration, or instrument-related variability.

Unlike `torch.roll`, fractional shift supports non-integer shifts and uses linear interpolation.

### Construction

For each selected spectrum $`\mathbf{x}`$, a shift amount is sampled:

```math
\delta \sim \mathcal{U}(\delta_{\min}, \delta_{\max}).
```

The shifted candidate $`\mathbf{x}_{\mathrm{shift}}`$ is constructed by interpolation:

```math
(\mathbf{x}_{\mathrm{shift}})_\ell=(1-\alpha_\ell)x_{\lfloor s_\ell \rfloor}+\alpha_\ell x_{\lfloor s_\ell \rfloor+1},
```

where

```math
s_\ell = \ell - \delta \quad\text{and}\quad\alpha_\ell = s_\ell - \lfloor s_\ell \rfloor.
```

Boundary indices are clamped to the valid wavelength-index range.

After the candidate shift is generated, $`\mathbf{x}_{\mathrm{shift}}`$ is reprojected to the SNV-compatible geometry.

### Angle-Limited Movement Toward the Shifted Candidate

The raw shifted candidate may be too far from the original spectrum. Therefore, the final augmented spectrum is not necessarily the full shifted candidate.

Instead, the module moves from $`\mathbf{x}`$ toward $`\mathbf{x}_{\mathrm{shift}}`$ by a sampled angle:

```math
\theta_{\mathrm{shift}}\sim\mathcal{U}(\theta_{\min}, \theta_{\max}).
```

If the candidate is closer than the sampled angle, the movement is clipped at the candidate itself.

This gives a controlled operation:

```math
\mathbf{x}_{\mathrm{aug}}=\mathrm{SLERP}(\mathbf{x},\mathbf{x}_{\mathrm{shift}},\theta_{\mathrm{shift}}).
```

This separates:

* `shift_delta_range`: how far to shift the candidate,
* `shift_angle_deg_range`: how far the final output is allowed to move from the original spectrum.

### Practical Role

Use fractional shift to improve robustness to:

* small wavelength-axis misalignment,
* peak-position jitter,
* interpolation differences,
* mild calibration variability.

For ChemoMAE, this is usually more appropriate than artificial low-frequency tilt because shift does not directly impose a global baseline trend.

---

## Augmentation 2 — Tangent Gaussian Noise

### Idea

Tangent Gaussian noise introduces small random local perturbations while respecting the spherical geometry.

Instead of adding Euclidean Gaussian noise directly,

```math
\mathbf{x}_{\mathrm{aug}} = \mathbf{x} + \boldsymbol{\varepsilon},
```

the implementation:

1. samples an ambient Gaussian vector,
2. centers it,
3. projects it onto the tangent space at the current spectrum,
4. normalizes the tangent direction,
5. rotates the spectrum by a sampled angle.

### Construction

For each selected spectrum $`\mathbf{x}`$ , sample

```math
\mathbf{g} \sim \mathcal{N}(\mathbf{0}, I_L).
```

Center the direction:

```math
\tilde{\mathbf{g}}
=
\mathbf{g}
-
\frac{1}{L}
(\mathbf{1}^{\top}\mathbf{g})\mathbf{1}.
```

Project it onto the tangent space:

```math
\mathbf{v}=\tilde{\mathbf{g}}-\frac{\tilde{\mathbf{g}}^{\top}\mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}.
```

Normalize the tangent direction:

```math
\mathbf{u}
=
\frac{\mathbf{v}}{\lVert \mathbf{v} \rVert_2}.
```

Then sample an angle:

```math
\theta_{\mathrm{noise}}
\sim
\mathcal{U}(\theta_{\min}, \theta_{\max})
```

and rotate along the tangent direction:

```math
\mathbf{x}_{\mathrm{noise}}=r\left(\cos(\theta_{\mathrm{noise}})\frac{\mathbf{x}}{r}+\sin(\theta_{\mathrm{noise}})\mathbf{u}\right).
```

Finally, $`\mathbf{x}_{\mathrm{noise}}`$ is reprojected to the SNV-compatible geometry when re-centering and re-normalization are enabled.

### Practical Role

Use tangent Gaussian noise to improve robustness to:

* small observation noise,
* weak local fluctuations,
* minor random spectral variations that should not change the semantic identity of the spectrum.

This augmentation is the spherical analogue of Gaussian noise, but with magnitude controlled by geodesic angle instead of Euclidean variance.

---

## Execution Order

The module supports two execution modes.

### Fixed Order

When

```python
shuffle_order_per_batch = False
```

the operations are applied in this fixed order:

```math
\text{fractional shift}\rightarrow\text{tangent Gaussian noise}.
```

This is the recommended initial setting for ChemoMAE pretraining because it has a clear observation-process interpretation:

```math
\text{wavelength misalignment}
\rightarrow
\text{measurement noise}.
```

With reprojection enabled, the sequence becomes:

```math
\text{shift}\rightarrow\text{recenter/renorm}\rightarrow\text{noise}\rightarrow\text{recenter/renorm}.
```

### Random Order

When

```python
shuffle_order_per_batch = True
```

the operation order is sampled once per batch. The possible orders are:

```math
\text{shift}\rightarrow\text{noise}\quad\mathrm{or}\quad\text{noise}\rightarrow\text{shift}.
```

The order is not sampled independently for each sample. However, each operation still has an independent per-sample application mask and independently sampled strength parameters.

Therefore, even with fixed operation order, the batch contains a mixture of:

* no augmentation,
* shift only,
* noise only,
* shift + noise.

---

## API

### Dataclass: `SpectraAugmenterConfig`

```python
@dataclass(frozen=True)
class SpectraAugmenterConfig:
    shift_prob: float = 0.5
    shift_delta_range: tuple[float, float] = (-4.0, 4.0)
    shift_angle_deg_range: tuple[float, float] = (1.0, 4.0)

    noise_prob: float = 0.5
    noise_angle_deg_range: tuple[float, float] = (0.5, 3.0)

    shuffle_order_per_batch: bool = True
    recenter_after_each_op: bool = True
    renorm_to_input_norm: bool = True
    eps: float = 1.0e-8
```

### Parameters

| Name                      | Type                  | Description                                                                       |
| ------------------------- | --------------------- | --------------------------------------------------------------------------------- |
| `shift_prob`              | `float`               | Probability of applying fractional shift to each sample.                          |
| `shift_delta_range`       | `tuple[float, float]` | Range of candidate fractional shift amounts in channel-index units.               |
| `shift_angle_deg_range`   | `tuple[float, float]` | Range of movement angles from the original spectrum toward the shifted candidate. |
| `noise_prob`              | `float`               | Probability of applying tangent Gaussian noise to each sample.                    |
| `noise_angle_deg_range`   | `tuple[float, float]` | Range of geodesic rotation angles for tangent Gaussian noise.                     |
| `shuffle_order_per_batch` | `bool`                | Whether to randomize the order of shift and noise once per batch.                 |
| `recenter_after_each_op`  | `bool`                | Whether to re-center each spectrum to mean zero after each augmentation.          |
| `renorm_to_input_norm`    | `bool`                | Whether to re-normalize each spectrum to the input norm after each augmentation.  |
| `eps`                     | `float`               | Numerical stability constant used in normalization and projection.                |

---

### Constraints

* `shift_prob` and `noise_prob` must lie in `[0, 1]`.
* `shift_delta_range` must satisfy `low <= high`.
* `shift_angle_deg_range` and `noise_angle_deg_range` must satisfy:

  * lower bound `>= 0`,
  * upper bound `<= 180`,
  * lower bound `<=` upper bound.
* `eps > 0`.

### Class: `SpectraAugmenter`

```python
class SpectraAugmenter(nn.Module):
    def __init__(self, config: SpectraAugmenterConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### Input

* `x`: `torch.Tensor` of shape `(B, L)`
* floating dtype required

### Output

* augmented tensor of shape `(B, L)`

### Behavior

* If `self.training == False`, `forward(x)` returns `x` unchanged.
* Each augmentation is applied independently according to its per-sample probability.
* All operations are batch-vectorized.
* With `recenter_after_each_op=True`, each augmented sample is returned to zero mean.
* With `renorm_to_input_norm=True`, each augmented sample is returned to the input per-sample L2 norm.
* The module is compatible with `augmenter.to(device)`, `augmenter.train()`, and `augmenter.eval()` because it subclasses `nn.Module`.

---

## Usage Example

```python
import torch

from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

cfg = SpectraAugmenterConfig(
    shift_prob=0.5,
    shift_delta_range=(-2.0, 2.0),
    shift_angle_deg_range=(0.5, 3.0),
    noise_prob=0.5,
    noise_angle_deg_range=(0.5, 3.0),
    shuffle_order_per_batch=False,
    recenter_after_each_op=True,
    renorm_to_input_norm=True,
)

augmenter = SpectraAugmenter(cfg)
augmenter.train()

x = torch.randn(64, 256, dtype=torch.float32)
x = x - x.mean(dim=1, keepdim=True)
x = x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1.0e-8) # SNV preprocessing

x_aug = augmenter(x)
```

---

## Design Notes

### Why angle-based strength?

Earlier versions used cosine similarity ranges. While cosine similarity is mathematically equivalent to angle on the sphere, angle is often easier to tune directly.

The relationship is:

```math
c = \cos(\theta).
```

For weak augmentations:

```math
\theta = 1^\circ
\quad\Rightarrow\quad
c \approx 0.99985
```

```math
\theta = 3^\circ
\quad\Rightarrow\quad
c \approx 0.99863
```

```math
\theta = 5^\circ
\quad\Rightarrow\quad
c \approx 0.99619
```

Thus, small degree values correspond to very high cosine similarity.

---

### Why fractional shift?

Fractional shift is a physically plausible spectral augmentation. It models small wavelength-axis variation without imposing an artificial global baseline trend.

It is especially suitable when spectra are smooth and peak locations may shift slightly due to measurement or interpolation effects.

---

### Why tangent Gaussian noise?

Tangent Gaussian noise improves robustness to small random variations while preserving the main geometry of SNV-normalized spectra.

Because it operates through tangent-space rotation, it avoids unconstrained additive noise that would otherwise change the norm and potentially the mean.

---

### Why keep augmentations weak?

ChemoMAE already receives a strong learning signal from masked reconstruction. Augmentation should not dominate this task.

The intended role is:

```math
\text{masked reconstruction}+\text{weak denoising regularization}.
```

Strong augmentations may cause the model to reconstruct targets from overly distorted inputs and could suppress degradation-related structure.

---

## When to Use `SpectraAugmenter` in ChemoMAE Pipelines

### Use during training only

The module is intended for stochastic training-time augmentation.

In evaluation mode, it returns the input unchanged:

```python
augmenter.eval()
x_out = augmenter(x)
```

gives $`\mathbf{x}_{\mathrm{out}} = \mathbf{x}`$ .

### Do not use for deterministic feature extraction

For latent extraction, spectra should be passed without stochastic augmentation.

For ChemoMAE feature extraction, use either:

```python
augmenter.eval()
```

or avoid passing an augmenter entirely.

Also ensure that the model itself is called with an all-visible mask if deterministic latent extraction is required.

### Recommended after SNV preprocessing

This module assumes spectra are already SNV-normalized.

The recommended input geometry is:

```math
\mathbf{1}^{\top}\mathbf{x} \approx 0\quad\text{and}\quad\lVert \mathbf{x} \rVert_2 \approx r.
```

---

## Common Pitfalls

### Confusing `shift_delta_range` and `shift_angle_deg_range`

`shift_delta_range` controls the candidate shift.

`shift_angle_deg_range` controls how far the final output moves toward that candidate.

A large candidate shift does not necessarily mean a large final perturbation if the angle range is small.

### Applying augmentation before SNV

This module is designed for SNV-normalized spectra.

If it is applied before SNV, the geometric assumptions behind re-centering, re-normalization, and tangent-space rotation become less meaningful.

### Applying augmentation during evaluation

The module is inactive in `eval()` mode. This is intentional.

Feature extraction and validation should remain deterministic unless stochastic evaluation is explicitly intended.

### Treating this as contrastive multi-view augmentation

This module does not create paired views for contrastive learning.

It is designed as weak input corruption for masked reconstruction. If later using view-consistency objectives, a separate two-view augmentation interface may be more appropriate.

---

## Minimal Test Snippets

```python
import torch

from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

x = torch.randn(32, 256, dtype=torch.float32)
x = x - x.mean(dim=1, keepdim=True)
x = x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1.0e-8)

cfg = SpectraAugmenterConfig(
    shift_prob=1.0,
    shift_delta_range=(-4.0, 4.0),
    shift_angle_deg_range=(2.0, 2.0),
    noise_prob=1.0,
    noise_angle_deg_range=(1.0, 1.0),
    shuffle_order_per_batch=False,
    recenter_after_each_op=True,
    renorm_to_input_norm=True,
)

aug = SpectraAugmenter(cfg)

# train mode -> stochastic augmentation
aug.train()
x_aug = aug(x)
assert x_aug.shape == x.shape

# mean preservation
x_mean = x.mean(dim=1)
x_aug_mean = x_aug.mean(dim=1)
torch.testing.assert_close(
    x_aug_mean,
    torch.zeros_like(x_aug_mean),
    rtol=1e-5,
    atol=1e-6,
)

# norm preservation
x_norm = torch.linalg.norm(x, dim=1)
x_aug_norm = torch.linalg.norm(x_aug, dim=1)
torch.testing.assert_close(x_norm, x_aug_norm, rtol=1e-5, atol=1e-6)

# eval mode -> identity
aug.eval()
x_eval = aug(x)
torch.testing.assert_close(x_eval, x)

# angle from original should be small for weak settings
cos = torch.sum(x * x_aug, dim=1) / (
    torch.linalg.norm(x, dim=1) * torch.linalg.norm(x_aug, dim=1) + 1.0e-8
)
cos = cos.clamp(-1.0, 1.0)
angle_deg = torch.rad2deg(torch.arccos(cos))

assert torch.all(angle_deg >= 0.0)
assert torch.all(angle_deg < 10.0)
```

---

## Version (v0.1.9)

* Updated in `chemomae.training.augmenter` for angle-controlled `shift + noise` augmentation.
* Replaces the previous `noise + tilt` cosine-controlled design.