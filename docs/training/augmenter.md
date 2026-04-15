# Spectral Augmentation on the Hypersphere — Spherical Gaussian Noise and Geodesic Tilt

> Module: `chemomae.training.augmenter`

This document describes **`SpectraAugmenter`**, a hypersphere-aware spectral augmentation module for **SNV-normalized spectra**.
The implementation provides two geometry-consistent augmentations:

* **Spherical Gaussian noise**
* **Geodesic tilt**

Both transformations preserve the **per-spectrum L2 norm** and operate by moving each spectrum **along the hypersphere** , rather than perturbing it in unconstrained Euclidean space. This makes the module suitable for ChemoMAE pipelines where spectra are interpreted under spherical geometry after SNV preprocessing.  

---

## Overview

Consider a batch of SNV-normalized spectra

```math
X = \{\mathbf{x}_1, \dots, \mathbf{x}_B\} \subset \mathbb{R}^L
```

where each spectrum has approximately constant L2 norm:

```math
\lVert \mathbf{x}_i \rVert_2 \approx r
```

for some fixed radius $`r > 0`$ .
In this setting, spectra can be interpreted as lying on a hypersphere:

```math
\mathbf{x}_i \in \mathbb{S}^{L-1}(r)
```

A standard additive perturbation,

```math
\mathbf{x}_i' = \mathbf{x}_i + \boldsymbol{\varepsilon}_i
```

generally moves the sample **off the hypersphere** and breaks the geometric structure induced by SNV.

`SpectraAugmenter` instead perturbs spectra by:

1. constructing a direction in the **tangent space** at the current point,
2. normalizing that direction,
3. applying a **geodesic rotation** on the sphere.

Thus, the augmented spectrum remains on the same-radius hypersphere:

```math
\lVert \mathbf{x}_i' \rVert_2 = \lVert \mathbf{x}_i \rVert_2
```

for both augmentations.   

---

## Strength Control via Cosine Similarity

Unlike angle-based APIs, this implementation controls augmentation strength through a target cosine similarity range.

For a spectrum $`\mathbf{x}`$ and its augmented version $`\mathbf{x}'`$ , the cosine similarity is

```math
\cos(\theta)=\frac{\mathbf{x}^\top \mathbf{x}'}{\lVert \mathbf{x} \rVert_2 \lVert \mathbf{x}' \rVert_2}
```

Because norm is preserved, this is equivalent to geodesic rotation by angle $`\theta`$ , with

```math
\theta = \arccos(c)
```

where $`c \in (0,1]`$ is the sampled target cosine similarity.

Interpretation:

* $`c \approx 1.0`$ : extremely weak perturbation
* smaller $`c`$ : stronger perturbation

Thus, `noise_cos_range` and `tilt_cos_range` directly specify how close the augmented spectrum should remain to the original one under cosine geometry.   

---

## Tangent-Space Construction

At a spectrum $`\mathbf{x}`$ , the tangent space of the sphere is

```math
T_{\mathbf{x}}\mathbb{S}^{L-1}(r)=\left\{\mathbf{v} \in \mathbb{R}^L \;\middle|\;\mathbf{v}^\top \mathbf{x} = 0 \right\}
```

Given an arbitrary direction $`\mathbf{d}`$ , the implementation projects it onto the tangent space via

```math
\mathbf{v}=\mathbf{d}-\frac{\mathbf{d}^\top \mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}
```

This is the core operation used by **both** augmentations:

* random Gaussian directions are projected for spherical Gaussian noise,
* a fixed linear tilt basis is projected for geodesic tilt. 

After projection, the tangent vector is row-wise normalized, with a fallback tangent direction used in degenerate cases.  

---

## Geodesic Rotation

Once a unit tangent direction $`\mathbf{u}`$ is available, the implementation rotates the spectrum along the sphere by angle $`\theta`$ :

```math
\mathbf{x}'=r\left(\cos(\theta)\frac{\mathbf{x}}{r}+\sin(\theta)\mathbf{u}\right),\qquad r = \lVert \mathbf{x} \rVert_2
```

This guarantees:

```math
\lVert \mathbf{x}' \rVert_2 = r
```

so the augmented sample stays on the same hypersphere. 

---

## Augmentation 1 — Spherical Gaussian Noise

### Idea

This augmentation introduces **random local perturbations** that respect spherical geometry.

Instead of adding Euclidean Gaussian noise directly to the spectrum, the implementation:

1. samples an ambient Gaussian vector,
2. projects it onto the tangent space at the current spectrum,
3. normalizes the tangent direction,
4. rotates the spectrum along the sphere by a cosine-controlled amount. 

### Construction

For each spectrum $`\mathbf{x}`$ ,

```math
\mathbf{g} \sim \mathcal{N}(\mathbf{0}, I_L)
```

is sampled, then projected:

```math
\mathbf{v}=\mathbf{g}-\frac{\mathbf{g}^\top \mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}
```

and normalized to obtain a unit tangent direction $`\mathbf{u}`$ .
A target cosine similarity $`c`$ is sampled from `noise_cos_range`, converted to

```math
\theta = \arccos(c),
```

and used in geodesic rotation.

### Interpretation

This augmentation models **small random spectral fluctuations** while preserving hypersphere structure.
It is the spherical analogue of Gaussian noise, but with magnitude controlled in cosine space rather than via unrestricted additive variance.

### Practical role

Use spherical Gaussian noise when you want robustness to:

* small observation variability,
* local random perturbations,
* slight deviations that should not change semantic spectral identity.

---

## Augmentation 2 — Geodesic Tilt

### Idea

This augmentation introduces a **structured low-frequency perturbation** corresponding to a gentle spectral slope or baseline-like trend.

Instead of choosing a random tangent direction, the implementation builds a fixed linear basis over wavelength coordinates:

```math
\mathbf{t} = \mathrm{linspace}(-1, 1, L)
```

then centers and normalizes it. 

### Construction

For each spectrum $`\mathbf{x}`$ , the tilt basis $`\mathbf{t}`$ is projected onto the tangent space:

```math
\mathbf{v}_{\text{tilt}}=\mathbf{t}-\frac{\mathbf{t}^\top \mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}
```

This gives a sample-specific tangent direction corresponding to “tilting” the spectrum while remaining compatible with the sphere.

A target cosine similarity is sampled from `tilt_cos_range`, converted to a rotation angle, and then multiplied by a random sign so that tilt may occur in either direction:

* positive slope
* negative slope 

### Interpretation

This augmentation models **baseline-like or low-frequency spectral variation** under spherical geometry.

### Practical role

Use geodesic tilt when you want robustness to:

* global slope-like distortions,
* low-frequency shape changes,
* baseline variation that should not dominate representation learning.

---

## Execution Order

When both augmentations are enabled, the current implementation applies them **sequentially** in this fixed order:

```math
\text{spherical Gaussian noise}\;\rightarrow\;\text{geodesic tilt}
```

That is, tilt is applied to the output of noise, not to the original spectrum. Therefore, the tangent-space projection used by tilt is computed at the already perturbed point. 

If the order is later randomized, the interpretation becomes:

* one geodesic move along a random tangent direction,
* followed by one geodesic move along a structured tilt direction,

with the sequence sampled per forward pass.

---

## API

### Dataclass: `SpectraAugmenterConfig`

```python
@dataclass(frozen=True)
class SpectraAugmenterConfig:
    noise_prob: float = 0.0
    noise_cos_range: tuple[float, float] = (1.0, 1.0)
    tilt_prob: float = 0.0
    tilt_cos_range: tuple[float, float] = (1.0, 1.0)
    eps: float = 1e-12
```

#### Parameters

| Name              | Type                  | Description                                                        |
| ----------------- | --------------------- | ------------------------------------------------------------------ |
| `noise_prob`      | `float`               | Probability of applying spherical Gaussian noise.                  |
| `noise_cos_range` | `tuple[float, float]` | Target cosine similarity range for spherical Gaussian noise.       |
| `tilt_prob`       | `float`               | Probability of applying geodesic tilt.                             |
| `tilt_cos_range`  | `tuple[float, float]` | Target cosine similarity range for geodesic tilt.                  |
| `eps`             | `float`               | Numerical stability constant used in normalization and projection. |

#### Constraints

* `noise_prob`, `tilt_prob` must lie in `[0, 1]`
* all cosine ranges must lie in `(0, 1]`
* range lower bound must be `<=` upper bound
* `eps > 0` 

---

### Class: `SpectraAugmenter`

```python
class SpectraAugmenter(nn.Module):
    def __init__(self, config: SpectraAugmenterConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

#### Input

* `x`: `torch.Tensor` of shape `(B, L)`
* floating dtype required

#### Output

* augmented tensor of shape `(B, L)`

#### Behavior

* if `self.training == False`, `forward(x)` returns `x` unchanged
* each augmentation is applied independently according to its probability
* all operations are batch-vectorized
* the output stays on the same per-sample L2 sphere as the input 

---

## Usage Example

```python
import torch
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

cfg = SpectraAugmenterConfig(
    noise_prob=0.5,
    noise_cos_range=(0.995, 0.9995),
    tilt_prob=0.3,
    tilt_cos_range=(0.997, 0.9998),
)

augmenter = SpectraAugmenter(cfg)
augmenter.train()

x = torch.randn(64, 256, dtype=torch.float32)
x_aug = augmenter(x)
```

### Interpretation of the example

* `noise_prob=0.5`: half of samples are exposed to spherical Gaussian noise on average
* `noise_cos_range=(0.995, 0.9995)`: noise remains very close to the original spectrum
* `tilt_prob=0.3`: tilt is applied less often
* `tilt_cos_range=(0.997, 0.9998)`: tilt is even weaker than noise

---

## Design Notes

### Why cosine-based strength?

Cosine similarity is more interpretable than raw geodesic angle in spectral workflows:

* `1.000` means almost unchanged
* slightly below `1.000` means mild perturbation
* lower values mean stronger movement on the sphere

This makes hyperparameter tuning easier than directly specifying angles in radians. 

### Why not Euclidean additive noise?

Because additive noise generally violates the constant-radius structure induced by SNV.
The augmentations here are designed specifically to remain compatible with spherical latent geometry.

### Why both augmentations?

They model different types of variation:

* **spherical Gaussian noise**: random local perturbation
* **geodesic tilt**: structured low-frequency perturbation

Together, they provide a compact but meaningful augmentation set for ChemoMAE pretraining.

---

## When to Use `SpectraAugmenter` in ChemoMAE Pipelines

* **Use during training only**
  The module is intended for stochastic training-time augmentation. In evaluation mode, it returns the input unchanged. 

* **Do not use for deterministic feature extraction**
  For latent extraction or final evaluation, spectra should be passed without stochastic augmentation.

* **Recommended after SNV preprocessing**
  These augmentations assume a hyperspherical interpretation of spectra, which is most natural after SNV normalization.

* **Keep perturbations weak initially**
  Since ChemoMAE already uses masking as the primary learning signal, these augmentations should act as auxiliary regularizers rather than dominate the task.

---

## Common Pitfalls

* **Using too strong a cosine range**
  Values too far below `1.0` may over-distort spectra and interfere with masked reconstruction.

* **Interpreting “Gaussian” too literally**
  The current implementation uses a Gaussian-derived tangent direction, but the final perturbation strength is controlled by cosine similarity, not by a free Euclidean variance parameter. 

* **Applying augmentation at evaluation time**
  The module is designed to be inactive under `eval()` mode; feature extraction and testing should remain deterministic.

* **Assuming noise and tilt are interchangeable**
  They serve different purposes: one is random and local, the other is structured and low-frequency.

---

## Minimal Test Snippets

```python
import torch
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig

x = torch.randn(32, 256, dtype=torch.float32)
x = x / (torch.linalg.norm(x, dim=1, keepdim=True) + 1e-12)

cfg = SpectraAugmenterConfig(
    noise_prob=1.0,
    noise_cos_range=(0.999, 0.999),
    tilt_prob=1.0,
    tilt_cos_range=(0.999, 0.999),
)
aug = SpectraAugmenter(cfg)

# train mode -> stochastic augmentation
aug.train()
y = aug(x)
assert y.shape == x.shape

# norm preservation
x_norm = torch.linalg.norm(x, dim=1)
y_norm = torch.linalg.norm(y, dim=1)
torch.testing.assert_close(x_norm, y_norm, rtol=1e-5, atol=1e-6)

# eval mode -> identity
aug.eval()
z = aug(x)
torch.testing.assert_close(z, x)

# cosine similarity should remain close to 1 for weak settings
cos = torch.sum(x * y, dim=1) / (
    torch.linalg.norm(x, dim=1) * torch.linalg.norm(y, dim=1) + 1e-12
)
assert torch.all(cos <= 1.0 + 1e-6)
```

---

## Version

* Introduced in `chemomae.training.augmenter` (v0.1.8) for hypersphere-aware spectral augmentation in ChemoMAE training.
