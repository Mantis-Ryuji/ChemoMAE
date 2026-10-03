# Spectral Augmentation on the Hypersphere — Fractional Shift and Tangent Gaussian Noise

> Module: `chemomae.training.augmenter`

This document describes **`SpectraAugmenter`**, a hypersphere-aware spectral augmentation module for **SNV-normalized spectra** in ChemoMAE training.

The implementation provides two lightweight training-time augmentations:

- **Fractional shift**
- **Tangent Gaussian noise**

Both transformations are designed for spectra that have already been
standardized by SNV. With re-centering and re-normalization enabled, they
preserve the input mean and norm instead of reintroducing the offset and scale
variation removed by preprocessing. They supply controlled changes of
normalized spectral shape for a reconstruction task.

The intended role of this module is **additional input corruption for masked
denoising**, or for full-spectrum denoising when that loss region is selected.
The reconstruction target is the spectrum before these additional
perturbations. The operations are inspired by signal variation and wavelength
misalignment, but they do not constitute a calibrated measurement-error model
or guarantee preservation of chemical state.

---

## Overview

Consider a batch of SNV-normalized spectra

$$
X = \lbrace\mathbf{x}_1, \dots, \mathbf{x}_B\rbrace \subset \mathbb{R}^L
$$

where each spectrum satisfies approximately

$$
\frac{1}{L}\sum_{\ell=1}^{L} x_{i,\ell} \approx 0 \quad\text{and}\quad\lVert \mathbf{x}_i \rVert_2 \approx r
$$

for some nearly constant radius $r > 0$.

In the idealized nonconstant case of SNV with sample standard deviation and no
epsilon, each spectrum lies on the intersection of:

1. the zero-mean hyperplane, and
2. a fixed-radius hypersphere.

That is,

$$
\mathbf{x}_i \in \mathcal{M}=\left\lbrace\mathbf{x} \in \mathbb{R}^L\quad\middle|\quad\mathbf{1}^{\top}\mathbf{x}=0,\quad\lVert \mathbf{x} \rVert_2=r\right\rbrace.
$$

This is a sphere within the zero-mean hyperplane. For the idealized SNV
definition its radius is $r=\sqrt{L-1}$. The library's
[SNV implementation](../preprocessing/snv.md) adds epsilon to the sample
standard deviation, so actual norms are approximately common for nonconstant
spectra. The augmenter restores each input spectrum's own norm rather than
forcing all rows to this ideal radius. Constant and length-one SNV outputs
have zero norm and do not satisfy the nonzero-sphere assumption.

A naive Euclidean perturbation,

$$
\mathbf{x}_i' = \mathbf{x}_i + \boldsymbol{\varepsilon}_i,
$$

generally violates this structure because it may change both the sample mean and the L2 norm.

`SpectraAugmenter` instead applies weak spectral perturbations and then, when enabled, projects the result back to the SNV-compatible space by:

1. re-centering each spectrum to mean zero,
2. re-normalizing it to the original per-sample L2 norm.

---

## Design Goal

The main learning signal in ChemoMAE is reconstruction over the region selected by `TrainerConfig.loss_region`.

In masked mode, augmentation is applied to the full input before masking. The
model uses the transformed visible bands to predict the original masked bands.
In all-region mode, every output element is compared with the unaugmented
spectrum. In both modes, the target is the observed input before augmentation,
not an independently measured noise-free spectrum. The masked task is:

$$
A(\mathbf{x})_{\Omega_v}\longrightarrow\mathbf{x}_{\Omega_m},
$$

where:

* $A$ is the augmentation operator,
* $\Omega_v$ is the visible wavelength region,
* $\Omega_m$ is the masked wavelength region.

The additional denoising task is intended to encourage predictive relationships
between bands that remain useful under the specified perturbations. Whether
this also improves the spatial coherence of a downstream clustering is an
empirical question; stability under these perturbations and spatial coherence
are different properties.

The module supplies two operations:

$$
\text{fractional shift} + \text{tangent Gaussian noise}.
$$

The choice to preserve mean and norm defines the neighborhood of inputs used
for training; it is a design choice for this reconstruction task.

---

## Strength Control

This implementation uses different control variables for fractional shift and tangent Gaussian noise.

### Fractional shift

Fractional shift is controlled only by the shift amount

$$
\delta \quad \text{[channel index]}.
$$

The corresponding API parameter is:

```python
shift_delta_range: tuple[float, float]
```

This means the operation directly controls the wavelength-axis displacement. The final shifted spectrum is not additionally clipped by cosine similarity or geodesic angle.

This design is intentional. Fractional shift is an axis-domain operation, and its physically interpretable control variable is the displacement along the wavelength or channel axis. Controlling the same operation again by spherical angle can unintentionally weaken the perturbation because the angular distance induced by a fixed shift depends strongly on the spectrum shape.

### Tangent Gaussian noise

Tangent Gaussian noise is controlled by geodesic angle.

For an input spectrum $\mathbf{x}$ and augmented spectrum $\mathbf{x}_{\mathrm{aug}}$, the angle is defined through

$$
\cos(\theta)=\frac{\mathbf{x}^{\top}\mathbf{x}_{\mathrm{aug}}}{\lVert \mathbf{x} \rVert_2 \lVert \mathbf{x}_{\mathrm{aug}} \rVert_2}.
$$

The API specifies the noise angle range in **degrees**:

```python
noise_angle_deg_range: tuple[float, float]
```

Internally, sampled angles are converted to radians.

Angle-based control is appropriate for tangent Gaussian noise because the perturbation direction is random and does not have a natural physical unit like channel displacement.

The associated research protocol used a uniform TGN angle from 0 to 5 degrees
and a uniform FS displacement from -2 to 2 channels. Each enabled operation
was applied independently with probability 0.5 per spectrum, with the order
randomized per batch. These are experiment settings: the library's default
`noise_angle_deg_range` is `(0.5, 3.0)`. Set the configuration explicitly when
reproducing a particular protocol.

---

## SNV-Compatible Reprojection

After each augmentation, the module can apply re-centering:

$$
\mathbf{x}_{\mathrm{cand}}\leftarrow\mathbf{x}_{\mathrm{cand}}-\frac{1}{L}\left(\mathbf{1}^{\top}\mathbf{x}_{\mathrm{cand}}\right)\mathbf{1},
$$

followed by re-normalization:

$$
\mathbf{x}_{\mathrm{cand}}\leftarrow\lVert \mathbf{x} \rVert_2\frac{\mathbf{x}_{\mathrm{cand}}}{\lVert \mathbf{x}_{\mathrm{cand}} \rVert_2}.
$$

Here:

* $\mathbf{x}$ is the input to the current augmentation operation,
* $\mathbf{x}_{\mathrm{cand}}$ is the intermediate augmented candidate.

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

At a spectrum $\mathbf{x}$, the tangent space of the sphere is

$$
T_{\mathbf{x}}\mathbb{S}^{L-1}(r)=\left\lbrace\mathbf{v} \in \mathbb{R}^L\quad\middle|\quad\mathbf{v}^{\top}\mathbf{x}=0\right\rbrace.
$$

Given an arbitrary direction $\mathbf{d}$, the projection onto this tangent space is

$$
\mathbf{v}=\mathbf{d}-\frac{\mathbf{d}^{\top}\mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}.
$$

In this implementation, the random direction is first centered before tangent
projection. For a zero-mean input, the projected direction is both zero-mean
and orthogonal to the input, so it belongs to the tangent space of the sphere
within the zero-mean hyperplane.

---

## Geodesic Rotation for Noise

Given a unit tangent direction $\mathbf{u}$, the spectrum is rotated along the sphere by angle $\theta$:

$$
\mathbf{x}_{\mathrm{aug}}=r\left(\cos(\theta)\frac{\mathbf{x}}{r}+\sin(\theta)\mathbf{u}\right),\qquad r = \lVert \mathbf{x} \rVert_2.
$$

This operation preserves the L2 norm before re-centering. Since re-centering may slightly change the norm, the implementation can re-normalize the result to the input norm afterward.

This geodesic rotation is used for tangent Gaussian noise, not for fractional shift.

---

## Augmentation 1 — Fractional Shift

### Idea

Fractional shift supplies a controlled displacement along the channel axis.

On an equally spaced wavelength grid, channel displacement also describes a
wavelength displacement. It is inspired by wavelength-position variation;
the sampled displacement range is a training choice, not an estimate of the
instrument's error distribution.

Unlike `torch.roll`, fractional shift supports non-integer shifts and uses linear interpolation.

### Construction

For each selected spectrum $\mathbf{x}$, a shift amount is sampled:

$$
\delta \sim \mathcal{U}(\delta_{\min}, \delta_{\max}).
$$

The shifted candidate $\mathbf{x}_{\mathrm{shift}}$ is constructed by interpolation:

$$
(\mathbf{x}_{\mathrm{shift}})_\ell=(1-\alpha_\ell)x_{\lfloor s_\ell \rfloor}+\alpha_\ell x_{\lfloor s_\ell \rfloor+1},
$$

where

$$
s_\ell = \ell - \delta \quad\text{and}\quad\alpha_\ell = s_\ell - \lfloor s_\ell \rfloor.
$$

Boundary indices are clamped to the valid wavelength-index range.

After the candidate shift is generated, $\mathbf{x}_{\mathrm{shift}}$ is reprojected to the SNV-compatible geometry.

The final output of this augmentation is

$$
\mathbf{x}_{\mathrm{aug}}=\Pi_{\mathcal{M}}(\mathbf{x}_{\mathrm{shift}}),
$$

where $\Pi_{\mathcal{M}}$ denotes the optional re-centering and re-normalization operation.

### Practical Role

Fractional shift varies the alignment of spectral features while retaining
the configured mean/norm geometry after reprojection. A fixed channel shift
can produce different angular changes for different spectral shapes, so its
strength is specified in channel units rather than as a spherical angle.

Because this operation precedes masking, interpolation can move information
across the boundary of a masked band. The training task uses the shifted
visible bands to reconstruct the original target bands, including this effect.

---

## Augmentation 2 — Tangent Gaussian Noise

### Idea

Tangent Gaussian noise rotates the input within the sphere in its zero-mean
hyperplane. The Gaussian distribution supplies the random direction before
projection; it does not describe an additive Gaussian error on each output
channel.

Instead of adding Euclidean Gaussian noise directly,

$$
\mathbf{x}_{\mathrm{aug}} = \mathbf{x} + \boldsymbol{\varepsilon},
$$

the implementation:

1. samples an ambient Gaussian vector,
2. centers it,
3. projects it onto the tangent space at the current spectrum,
4. normalizes the tangent direction,
5. rotates the spectrum by a sampled angle.

### Construction

For each selected spectrum $\mathbf{x}$, sample

$$
\mathbf{g} \sim \mathcal{N}(\mathbf{0}, I_L).
$$

Center the direction:

$$
\tilde{\mathbf{g}}=\mathbf{g}-\frac{1}{L}(\mathbf{1}^{\top}\mathbf{g})\mathbf{1}.
$$

Project it onto the tangent space:

$$
\mathbf{v}=\tilde{\mathbf{g}}-\frac{\tilde{\mathbf{g}}^{\top}\mathbf{x}}{\lVert \mathbf{x} \rVert_2^2}\mathbf{x}.
$$

Normalize the tangent direction:

$$
\mathbf{u}=\frac{\mathbf{v}}{\lVert \mathbf{v} \rVert_2}.
$$

Then sample an angle:

$$
\theta_{\mathrm{noise}}
\sim
\mathcal{U}(\theta_{\min}, \theta_{\max})
$$

and rotate along the tangent direction:

$$
\mathbf{x}_{\mathrm{noise}}=r\left(\cos(\theta_{\mathrm{noise}})\frac{\mathbf{x}}{r}+\sin(\theta_{\mathrm{noise}})\mathbf{u}\right).
$$

Finally, $\mathbf{x}_{\mathrm{noise}}$ is reprojected to the SNV-compatible geometry when re-centering and re-normalization are enabled.

### Practical Role

Tangent Gaussian noise provides random changes of normalized spectral shape
with magnitude controlled by geodesic angle. For a valid tangent direction,
this angle is the angular separation from the input before numerical
reprojection. It preserves mean and norm under the stated assumptions.

---

## Execution Order

The module supports two execution modes.

### Fixed Order

When

```python
shuffle_order_per_batch = False
```

the operations are applied in this fixed order:

$$
\text{fractional shift}\rightarrow\text{tangent Gaussian noise}.
$$

With reprojection enabled, the sequence becomes:

$$
\text{shift}\rightarrow\text{recenter/renorm}\rightarrow\text{noise}\rightarrow\text{recenter/renorm}.
$$

### Random Order

When

```python
shuffle_order_per_batch = True
```

the operation order is sampled once per batch. The possible orders are:

$$
\text{shift}\rightarrow\text{noise}\quad\mathrm{or}\quad\text{noise}\rightarrow\text{shift}.
$$

The order is not sampled independently for each sample. However, each operation still has an independent per-sample application mask and independently sampled strength parameters.

---

## API

### Dataclass: `SpectraAugmenterConfig`

```python
@dataclass(frozen=True)
class SpectraAugmenterConfig:
    shift_prob: float = 0.5
    shift_delta_range: tuple[float, float] = (-2.0, 2.0)

    noise_prob: float = 0.5
    noise_angle_deg_range: tuple[float, float] = (0.5, 3.0)

    shuffle_order_per_batch: bool = True
    recenter_after_each_op: bool = True
    renorm_to_input_norm: bool = True
    eps: float = 1.0e-8
```

### Parameters

| Name                      | Type                  | Description                                                                      |
| ------------------------- | --------------------- | -------------------------------------------------------------------------------- |
| `shift_prob`              | `float`               | Probability of applying fractional shift to each sample.                         |
| `shift_delta_range`       | `tuple[float, float]` | Range of fractional shift amounts in channel-index units.                        |
| `noise_prob`              | `float`               | Probability of applying tangent Gaussian noise to each sample.                   |
| `noise_angle_deg_range`   | `tuple[float, float]` | Range of geodesic rotation angles for tangent Gaussian noise.                    |
| `shuffle_order_per_batch` | `bool`                | Whether to randomize the order of shift and noise once per batch.                |
| `recenter_after_each_op`  | `bool`                | Whether to re-center each spectrum to mean zero after each augmentation.         |
| `renorm_to_input_norm`    | `bool`                | Whether to re-normalize each spectrum to the input norm after each augmentation. |
| `eps`                     | `float`               | Numerical stability constant used in normalization and projection.               |

---

### Constraints

* `shift_prob` and `noise_prob` must lie in `[0, 1]`.
* `shift_delta_range` must have finite endpoints and satisfy `low <= high`.
* `noise_angle_deg_range` must satisfy:

  * finite endpoints,
  * lower bound `>= 0`,
  * upper bound `<= 180`,
  * lower bound `<=` upper bound.
* `eps` must be finite and strictly positive.

### Class: `SpectraAugmenter`

```python
class SpectraAugmenter(nn.Module):
    def __init__(
        self, config: SpectraAugmenterConfig, *,
        generator: torch.Generator | None = None,
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...
```

### Input

* `x`: `torch.Tensor` of shape `(B, L)`
* floating dtype required

For float16/bfloat16 input, fractional-shift draws, channel coordinates, and
interpolation arithmetic use float32 before converting back to the input dtype.
This avoids rounding away half-channel shifts or aliasing neighboring channel
indices. Float32/float64 input keeps its corresponding shift arithmetic.

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
    noise_prob=0.5,
    noise_angle_deg_range=(0.5, 3.0),
    shuffle_order_per_batch=True,
    recenter_after_each_op=True,
    renorm_to_input_norm=True,
)

augmenter = SpectraAugmenter(cfg)
augmenter.train()

x = torch.randn(64, 256, dtype=torch.float32)
x = x - x.mean(dim=1, keepdim=True)
x = x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1.0e-8)

x_aug = augmenter(x)
```

---

## Caller-owned random streams

Every random draw can use an explicit `torch.Generator`: operation order,
per-spectrum application masks, shift amounts, Gaussian directions, and noise
angles. Supply a module default so that `Trainer`, `Tester`, and `Extractor`
can keep calling `augmenter(x)` without hidden global-state replacement:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
augmentation_stream = torch.Generator(device=device).manual_seed(42)
augmenter = SpectraAugmenter(cfg, generator=augmentation_stream).train()
x = x.to(device)
x_aug = augmenter(x)

# A separate stream overrides the module default for this call only.
other_stream = torch.Generator(device=device).manual_seed(73)
other_view = augmenter(x, generator=other_stream)
```

`forward(..., generator=None)` selects the module default. If neither is
supplied, random draws use the input device's global Torch generator; the module
does not reseed or replace global state. Explicit streams leave unrelated
global streams untouched. Use independent generators for masking, augmentation,
and sampling when those operations must not affect each other's sequences.

The active stream must match the input's device type and CUDA index. Device
mismatches raise a clear error during active augmentation. `augmenter.to(...)`
does not move a generator: create a stream on the destination device explicitly.
In evaluation mode the input is returned unchanged and no stream advances.

Generator state is caller-owned and is not part of `augmenter.state_dict()`.
Persist the selected stream separately when continuing an augmentation sequence:

```python
state = augmentation_stream.get_state().clone()
expected_next = augmenter(x)
augmentation_stream.set_state(state)
repeated_next = augmenter(x)
torch.testing.assert_close(repeated_next, expected_next)
```

Restoring a seed or state only reproduces a sequence under matching inputs,
configuration, modes, batch partitioning, call order, device, dtype, and
software. Equal seeds do not establish equality across CPU/CUDA or Torch
versions. Skipped operations and degenerate tangent directions can change how
many draws a call consumes. This API changes RNG ownership; fractional-shift
interpolation, tangent projection, geodesic rotation, and reprojection remain
the same mathematical operations.

---

## Design Notes

### Why remove angle control from fractional shift?

Fractional shift is naturally parameterized by the displacement amount $\delta$ along the wavelength axis.

Earlier versions controlled shift in two stages:

1. create a shifted candidate using `shift_delta_range`,
2. move toward that candidate using `shift_angle_deg_range`.

This made the final perturbation depend not only on the selected shift amount, but also on the angular distance between the original and shifted spectra.

For spectra, this can be undesirable because the same channel shift may induce very different cosine or angular changes depending on spectral shape, local slope, peak sharpness, and boundary behavior. As a result, angle-limited shift can weaken the intended wavelength-axis perturbation in a data-dependent way.

Therefore, the current design uses:

$$
\delta \sim \mathcal{U}(\delta_{\min}, \delta_{\max})
$$

as the only shift-strength parameter.

The shift operation is then followed only by SNV-compatible reprojection.

---

### Why keep angle control for tangent Gaussian noise?

Tangent Gaussian noise does not have a natural physical unit like channel displacement.

The random direction is sampled in the ambient feature space and then projected to the tangent space. Therefore, controlling its magnitude by geodesic angle is appropriate:

$$
c = \cos(\theta).
$$

For weak augmentations:

$$
\theta = 1^\circ
\quad\Rightarrow\quad
c \approx 0.99985
$$

$$
\theta = 3^\circ
\quad\Rightarrow\quad
c \approx 0.99863
$$

$$
\theta = 5^\circ
\quad\Rightarrow\quad
c \approx 0.99619
$$

Thus, small degree values correspond to very high cosine similarity.

---

### Why fractional shift?

Fractional shift introduces wavelength-axis variation without directly adding
a baseline offset or slope. Linear interpolation and endpoint clamping define
the actual transformation, so its suitability and strength should be assessed
for the supplied wavelength grid and spectral features.

---

### Why tangent Gaussian noise?

Tangent Gaussian noise introduces random variation within the geometry of
SNV-transformed spectra when the input is zero-mean and the direction is
nondegenerate.

Because it operates through tangent-space rotation, it avoids unconstrained additive noise that would otherwise change the norm and potentially the mean.

---

### Why keep augmentations weak?

The reconstruction target remains the unaugmented spectrum. Mild settings
retain a close neighborhood of that target while adding variation to the
visible input.

The intended role is:

$$
\text{reconstruction}+\text{weak denoising regularization}.
$$

Stronger settings change the reconstruction task and may alter which spectral
differences the representation retains. The appropriate strength depends on
the data and evaluation goal; the API defaults are not an empirically optimal
setting for every dataset.

---

## When to Use `SpectraAugmenter` in ChemoMAE Pipelines

### Use during training only

The module is intended for stochastic training-time augmentation.

In evaluation mode, it returns the input unchanged:

```python
augmenter.eval()
x_out = augmenter(x)
```

gives $\mathbf{x}_{\mathrm{out}} = \mathbf{x}$.

### Do not use for deterministic feature extraction

For latent extraction, spectra should be passed without stochastic augmentation.

For repeatable extraction with `Extractor`, omit the augmenter. Passing an
augmenter to `Extractor` or `Tester` explicitly activates it for each batch,
even if it was previously in evaluation mode; its original mode is restored.
Calling `augmenter.eval()` directly still makes standalone `augmenter(x)` an
identity operation. `Extractor` calls the model's all-visible `encode` API.

### Recommended after SNV preprocessing

This module assumes spectra are already SNV-normalized.

The recommended input geometry is:

$$
\mathbf{1}^{\top}\mathbf{x} \approx 0\quad\text{and}\quad\lVert \mathbf{x} \rVert_2 \approx r.
$$

---

## Common Pitfalls

### Confusing shift amount and noise angle

`shift_delta_range` controls wavelength-axis displacement in channel-index units.

`noise_angle_deg_range` controls the geodesic rotation angle for tangent Gaussian noise.

These parameters should not be interpreted as the same kind of perturbation strength.

### Applying augmentation before SNV

This module is designed for SNV-normalized spectra.

If it is applied before SNV, the geometric assumptions behind re-centering, re-normalization, and tangent-space rotation become less meaningful.

### Applying augmentation during evaluation

The module is inactive in `eval()` mode. This is intentional.

Feature extraction and validation should remain deterministic unless stochastic evaluation is explicitly intended.

### Treating this as contrastive multi-view augmentation

This module does not create paired views for contrastive learning.

It is designed as weak input corruption for reconstruction training. If later using view-consistency objectives, a separate two-view augmentation interface may be more appropriate.

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
    noise_prob=1.0,
    noise_angle_deg_range=(1.0, 1.0),
    shuffle_order_per_batch=True,
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

# angle from original should remain moderate for weak settings
cos = torch.sum(x * x_aug, dim=1) / (
    torch.linalg.norm(x, dim=1) * torch.linalg.norm(x_aug, dim=1) + 1.0e-8
)
cos = cos.clamp(-1.0, 1.0)
angle_deg = torch.rad2deg(torch.arccos(cos))

assert torch.all(angle_deg >= 0.0)
assert torch.all(angle_deg < 15.0)
```

---

## Version

### v0.2.1

* Documents use with masked and full-spectrum Trainer loss regions; augmentation behavior is unchanged.

### v0.2.0

* Updated `chemomae.training.augmenter` to use delta-controlled fractional shift and angle-controlled tangent Gaussian noise.
* Replaces the previous `noise + tilt` cosine-controlled design.
