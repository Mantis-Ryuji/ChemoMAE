# Spectral augmentation

> API reference for ChemoMAE v0.2.4.

Module: `chemomae.training.augmenter`.

`SpectraAugmenter` applies fractional channel shifts and tangent Gaussian noise
to floating spectral batches. It is designed for zero-mean spectra, such as
[SNV output](../preprocessing/snv.md). Optional re-centering and re-normalization
preserve that geometry. Use it with the training helpers or in a caller-owned
PyTorch pipeline; perturbation strengths are application choices.

## Quick start

This small CPU example checks shape, mean, and norm for combined augmentation.
The separate noise-only check tests its configured angle: fractional shift has
no angular cap, even when its channel displacement is small.

```python
import torch
from chemomae.training import SpectraAugmenter, SpectraAugmenterConfig

stream = torch.Generator(device="cpu").manual_seed(42)
x = torch.randn(4, 16, dtype=torch.float64, generator=stream)
x = x - x.mean(dim=1, keepdim=True)
x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
augmenter = SpectraAugmenter(
    SpectraAugmenterConfig(shift_prob=1.0, noise_prob=1.0),
    generator=stream,
).train()
augmented = augmenter(x)
assert augmented.shape == x.shape
torch.testing.assert_close(augmented.mean(dim=1), torch.zeros(4, dtype=x.dtype),
                           rtol=0, atol=1e-10)
torch.testing.assert_close(torch.linalg.vector_norm(augmented, dim=1),
                           torch.linalg.vector_norm(x, dim=1), rtol=1e-10, atol=1e-10)

noise_only = SpectraAugmenter(
    SpectraAugmenterConfig(shift_prob=0.0, noise_prob=1.0,
                           noise_angle_deg_range=(2.0, 2.0)),
    generator=torch.Generator(device="cpu").manual_seed(7),
).train()
noise_augmented = noise_only(x)
cosine = (x * noise_augmented).sum(dim=1) / (
    torch.linalg.vector_norm(x, dim=1) * torch.linalg.vector_norm(noise_augmented, dim=1)
)
angle_deg = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
torch.testing.assert_close(angle_deg, torch.full_like(angle_deg, 2.0),
                           rtol=0, atol=1e-6)
augmenter.eval()
assert augmenter(x) is x
```

Unit-norm rows make this example compact. The augmenter retains each row's
input norm; it does not require or impose a common unit radius.

## Configuration and defaults

| Setting | Default | Contract |
| --- | --- | --- |
| `shift_prob` | `0.5` | Per-spectrum probability of fractional shift; in `[0, 1]` |
| `shift_delta_range` | `(-2.0, 2.0)` | Uniform displacement in channel-index units; finite, ordered endpoints |
| `noise_prob` | `0.5` | Per-spectrum probability of tangent Gaussian noise; in `[0, 1]` |
| `noise_angle_deg_range` | `(0.5, 3.0)` | Uniform rotation angle in degrees; ordered endpoints within `[0, 180]` |
| `shuffle_order_per_batch` | `True` | Randomize the two operations' order once per batch |
| `recenter_after_each_op` | `True` | Subtract the candidate row mean after each applied operation |
| `renorm_to_input_norm` | `True` | Restore the input row norm after each applied operation |
| `eps` | `1e-8` | Finite positive threshold for normalization and degenerate directions |

`SpectraAugmenterConfig` is a frozen dataclass and validates probabilities,
intervals, and epsilon at construction. The library defaults specify a usable
configuration, not an optimal perturbation protocol for every dataset.

```python
class SpectraAugmenter(torch.nn.Module):
    def __init__(
        self, config: SpectraAugmenterConfig, *,
        generator: torch.Generator | None = None,
    ) -> None: ...

    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...
```

## Input, output, and modes

Input must be a floating Torch tensor of shape `(B, L)`. Training mode requires
`B >= 1` and `L >= 2`, even when both application probabilities are zero.
Evaluation mode validates the tensor's rank and floating dtype, then returns
`x` itself without those size checks or random draws. An active call returns
the same shape, dtype, and device as its input.

The intended geometric input is zero-mean with nonzero row norms. The module
does not check whether SNV was applied, nor does it validate input finiteness.
Callers using it directly own those checks. Constant/zero-norm spectra and
degenerate tangent directions do not have the ordinary spherical-angle contract.

With `shuffle_order_per_batch=False`, the order is shift then noise. Otherwise
the order is drawn once per batch, while each operation has its own independent
per-spectrum application mask and strength draws. Reprojection follows each
applied operation. Rows not selected for an operation remain unchanged.

For float16/bfloat16 input, shift amounts, channel coordinates, and interpolation
arithmetic use float32 before casting back. Float32/float64 input retains its
corresponding shift arithmetic. This avoids losing fractional channel positions;
it does not make all augmentation arithmetic float32. Tangent-noise arithmetic
uses the input dtype, and geometric identities hold up to numerical precision.

## Random streams and continuation

Every random draw uses the resolved `torch.Generator`: operation order,
application masks, displacements, Gaussian directions, and noise angles.
A per-call generator overrides the module default. `generator=None` selects the
module default; when neither is supplied, draws use the input device's global
Torch stream. The augmenter does not seed or replace global RNG state.

```python
# Reuse x and augmenter from Quick start.
augmenter.train()
state = stream.get_state().clone()
expected_next = augmenter(x)
stream.set_state(state)
repeated_next = augmenter(x)
torch.testing.assert_close(repeated_next, expected_next, rtol=0, atol=0)

other_stream = torch.Generator(device=x.device).manual_seed(73)
other_view = augmenter(x, generator=other_stream)
```

An active stream must match the input device type and CUDA index; mismatches
raise an error. `augmenter.to(device)` does not move a generator. Create a new
stream on the destination device explicitly. Evaluation mode does not advance
either the default or override generator.

Generator state is caller-owned and absent from `augmenter.state_dict()`.
Persist it separately, or use [Trainer extension hooks](trainer.md#custom-ordering-masks-and-caller-state).
Matching a seed or snapshot reproduces a sequence only under matching inputs,
configuration, modes, batch partitioning, call order, dtype, device, and software.
Skipped operations and degenerate directions can change draw consumption.
Use separate generators when masking, augmentation, and sampling must have
independent sequences. See [RNG utilities](../utils/seed.md).

## Geometry and reprojection

For a nonconstant spectrum of length $L$, ideal SNV with sample standard
deviation and no epsilon gives zero mean and norm $r=\sqrt{L-1}$. These rows lie
on a sphere inside the zero-mean hyperplane:

$$
\mathcal{M}_r=\left\lbrace\mathbf{x}\in\mathbb{R}^L\quad\middle|\quad
\mathbf{1}^{\top}\mathbf{x}=0,\quad\lVert\mathbf{x}\rVert_2=r\right\rbrace.
$$

The library's SNV adds epsilon to the standard deviation, so actual norms are
only approximately common. The augmenter uses each input row's own norm.
SNV outputs for constant or length-one spectra have zero norm and lie outside
the nonzero-sphere assumption.

For an input row $\mathbf{x}$ and candidate $\mathbf{y}$, the enabled reprojection
steps are:

$$
\mathbf{y}\leftarrow\mathbf{y}-\frac{\mathbf{1}^{\top}\mathbf{y}}{L}\mathbf{1},
$$

$$
\mathbf{y}\leftarrow\lVert\mathbf{x}\rVert_2
\frac{\mathbf{y}}{\lVert\mathbf{y}\rVert_2}.
$$

These expressions describe nondegenerate rows. The implementation clamps norm
denominators by `eps`; if the candidate norm is at most `eps`, re-normalization
returns the reference row. For ordinary zero-mean, nonzero-norm inputs, the two
steps restore zero mean and the input norm up to rounding. For a nonzero-mean
input, re-centering deliberately changes the mean, and fallback may retain it.

### Fractional shift

For each selected row, draw $\delta\sim\mathcal{U}(\delta_{\min},\delta_{\max})$.
Using zero-based channel indices $\ell=0,\ldots,L-1$, define

$$
s_\ell=\ell-\delta,\qquad
\alpha_\ell=s_\ell-\lfloor s_\ell\rfloor.
$$

Linear interpolation gives

$$
y_\ell=(1-\alpha_\ell)x_{\lfloor s_\ell\rfloor}
+\alpha_\ell x_{\lfloor s_\ell\rfloor+1},
$$

with both source indices clamped to $[0,L-1]$. Positive displacement moves
features toward larger output indices. Enabled reprojection then restores the
specified mean/norm geometry.

Displacement is measured in channel units. It represents a constant wavelength
displacement only on an equally spaced wavelength grid. Interpolation and
endpoint clamping are part of the transformation, so suitable strengths depend
on the grid and spectral shape. There is no cosine or angular cap: a small
channel shift can produce a large angular change, especially for rapidly
varying input. Angle-limited shift from earlier designs is not this operation.

### Tangent Gaussian noise

For a zero-mean input $\mathbf{x}$ with norm $r>0$, draw an ambient Gaussian
direction and remove its mean and component parallel to the input:

$$
\mathbf{g}\sim\mathcal{N}(\mathbf{0},I_L),\qquad
\tilde{\mathbf{g}}=\mathbf{g}-\frac{\mathbf{1}^{\top}\mathbf{g}}{L}\mathbf{1},
$$

$$
\mathbf{v}=\tilde{\mathbf{g}}-
\frac{\tilde{\mathbf{g}}^{\top}\mathbf{x}}{\lVert\mathbf{x}\rVert_2^2}\mathbf{x},
\qquad\mathbf{u}=\frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2}.
$$

For a valid direction, $\mathbf{u}$ is both zero-mean and orthogonal to
$\mathbf{x}$. Draw an angle uniformly from `noise_angle_deg_range`, convert it
to radians, and rotate:

$$
\mathbf{y}=r\left(\cos(\theta)\frac{\mathbf{x}}{r}
+\sin(\theta)\mathbf{u}\right).
$$

Before numerical reprojection, this preserves the norm and gives angular
separation $\theta$ for the stated nondegenerate geometry. Optional reprojection
follows. A tangent direction with norm at most `eps` is not rotated; the code
also uses epsilon guards in the projection and normalization. The angle
identity should not be assumed for zero or near-zero rows or degenerate
directions, including the zero-dimensional tangent space of a two-channel
zero-mean sphere.

The Gaussian draw defines a direction, not independent additive Gaussian error
on every output channel. The angle controls TGN alone, not the angular distance
of a composition with fractional shift.

## Integration and choosing a perturbation task

[Trainer](trainer.md) augments the complete input before masking and keeps the
pre-augmentation spectrum as its reconstruction target. With `loss_region="masked"`,
the selected hidden channels define the loss; `"all"` compares every output
channel. Use `n_mask=0` as well for an all-visible denoising autoencoder.
The unaugmented target can still contain measurement variation.

Because shifting precedes masking, interpolation can move information across
masked-band boundaries. That behavior is part of the configured prediction
task. The operations offer controlled changes of spectral shape; they are not
a calibrated instrument-error model or a guarantee of chemical-state preservation.
Choose their strengths against the application's objective and data.

Calling `augmenter.eval()` makes standalone calls identity operations.
[Tester](tester.md) and [Extractor](extractor.md) explicitly activate a supplied
augmenter temporarily and then restore its mode. Omit it for unperturbed
evaluation or feature extraction. With stochastic evaluation, record random
streams, repetitions, and batch partitioning alongside the configuration.

The module returns one transformed batch per call. A caller-owned loop may make
multiple calls to create views or use other objectives; it owns how views,
targets, gradients, and losses are combined.
