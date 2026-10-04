# Standard Normal Variate

Module: `chemomae.preprocessing.snv`.

`snv` and `SNVScaler` standardize each spectrum independently using its own mean
and sample standard deviation. Choose SNV when removing per-spectrum offset and
scale is appropriate for the application. It is an optional preprocessing step;
ChemoMAE and the other package components do not apply it automatically.

Both NumPy and native Torch are supported. Torch operations stay on the input
device and preserve autograd, without converting spectral data to NumPy or CPU.

## Quick start

This CPU example standardizes two spectra, reconstructs them from returned
statistics, and checks agreement between NumPy and Torch.

```python
import numpy as np
import torch
from chemomae.preprocessing import SNVScaler, snv

spectra = np.array(
    [[1.0, 2.0, 4.0], [3.0, 1.0, 5.0]], dtype=np.float32,
)
scaler = SNVScaler(transform_stats=True)
normalized, mean, scale = scaler.fit_transform(spectra)
restored = scaler.inverse_transform(normalized, mu=mean, sd=scale)
torch_normalized = snv(torch.from_numpy(spectra))

assert normalized.shape == spectra.shape
assert normalized.dtype == np.float32
assert mean.shape == scale.shape == (2, 1)
assert torch_normalized.device.type == "cpu"
np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
np.testing.assert_allclose(restored, spectra, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(torch_normalized.numpy(), normalized, rtol=1e-6, atol=1e-6)
```

## API contract

| Entry point | Behavior |
| --- | --- |
| `snv(x, eps=1e-12)` | Return standardized spectra in the input framework. |
| `SNVScaler(eps=1e-12, copy=True, transform_stats=False)` | Stateless transformer with optional inverse statistics. |
| `fit(X, y=None)` | Validate input and epsilon, then return the scaler; learn no dataset statistics. |
| `transform(X)` | Return normalized spectra, or `(normalized, mean, scale)` when `transform_stats=True`. |
| `fit_transform(X, y=None)` | Apply the same operation as `transform`; `y` is ignored. |
| `inverse_transform(Y, *, mu, sd)` | Reconstruct using the supplied mean and effective scale. |
| `get_params(deep=True)`, `set_params(**params)` | Expose validated constructor parameters for transformer/pipeline conventions. |

`eps` must be finite and strictly positive. The denominator is **standard
deviation plus epsilon**. The returned statistic named `scale` is `std + eps`;
pass it as `sd` to `inverse_transform`, which does not add epsilon again.

`copy=True` makes an initial working copy. `copy=False` allows reusing the input
as the computation source when its dtype matches; neither setting writes into
the input, and normalization allocates an output. There is no scikit-learn runtime
dependency.

### Input, dtype, and device

Both APIs accept a floating NumPy array or dense Torch tensor with shape `(L,)`
or `(N, L)`. Higher-dimensional data must be reshaped by the caller. For example,
an application using a spectral image can select valid spectra into `(N, L)`
while retaining their pixel indices separately.

| Input dtype | Computation, output, and statistics dtype |
| --- | --- |
| NumPy/Torch float64 | float64 |
| NumPy/Torch float32 | float32 |
| NumPy/Torch float16 | float32 |
| Torch bfloat16 | float32 |

Torch output and statistics remain on the input device. NumPy output and
statistics remain NumPy arrays. Half inputs are promoted so that small positive
scales, including the default epsilon, do not disappear into half precision.
Cast explicitly afterward if a later operation needs a different dtype.

Integer, bool, complex, unsupported floating dtypes, nonfinite values, empty
batches/spectra, and higher-rank inputs raise descriptive errors. An epsilon that
underflows to a zero scale, or nonfinite intermediate/output arithmetic, also
raises an error. Noncontiguous arrays and tensors are supported.

### Inverse statistics

| Spectrum shape | Returned mean and scale shapes |
| --- | --- |
| `(L,)` | `()`, zero-dimensional arrays/tensors. |
| `(N, L)` | `(N, 1)`. |

For inverse transformation, array/tensor statistics must use the same framework
and, for Torch, the same device as the spectra. Real scalars can describe a shared
mean or scale. Array/tensor statistics must have the shapes above; shape `(1,)`
is also accepted for a 1D spectrum. A 2D input does not accept an ambiguous `(N,)`
statistic that could accidentally broadcast over channels. Scales must be finite
and positive; means must be finite. Statistics are cast to the inverse
computation dtype using the precision policy above.

The transformer does not retain per-spectrum statistics between calls. Preserve
the returned mean and scale if inverse transformation will be needed later.

## Mathematical definition

For a spectrum $x_i$ with $L$ channels, define

$$
\mu_i = \frac{1}{L}\sum_{j=1}^{L}x_{ij},
\qquad
d = \begin{cases}
1, & L \geq 2, \\
0, & L=1,
\end{cases}
\qquad
s_i = \sqrt{\frac{1}{L-d}\sum_{j=1}^{L}(x_{ij}-\mu_i)^2}.
$$

The normalized spectrum and effective scale are

$$
a_i = s_i + \varepsilon,
\qquad
y_{ij} = \frac{x_{ij}-\mu_i}{a_i},
\qquad
x_{ij} = y_{ij}a_i + \mu_i.
$$

The implementation adds epsilon to the standard deviation, rather than clamping
it or adding epsilon to the variance. Length-one and constant spectra produce
zeros and retain their original mean for inverse transformation. The inverse
identity is subject to floating-point rounding in actual computation.

### Geometry

For nonconstant spectra with $L\geq2$,

$$
\lVert y_i\rVert_2 = \sqrt{L-1}\frac{s_i}{s_i+\varepsilon}.
$$

The norm is close to $\sqrt{L-1}$ only when $s_i$ is much larger than epsilon.
SNV does not produce unit-length vectors, and centering can change angles between
spectra. Apply a separate L2 normalization when a downstream method requires
unit vectors; constant spectra still have zero norm.

In the idealized nonconstant case without epsilon, mean removal places spectra
in the zero-mean hyperplane, and sample-standard-deviation scaling places them
on a common-radius sphere within that hyperplane. The implementation retains
this interpretation approximately, with the norm correction above. Degenerate
zero outputs lie outside that nonzero sphere.

This geometry can be useful when comparing spectral shape by direction. It does
not mean that offset and scale are irrelevant to every application. ChemoMAE's
optional latent normalization is a separate operation: latents have no zero-mean
constraint, and their cosine similarities need not match those of the inputs.

## Numerical behavior and Torch gradients

Reductions subtract a per-spectrum anchor first to limit cancellation from a
common offset and to keep constant rows exactly zero. NumPy and Torch implement
the same definition and dtype policy, but reduction implementations can differ;
bitwise equivalence across frameworks or devices is not promised. Use
tolerance-based comparisons appropriate to the selected dtype. Float64 is useful
when channel differences are small relative to the input magnitude; values
already quantized in a float16/float32 input cannot be recovered by promotion.

Normalization, mean, and scale participate in the Torch computation graph. This
self-contained CPU example propagates a gradient through one normalized channel:

```python
import torch
from chemomae.preprocessing import SNVScaler

spectra = torch.tensor(
    [[1.0, 2.0, 4.0], [3.0, 1.0, 5.0]],
    dtype=torch.float32,
    requires_grad=True,
)
scaler = SNVScaler(transform_stats=True)
normalized, mean, scale = scaler.transform(spectra)
loss = normalized[:, 0].sum()
loss.backward()
assert spectra.grad is not None and torch.isfinite(spectra.grad).all()
```

Detach explicitly when consuming the result as fixed features. For CUDA tensors,
validity checks inspect scalar conditions and can synchronize the device;
spectral arrays and statistics are not implicitly transferred to the host.

Time and temporary memory are $O(NL)$, including working, centered, and output
arrays; per-spectrum statistics occupy $O(N)$. This helper is not a streaming
loader. Apply it batch by batch when the full array does not fit on the device.

## Composition notes

SNV learns no cohort statistics in `fit`, so held-out spectra use their own row
statistics. Dataset splits and group independence remain application decisions.
Keep channel order and masks aligned with the spectra, and avoid accidentally
standardizing arrays that were already transformed.

If SNV is chosen, its output can be passed to [FPS](dowmsampling.md), a spectral
model, or another compatible downstream method. Some consumers normalize rows
internally; others require unit input explicitly. Follow the consumer's own
contract rather than treating SNV as interchangeable with L2 normalization.

## Related references

- [ChemoMAE](../models/chemo_mae.md)
- [Spectral augmentation](../training/augmenter.md)
- [Cosine farthest-point sampling](dowmsampling.md)
- Implementation checks: `tests/preprocessing/test_snv.py`
