# Standard Normal Variate: NumPy and native Torch

Module: `chemomae.preprocessing.snv`.

`snv` and `SNVScaler` normalize each spectrum independently. Torch operations
stay on the input device and preserve autograd; they do not detach the tensor,
convert it to NumPy, or move spectral data to CPU.

## Mathematical definition

For a spectrum $x_i$ with $L$ channels, define

$$
\mu_i = \frac{1}{L}\sum_{j=1}^{L}x_{ij},
\qquad
d = \begin{cases}1, & L \geq 2,\\0, & L=1,\end{cases}
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

Here $\varepsilon$ must be finite and strictly positive. This is **standard
deviation plus epsilon**, not a clamped standard deviation or a variance with
epsilon added before the square root. The returned `sd` statistic is $a_i$;
`inverse_transform` uses it directly without adding epsilon again.

Length-one and constant spectra produce zeros and retain their original mean
for inverse transformation. For nonconstant spectra with $L \geq 2$,

$$
\lVert y_i\rVert_2 = \sqrt{L-1}\frac{s_i}{s_i+\varepsilon}.
$$

The norm is therefore close to $\sqrt{L-1}$ only when $s_i$ is much larger than
$\varepsilon$. SNV does not produce unit-length vectors, and centering can change
angles between spectra. Apply a separate L2 normalization when a downstream
method requires unit vectors; constant spectra still have zero norm.

## Input and precision contract

Both APIs accept a floating NumPy array or dense Torch tensor with shape `(L,)`
or `(N, L)`. Normalize HSI cubes by reshaping the selected valid spectra into
`(N, L)` and preserving their pixel indices outside this helper.

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

Reductions subtract a per-spectrum anchor first to limit cancellation from a
common offset and to keep constant rows exactly zero. NumPy and Torch implement
the same definition and dtype policy, but reduction implementations can differ;
bitwise equivalence across frameworks or devices is not promised. Use
tolerance-based comparisons appropriate to the selected dtype. Float64 is useful
when channel differences are small relative to the input magnitude; values
already quantized in a float16/float32 input cannot be recovered by promotion.

Integer, bool, complex, unsupported floating dtypes, nonfinite values, empty
batches/spectra, and higher-rank inputs raise descriptive errors. An epsilon that
underflows to a zero scale, or nonfinite intermediate/output arithmetic, also
raises an error instead of returning invalid normalized values.

No input is modified. Noncontiguous arrays and tensors are supported.

## Functional API

```python
import numpy as np
from chemomae.preprocessing.snv import snv

spectra = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 5.0]], dtype=np.float32)
normalized = snv(spectra, eps=1e-12)
```

`snv(x, eps=1e-12)` returns the normalized spectra in the input framework, with
the precision policy above.

## Transformer API and inverse statistics

`SNVScaler(eps=1e-12, copy=True, transform_stats=False)` is stateless.
`fit(X, y=None)` validates the input and returns the scaler without learning
dataset statistics. `fit_transform(X, y=None)` applies the same operation as
`transform(X)`. The optional `y` is ignored. `get_params` and `set_params` expose
validated constructor parameters for transformer/pipeline conventions without
requiring scikit-learn at runtime.

`copy=True` makes an initial working copy. `copy=False` allows reusing the input
as the computation source when its dtype matches; neither setting writes into
the input, and normalization allocates an output.

With `transform_stats=False`, `transform` returns only normalized spectra. With
`transform_stats=True`, it returns `(normalized, mean, scale)`:

| Input shape | Mean and scale shapes |
| --- | --- |
| `(L,)` | `()`, zero-dimensional arrays/tensors |
| `(N, L)` | `(N, 1)` |

```python
from chemomae.preprocessing.snv import SNVScaler

scaler = SNVScaler(transform_stats=True)
normalized, mean, scale = scaler.fit_transform(spectra)
reconstructed = scaler.inverse_transform(normalized, mu=mean, sd=scale)
np.testing.assert_allclose(reconstructed, spectra, rtol=1e-6, atol=1e-6)
```

For inverse transformation, statistics must use the same framework and, for
Torch, the same device as the spectra. Scalars can describe a shared mean/scale.
Array/tensor statistics must have the shapes above; shape `(1,)` is also accepted
for a 1D spectrum. A 2D input does not accept an ambiguous `(N,)` statistic that
could accidentally broadcast over channels. Scales must be finite and positive.

## Torch, CUDA, and gradients

```python
import torch
from chemomae.preprocessing.snv import SNVScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spectra = torch.tensor(
    [[1.0, 2.0, 4.0], [3.0, 1.0, 5.0]],
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)
scaler = SNVScaler(transform_stats=True)
normalized, mean, scale = scaler.transform(spectra)
reconstructed = scaler.inverse_transform(normalized, mu=mean, sd=scale)
loss = normalized[:, 0].sum()
loss.backward()
```

Normalization, mean, and scale participate in the Torch computation graph.
Users who need detached features should detach explicitly at their own boundary.
Validity checks inspect scalar conditions and can synchronize a CUDA device;
spectral arrays and statistics are never transferred to host implicitly.

Time and temporary memory are $O(NL)$, including working/centered/output arrays;
per-spectrum statistics occupy $O(N)$. This helper is not a streaming loader.
Call it batch by batch when the full array does not fit on the computation device.

## Workflow notes

SNV computes each spectrum's own statistics, so `fit` does not estimate cohort
statistics to reuse on held-out data. Split and specimen/group decisions remain
with the consuming workflow. Keep wavelengths and masks aligned with spectra,
and do not apply SNV a second time to supplied SNV arrays unintentionally.

The redesigned precision/statistics contract differs from v0.2.2: float64 output
is no longer reduced through float32, half output is promoted, and Torch
statistics are tensors rather than NumPy arrays. Historical experiment artifacts
are not converted by this helper.

## Verification

Focused tests are in `tests/preprocessing/test_snv.py`, covering the independent
sample-standardization definition, backend/dtype agreement, constant/length-one
spectra, round trips, gradients, native Torch execution, validation, and optional
CUDA device checks. Run the checks from the project root:

```powershell
pytest tests/preprocessing/test_snv.py -q
```

These tests and the GitHub/Colab rendering checks must be run in their target
environments; source changes alone do not establish numerical or CUDA validation.
