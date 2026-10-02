# Local Label Agreement — Spatial Cluster Evaluation

> Module: `chemomae.clustering.spatial`

`local_label_agreement` evaluates a spatial label map using the occupancy-corrected
LLA in Thesis equation (11). It uses binary class-map convolutions on CPU or CUDA.
The labels need not be consecutive, and label zero is a valid class. The evaluated
region is always supplied explicitly as `valid_mask`.

## Example

```python
import numpy as np
from chemomae.clustering import local_label_agreement

labels = np.array([[0, 0, 0, 8]], dtype=np.int64)
valid = np.ones_like(labels, dtype=bool)
result = local_label_agreement(labels, valid, windows=(3, 5, 9), device="cpu")

print(result.class_labels, result.class_counts)  # (0, 8), (3, 1)
print(result.chance_agreement)                  # 0.5
for window in result.windows:
    print(window.window, window.score, window.raw_agreement,
          window.valid_pairs, window.undefined_reasons)
```

The hand-counted corrected scores for this example are $1/3$, $1/5$, and $0$.
Raw agreement is returned separately. Do not report raw agreement as the
equation-(11) LLA or average window scores implicitly.

For an existing CUDA label map, omit `device` to compute on that device:

```python
import torch

if torch.cuda.is_available():
    result = local_label_agreement(
        torch.as_tensor(labels, device="cuda"),
        torch.as_tensor(valid, device="cuda"),
        class_chunk=4,
    )
```

This example is synthetic. Real LLA requires actual spatial neighbors from one
image. Reordering tabular spectra into a grid does not create spatial evidence.

## Definition and convolution

Let $V$ be the valid-pixel set, $N=|V|$, and $n_k$ the count of class $k$ in $V$.
Define $M(p)=1$ on $V$ and zero elsewhere, and $B_k(p)=1$ when $p\in V$ has label
$k$. For an odd square width $w$, $K_w$ is one at every kernel position except
its zero-valued center. Out-of-image positions are zero-padded.

The total valid directed pairs $D_w$, matching directed pairs $Q_w$, raw agreement
$A_w$, and finite-sample chance agreement $P$ are:

$$
D_w = \sum_p M(p)\,(K_w * M)(p),
\qquad
Q_w = \sum_k \sum_p B_k(p)\,(K_w * B_k)(p).
$$

$$
A_w = \frac{Q_w}{D_w},
\qquad
P = \frac{\sum_k n_k(n_k-1)}{N(N-1)},
\qquad
\operatorname{LLA}_w = \frac{A_w-P}{1-P}.
$$

Each directed neighbor pair has equal weight. This is not an average of
per-pixel agreement fractions: edge pixels and holes have fewer neighbors.
Diagonals are included, centers are excluded, and image edges never wrap.
Excluded pixels cannot be either end of a counted pair. Isolated valid pixels
still contribute to $N$, occupancy, and $P$. The finite-sample correction is not
the approximation $\sum_k(n_k/N)^2$.

## Input contract

```python
local_label_agreement(
    labels, valid_mask, *, windows=(3, 5, 9), device=None, class_chunk=16,
)
```

| Input | Contract |
| --- | --- |
| `labels` | Two-dimensional NumPy integer array or Torch integer tensor. IDs fit signed int64; Torch supports uint8 and signed int8/int16/int32/int64. Zero, negative, and nonconsecutive IDs are valid. |
| `valid_mask` | Matching two-dimensional boolean or binary-integer array/tensor. At least one pixel must be valid. No background ID is inferred. |
| `windows` | Nonempty sequence of distinct positive odd widths, retained in caller order. Width 1 has no neighbor pairs. |
| `device` | CPU or CUDA. `None` follows a labels tensor, otherwise a mask tensor, otherwise CPU. Explicit unavailable CUDA is rejected. |
| `class_chunk` | Positive maximum number of binary class maps processed together. `None` processes all used classes together. |

No inputs are modified. There is no fitting, randomness, file I/O, or automatic
aggregation across maps. Very large widths that exceed exact FP32 local-count
capacity, or pair-count bounds exceeding int64, are rejected.

## Results and undefined values

`LLAResult` is an immutable dataclass containing Python scalars and tuples.

| Image field | Meaning |
| --- | --- |
| `valid_pixels` | $N$, including isolated pixels. |
| `class_labels`, `class_counts`, `occupancy` | Sorted original IDs, $n_k$, and $n_k/N$, in matching order. |
| `used_classes`, `maximum_occupancy` | Number of present classes and largest occupancy. |
| `chance_agreement` | $P$; NaN when $N<2$. |
| `windows` | One `LLAWindowResult` per requested width; no automatic average. |

Each window result includes `window`, corrected `score`, `raw_agreement`, integer
`matching_pairs` and `valid_pairs`, `pixels_with_neighbors`,
`neighbor_pixel_fraction`, `raw_undefined_reason`, and `undefined_reasons`.
The neighbor fraction measures how many valid pixels have at least one neighbor;
it does not alter occupancy or pair weights.

Corrected LLA is NaN for fewer than two valid pixels, no valid neighbor pairs, or
a single occupied class. Applicable reason strings are returned together:
`"fewer_than_two_valid_pixels"`, `"no_valid_neighbor_pairs"`, and `"single_class"`.
Raw agreement is NaN only when no valid pair exists. A single-class map with
neighbors has raw agreement 1 and undefined corrected LLA. An empty mask is an
input error. Defined negative LLA is retained without clipping.

LLA describes spatial coherence conditioned on occupancy. It does not establish
chemical correctness, semantic label accuracy, or the best clustering setting.
Specify window widths and undefined-value aggregation in the experiment protocol.

## Precision and memory

`conv2d` computes binary local counts in FP32 with ambient autocast and cuDNN TF32
disabled within a scoped context. Local results are checked for integrality,
converted to int64, and summed as integers. A noninteger local result raises an
error instead of silently rounding it into a count.

Compact counts are transferred to the CPU. Final correction uses Python's
unbounded integer products before the final division:

$$
T=N(N-1),\qquad S=\sum_k n_k(n_k-1),\qquad
\operatorname{LLA}_w = \frac{Q_wT-D_wS}{D_w(T-S)}.
$$

This avoids int64 product overflow and cancellation from subtracting a rounded
chance probability near one. Output scores are Python floats, rather than the
FP32 final division used by the original experiment helper. The separately
reported float `chance_agreement` can round to one under extreme imbalance;
the score's definedness is determined from exact class counts.

Class chunking bounds temporary class-map storage by the chunk size, not the
complete computation's memory. Full labels/masks and dense class indices remain
resident, and convolution workspace is backend-dependent. No `unfold` tensor
of shape proportional to image pixels times neighborhood area is constructed.
Image tiling is not implemented. CUDA speed and memory benchmarks remain pending;
small images can favor CPU once transfer overhead is included.

The focused tests contain an independent pixel-pair enumeration oracle, boundary
and mask cases, finite-sample and negative-score checks, class-chunk invariance,
large-integer finalization, and optional CPU/CUDA comparison. They have not been
executed as part of this implementation update.
