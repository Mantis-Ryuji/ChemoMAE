# Extractor

Module: `chemomae.training.extractor`.

`Extractor` calls the public all-visible `ChemoMAE.encode` API for each input
batch. Use `iter_transform` to consume features without collecting the dataset,
or `transform` to return one array. `extract` and calling the extractor directly
are aliases for `transform`.

The detached features can feed clustering, a trainable downstream head, or a
caller-owned storage pipeline. The helper uses the model's current weights;
it does not select or load a checkpoint.

## Quick start

This CPU example compares batch streaming with aggregate output. The model has
illustrative random weights; load trained weights for an application.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from chemomae.models import ChemoMAE
from chemomae.training import Extractor, ExtractorConfig

model = ChemoMAE(
    seq_len=16, n_patches=4, n_mask=1, d_model=16, num_layers=1, latent_dim=4,
)
spectra = torch.randn(6, 16)
loader = DataLoader(TensorDataset(spectra), batch_size=2, shuffle=False)
extractor = Extractor(model, ExtractorConfig(representation="raw_latent"))
features = extractor.transform(loader)
streamed = torch.cat(list(extractor.iter_transform(loader)), dim=0)
assert features.shape == (6, 4) and features.device.type == "cpu"
assert not features.requires_grad and torch.isfinite(features).all()
torch.testing.assert_close(streamed, features)
assert model.training  # Each inference call restored the original mode.
```

## Configuration and defaults

| Setting | Default | Contract |
| --- | --- | --- |
| `device` | `None` | Follow the model's device when extraction starts; CPU and CUDA are supported. |
| `amp` | `False` | CUDA autocast is opt-in. CPU AMP is rejected. |
| `amp_dtype` | `"bf16"` | `"bf16"` or `"fp16"`; CUDA BF16 support is checked. |
| `representation` | `"latent"` | Select the output of `ChemoMAE.encode`. |
| `output_type` | `"tensor"` | `"tensor"` or `"numpy"`, for streaming and aggregate output. |
| `output_device` | `None` | Follow inference device for tensors; use CPU for NumPy. |
| `output_dtype` | `torch.float32` | Storage dtype: float16, bfloat16, float32, or float64. |
| `save_path` | `None` | Aggregate saving only; streaming ignores this setting. |
| `progress` | `False` | Show a tqdm progress bar when enabled. |

`device=None` does not automatically select CUDA. For explicit GPU inference,
set `device="cuda"` or move the model to CUDA before constructing the extractor.
An explicit device move persists after extraction. The helper restores modes,
but does not move model or augmenter parameters back to their original devices.

Inputs are cast to the model's floating-point parameter dtype. AMP controls
encoder arithmetic, while `output_dtype` controls the stored result. Converting
BF16 inference output to float64 cannot recover precision lost during inference.
With `amp=False`, ambient CPU and CUDA autocast are explicitly disabled for each
batch, rather than silently inherited from a surrounding context.

NumPy output requires a CPU output device and cannot represent bfloat16. The same
bfloat16 restriction applies to `.npy` saving, regardless of the returned type.

## Representation contract

| `representation` | Output | Width |
| --- | --- | --- |
| `"latent"` | Projected latent, normalized according to the model's `latent_normalize` setting | `latent_dim` |
| `"raw_latent"` | Projected latent before normalization | `latent_dim` |
| `"normalized_latent"` | Projected latent with L2 normalization, regardless of the model setting | `latent_dim` |
| `"cls"` | Encoder CLS output before latent projection | `d_model` |

For raw latent row $z_i$, the normalized output is

$$
\bar{z}_i = \frac{z_i}{\max(\lVert z_i \rVert_2, \epsilon)}.
$$

The implementation uses PyTorch's `F.normalize` contract, including its default
`eps`. All patches are visible; extraction does not draw random masks or run the
decoder. `ChemoMAE.encode` itself leaves gradients and modes under caller control;
`Extractor` supplies the inference scopes.

Normalized latent features support direction-based comparisons. They are not
constrained to have zero mean, and their cosine similarities need not equal
those in input space. Choose a representation explicitly and keep it fixed
when comparing downstream results or applying a saved model to new features.

## Stream features and keep memory bounded

```python
from contextlib import closing

extractor = Extractor(
    model,
    ExtractorConfig(
        device="cuda",
        output_device="cuda",
        representation="normalized_latent",
        output_dtype=torch.float32,
    ),
)

with closing(extractor.iter_transform(loader)) as batches:
    for feature_batch in batches:
        # Consume this batch on GPU or write it to your own sink.
        batch_mean = feature_batch.mean(dim=0)
        # A caller can stop early here; closing releases the progress bar.
```

`iter_transform` yields one output per input batch without a full-dataset feature
list or concatenation. Its feature storage is bounded by the current batches
and device transfers; the model, input loader/prefetching, and anything retained
by the consumer still use memory. An empty input batch yields an empty feature
batch, and an empty loader yields nothing. Streaming never writes `save_path`.

For a caller-owned NumPy writer, request CPU arrays explicitly:

```python
extractor = Extractor(
    model,
    ExtractorConfig(output_type="numpy", output_dtype=torch.float32),
)
for feature_batch in extractor.iter_transform(loader):
    # Pass feature_batch to your own writer. No dataset aggregation occurs here.
    assert feature_batch.ndim == 2
```

## Ordering, modes, and early termination

Each batch must be a floating-point tensor of shape `(B, model.seq_len)`, or a
tuple/list with that tensor first. Additional items such as labels or pixel IDs
are ignored. Dense finite inputs are required; overflow while casting to the
model dtype, nonfinite augmentation/encoder output, and overflow in the selected
storage dtype raise errors before saving or yielding that batch. The extractor
preserves exactly the order supplied by the iterable;
it does not sort or shuffle. Use `shuffle=False` and retain your own row/pixel
indices when rebuilding a spatial map. Loader worker behavior and split selection
remain the caller's responsibility.

Each nonempty batch temporarily sets the model to evaluation mode and disables
gradient recording. All model and augmenter training flags, including mixed
child modes, are restored before a feature batch is yielded and if encoding
raises an exception. Grad, inference, and autocast scopes also end before yield.
Early termination therefore leaves no suspended model-mode changes. Close the
iterator, preferably with `contextlib.closing`, to release its optional progress
bar promptly. The helper does not close caller-owned writers or loader resources.

Outputs are ordinary detached tensors, even when extraction is called inside an
inference-mode context. They can feed a trainable downstream head without cloning
an inference tensor. Encoder gradients are not recorded. For end-to-end supervised
fine-tuning, call `model.encode` directly under your own training/grad scopes.
Do not train or extract concurrently using the same model instance.

Without an augmenter, random masking and dropout are disabled. This supports
repeatable stream/aggregate comparisons under matching conditions; it does not
promise bitwise equality across devices, Torch versions, precision modes, or
nondeterministic kernels.

Passing `augmenter=...` explicitly enables augmentation in training mode only
while each batch is computed. This can be stochastic. Supply a caller-owned
default generator to `SpectraAugmenter` to control its random stream; it must
match the inference device. The extractor never seeds or replaces RNG state.
Callers must choose and record their augmentation/RNG protocol. An augmenter
must preserve the input tensor's shape.

## Aggregate output and saving

```python
extractor = Extractor(
    model,
    ExtractorConfig(
        output_type="numpy",
        output_dtype=torch.float64,
        representation="raw_latent",
        save_path="features/latent.npy",
    ),
)
features = extractor(loader)
```

`transform` holds all batches and then allocates their concatenation on
`output_device`, so it can require roughly twice the feature-array storage during
concatenation. Choose streaming when the dataset does not fit that budget.
An empty loader returns `(0, latent_dim)` or `(0, d_model)` with the configured
type/device/dtype.

A `.npy` destination saves a NumPy array; other suffixes save a CPU tensor with
`torch.save`. Saving creates parent directories and overwrites the requested
file. GPU aggregate output is copied to CPU for saving, but the returned tensor
stays on the configured device. The saved file contains features only; store
row IDs, preprocessing, weights, representation, precision, and protocol metadata
separately when they are needed to interpret it.

For changes from earlier releases, see the [changelog](../../CHANGELOG.md).
