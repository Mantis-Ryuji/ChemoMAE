# Extractor — Latent Feature Extraction

> Module: `chemomae.training.extractor`

This document describes the `Extractor` and its configuration (`ExtractorConfig`), which provide an efficient way to extract latent embeddings (`Z`) from trained **ChemoMAE** models under **all-visible mode**.

The current `Extractor` also supports an optional `SpectraAugmenter`.  
When an augmenter is provided, augmentation is applied to the input spectrum before latent feature extraction.

---

## Overview

The `Extractor` obtains **latent representations** from a trained ChemoMAE encoder without ChemoMAE masking.

These embeddings can be used for downstream analysis such as:

* clustering (`CosineKMeans`, vMF Mixture, etc.),
* dimensionality reduction (`UMAP`, `t-SNE`, PCA),
* representation inspection,
* segmentation pipelines,
* downstream evaluation.

### Key features

* **All-visible encoding** — uses an all-ones visible mask instead of random MAE masking.
* **Optional `SpectraAugmenter` support** — applies spectral augmentation before encoding when provided.
* **AMP support** — accelerated inference with `bf16` or `fp16` on CUDA.
* **Flexible output** — returns `torch.Tensor` or `numpy.ndarray`.
* **Optional saving** — results can be automatically written to disk (`.npy` or `.pt`).
* **CPU aggregation** — extracted features are detached and moved to CPU before concatenation.

### Determinism

Without an augmenter, extraction is deterministic with respect to ChemoMAE masking because the extractor always uses an all-visible mask.

With an augmenter, extraction may become stochastic, because `SpectraAugmenter` can sample random shift/noise perturbations.  
This is intentional when augmented feature extraction is requested.

---

## Configuration — `ExtractorConfig`

```python
@dataclass
class ExtractorConfig:
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    save_path: Optional[str | Path] = None
    return_numpy: bool = False
```

### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | `str` or `torch.device` | `"cuda"` | Device used for feature extraction. |
| `amp` | `bool` | `True` | Enables automatic mixed precision during CUDA extraction. |
| `amp_dtype` | `"bf16"` or `"fp16"` | `"bf16"` | Precision type for autocast. `bf16` is recommended on recent GPUs. |
| `save_path` | `str`, `Path`, or `None` | `None` | Optional output path. If given, results are saved after extraction. |
| `return_numpy` | `bool` | `False` | If `True`, returns `np.ndarray`; otherwise returns `torch.Tensor`. |

---

## Class: `Extractor`

### Initialization

```python
extractor = Extractor(
    model,
    cfg=ExtractorConfig(),
    augmenter=None,
)
```

| Argument | Description |
| --- | --- |
| `model` | Trained ChemoMAE model. Must implement `encoder(x, visible_mask) -> z`. |
| `cfg` | Optional configuration controlling device, AMP, saving, and return format. |
| `augmenter` | Optional `SpectraAugmenter` applied before encoder inference. |

The model is moved to `cfg.device` and set to `eval()` during extraction.

If `augmenter` is provided, it is also moved to `cfg.device`.

---

## Call Interface

```python
Z = extractor(loader)
```

The extractor iterates through batches from `loader`.

For each batch:

1. Extracts `x` from the batch.
2. Moves `x` to `cfg.device`.
3. Applies optional augmentation:

   ```python
   x_input = augmenter(x) if augmenter is not None else x
   ```

4. Constructs an all-ones visible mask:

   ```python
   visible_mask = torch.ones(B, L, dtype=torch.bool, device=device)
   ```

5. Calls:

   ```python
   z = model.encoder(x_input, visible_mask)
   ```

6. Detaches `z`, casts it to `float32`, moves it to CPU, and appends it to the feature list.

After all batches are processed, all features are concatenated along the sample dimension.

---

## Augmenter Handling

`Extractor` supports optional `SpectraAugmenter`:

```python
from chemomae.training import SpectraAugmenter, SpectraAugmenterConfig

aug_cfg = SpectraAugmenterConfig(
    shift_prob=0.5,
    shift_delta_range=(-2.0, 2.0),
    noise_prob=0.5,
    noise_angle_deg_range=(0.5, 3.0),
)

augmenter = SpectraAugmenter(aug_cfg)

extractor = Extractor(
    model,
    cfg,
    augmenter=augmenter,
)
```

### Important mode behavior

`SpectraAugmenter` is implemented as an `nn.Module`.

It returns the input unchanged when it is in `eval()` mode.  
Therefore, when an augmenter is provided, `Extractor` temporarily sets only the augmenter to `train()` during extraction:

```python
model.eval()
augmenter.train()
```

After extraction, the original train/eval state of the augmenter is restored.

This means:

* the ChemoMAE model remains in evaluation mode,
* the augmenter is active during extraction,
* the augmenter state is restored after extraction.

### Target semantics

`Extractor` only extracts latent features and does not compute reconstruction loss.  
Thus, unlike `Trainer` and `Tester`, there is no reconstruction target.

The augmented spectrum is used only as the encoder input:

```python
z = model.encoder(x_input, visible_mask)
```

---

## Saving Behavior

If `cfg.save_path` is specified:

| File extension | Behavior |
| --- | --- |
| `.npy` | Saved using `numpy.save(save_path, Z.numpy())`. |
| otherwise | Saved using `torch.save(Z, save_path)`. |

Parent directories are created automatically.

The return value is controlled only by `cfg.return_numpy`, not by the saving format.

For example:

```python
cfg = ExtractorConfig(save_path="latent.npy", return_numpy=False)
Z = extractor(loader)
```

This saves `latent.npy` but still returns a `torch.Tensor`.

---

## Usage Examples

### Extract features to memory

```python
from chemomae.training import Extractor, ExtractorConfig

cfg = ExtractorConfig(
    device="cuda",
    amp=True,
    amp_dtype="bf16",
    return_numpy=True,
)

extractor = Extractor(model, cfg)
Z = extractor(loader)  # np.ndarray, shape (N, D)
```

### Save features to disk

```python
cfg = ExtractorConfig(
    device="cuda",
    amp=True,
    save_path="latent.npy",
    return_numpy=False,
)

extractor = Extractor(model, cfg)
Z = extractor(loader)  # torch.Tensor

# latent.npy is written to disk.
```

### Save as Torch tensor

```python
cfg = ExtractorConfig(
    device="cuda",
    save_path="latent.pt",
)

extractor = Extractor(model, cfg)
Z = extractor(loader)

# latent.pt contains a torch.Tensor.
```

### Extract augmented features

```python
from chemomae.training import (
    Extractor,
    ExtractorConfig,
    SpectraAugmenter,
    SpectraAugmenterConfig,
)

aug_cfg = SpectraAugmenterConfig(
    shift_prob=0.5,
    shift_delta_range=(-2.0, 2.0),
    noise_prob=0.5,
    noise_angle_deg_range=(0.5, 3.0),
    recenter_after_each_op=True,
    renorm_to_input_norm=True,
)

augmenter = SpectraAugmenter(aug_cfg)

cfg = ExtractorConfig(
    device="cuda",
    amp=True,
    amp_dtype="bf16",
    save_path="latent_aug.npy",
    return_numpy=True,
)

extractor = Extractor(
    model,
    cfg,
    augmenter=augmenter,
)

Z_aug = extractor(loader)
```

This can be useful for robustness checks or test-time augmentation style analyses.

---

## Design Notes

### All-visible encoder call

The extractor bypasses ChemoMAE's random mask generation and directly calls:

```python
model.encoder(x_input, visible_mask)
```

where:

```python
visible_mask = torch.ones(B, L, dtype=torch.bool, device=device)
```

This ensures the encoder observes the full spectrum.

### Augmentation and determinism

Without an augmenter, all-visible extraction removes mask randomness.

With an augmenter, the extracted features may vary across calls unless random seeds are controlled.  
This is expected because spectral shift/noise augmentation is stochastic.

### AMP and dtype handling

Autocast is used only when:

* `cfg.amp=True`, and
* `cfg.device` resolves to CUDA.

Extracted features are cast to `float32` before CPU aggregation.  
This avoids issues when saving `bf16` tensors as NumPy arrays.

### Device safety

All extracted features are moved to CPU before concatenation and saving.

### Format flexibility

Saving format and return type are independent.

---

## Minimal Tests

### Basic extraction

```python
cfg = ExtractorConfig(
    device="cpu",
    return_numpy=True,
    save_path=None,
)

extractor = Extractor(model, cfg)
Z = extractor(loader)

assert isinstance(Z, np.ndarray)
assert Z.ndim == 2
```

### Augmenter path

```python
augmenter = SpectraAugmenter(SpectraAugmenterConfig())

cfg = ExtractorConfig(
    device="cpu",
    return_numpy=False,
)

extractor = Extractor(
    model,
    cfg,
    augmenter=augmenter,
)

Z = extractor(loader)

assert isinstance(Z, torch.Tensor)
assert Z.ndim == 2
```

---

## Version v0.2.0

Updated for the optional augmenter-enabled ChemoMAE extraction pipeline.

Changes:

* added optional `SpectraAugmenter` support,
* documented augmenter train/eval mode handling,
* clarified determinism with and without augmentation,
* clarified CPU aggregation and dtype handling,
* added augmented extraction usage examples.