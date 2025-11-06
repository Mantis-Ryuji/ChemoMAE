# Extractor — Latent Feature Extraction

> Module: `chemomae.training.extractor`

This document describes the `Extractor` and its configuration (`ExtractorConfig`), which provide an efficient way to extract latent embeddings (`Z`) from trained **ChemoMAE** models under **all-visible mode**.

---

## Overview

The `Extractor` obtains **latent representations** from a trained ChemoMAE encoder without masking.
These embeddings can be used for downstream analysis such as **clustering** (CosineKMeans, vMF Mixture) or **dimensionality reduction** (UMAP, t-SNE).

### Key features

* **All-visible encoding:** No masking; produces deterministic latent vectors.
* **AMP support:** Accelerated inference with `bf16` or `fp16`.
* **Flexible output:** Returns `torch.Tensor` or `numpy.ndarray`.
* **Optional saving:** Results can be automatically written to disk (`.npy` or `.pt`).

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

#### Parameters

| Name           | Type                      | Default  | Description                                                                            |
| -------------- | ------------------------- | -------- | -------------------------------------------------------------------------------------- |
| `device`       | `str` or `torch.device`   | `"cuda"` | Device used for feature extraction.                                                    |
| `amp`          | `bool`                    | `True`   | Enable automatic mixed precision during inference.                                     |
| `amp_dtype`    | `"bf16"` or `"fp16"`      | `"bf16"` | Precision type for autocast (`bf16` recommended).                                      |
| `save_path`    | `str` or `Path` or `None` | `None`   | Optional output file path. If given, results are automatically saved after extraction. |
| `return_numpy` | `bool`                    | `False`  | Whether to return results as `np.ndarray` instead of `torch.Tensor`.                   |

---

## Class: `Extractor`

### Initialization

```python
extractor = Extractor(model, cfg=ExtractorConfig())
```

| Argument | Description                                                                      |
| -------- | -------------------------------------------------------------------------------- |
| `model`  | Trained ChemoMAE model (must implement `encoder(x, visible_mask)` → latent `z`). |
| `cfg`    | Optional configuration controlling device, AMP, and output options.              |

The model is moved to `cfg.device` and set to `eval()`.

---

### Call Interface

```python
Z = extractor(loader)
```

* Iterates through batches from `loader`.
* Each batch:

  * Transfers input to device.
  * Constructs an **all-ones visible mask** `(B, L)` (i.e., fully visible sequence).
  * Calls `model.encoder(x, visible_mask)` to obtain latent `z`.
* Collects and concatenates all `z` across batches.
* Returns either a `torch.Tensor` or `numpy.ndarray` depending on `return_numpy`.

---

## Saving Behavior

If `cfg.save_path` is specified:

| File extension | Behavior                               |
| -------------- | -------------------------------------- |
| `.npy`         | Saved using `numpy.save(save_path, Z)` |
| otherwise      | Saved using `torch.save(Z, save_path)` |

* Parent directories are created automatically.
* The return value (`torch.Tensor` or `np.ndarray`) is unaffected by saving format.

---

## Usage Examples

### Extract features to memory

```python
from chemomae.training import Extractor, ExtractorConfig

cfg = ExtractorConfig(device="cuda", return_numpy=True)
extractor = Extractor(model, cfg)
Z = extractor(loader)  # np.ndarray, shape (N, D)
```

### Save features to disk

```python
cfg = ExtractorConfig(device="cuda", save_path="latent.npy")
extractor = Extractor(model, cfg)
Z = extractor(loader)  # torch.Tensor
# latent.npy written to disk
```

### Save as Torch tensor

```python
cfg = ExtractorConfig(device="cuda", save_path="latent.pt")
extractor = Extractor(model, cfg)
Z = extractor(loader)
# latent.pt contains a torch.Tensor
```

---

## Design Notes

* **Deterministic encoding:**
  Uses an all-visible mask to ensure consistent embeddings across runs.

* **AMP efficiency:**
  Reduces memory footprint and speeds up forward passes.

* **Device safety:**
  All outputs are transferred to CPU before concatenation and saving.

* **Format flexibility:**
  Saving format and return type are independent — for example, you can save `.npy` but return a Tensor.

---

## Minimal Tests

```python
cfg = ExtractorConfig(device="cpu", return_numpy=True, save_path=None)
extractor = Extractor(model, cfg)
Z = extractor(loader)
assert isinstance(Z, np.ndarray)
assert Z.ndim == 2
```

---

## Version

* Introduced in `chemomae.training.extractor` — initial public draft.