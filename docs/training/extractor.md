# Extractor — Latent Feature Extraction

> Module: `chemomae.training.extractor`

This document describes the **Extractor** and its **ExtractConfig**, which enable deterministic latent feature extraction from trained ChemoMAE models.

---

## Overview

The `Extractor` provides a utility to obtain **latent embeddings (Z)** from a trained ChemoMAE in **all-visible mode**:

* **Deterministic:** Uses a full visible mask (all tokens visible), independent of random masking.
* **AMP support:** Compatible with bf16/fp16 inference.
* **Output control:** Can return `torch.Tensor` or `numpy.ndarray`, and optionally save results to disk.

This is particularly useful for downstream tasks such as clustering (e.g., CosineKMeans, vMF mixture) or visualization (e.g., UMAP, t-SNE).

---

## Configuration — `ExtractConfig`

```python
@dataclass
class ExtractConfig:
    device: str | torch.device = "cuda"
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    save_path: Optional[str | Path] = None
    return_numpy: bool = False
```

**Fields**

* `device`: Inference device (e.g., `"cuda"`, `"cpu"`).
* `amp`: Enable automatic mixed precision.
* `amp_dtype`: AMP precision (`bf16` recommended).
* `save_path`: Optional file path to save extracted features.

  * `.npy` → saved with `numpy.save`
  * otherwise → saved with `torch.save`
* `return_numpy`: If `True`, return a NumPy array; otherwise, return a Torch tensor.

---

## Class: `Extractor`

### Initialization

```python
extractor = Extractor(model, cfg=ExtractConfig())
```

* `model`: Trained ChemoMAE.
* `cfg`: Extraction configuration.

### Call

```python
Z = extractor(loader)
```

* Iterates over batches from `loader`.
* For each batch:

  * Moves inputs to device.
  * Builds an all-ones visible mask `(B, L)`.
  * Calls `model.encode(x, visible_mask)` to compute latent features.
  * Collects and concatenates results across batches.
* Returns either a Torch tensor `(N, D)` or NumPy array, depending on `return_numpy`.

### Saving

If `save_path` is provided:

* Automatically saves after extraction.
* Parent directories are created if missing.
* Format inferred from file suffix.

---

## Usage Examples

### Extract features to memory

```python
from chemomae.training import Extractor, ExtractConfig

cfg = ExtractConfig(device="cuda", return_numpy=True)
extractor = Extractor(model, cfg)
Z = extractor(loader)  # np.ndarray, shape (N, D)
```

### Save features to disk

```python
cfg = ExtractConfig(device="cuda", save_path="latent.npy", return_numpy=False)
extractor = Extractor(model, cfg)
Z = extractor(loader)  # torch.Tensor
# latent.npy written to disk as np.ndarray
```

### Torch save format

```python
cfg = ExtractConfig(device="cuda", save_path="latent.pt")
extractor = Extractor(model, cfg)
Z = extractor(loader)
# latent.pt contains torch.Tensor
```

---

## Design Notes

* **All-visible encoding:** Guarantees determinism and avoids dependency on random masking.
* **AMP:** Reduces VRAM usage and speeds up inference.
* **Separation of save/return formats:** You can save `.npy` but still return a Torch tensor, and vice versa.
* **Output device:** Final results are always moved to CPU before concatenation.

---

## Minimal Tests

```python
cfg = ExtractConfig(device="cpu", return_numpy=True, save_path=None)
extractor = Extractor(model, cfg)
Z = extractor(loader)
assert isinstance(Z, np.ndarray)
```

---

## Version

* Introduced in `chemomae.training.extractor` — initial public draft.
