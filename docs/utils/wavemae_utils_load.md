# Loading Default Pretrained Model — WaveMAE

> Module: `wavemae.utils.load`

This document describes `load_default_pretrained`, a convenience function that constructs a **default WaveMAE** and attempts to load pretrained weights packaged with the library.

---

## Overview

* Builds a **WaveMAE** with default hyperparameters.
* Attempts to locate weights in the package `assets/` directory (e.g., `wavemae_base_256.pt`).
* Optionally verifies integrity via `.sha256` files.
* Returns both the model and a metadata dictionary summarizing what was loaded.
* If weights are missing or invalid, returns a randomly initialized model with a warning in `meta["warning"]`【90†source】.

---

## API

### `load_default_pretrained(weight_path: Optional[str|Path] = None, *, device: Optional[str|torch.device] = None, strict: bool = True) -> Tuple[WaveMAE, Dict[str, Any]]`

**Parameters**

* `weight_path`: Optional explicit `.pt` file. If `None`, auto-searches in package `assets/`.
* `device`: Destination device for model ("cuda", "cpu", or `None` = auto-detect).
* `strict`: Whether to enforce strict key matching in `load_state_dict`.

**Returns**

* `model`: WaveMAE instance with default config and (if available) pretrained weights loaded.
* `meta`: Dictionary with fields:

  * `name`: basename of the loaded weight file, or "(none)"
  * `config`: default hyperparameters (`seq_len`, `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`, `use_learnable_pos`, `latent_dim`, `dec_hidden`, `dec_dropout`, `n_blocks`, `n_mask`)
  * `repos`: fixed GitHub links (`{"library": ..., "pretraining": ...}`)
  * `device`: device string
  * `strict`: strict flag
  * `pretrained_loaded`: bool indicating load success
  * `warning`: (optional) error/warning message if weights not loaded【90†source】

---

## Example Usage

### Default load (auto assets)

```python
from wavemae.utils.load import load_default_pretrained

model, meta = load_default_pretrained(device="cuda")
print(meta["pretrained_loaded"])  # True if weights found and loaded
```

### Explicit weight path

```python
model, meta = load_default_pretrained("/path/to/custom.pt", device="cpu", strict=False)
```

### Handling missing weights

```python
model, meta = load_default_pretrained(device="cpu")
if not meta["pretrained_loaded"]:
    print("Warning:", meta.get("warning"))
```

---

## Internal Notes

* **Default config** is centralized in `_DEFAULT_CFG`.
* **Assets dir** is located with `importlib.resources.files("wavemae")/assets`, falling back to `src/wavemae/assets` for editable installs.
* **Integrity check**: if `<file>.sha256` exists, the SHA256 is compared before loading.
* **Load formats**: supports plain `state_dict`, `{"state_dict": ...}`, or `{"model": {...}}` dicts.
* **Device logic**: if `device=None`, auto-selects CUDA if available, else CPU.

---

## Minimal Test

```python
model, meta = load_default_pretrained(device="cpu")
assert isinstance(meta["config"], dict)
assert "pretrained_loaded" in meta
```

---

## Version

* Introduced in `wavemae.utils.load` — initial public draft.
