# ChemoMAE — Masked Autoencoder for 1D Spectra

> Module: `chemomae.models.chemo_mae`

This document describes **ChemoMAE**, a Transformer-based masked autoencoder specialized for **one-dimensional spectral data** such as near-infrared (NIR) spectra or 1D hyperspectral bands.

<p align="center">
<img src="../../images/ChemoMAE.svg">
</p>

---

## Overview

**ChemoMAE** adapts the **Masked Autoencoder (MAE)** framework (He et al., 2022) to *1D spectral sequences*.
Each spectrum of length `L` is divided into **`n_patches` contiguous patches**, and a subset of them is randomly masked at training time.

Only **visible patches** are passed to the Transformer encoder, while the decoder reconstructs the **entire sequence** from the latent embedding.
The reconstruction loss is computed **only on masked regions**, encouraging the encoder to learn chemically meaningful and context-aware structure.

---

## Key Ideas

### Patch-wise masking

The sequence is reshaped into:

```
(B, L) → (B, n_patches, patch_size)
```

Then `n_mask` patches are randomly hidden per sample.
This creates a reconstruction problem on a **mesoscopic scale**, aligned with the physical smoothness of real spectra.

### Encoder

* Patch embeddings → positional encoding
* Only visible patches + a `[CLS]` token are passed to a Transformer encoder
* The `[CLS]` output is projected to `latent_dim` and **L2-normalized** → latent vector on the **unit hypersphere**
  (ideal for cosine metrics, CosineKMeans, vMF mixtures)

### Decoder

A lightweight **2-layer MLP decoder** that maps the latent vector directly to the full-length spectrum `(B, L)`.
The decoder intentionally avoids any patch reconstruction structure to place the learning burden on the encoder.

---

## Architecture

### Positional Encoding

* **Learnable positional embeddings** by default
* Alternatively: **fixed sinusoidal** positional encoding
* Only `n_patches` positions are encoded (not length `L`)

---

## Masking

Masking is performed by:

```
make_patch_mask(batch_size, seq_len, n_patches, n_mask)
```

* Returns `(B, L)` boolean mask (`True = masked`)
* The model internally converts it to a **visible mask** (`True = visible`)
* `seq_len` must be divisible by `n_patches`

---

## Encoder — `ChemoEncoder`

**Input**

* Spectra `(B, L)`
* Visible mask `(B, L)`

**Pipeline**

1. Patchify the spectrum
2. Linear projection → patch embeddings
3. Gather only visible patches (`V ≤ n_patches`)
4. Add `[CLS]` token
5. Transformer encoder
6. `[CLS]` → linear → **L2 normalization**

**Output**

* Latent vectors `(B, latent_dim)` lying on the **unit hypersphere**

---

## Decoder — `ChemoDecoder`

**Input**
`(B, latent_dim)`

**Output**
`(B, L)` full-length reconstruction

The decoder is intentionally simple to emphasize encoder learning.

---

## API

### Class: `ChemoMAE`

```python
mae = ChemoMAE(
    seq_len=256,
    d_model=256,
    nhead=4,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    latent_dim=16,
    n_patches=32,
    n_mask=16,
)
```

### Parameters

| Name                | Type  | Default | Description                                        |
| ------------------- | ----- | ------- | -------------------------------------------------- |
| `seq_len`           | int   | 256     | Length of the input spectrum.                      |
| `d_model`           | int   | 256     | Transformer embedding dimension.                   |
| `nhead`             | int   | 4       | Number of attention heads.                         |
| `num_layers`        | int   | 4       | Transformer encoder layers.                        |
| `dim_feedforward`   | int   | 1024    | FFN hidden dimension.                              |
| `dropout`           | float | 0.1     | Dropout in encoder layers.                         |
| `latent_dim`        | int   | 16      | Dimension of L2-normalized latent embedding.       |
| `n_patches`         | int   | 32      | Number of patches; must divide `seq_len`.          |
| `n_mask`            | int   | 16      | Number of patches to mask.                         |

### Methods

* `forward(x, visible_mask=None, *, n_mask=None)` → `(x_recon, z, visible_mask)`
* `reconstruct(x, visible_mask=None, *, n_mask=None)` → `x_recon`
* `make_visible(batch_size, *, n_mask=None, device=None)` → `visible_mask`

---

## Usage Examples

### Training

```python
import torch
from chemomae.models import ChemoMAE

mae = ChemoMAE(seq_len=256, latent_dim=16, n_patches=16, n_mask=4)
x = torch.randn(8, 256)

x_recon, z, visible = mae(x)  # visible=True → used in encoder

# Masked reconstruction loss
sqerr = (x_recon - x).pow(2)
loss = sqerr[~visible].mean()
loss.backward()
```

---

## Downstream Applications

* **Clustering:**
  CosineKMeans, vMF mixture → latent space is hyperspherical

* **Visualization:**
  UMAP / t-SNE using `metric="cosine"`

* **Spectral segmentation / change detection:**
  Patch masking + latent vectors capture mesoscopic chemical transitions

---

## Design Notes

### Hyperspherical latent

L2 normalization ensures:

```
‖z‖ = 1
cosine similarity = z_i · z_j
```

Ideal for cosine geometry and directional clustering.

### Clean architecture

MAE training utilities (EMA, AMP, checkpointing, loss) are kept outside the model.

### Determinism

Masking is RNG-driven; use fixed seeds or provide explicit `visible_mask` for reproducibility.

---

## Minimal Tests

```python
import torch
from chemomae.models import ChemoMAE

mae = ChemoMAE(seq_len=128, latent_dim=8, n_patches=8, n_mask=6)
x = torch.randn(4, 128)

x_rec, z, visible = mae(x)

assert x_rec.shape == x.shape
assert z.shape == (4, 8)
assert visible.shape == (4, 128)
assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)

# masked loss
sqerr = (x_rec - x).pow(2)
loss = (sqerr[~visible]).mean()
assert torch.isfinite(loss)
```

---

## Version

* v0.1.4