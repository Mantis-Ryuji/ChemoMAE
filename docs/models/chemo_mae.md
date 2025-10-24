# ChemoMAE — Masked Autoencoder for 1D Spectra

> Module: `chemomae.models.chemo_mae`

This document describes **ChemoMAE**, a Transformer‑based masked autoencoder designed for one‑dimensional spectral data (e.g., near‑infrared (NIR) spectra or hyperspectral bands).

<p align="center">
<img src="../../images/ChemoMAE.svg">
</p>

---

## Overview

ChemoMAE adapts the **Masked Autoencoder (MAE)** framework (He et al., 2022) to 1D spectra. Instead of image patches, spectra are divided into **contiguous blocks** along the spectral axis, and a large fraction of them are randomly masked during training. The encoder sees only the **visible tokens**, while the decoder attempts to reconstruct the entire sequence. Importantly, the loss is computed only on the masked regions, ensuring that the encoder is forced to learn informative spectral representations.

### Key ideas

* **Block‑wise masking:** The sequence of length $L$ is split into `n_blocks`; `n_mask` of them are hidden per sample. This encourages learning robust, context‑aware latent representations.
* **Encoder:** Transformer encoder operating only on visible tokens plus a prepended [CLS] token. The CLS output is projected to latent_dim and L2-normalized, yielding an embedding constrained to the unit hypersphere. This makes the representation naturally compatible with hyperspherical geometry and cosine-similarity–based clustering.
* **Decoder:** A linear projection decoder that reconstructs the full spectral sequence of length $`L`$ from the latent embedding. The decoder is used only during training, and the loss is computed on the masked regions.

---

## Architecture

### Positional Encoding

* Supports either **learnable positional embeddings** (default) or **fixed sinusoidal encoding**.
* Shape: `(1, L, d_model)` broadcastable across batch.

### Masking

* Implemented via `make_block_mask(batch_size, seq_len, n_blocks, n_mask)`.
* Returns `(B, L)` boolean mask with `True=masked`.
* ChemoMAE internally converts this to a visible mask (`True=visible`).

### Encoder — `ChemoEncoder`

* Input: spectra `(B, L)` with visible mask `(B, L)`.
* Projects to embedding dimension `d_model`, adds positional encoding.
* Packs only visible tokens + CLS.
* Processes with a Transformer encoder (`num_layers`, `nhead`, `dim_feedforward`, `dropout`).
* Output: L2‑normalized latent `(B, latent_dim)`, suitable for cosine‑based metrics.

### Decoder — `ChemoDecoderMLP`

* Input: latent `(B, latent_dim)`.
* Linear Projection Decoder
* Output: reconstruction `(B, L)`.
* Lightweight by design: the learning burden is shifted to the encoder.

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
    use_learnable_pos=True,
    latent_dim=64,
    dec_hidden=256,
    dec_dropout=0.1,
    n_blocks=32,
    n_mask=16,
)
```

**Parameters**

* `seq_len` (`int`): Input sequence length.
* `d_model` (`int`, default=384): Transformer embedding dimension.
* `nhead` (`int`, default=6): Number of self‑attention heads (`d_model % nhead == 0`).
* `num_layers` (`int`, default=6): Encoder depth.
* `dim_feedforward` (`int`, default=1536): Hidden dimension in FFN layers.
* `dropout` (`float`, default=0.1): Dropout rate.
* `use_learnable_pos` (`bool`, default=True): If `False`, uses fixed sinusoidal embeddings.
* `latent_dim` (`int`, default=64): Latent embedding dimension.
* `dec_hidden` (`int`, default=256): Decoder hidden dimension.
* `dec_dropout` (`float`, default=0.1): Decoder dropout rate.
* `n_blocks` (`int`, default=32): Number of equal blocks along the sequence.
* `n_mask` (`int`, default=16): Default number of blocks to mask.

**Methods**

* `forward(x, visible_mask=None, *, n_mask=None)` → `(x_recon, z, visible_mask)`
* `reconstruct(x, visible_mask=None, *, n_mask=None)` → `x_recon`
* `make_visible(batch_size, *, n_mask=None, device=None)` → `visible_mask`

---

## Usage Examples

### Training loop

```python
import torch
from chemomae.models import ChemoMAE

mae = ChemoMAE(seq_len=256, latent_dim=64, n_blocks=16, n_mask=4)
x = torch.randn(8, 256)

# Forward pass with automatic masking
x_recon, z, visible = mae(x)

# Loss: reconstruction error only on masked positions
loss = ((x_recon - x) ** 2)[~visible].sum() / x.size(0)
loss.backward()
```

### Downstream tasks

* **Clustering:** CosineKMeans, vMF Mixture.
* **Dimensionality reduction:** UMAP, t‑SNE with metric="cosine".
* **Transfer learning:** Latent features can serve as spectral embeddings.

---

## Design Notes

* **L2‑normalized latent:** All embeddings lie on a hypersphere, directly usable for cosine metrics.
* **Separation of concerns:** Loss and training logic are external. This keeps ChemoMAE flexible for different objectives.
* **Compatibility:** AMP (bf16/fp16) supported via external `torch.autocast`. EMA and checkpointing handled in training utilities.

---

## Minimal Tests

```python
mae = ChemoMAE(seq_len=128, latent_dim=32, n_blocks=8, n_mask=6)
x = torch.randn(4, 128)

x_rec, z, visible = mae(x)
assert x_rec.shape == x.shape
assert z.shape == (4, 32)
assert visible.shape == (4, 128)
assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)
```

---

## Version

* Introduced in `chemomae.models.chemo_mae` — initial public draft.
