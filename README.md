<h1 align="center">ChemoMAE</h1>

[![PyPI version](https://img.shields.io/pypi/v/chemomae.svg)](https://pypi.org/project/chemomae/)
[![CI](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/chemomae.svg)](https://pypi.org/project/chemomae/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)


> **ChemoMAE**: A Research-Oriented PyTorch Toolkit and Models for **1D Spectral Representation Learning and Hyperspherical Clustering**.

[**Research Repository**](https://github.com/Mantis-Ryuji/WoodDegradationSeg-NIRHSI): Unsupervised Segmentation of Wood Degradation Patterns with NIR-HSI (preparing)

---

## Why ChemoMAE?

Traditional chemometrics has long relied on **linear methods** such as PCA and PLS.
These remain foundational, but they often struggle to capture **nonlinear structure** and **high-dimensional variability** in modern spectral datasets.<br>
**ChemoMAE** is designed to respect the **hyperspherical geometry** induced by the Standard Normal Variate (SNV) transformation —
a fundamental preprocessing step in chemometrics that normalizes each spectrum to zero mean and unit variance,
placing all samples on a **hypersphere of constant norm**. <br>
ChemoMAE learns representations consistent with this geometry and preserves it across downstream tasks.

### 1. Extending Chemometrics with Deep Learning

A **Transformer-based Masked Autoencoder (MAE)** specialized for **1D spectra** enables flexible, data-driven representation learning.
We apply **block-wise masking** to SNV-preprocessed spectra, optimizing the **MSE loss** only over the hidden spectral regions.
The encoder produces **unit-norm embeddings** `z` that capture **directional spectral information**.
This design aligns naturally with the hyperspherical geometry induced by SNV, yielding representations that are inherently suitable for **cosine similarity** and **hyperspherical clustering**.


### 2. Hyperspherical Geometry Toolkit (for downstream use)

Because embeddings are **L2-normalized**, they lie on the **unit hypersphere**.
Built-in clustering modules such as **Cosine-KMeans** and **vMFMixture** operate natively within this geometry,
enabling clustering that faithfully reflects **spectral shape** after SNV preprocessing.

---

## Quick Start

Install ChemoMAE **(preparing)**

```bash
pip install chemomae
```

---

### ChemoMAE Example

<details>
<summary><b>Example</b></summary>

```python
# === 1. SNV Preprocessing ===
# Import the Standard Normal Variate (SNV) scaler.
# SNV standardizes each spectrum to have zero mean and unit variance:
#   x_snv = (x - mean(x)) / std(x)
#
# This removes baseline and scaling effects while preserving the spectral shape (direction).
# After SNV, all spectra have an identical L2 norm of sqrt(C - 1)
#   e.g., for 256-dimensional spectra, ||x_snv||₂ = √255 ≈ 15.97
# Hence, SNV maps spectra onto a constant-radius hypersphere.
from chemomae.preprocessing import SNVScaler

# X_*: reflectance data (np.ndarray)
# Expected shape: (N, 256)  -> N samples, 256 wavelength bands
preprocessed = []
for X in [X_train, X_val, X_test]:
    sc = SNVScaler()
    X_snv = sc.transform(X)
    preprocessed.append(X_snv)

# Unpack processed datasets
X_train_snv, X_val_snv, X_test_snv = preprocessed
```

```python
# === 2. Dataset and DataLoader Preparation ===
# Convert preprocessed numpy arrays to PyTorch tensors.
# DataLoader wraps datasets with batching, shuffling, and GPU pipeline support.

from chemomae.utils import set_global_seed
import torch
from torch.utils.data import DataLoader, TensorDataset

set_global_seed(42)  # Ensure reproducibility

train_ds = TensorDataset(torch.as_tensor(X_train_snv, dtype=torch.float32))
val_ds   = TensorDataset(torch.as_tensor(X_val_snv,   dtype=torch.float32))
test_ds  = TensorDataset(torch.as_tensor(X_test_snv,  dtype=torch.float32))

# Define loaders (batch size and shuffle behavior)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, drop_last=False)
```

```python
# === 3. Model, Optimizer, and Scheduler Setup ===
# Define ChemoMAE (Masked AutoEncoder for spectral data).
# This model learns to reconstruct masked blocks, capturing spectral structure.

from chemomae.models import ChemoMAE
from chemomae.training import build_optimizer, build_scheduler

model = ChemoMAE(
    seq_len=256,             # input sequence length
    d_model=256,             # Transformer hidden dimension
    nhead=4,                 # number of attention heads
    num_layers=4,            # encoder depth
    dim_feedforward=1024,    # MLP dimension
    dropout=0.1,
    use_learnable_pos=True,  # learnable positional encoding
    latent_dim=64,           # latent vector dimension
    dec_hidden=256,          # decoder hidden size
    dec_dropout=0.1,
    n_blocks=32,             # number of total blocks
    n_mask=16                # number of masked blocks per sample
)

# Optimizer: AdamW with decoupled weight decay
opt = build_optimizer(
    model, 
    lr=3e-4, 
    weight_decay=1e-4, 
    betas=(0.9, 0.95)  # standard for MAE pretraining
)

# Learning rate schedule: warmup + cosine annealing
sched = build_scheduler(
    opt,
    steps_per_epoch=max(1, len(train_loader)),
    epochs=500,
    warmup_epochs=10,    # linear warmup for 10 epochs
    min_lr_scale=0.1     # final LR = base_lr * 0.1
)
```

```python
# === 4. Training Setup (Trainer + Config) ===
# Trainer orchestrates the full training loop with:
# - AMP (Automatic Mixed Precision)
# - EMA (Exponential Moving Average of model weights)
# - Early stopping and learning-rate scheduling
# - Checkpointing and full logging for reproducibility

from chemomae.training import TrainerConfig, Trainer

trainer_cfg = TrainerConfig(
    out_dir = "runs",               # Root directory for all outputs and logs
    device = "cuda",                # Training device (auto-detected if None)
    amp = True,                     # Enable mixed precision (AMP)
    amp_dtype = "bf16",             # AMP precision type (bf16 is stable and efficient)
    enable_tf32 = False,            # Disable TF32 to maintain numerical reproducibility
    grad_clip = 1.0,                # Gradient clipping threshold (norm-based)
    use_ema = True,                 # Enable EMA to smooth parameter updates
    ema_decay = 0.999,              # EMA decay rate
    loss_type = "mse",              # Masked reconstruction loss type
    reduction = "mean",             # Reduction method for masked loss
    early_stop_patience = 50,       # Stop if val_loss doesn't improve for 50 epochs
    early_stop_start_ratio = 0.5,   # Start monitoring early stopping after half of total epochs
    early_stop_min_delta = 0.0,     # Required minimum improvement in validation loss
    resume_from = "auto"            # Resume from the latest checkpoint if available
)

trainer = Trainer(
    model, 
    opt, 
    train_loader, 
    val_loader, 
    scheduler=sched, 
    cfg=trainer_cfg
)

# ---------------------------------------------------------------------
# During training, ChemoMAE produces the following outputs under out_dir:
#
#  runs/
#  ├── training_history.json
#  │     ↳ Records per-epoch statistics:
#  │        [{"epoch": 1, "train_loss": ..., "val_loss": ..., "lr": ...}, ...]
#  │        → useful for visualizing loss curves and learning rate schedules.
#  │
#  ├── best_model.pt
#  │     ↳ Model weights only (state_dict). Compact and ideal for inference.
#  │        Saved whenever validation loss reaches a new minimum.
#  │
#  └── checkpoints/
#         ├── last.pt
#         │     ↳ Full checkpoint (model + optimizer + scheduler + scaler + EMA + RNG + history)
#         │        Saved every epoch to allow full recovery (resume_from="auto").
#         │
#         └── best.pt
#               ↳ Full checkpoint at the best validation loss.
#                  Includes everything in last.pt but frozen at the optimal epoch.
# ---------------------------------------------------------------------

# Begin training for 500 epochs (or until early stopping triggers)
_ = trainer.fit(epochs=500)
```

```python
# === 5. Evaluation (Tester) ===
# The Tester evaluates the trained model on unseen test data.

from chemomae.training import TesterConfig, Tester

tester_cfg = TesterConfig(
    out_dir = "runs",
    device = "cuda",
    amp = True,
    amp_dtype = "bf16",
    loss_type = "mse",
    reduction = "mean",
    fixed_visible = None,         # optionally fix visible blocks during masking
    log_history = True,           # append evaluation results to history file
    history_filename = "training_history.json"
)

tester = Tester(model, tester_cfg)

# Compute reconstruction loss on test set
test_loss = tester(test_loader)
print(f"Test Loss : {test_loss:.2f}")
```

```python
# === 6. Latent Extraction ===
# Extract latent embeddings from the trained ChemoMAE model.

from chemomae.training import ExtractConfig, Extractor

extractor_cfg = ExtractConfig(
    device = "cuda",
    amp = True,
    amp_dtype = "bf16",
    save_path = None,      # optional file output (e.g. "latent_test.npy")
    return_numpy = False   # return as torch.Tensor instead of np.ndarray
)

extractor = Extractor(model, extractor_cfg)

latent_test = extractor(test_loader)
```

```python
# === 7. Clustering with CosineKMeans ===
# Cluster the latent vectors based on cosine similarity.
# The elbow method automatically determines an optimal K by analyzing inertia.

from chemomae.clustering import CosineKMeans, elbow_ckmeans

k_list, inertias, K, idx, kappa = elbow_ckmeans(
    CosineKMeans, 
    latent_test, 
    device="cuda", 
    k_max=50,              # maximum clusters to test
    chunk=5000000,         # GPU chunking for large datasets
    random_state=42
)

# Initialize and fit final clustering model
ckm = CosineKMeans(
    n_components=K, 
    tol=1e-4,
    max_iter=500,
    device="cuda",
    random_state=42
)

ckm.fit(latent_test, chunk=5000000)
ckm.save_centroids("runs/ckm.pt")

# Later, reload and predict cluster labels
# ckm.load_centroids("runs/ckm.pt")
labels = ckm.predict(latent_test, chunk=5000000)
```

```python
# === 8. Clustering with vMF Mixture (von Mises–Fisher) ===
# For hyperspherical latent representations, the vMF mixture model provides a probabilistic alternative.

from chemomae.clustering import VMFMixture, elbow_vmf

k_list, scores, K, idx, kappa = elbow_vmf(
    VMFMixture, 
    latent_test, 
    device="cuda", 
    k_max=50,
    chunk=5000000,
    random_state=42,
    criterion="bic"         # choose best K using Bayesian Information Criterion
)

vmf = VMFMixture(
    n_components=K, 
    tol=1e-4,
    max_iter=500,
    device="cuda",
    random_state=42
)

vmf.fit(latent_test, chunk=5000000)
vmf.save("runs/vmf.pt")

# Reload if needed and predict cluster assignments
# vmf.load("runs/vmf.pt")
labels = vmf.predict(latent_test, chunk=5000000)
```

</details>

---

## Library Features

<details>
<summary><b><code>chemomae.preprocessing</code></b></summary>
<br>

- **SNVScaler**
- [Document](docs/preprocessing/snv.md)
- [Implementation](src/chemomae/preprocessing/snv.py)

```python
```

- **cosine_fps_downsample**
- [Document](docs/preprocessing/dowmsampling.md)
- [Implementation](src/chemomae/preprocessing/downsampling.py)

```python
```
</details>


<details>
<summary><b><code>chemomae.models</code></b></summary>
<br>

- **ChemoMAE**
- [Document](docs/models/chemo_mae.md)
- [Implementation](src/chemomae/models/chemo_mae.py)

```python
```
</details>


<details>
<summary><b><code>chemomae.training</code></b></summary>
<br>

- **build_optimizer & build_scheduler**
- [Document](docs/training/optim.md)
- [Implementation](src/chemomae/training/optim.py)

```python
```

- **Trainer**
- [Document](docs/training/trainer.md)
- [Implementation](src/chemomae/training/trainer.py)

```python
```

- **Tester**
- [Document](docs/training/tester.md)
- [Implementation](src/chemomae/training/tester.py)

```python
```

- **Extractor**
- [Document](docs/training/extractor.md)
- [Implementation](src/chemomae/training/extractor.py)

```python
```
</details>


<details>
<summary><b><code>chemomae.clustering</code></b></summary>
<br>

- **CosineKMeans & elbow_ckmeans**
- [Document](docs/clustering/cosine_kmeans.md)
- [Implementation](src/chemomae/clustering/cosine_kmeans.py)

```python
```

- **VMFMixture & elbow_vmf**
- [Document](docs/clustering/vmf_mixture.md)
- [Implementation](src/chemomae/clustering/vmf_mixture.py)

```python
```

- **silhouette_samples_cosine_gpu & silhouette_score_cosine_gpu**
- [Document](docs/clustering/metric.md)
- [Implementation](src/chemomae/clustering/metric.py)

```python
```
</details>


<details>
<summary><b><code>chemomae.utils</code></b></summary>
<br>

- **set_global_seed**
- [Document](docs/utils/seed.md)
- [Implementation](src/chemomae/utils/seed.py)

```python
```
</details>

---

## [License](LICENSE)

ChemoMAE is released under the **Apache License 2.0**,
a permissive open-source license that allows both academic and commercial use with minimal restrictions.

Under this license, you are free to:

* **Use** the source code for research or commercial projects.
* **Modify** and adapt it to your own needs.
* **Distribute** modified or unmodified versions, provided that the original copyright notice and license text are preserved.

However, there is **no warranty** of any kind —
the software is provided “*as is*,” without guarantee of fitness for any particular purpose or liability for damages.

For complete terms, see the official license text:
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)
