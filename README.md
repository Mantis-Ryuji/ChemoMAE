<h1 align="center">ChemoMAE</h1>

[![PyPI version](https://img.shields.io/pypi/v/chemomae.svg)](https://pypi.org/project/chemomae/)
[![torch](https://img.shields.io/badge/PyTorch-2.6-orange)](#)
[![CI](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/chemomae.svg)](https://pypi.org/project/chemomae/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)


> **ChemoMAE**: A Research-Oriented PyTorch Toolkit and Models for **1D Spectral Representation Learning and Hyperspherical Clustering**.

[**Research Repository**](https://github.com/Mantis-Ryuji/WoodDegradationSeg-NIRHSI): **Unsupervised Segmentation of Wood Degradation Patterns with NIR-HSI**

---

## Why ChemoMAE?

Traditional chemometrics has long relied on **linear methods** such as PCA and PLS.
While these methods remain foundational, they often struggle to capture the **nonlinear structures** and **high-dimensional variability** present in modern spectral datasets.<br>
ChemoMAE is designed to respect the **hyperspherical geometry** induced by the Standard Normal Variate (SNV) transformation.
This core chemometric preprocessing step standardizes each spectrum to zero mean and unit variance, thereby placing all samples on a hypersphere of constant norm.<br>
ChemoMAE learns representations that are consistent with this geometry and preserves it across downstream tasks.

### 1. Extending Chemometrics with Deep Learning

A **Transformer-based Masked Autoencoder (MAE)** specialized for **1D spectra** enables flexible, data-driven representation learning.<br>
We apply block-wise masking to SNV-preprocessed spectra and optimize the **mean squared error (MSE)** only over the masked spectral regions.
The encoder produces **unit-norm embeddings** `z` that capture **directional spectral features**.
This architecture naturally aligns with the **hyperspherical geometry** induced by SNV, resulting in representations inherently suited for **cosine similarity** and **hyperspherical clustering**.

### 2. Hyperspherical Geometry Toolkit (for downstream use)

The embeddings, being L2-normalized, reside on a **unit hypersphere**.
Built-in clustering modules — **Cosine K-Means** and **vMF Mixture** — leverage this geometry natively, yielding clusters that faithfully capture **spectral shape variations**.

---

## Quick Start

Install ChemoMAE

```bash
pip install chemomae
```

---

### ChemoMAE Example

<details>
<summary><b>Example</b></summary>

#### 1. SNV Preprocessing 

Import the `SNVscaler`. <br>
SNV standardizes each spectrum to have zero mean and unit variance. This removes baseline and scaling effects while preserving the spectral shape.
After SNV, all spectra have an identical L2 norm of $`\sqrt{L - 1}`$ <br>
(e.g., for 256-dimensional spectra, ||x_snv||₂ = √255 ≈ 15.97) <br>
Hence, SNV maps spectra onto a constant-radius hypersphere.

```python
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
#### 2. Dataset and DataLoader Preparation

Convert preprocessed numpy arrays to PyTorch tensors. <br>
Build a PyTorch DataLoader from preprocessed NumPy arrays to handle batching and shuffling.

```python
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
#### 3. Model, Optimizer, and Scheduler Setup

Define ChemoMAE (Masked AutoEncoder for 1D spectra).
This model learns to reconstruct masked spectral blocks while learning representations constrained on the unit hypersphere.

```python
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
#### 4. Training Setup (Trainer + Config)

Trainer orchestrates the full training loop with:
- AMP (Automatic Mixed Precision)
- EMA (Exponential Moving Average of model weights)
- Early stopping and learning-rate scheduling
- Checkpointing and full logging for reproducibility

```python
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
#### 5. Evaluation (Tester + Config)

The Tester evaluates the trained model on test data.

```python
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
#### 6. Latent Extraction (Extractor + Config)

Extract latent embeddings from the trained ChemoMAE model **without masking**.

```python
from chemomae.training import ExtractorConfig, Extractor

extractor_cfg = ExtractorConfig(
    device = "cuda",
    amp = True,
    amp_dtype = "bf16",
    save_path = None,      # optional file output (e.g. "latent_test.npy")
    return_numpy = False   # return as torch.Tensor instead of np.ndarray
)

extractor = Extractor(model, extractor_cfg)

latent_test = extractor(test_loader)
```
#### 7. Clustering with Cosine K-Means

Cluster the latent vectors based on cosine similarity. <br>
The elbow method automatically determines an optimal K by analyzing inertia.

```python
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
#### 8. Clustering with vMF Mixture (von Mises–Fisher)

For hyperspherical latent representations, the vMF mixture model provides a probabilistic alternative.

```python
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

---

### `SNVScaler`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/preprocessing/snv.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/preprocessing/snv.py)

`SNVScaler` performs **row-wise mean subtraction and variance scaling** — each spectrum is centered and divided by its **unbiased standard deviation** (`ddof=1`). 
It is a **stateless** transformer supporting both **NumPy** and **PyTorch**, automatically preserving the original **framework, device, and dtype**. <br>
When `transform_stats=True`, it returns `(Y, mu, sd)`, where `sd` already includes `eps` and can be directly used for reconstruction. <br>
After SNV, all rows have **zero mean** and **unit variance**, producing a constant L2 norm of $`\sqrt{L-1}`$ and thus mapping spectra onto a constant-radius **hypersphere** — ideal for cosine-based clustering (e.g., `CosineKMeans`, `vMFMixture`).

```python
# === Basic usage (NumPy) ===
import numpy as np
from chemomae.preprocessing import SNVScaler

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)

# Stateless transform
scaler = SNVScaler()
Y = scaler.transform(X)  # same dtype and shape

# Each row now has mean ≈ 0 and variance ≈ 1 (ddof=1)
# The L2 norm becomes sqrt(L - 1), constant across all rows.

# === Round-trip reconstruction ===
scaler = SNVScaler(transform_stats=True)
Y, mu, sd = scaler.transform(X)
X_rec = scaler.inverse_transform(Y, mu=mu, sd=sd)

# === PyTorch-compatible ===
import torch
Xt = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]], device="cuda", dtype=torch.float32)

scaler = SNVScaler(transform_stats=True)
Yt, mu_t, sd_t = scaler.transform(Xt)
Xt_rec = scaler.inverse_transform(Yt, mu=mu_t, sd=sd_t)
```

**Key Features**

* **Unbiased standard deviation:** uses `ddof=1` for `L≥2`; automatically switches to `ddof=0` when `L=1`.
* **`eps` handling:** `eps` is added to `sd` internally for numerical stability; the returned `sd` already includes it.
* **Precision:** computations run in `float64`.
* **Torch integration:** device and dtype are preserved when returning tensors.

**When to Use**

* As a **standard preprocessing step** for NIR spectra to remove per-sample offsets and scaling effects.
* Recommended prior to **cosine similarity–based** models (ChemoMAE, CosineKMeans, vMFMixture) to align data with hyperspherical geometry.

---

### `cosine_fps_downsample`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/preprocessing/dowmsampling.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/preprocessing/downsampling.py)

`cosine_fps_downsample` performs **Farthest-Point Sampling (FPS)** under **hyperspherical geometry**, selecting points that are most directionally distinct on the unit hypersphere. <br>
Internally, all rows are **L2-normalized** for selection, but the returned subset is drawn from the **original-scale** `X`. <br>
It supports both NumPy and PyTorch inputs, automatically leveraging CUDA when available, and keeps Torch tensors on their original device/dtype. <br>
This method is particularly useful for **reducing redundancy** in NIR-HSI datasets while preserving **angular diversity**, making it ideal for self-supervised spectral learning pipelines.

```python
# === Basic usage (NumPy) ===
import numpy as np
from chemomae.preprocessing import cosine_fps_downsample

X = np.random.randn(1000, 128).astype(np.float32)
X_sub = cosine_fps_downsample(X, ratio=0.1, seed=42)  # -> (100, 128)

# === Torch input (device preserved) ===
import torch
Xt = torch.randn(5000, 128, device="cuda", dtype=torch.float32)
Xt_sub = cosine_fps_downsample(Xt, ratio=0.1, return_numpy=False)
# -> torch.Tensor on CUDA, shape (500, 128)

# === With SNV (recommended before cosine geometry) ===
from chemomae.preprocessing import SNVScaler
X_snv = SNVScaler().transform(X)
X_down = cosine_fps_downsample(X_snv, ratio=0.1)
```

**Key Features**

* **Internal normalization:** always performed; ensures scale invariance (selection depends only on direction).
* **Output:** taken from the *original* `X` (not normalized).

**When to Use**

* For **diversity-driven subsampling** of SNV- or L2-normalized spectra.
* Recommended at the **per-sample or per-tile** level in NIR-HSI datasets to reduce local redundancy and stabilize batch-wise coverage.
</details>


<details>
<summary><b><code>chemomae.models</code></b></summary>

---

### `ChemoMAE`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/models/chemo_mae.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/models/chemo_mae.py)

**ChemoMAE** is a **Masked Autoencoder for 1D spectra**.<br>
It performs **block-wise masking** along the spectral axis, encoding only the **visible tokens** and the **[CLS] token**, and reconstructs the full sequence through a **linear projection decoder**.<br>
The encoder incorporates **positional embeddings** to capture local spectral order, while the **[CLS]** output is projected to a `latent_dim` vector and **L2-normalized**, yielding embeddings that lie on the **unit hypersphere** — **naturally suited for cosine-based clustering and similarity metrics**.

```python
# === Basic training usage ===
import torch
from chemomae.models import ChemoMAE

mae = ChemoMAE(
    seq_len=256, 
    latent_dim=64,
    d_model=256, 
    nhead=4, 
    num_layers=4, 
    dim_feedforward=1024,
    n_blocks=32, 
    n_mask=16
)

x = torch.randn(8, 256)              # (B, L)
x_rec, z, visible = mae(x)           # visible auto-generated if None

# Loss on masked positions only
loss = ((x_rec - x) ** 2)[~visible].sum() / x.size(0)
loss.backward()

# === Feature extraction (all visible) ===
visible_all = torch.ones_like(visible, dtype=torch.bool)
z_all = mae.encoder(x, visible_all)   # L2-normalized latent, ready for cosine metrics

# === Reconstruction-only API ===
x_rec2 = mae.reconstruct(x, n_mask=16)
```

**Key Features**

* **Block-wise masking:** split length-`L` spectra into `n_blocks`; hide `n_mask` blocks per sample. 
* **Encoder (`ChemoEncoder`):** transforms only visible tokens + CLS; outputs **L2-normalized** latent `(B, latent_dim)`. 
* **Decoder (`ChemoDecoderLP`):** linear projection decoder that reconstructs `(B, L)`; **loss computed externally**, typically on masked regions. 
* **Positional encoding:** choose **learnable** or **fixed sinusoidal** embeddings. 
* **Cosine-friendly latents:** unit-sphere embeddings pair well with **CosineKMeans / vMF Mixture** and UMAP/t-SNE (`metric="cosine"`). 

**When to Use**

* Learning **geometry-aware spectral embeddings** from SNV/L2-normalized spectra for clustering, retrieval, or downstream supervised tasks.
</details>


<details>
<summary><b><code>chemomae.training</code></b></summary>

---

### `build_optimizer` & `build_scheduler`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/optim.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/training/optim.py)

`build_optimizer` and `build_scheduler` are utility functions designed to construct a **standardized optimization pipeline** for Transformer-style models such as **ChemoMAE**. <br>
They provide a simple and consistent API for creating a parameter-grouped **AdamW optimizer** (with weight-decay exclusions) and a **linear-warmup → cosine-decay learning-rate scheduler**, ensuring stable and smooth training dynamics for spectral MAE models.

```python
# === Basic usage (ChemoMAE training) ===
import torch
from chemomae.models import ChemoMAE
from chemomae.training.optim import build_optimizer, build_scheduler

# 1) Initialize model
model = ChemoMAE(seq_len=256)

# 2) Build optimizer (AdamW with grouped decay/no-decay)
optimizer = build_optimizer(
    model,
    lr=3e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)

# 3) Build scheduler (linear warmup → cosine decay)
steps_per_epoch = 1000
scheduler = build_scheduler(
    optimizer,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    warmup_epochs=5,
    min_lr_scale=0.1,
)

# === Training loop sketch ===
for epoch in range(100):
    for step in range(steps_per_epoch):
        loss = train_step(model)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

**Key Features**

* **AdamW with grouped parameters:**
  Automatically splits parameters into two groups —
  *decay group* (regular weights) and *no-decay group* (bias, LayerNorm, positional or CLS embeddings).
  This prevents over-regularization of normalization and bias terms.

* **Cosine learning-rate schedule:**
  Implements a warmup phase followed by cosine decay to `min_lr_scale × base_lr`, realized via `LambdaLR`.

* **Linear warmup:**
  Gradually ramps up LR during `warmup_epochs` to avoid instability in early training.


**When to Use**

* Training spectral Transformer models (e.g., ChemoMAE) requiring **cosine annealing** and **warmup scheduling**.
* Any experiment needing **stable early convergence** and **smooth LR decay** under weight-decay-aware optimization.

---

### `TrainerConfig` & `Trainer`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/trainer.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/training/trainer.py)

`TrainerConfig` and `Trainer` together form the **core training engine** of ChemoMAE. <br>
They provide a training loop for **masked reconstruction training**, with full support for **AMP (bf16/fp16)**, **TF32 acceleration**, **EMA parameter tracking**, **gradient clipping**, and **checkpointing / resume**. <br>
Training and validation history are stored automatically as JSON, enabling reproducible and resumable experiments.


```python
# === Basic usage (ChemoMAE reconstruction) ===
from chemomae.models import ChemoMAE
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.training.trainer import Trainer, TrainerConfig

# 1) Model and configuration
model = ChemoMAE(seq_len=256, latent_dim=64, n_blocks=32, n_mask=24)
cfg = TrainerConfig(
    out_dir = "runs",
    device = "cuda",
    amp = True,
    amp_dtype = "bf16",  # "bf16" | "fp16"
    enable_tf32 = False,
    grad_clip = 1.0,
    use_ema = True,
    ema_decay = 0.999,
    loss_type = "mse",   # "sse" | "mse"
    reduction = "mean",  # for sse/mse
    early_stop_patience = 20,
    early_stop_start_ratio = 0.5,
    early_stop_min_delta = 0.0,
    resume_from = "auto"
)

# 2) Optimizer and scheduler
optimizer = build_optimizer(model, lr=3e-4, weight_decay=1e-4)
scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader), epochs=100, warmup_epochs=5)

# 3) Trainer initialization
trainer = Trainer(model, optimizer, train_loader, val_loader, scheduler=scheduler, cfg=cfg)

# 4) Run training loop
history = trainer.fit(epochs=100)
print("Best validation:", history["best"])
```

**Key Features**

* **Automatic device & precision:**
  Detects CUDA/MPS/CPU automatically; supports AMP (`bf16` or `fp16`) and optional **TF32** acceleration.
  Uses `torch.amp.autocast` internally for efficient mixed-precision computation.


* **EMA tracking:**
  Maintains an exponential moving average of parameters (`ema_decay≈0.999`),
  automatically applied during validation and restored afterward.

* **Gradient safety:**
  Global gradient clipping (`clip_grad_norm_`) and automatic unscaling for fp16 stability.

* **Checkpointing & resume:**
  Saves full state (`model`, `optimizer`, `scheduler`, `scaler`, `EMA`, `history`) as
  `{out_dir}/checkpoints/last.pt` and `{out_dir}/checkpoints/best.pt`.
  `resume_from="auto"` resumes automatically from the latest checkpoint.

* **Early stopping:**
  Configurable via `early_stop_patience`, `early_stop_start_ratio`, and `early_stop_min_delta`.
  Training halts automatically when validation loss fails to improve.

* **History logging:**
  Per-epoch JSON log (`training_history.json`) including `train_loss`, `val_loss`, and `lr`,
  with atomic write safety for concurrent runs.

* **Loss flexibility:**
  Supports both `masked_mse` and `masked_sse`; computes loss only on masked (unseen) positions:
  $`L = \text{reduction}( (x_\text{recon} - x)^2 \odot (1 - \text{visible}) )`$.


**When to Use**

* For **masked reconstruction** training of 1D spectral MAE models (e.g., ChemoMAE).
* When requiring **precision control**, **EMA stabilization**, or **reproducible checkpoints**.
* Suitable for self-supervised pretraining.

---

### `TesterConfig` & `Tester`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/tester.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/training/tester.py)

`TesterConfig` and `Tester` provide a lightweight, reproducible **evaluation loop** for trained ChemoMAE models. <br>
They compute **masked reconstruction loss** (SSE/MSE) over a DataLoader with **AMP (bf16/fp16)** support, optional **fixed visible masks**, and **JSON logging** to append results under `out_dir`.  

```python
# === Basic evaluation (MSE over masked tokens) ===
import torch
from chemomae.training.tester import Tester, TesterConfig

cfg = TesterConfig(
    out_dir="runs",
    device="cuda",
    amp=True, 
    amp_dtype="bf16",
    loss_type="mse",          # or "sse"
    reduction="mean",         # "sum" | "mean" | "batch_mean"
)

tester = Tester(model, cfg)   # model: trained ChemoMAE
avg_loss = tester(test_loader)
print("Test loss:", avg_loss)  # float
```

```python
# === With a fixed visible mask (disable model's random masking) ===
import torch
seq_len = 256
visible = torch.ones(seq_len, dtype=torch.bool)   # (L,) or (B, L)

cfg = TesterConfig(fixed_visible=visible, loss_type="sse", reduction="batch_mean")
tester = Tester(model, cfg)
avg_loss = tester(test_loader)
```

**Key Features**

* **Masked-only error:** computes loss on **unseen (masked)** positions, consistent with MAE training. 
* **Loss options:** `loss_type ∈ {"mse","sse"}` with `reduction ∈ {"sum","mean","batch_mean"}` for flexible aggregation. 
* **Precision & speed:** **AMP** (`bf16` or `fp16`) via `torch.amp.autocast`; easy GPU/CPU switching through `device`. 
* **Fixed visibility masks:** optionally evaluate under a **given visible mask** instead of model-internal masking. 
* **History logging:** appends results to `{out_dir}/{history_filename}` (JSON) with safe atomic writes. 

**When to Use**

* Benchmarking **reconstruction quality** of ChemoMAE checkpoints under consistent masking protocols. 
* Running **reproducible test passes** (with fixed masks) or automated CI evaluations that log into a shared `{out_dir}/` directory. 

---

### `ExtractorConfig` & `Extractor`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/extractor.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/training/extractor.py)

`ExtractorConfig` and `Extractor` provide a **deterministic latent feature extraction** pipeline from a trained **ChemoMAE** in **all-visible mode** (no random masking).<br>
Supports **AMP (bf16/fp16)** inference, returns either **Torch** or **NumPy** arrays, and can **optionally save** features to disk with format inferred from file extension.  

```python
# === Basic usage: extract to memory ===
from chemomae.training.extractor import Extractor, ExtractorConfig

cfg = ExtractorConfig(
    device="cuda",   # "cuda" or "cpu"
    amp=True, 
    amp_dtype="bf16",
    return_numpy=True,   # return np.ndarray instead of torch.Tensor
    save_path=None       # don't save to disk
)
extractor = Extractor(model, cfg)     # model: trained ChemoMAE
Z = extractor(loader)                 # -> np.ndarray of shape (N, D)

# === Save to disk (.npy or .pt), independent of return type ===
cfg = ExtractorConfig(device="cuda", save_path="latent.npy", return_numpy=False)
Z_torch = Extractor(model, cfg)(loader)   # -> torch.Tensor; also writes "latent.npy"

# === Notes ===
# * The extractor builds an all-ones visible mask and calls model.encoder(x, visible).
# * Results are concatenated on CPU; AMP reduces VRAM/time on CUDA.
```

**Key Features**

* **All-visible encoding (deterministic):** builds an all-ones mask `(B, L)` and calls `model.encoder(x, visible)`; no randomness from masking. 
* **AMP inference:** optional `bf16`/`fp16` autocast on CUDA (`torch.amp.autocast`). 
* **Flexible I/O:** return **Torch** or **NumPy** (`return_numpy`), and **save** to `.npy` (via `np.save`) or others via `torch.save`. Saving and return formats are **independent**. 
* **Simple config:** `device`, `amp`, `amp_dtype`, `save_path`, `return_numpy`. 

**When to Use**

* To obtain **unit-sphere latents** (from ChemoMAE’s encoder) for **clustering** (CosineKMeans, vMF mixture) or **visualization** (UMAP/t-SNE with `metric="cosine"`). 

</details>


<details>
<summary><b><code>chemomae.clustering</code></b></summary>

---

### `CosineKMeans` & `elbow_ckmeans`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/cosine_kmeans.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/clustering/cosine_kmeans.py)

`CosineKMeans` implements **hyperspherical k-means** with cosine similarity: E-step assigns by maximum cosine; M-step updates centroids as **L2-normalized means**. 
It supports **k-means++ initialization**, **streaming CPU→GPU** for large datasets, and keeps centroids on the **unit sphere**. <br>
The objective reported as `inertia_` is the mean cosine dissimilarity ($`\mathrm{mean}(1-\cos)`$). <br>
`elbow_ckmeans` sweeps ($`K=1..k_{\max}`$) and selects an elbow via **curvature**.  

```python
# === Basic usage (fit → predict) ===
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans, elbow_ckmeans

X = torch.randn(10_000, 64)               # features (not necessarily normalized)
ckm = CosineKMeans(n_components=12, device="cuda", random_state=42)
ckm.fit(X)                                 # internal row-wise L2 normalization
labels = ckm.predict(X)                    # (N,)

# === Distances (1 - cos) ===
labels, dist = ckm.predict(X, return_dist=True)  # dist: (N, K)

# === Streaming for big data (CPU→GPU chunks) ===
ckm_big = CosineKMeans(n_components=50, device="cuda")
ckm_big.fit(X, chunk=10_000_000)            # bounded VRAM

# === Save / load centroids only ===
ckm.save_centroids("centroids.pt")
ckm2 = CosineKMeans(n_components=12).load_centroids("centroids.pt")

# === Model selection (elbow by curvature) ===
k_list, inertias, optimal_k, elbow_idx, kappa = elbow_ckmeans(
    CosineKMeans, X, device="cuda", k_max=30, chunk=1_000_000, verbose=True
)
print("Elbow K:", optimal_k)
```

**Key Features**

* **Objective & updates:** minimizes $`\mathrm{mean}(1-\cos(x,c))`$ ; E-step by argmax cosine, M-step by **normalized cluster means**. 
* **Internal normalization:** rows are L2-normalized internally; centroids are stored **unit-norm**. 
* **k-means++ init:** Uses cosine dissimilarity for sampling (not squared), ensuring reproducible seeding with `random_state`. 
* **Streaming (CPU→GPU):** `chunk>0` enables large-N clustering with bounded VRAM; also supported in `predict` and elbow sweep. 
* **Precision policy:** computations run in **fp32** internally (even with half/bf16 inputs). 
* **Empty clusters:** reinitialize by stealing **farthest samples** to keep K active. 
* **Elbow selection:** `elbow_ckmeans` returns `(k_list, inertias, optimal_k, elbow_idx, kappa)` using a **curvature-based** rule.  

**When to Use**

* Clustering **unit-sphere embeddings** (e.g., SNV-processed spectra or **ChemoMAE** latents) where **cosine geometry** is appropriate. 
* **Model selection** of K with an automatic, curvature-based elbow on large datasets (optionally with streaming).  

---

### `VMFMixture` & `elbow_vmf`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/vmf_mixture.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/clustering/vmf_mixture.py)

`VMFMixture` fits a von Mises–Fisher mixture model on the unit hypersphere using EM. <br>
Automatically normalizes inputs, enforces unit-norm means, estimates $\kappa$ from resultant lengths, and provides stable torch-only Bessel approximations with chunked E-steps. <br>
`elbow_vmf` sweeps $`K=1..k_{max}`$ and selects an elbow via **curvature**, using **BIC** or **mean NLL** as the score.  

```python
# === Basic usage (fit → predict / proba) ===
import torch
from chemomae.clustering.vmf_mixture import VMFMixture, elbow_vmf

# 1) Fit VMF on (N, D) features (not necessarily pre-normalized)
X = torch.randn(10_000_000, 64, device="cuda")
vmf = VMFMixture(n_components=32, device="cuda", random_state=42)
vmf.fit(X, chunk=1_000_000)                  # chunked E-step for large N

labels = vmf.predict(X, chunk=1_000_000)        # (N,)
resp   = vmf.predict_proba(X, chunk=1_000_000)  # (N, K)

# 2) Model selection by elbow (BIC or mean NLL)
k_list, scores, optimal_k, elbow_idx, kappa = elbow_vmf(
    VMFMixture, X, device="cuda", k_max=30, chunk=1_000_000,
    criterion="bic", random_state=42, verbose=True
)
print("Elbow K:", optimal_k)

# 3) Save / load a fitted model
vmf.save("vmf.pt")
vmf2 = VMFMixture.load("vmf.pt", map_location="cuda")
```

**Key Features**

* **EM on the sphere:** responsibilities on E-step; M-step updates unit directions and $`\kappa`$ from cluster **resultant length** (closed-form approx). 
* **Stable special functions:** torch-only blends for $`\log I_\nu(\kappa)`$ and the Bessel ratio $`\frac{I_{\nu+1}(\kappa)}{I_\nu(\kappa)}`$ (small/large-$`\kappa`$ expansions with smooth transition). 
* **Cosine k-means++ seeding:** hyperspherical initialization for mean directions. 
* **Chunked E-step:** stream CPU→GPU with `chunk` to handle very large datasets under limited VRAM. 
* **Diagnostics & criteria:** `loglik`, `bic`, and **curvature-based elbow** via `elbow_vmf(…, criterion={"bic","nll"})`.  
* **Lightweight persistence:** `save()` / `load()` restore mixture parameters and RNG state. 

**When to Use**

* Clustering **unit-sphere embeddings** (e.g., SNV/L2-normalized spectra or **ChemoMAE** latents) where **direction** (cosine geometry) is the signal. 
* **Model selection** of cluster count with **BIC**/**mean NLL** and an automatic **elbow** (curvature) on large datasets with streaming E-steps.  


---

### `silhouette_samples_cosine_gpu` & `silhouette_score_cosine_gpu`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/metric.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/clustering/metric.py)

Cosine-based **silhouette coefficients** with **GPU acceleration** for clustering evaluation. Rows are **L2-normalized internally** (zero rows stay zeros → cos=0, distance=1). <br>
The per-sample score uses $`d(x,y)=1-\cos(x,y)`$ with standard definitions of $`a_i`$, $`b_i`$, and $`s_i=\frac{b_i-a_i}{\max(a_i,b_i)}`$. <br>
The implementation is **O(NK)** and supports **chunked** computation to bound memory. 

```python
# === NumPy (CPU) ===
import numpy as np
from chemomae.clustering.metric import (
    silhouette_samples_cosine_gpu, silhouette_score_cosine_gpu
)

X = np.random.randn(100, 16).astype(np.float32)
labels = np.random.randint(0, 4, size=100)

s = silhouette_samples_cosine_gpu(X, labels, device="cpu")    # (100,)
score = silhouette_score_cosine_gpu(X, labels, device="cpu")  # float

# === PyTorch (GPU) ===
import torch
X_t = torch.randn(200, 32, device="cuda", dtype=torch.float32)
y_t = torch.randint(0, 5, (200,), device="cuda")

s_t = silhouette_samples_cosine_gpu(X_t, y_t, device="cuda", return_numpy=False)  # torch.Tensor on CUDA

# === Chunked evaluation (memory-bounded) ===
s_big = silhouette_samples_cosine_gpu(X_t, y_t, device="cuda", chunk=1_000_000)
```

**Key Features**

* **Cosine distance** only: internally computes $`1-\cos`$ after row-wise L2 normalization (zero vectors handled safely). 
* **GPU-vectorized** evaluation with optional **chunking** for $`b_i`$ to reduce peak VRAM. 
* **API parity** with `sklearn.metrics`: `silhouette_samples` & `silhouette_score` drop-in, specialized for cosine. 
* **Precision control**: supports `float16`/`bfloat16`/`float32` on GPU; final mean computed in `float32`. 

**When to Use**

* Validating clusters from **CosineKMeans** / **vMFMixture** on **unit-sphere**. 
* **Model selection**: compare mean silhouette across candidate K or algorithms under cosine geometry. 
</details>


<details>
<summary><b><code>chemomae.utils</code></b></summary>

---

### `set_global_seed`

* [Document](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/utils/seed.md)
* [Implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/src/chemomae/utils/seed.py)

`set_global_seed` provides a **unified seeding interface** for reproducible experiments across **Python**, **NumPy**, and **PyTorch**. <br>
It optionally enables **CuDNN deterministic mode**, ensuring full determinism in GPU computations (at the cost of performance). <br>
This function is typically called **once at the beginning** of every experiment to ensure consistent behavior across runs. 

```python
# === Basic usage ===
from chemomae.utils.seed import set_global_seed

# Fix randomness across Python, NumPy, and PyTorch
set_global_seed(42)  # deterministic CuDNN enabled by default

# === Disable CuDNN determinism (faster but non-reproducible) ===
set_global_seed(42, fix_cudnn=False)
```

**Key Features**

* **Unified random state control:** sets seeds for `random`, `numpy`, and `torch` (if available).
* **CuDNN determinism:**

  * `torch.backends.cudnn.deterministic = True`
  * `torch.backends.cudnn.benchmark = False`
    when `fix_cudnn=True`.
* **Torch-safe fallback:** if PyTorch is not installed, silently skips without error.
* **Environment hash fix:** enforces `PYTHONHASHSEED` for consistent hashing in Python.
* **Lightweight helper:** complements `enable_deterministic()` for runtime toggling. 

**When to Use**

* At the **start of any experiment** (training, testing, or clustering) to ensure reproducibility.
* Before launching **multi-GPU** or **EMA**-based runs where consistent initialization is critical.
* In conjunction with `enable_deterministic(True)` when strict determinism (bitwise identical results) is required. 
</details>

---

## License

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
