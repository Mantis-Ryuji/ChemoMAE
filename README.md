<h1 align="center">ChemoMAE</h1>

[![PyPI version](https://img.shields.io/pypi/v/chemomae.svg)](https://pypi.org/project/chemomae/)
[![CI](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/chemomae.svg)](https://pypi.org/project/chemomae/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/LICENSE)

**ChemoMAE is a PyTorch library for learning representations of one-dimensional spectra, clustering directional features, and evaluating spatial label maps.**
Its preprocessing, model, training, clustering, and metric components can be used
independently or combined in a pipeline.

<p align="center">
  <img src="https://raw.githubusercontent.com/Mantis-Ryuji/ChemoMAE/main/images/chemomae.svg" alt="Example hyperspectral spatial mapping workflow: encoder, masked spectral reconstruction, and clustering with labels mapped back to pixel locations" width="640">
</p>

> **Example application: unsupervised spatial mapping of hyperspectral images.**
> (a) A Transformer encoder projects spectra into L2-normalized representations.
> (b) In the illustrated training setup, augmented visible patches are used to reconstruct the masked regions of the original spectra.
> (c) The trained encoder receives complete spectra; their representations are clustered, and the cluster labels are mapped back to the original pixel locations.

These guides cover **v0.2.4, in preparation and not yet published to PyPI**.
README links lead to the maintained
repository documentation; the [v0.2.3 snapshot](https://github.com/Mantis-Ryuji/ChemoMAE/tree/v0.2.3/docs)
retains the documentation distributed with that release. See the
[release notes](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/CHANGELOG.md)
for changes and compatibility notes.

## Choose the components you need

| Task | Components and guide |
| --- | --- |
| Standardize each spectrum or select diverse rows | [SNV](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/preprocessing/snv.md), [cosine FPS](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/preprocessing/dowmsampling.md) |
| Learn a spectral representation | [ChemoMAE](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/models/chemo_mae.md), [reconstruction losses](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/models/losses.md) |
| Train with a supplied loop or your own PyTorch loop | [Trainer and hooks](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/trainer.md), [optimizer/scheduler](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/optim.md), [augmentation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/augmenter.md) |
| Evaluate reconstruction or extract features | [Tester](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/tester.md), [Extractor](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/extractor.md) |
| Cluster an existing feature matrix | [CosineKMeans](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/cosine_kmeans.md) for hard assignments; [VMFMixture](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/vmf_mixture.md) for mixture probabilities |
| Evaluate a partition or a spatial map | [Cosine silhouette](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/metric.md), [Local Label Agreement](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/clustering/spatial.md) |
| Save a model or control random streams | [Artifacts and checkpoints](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/models/persistence.md), [RNG utilities](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/utils/seed.md) |

SNV is optional: use it when removing each spectrum's mean and scale suits your
data. Clustering accepts feature matrices without requiring a ChemoMAE model.
Spatial evaluation requires a label map and validity mask; ordinary spectral
learning and feature extraction require no pixel coordinates.

## Quick Start

Install a CPU or CUDA build of PyTorch appropriate for your environment using
the [official installation selector](https://pytorch.org/get-started/locally/),
then install this v0.2.4 source checkout from the repository root:

```bash
python -m pip install -e .
```

Python >=3.10 and PyTorch >=2.1 are required. CI covers Python 3.10–3.13 with
selected CPU PyTorch builds, including Python 3.10/PyTorch 2.1; it does not test
every combination. With PyTorch 2.1, use NumPy `>=1.24,<2` for NumPy interoperability.

For the published v0.2.3 package, use `python -m pip install "chemomae==0.2.3"`
and its versioned documentation. The examples below use the new v0.2.4 API.

## ChemoMAE Example

This complete CPU example trains a small model, reloads its inference artifact,
and clusters its features. SNV and clustering illustrate optional composition.
The synthetic data, two epochs, and K=3 demonstrate APIs, not recommended tuning
values. Outputs go to a new temporary directory whose path is printed.

```python
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.preprocessing import snv
from chemomae.models import ChemoMAE
from chemomae.training import Trainer, TrainerConfig, Extractor, ExtractorConfig
from chemomae.clustering import CosineKMeans

torch.manual_seed(42)
data_stream = torch.Generator().manual_seed(7)
spectra = snv(torch.randn(80, 64, generator=data_stream))
train_x, test_x = spectra[:64], spectra[64:]

def loader(x: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(x), batch_size=16, shuffle=False)

device = torch.device("cpu")
run_dir = Path(tempfile.mkdtemp(prefix="chemomae-"))
model = ChemoMAE(
    seq_len=64, n_patches=8, n_mask=4, d_model=16,
    nhead=4, num_layers=1, latent_dim=8,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(
    model, optimizer, loader(train_x),
    cfg=TrainerConfig(
        out_dir=run_dir, device=device, resume_from=None, use_ema=False,
        progress=False, verbose=False,
    ),
)
result = trainer.fit(epochs=2)

assert result["final_artifact"] is not None
inference_model = ChemoMAE.load(result["final_artifact"], device=device)
extractor = Extractor(
    inference_model,
    ExtractorConfig(representation="normalized_latent", output_device="cpu"),
)
train_features = extractor(loader(train_x))
test_features = extractor(loader(test_x))

clusterer = CosineKMeans(n_components=3, max_iter=30, device=device, random_state=42)
clusterer.fit(train_features)
test_labels = clusterer.predict(test_features)
clusterer.save_centroids(run_dir / "clusters.pt")
assert train_features.shape == (64, 8) and test_features.shape == (16, 8)
assert test_labels.shape == (16,)
print(test_labels, clusterer.converged_, clusterer.stop_reason_)
print("Outputs:", run_dir)
```

### Choices to keep explicit

The example selects a fresh output directory, raw final weights, and CPU.
Keep these choices explicit when adapting it:

| You want to… | What to specify |
| --- | --- |
| Start a fresh training run | Use a new `out_dir` and `resume_from=None`. Existing training artifacts cause an error. The default `resume_from="auto"` can resume an earlier run in the same directory. |
| Continue a saved run | Create a new Trainer with `resume_from=checkpoint_path`. `fit(epochs=10)` trains through epoch 10: a checkpoint after epoch 7 leaves epochs 8–10 to run. |
| Reload for inference | Pass `result["final_artifact"]` directly to `ChemoMAE.load`; it is an absolute path to the selected raw/EMA artifact, or `None` when no artifact is exported. `final_model` continues to name a weight file for `load_state_dict`. |
| Bring NumPy spectra | Explicitly convert to the model's floating dtype before training, for example `torch.as_tensor(array, dtype=next(model.parameters()).dtype)`. NumPy arrays often start as float64 and remain float64 through SNV. |
| Choose CPU or CUDA | Move the model to the chosen device before constructing the optimizer, and pass `device` to the clusterer. Trainer/Extractor follow the model by default; CosineKMeans/VMFMixture default to `"cuda"`, including on a CPU-only machine. |

See [starting or resuming training](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/training/trainer.md#starting-fresh-or-resuming)
and the [output-file guide](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/models/persistence.md#raw-and-ema-exports)
for the exact defaults and raw/EMA file choices. CPU use requires explicitly
passing `device="cpu"` to clustering; CUDA use requires an available CUDA device.

Continue with the [staged workflow tutorial](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/tutorials/workflow.md)
for reconstruction evaluation and optional augmentation, resume, clustering,
spatial analysis, and reporting. The [documentation index](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/README.md)
also provides direct routes to each API.
For configuration comparisons and memory estimates, see
[planning a first experiment](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/tutorials/first_experiment.md).

## Model and workflow choices

ChemoMAE splits a spectrum into contiguous patches. Visible patch tokens and a
CLS token enter a Transformer encoder; the projected CLS vector is the shared
bottleneck from which a decoder reconstructs every spectral channel. The
[model guide](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/docs/models/chemo_mae.md)
describes masks, representations, and boundary behavior.

- **Reconstruction objective:** Trainer defaults to hidden-channel loss. For a
  full-spectrum autoencoder, choose `n_mask=0` and `loss_region="all"` explicitly.
- **Augmentation:** `SpectraAugmenter` offers fractional channel shifts and tangent
  noise. Trainer keeps the pre-augmentation input as the target when one is supplied.
- **Decoder and latent:** one decoder layer is affine; the default two-layer
  decoder is an MLP. Latents are L2-normalized by default; set
  `latent_normalize=False` when unconstrained latents suit the task.
- **Features:** `model.encode` and `Extractor` expose CLS, raw projected, and
  normalized representations with all patches visible. Use `model.encode` in a
  custom gradient-enabled loop for end-to-end fine-tuning.
- **Training control:** Trainer provides reconstruction training, hooks, EMA, and
  completed-epoch resume. Model selection, downstream objectives, and data splits
  belong to the application. Model, augmentation, and loss primitives also work
  in ordinary PyTorch loops.

When evaluating generalization to unseen groups, fit learned components on the
training groups and retain them for held-out prediction. Descriptive analysis
of an existing dataset may use that dataset for fitting. Reconstruction loss,
feature separation, and spatial coherence measure different properties; select
the diagnostics that address your intended use.

## Research background

ChemoMAE was extracted into a reusable library from spectral representation
learning research. The associated research repository is
[WoodDegradationMap](https://github.com/Mantis-Ryuji/WoodDegradationMap).
A manuscript with an NIR-HSI case study is in preparation; its publication link
will be added when available. Library defaults and examples are documented
independently of that study's settings.

## License

ChemoMAE is distributed under the [Apache License 2.0](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/LICENSE).
See [NOTICE](https://github.com/Mantis-Ryuji/ChemoMAE/blob/main/NOTICE) for notices.
