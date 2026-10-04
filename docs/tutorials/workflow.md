# Spectral learning and optional downstream workflows

This tutorial uses ChemoMAE v0.2.4 with small synthetic CPU inputs. Complete
sections 1–4 to train a model and extract features. The later sections show
optional preprocessing, augmentation, resume, evaluation, clustering, spatial
analysis, and reporting. You can also run all Python blocks in order in one
session; they define their inputs and use only runtime dependencies.

Seeds, model size, two epochs, and K=3 keep the example small. They are examples
of API usage, not recommended settings for a dataset. The model, clustering,
and metrics can each be used without the complete workflow.

## 1. Set up and prepare spectra

Install ChemoMAE v0.2.4:

```bash
pip install chemomae==0.2.4
```

Every row is a spectrum, with shape `(N, L)`. The example generates independent
rows from three templates. It uses SNV to compare relative spectral shapes;
omit or replace this preprocessing when mean and scale carry useful information.
All samples passed to a given model need the same channel count and ordering.
For held-out evaluation, define the appropriate independent
group splits before fitting learned components or sampling training pixels.

```python
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import chemomae
from chemomae.preprocessing import snv
from chemomae.models import ChemoMAE
from chemomae.training import (
    Trainer, TrainerConfig, Extractor, ExtractorConfig,
    SpectraAugmenter, build_optimizer, build_scheduler,
)
from chemomae.utils import set_global_seed

assert chemomae.__version__ == "0.2.4"
device = torch.device("cpu")
model_dtype = torch.float32
set_global_seed(42)
run_dir = Path(tempfile.mkdtemp(prefix="chemomae-workflow-"))
length = 64
snv_eps = 1e-12
data_stream = torch.Generator(device="cpu").manual_seed(24)
axis = torch.linspace(0.0, 1.0, length)
templates = torch.stack([
    1.0 + torch.exp(-((axis - center) / 0.12) ** 2)
    for center in (0.2, 0.5, 0.8)
])

def sample_spectra(count: int) -> torch.Tensor:
    identities = torch.randint(0, 3, (count,), generator=data_stream)
    noise = 0.03 * torch.randn(count, length, generator=data_stream)
    return templates[identities] + noise

# Simulate arrays from a NumPy-based measurement pipeline.
train_array = sample_spectra(64).numpy().astype(np.float64)
validation_array = sample_spectra(16).numpy().astype(np.float64)
test_array = sample_spectra(16).numpy().astype(np.float64)
train_x = torch.as_tensor(snv(train_array, eps=snv_eps), dtype=model_dtype)
validation_x = torch.as_tensor(snv(validation_array, eps=snv_eps), dtype=model_dtype)
test_x = torch.as_tensor(snv(test_array, eps=snv_eps), dtype=model_dtype)
assert train_x.dtype == model_dtype

def ordered_loader(spectra: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(spectra), batch_size=16, shuffle=False, num_workers=0)

train_loader = DataLoader(
    TensorDataset(train_x), batch_size=16, shuffle=True, num_workers=0,
)
print("Outputs:", run_dir)
```

SNV has no fitted population statistics. Constant rows become zero; see its
[precision, epsilon, and short-spectrum contract](../preprocessing/snv.md).
The template identities are not supplied to training or clustering.

When replacing these arrays with your NumPy spectra, keep the explicit dtype
conversion. SNV preserves NumPy float64, and `torch.from_numpy` alone also
preserves it. Here `model_dtype` controls both the input conversion and model
construction; the CPU DataLoader batches are then transferred by Trainer.
With ChemoMAE, Trainer preserves input dtype and reports incompatible
input/model dtypes; Extractor casts floating inputs to its model's dtype.
Explicit preparation makes the same arrays suitable for both paths.

The same explicit `device` is used for the model, training, extraction, and
clustering. Keep it at `"cpu"` for this walkthrough. When adapting to CUDA, move
the model before creating its optimizer and pass the chosen device to clustering.
Trainer/Extractor follow the model when device is omitted; clusterers default to
CUDA. See [device and run choices](../../README.md#choices-to-keep-explicit).

## 2. Train a reconstruction model

The spectrum length must be divisible by `n_patches`, and `nhead` must divide
`d_model`. Here eight patches contain eight channels each, and four are hidden.
Visible patch tokens and a CLS token enter the encoder; the decoder reconstructs
every channel from the projected CLS bottleneck. `loss_region="masked"` selects
hidden channels for the loss.

This example uses the default two-layer MLP decoder, no augmentation, and raw
final weights. One decoder layer selects an affine decoder. Move the model to
its computation device before constructing its optimizer.

```python
model_config = {
    "seq_len": length, "n_patches": 8, "n_mask": 4,
    "d_model": 16, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 32, "dropout": 0.0,
    "latent_dim": 8, "latent_normalize": True, "decoder_num_layers": 2,
}

def make_trainer(
    output_dir: Path,
    resume_from: Path | None = None,
    augmenter: SpectraAugmenter | None = None,
) -> Trainer:
    model = ChemoMAE(**model_config).to(device=device, dtype=model_dtype)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = build_scheduler(
        optimizer, steps_per_epoch=len(train_loader), epochs=2,
        warmup_epochs=0, min_lr_scale=0.1,
    )
    return Trainer(
        model, optimizer, train_loader, scheduler=scheduler, augmenter=augmenter,
        cfg=TrainerConfig(
            out_dir=output_dir, device=device, resume_from=resume_from,
            amp=False, enable_tf32=False, use_ema=False,
            loss_type="mse", loss_region="masked", reduction="mean",
            progress=False, verbose=False,
        ),
    )

trainer = make_trainer(run_dir)
result = trainer.fit(epochs=2)
assert result["completed"] and result["optimizer_updates"] == 8
assert result["amp_skips"] == 0
```

This factory sets `resume_from=None` for a fresh run in the new `run_dir`.
Re-running the training section alone requires a new output directory; existing
artifacts cause an error. The library default is `resume_from="auto"`, which can
resume the checkpoint in the same directory. The
[optional resume section](#optional-resume-a-completed-epoch) shows an explicit path.

`fit(epochs=2)` sets the final absolute epoch, so resuming after epoch 1 runs only
epoch 2. Scheduler and EMA updates, when
configured, follow successful optimizer updates; AMP overflow skips are counted
separately. See [LR indexing](../training/optim.md) and [Trainer hooks](../training/trainer.md).
For a full-spectrum autoencoder, explicitly use `n_mask=0` and
`loss_region="all"`. The loss region is never inferred from the mask count.

## 3. Save and reload the selected model

Trainer's default outputs include weights, a config-and-weights artifact,
history, and a training checkpoint. This example selects raw final weights.
Choose raw or EMA weights deliberately before downstream inference.
Here `use_ema=False`, so `result["final_model"]` is `"last_model.pt"`.
That is a weight file for `load_state_dict`. Use `result["final_artifact"]` to
load the corresponding config-and-weights artifact directly; it is an absolute
path, including when output filenames are customized. With EMA enabled and
exported, it selects the corresponding EMA artifact instead.

```python
assert result["final_model"] == "last_model.pt"
assert result["final_artifact"] is not None
inference_model = ChemoMAE.load(result["final_artifact"], device=device)
selected_artifact = Path(result["final_artifact"])
assert not inference_model.training
assert selected_artifact.is_file()
```

`ChemoMAE.load` restores the constructor configuration and saved floating dtype,
moves the model to the requested device, and returns it in eval mode. Training
checkpoints additionally retain optimizer and progress state for resume. Output
paths and enablement are configurable. See [persistence](../models/persistence.md).
`final_artifact` is `None` when no selected artifact was exported, such as when
artifact export is disabled or the model does not support that format.

## 4. Extract features

Extractor uses all-visible encoder inference. Choose CLS, raw projected,
normalized projected, or configured `latent` output. Features can feed a
visualization, clusterer, or downstream predictor; coordinates are unnecessary.

```python
extractor = Extractor(
    inference_model,
    ExtractorConfig(
        device=device, representation="normalized_latent",
        output_type="tensor", output_device="cpu", output_dtype=torch.float32,
    ),
)
train_features = extractor(ordered_loader(train_x))
test_features = extractor(ordered_loader(test_x))
assert train_features.shape == (64, 8)
assert test_features.shape == (16, 8)
assert torch.isfinite(test_features).all()
```

Extraction disables dropout and random masking, then restores the original
module modes. Zero projections remain zero; very small projections follow the
normalization epsilon. Latents have no zero-mean constraint, and their cosine
similarities need not match those of the input spectra.

For larger outputs, consume one batch at a time:

```python
stream = extractor.iter_transform(ordered_loader(test_x))
try:
    for features in stream:
        print(features.shape)
        # Pass each batch to your writer or downstream predictor.
finally:
    stream.close()
```

Streaming avoids concatenating the full feature output and does not save it
implicitly. Loader order is caller-controlled. For end-to-end fine-tuning, call
`model.encode` in your own gradient-enabled loop; Extractor returns detached
features. See the [representation and mode contracts](../training/extractor.md).

## Optional: select training rows with FPS

This section uses `train_x` from section 1. FPS selects diverse directions and
returns their original indices. It does not preserve sampling density or balance
groups. The remaining tutorial continues to use all training rows.

```python
from chemomae.preprocessing import cosine_fps_downsample

fps_x, fps_indices = cosine_fps_downsample(
    train_x, ratio=0.5, seed=42, device="cpu",
    return_numpy=False, return_indices=True,
)
assert fps_x.shape == (32, length)
assert torch.equal(fps_x, train_x[fps_indices])
```

To train on the subset, build a loader from `fps_x` explicitly and use that
loader's length for the scheduler. Keep indices when rows have associated metadata.

## Optional: add spectral augmentation

This section uses the factory from section 2 to configure a separate run.
Fractional shifts and tangent noise perturb the model input before masking;
Trainer retains the input from before augmentation as the reconstruction target.
That target still contains any variation present in the original input.

```python
from chemomae.training import SpectraAugmenterConfig

augmentation_config = SpectraAugmenterConfig(
    shift_prob=0.3, shift_delta_range=(-0.5, 0.5),
    noise_prob=0.5, noise_angle_deg_range=(0.5, 1.0),
)
augmented_trainer = make_trainer(
    run_dir / "augmented", augmenter=SpectraAugmenter(augmentation_config),
)
# Call augmented_trainer.fit(epochs=2) when you want to run this alternative.
assert augmented_trainer.augmenter is not None
```

Choose strengths for your data. Shift displacement is measured in channels;
noise angle is measured in degrees. A small shift does not imply a small angle
for every spectrum. Default reprojection preserves SNV-compatible geometry,
subject to the [augmentation boundary conditions](../training/augmenter.md).
Caller-owned generators need explicit persistence through checkpoint hooks.

## Optional: resume a completed epoch

This section uses section 2's factory and data for a separate demonstration.
Both constructions retain the same two-epoch scheduler budget. The first call
completes epoch one; the resumed call completes epoch two.

```python
resume_dir = run_dir / "resume-example"
first = make_trainer(resume_dir)
first.fit(epochs=1)
resumed = make_trainer(resume_dir, resume_dir / "checkpoints" / "last.pt")
resumed_result = resumed.fit(epochs=2)
assert resumed_result["completed"] and resumed_result["epochs"] == 2
assert resumed_result["optimizer_updates"] == 8
```

An interrupted incomplete epoch is replayed from the last completed checkpoint.
Checkpoints validate the saved model and training contracts and restore standard
global RNG. Recreate the same optimizer, scheduler recipe, data, and ordering.
Independent generators and worker/external state require caller restoration.
Matching seeds alone do not promise identical results across environments.
See [resume requirements](../models/persistence.md).

## Optional: evaluate reconstruction

This section uses the model from section 3 and held-out rows from section 1.
An all-visible mask with `loss_region="all"` evaluates full-spectrum MSE without
random masking or augmentation. It differs from the masked training objective.
The validation and test names illustrate separation from fitting; the example
performs no validation-based model selection.

```python
from chemomae.training import Tester, TesterConfig

tester = Tester(
    inference_model,
    TesterConfig(
        device=device, amp=False, loss_type="mse", loss_region="all",
        reduction="mean", fixed_visible=torch.ones(length, dtype=torch.bool),
        log_history=False, progress=False,
    ),
)
validation_mse = tester(ordered_loader(validation_x))
test_mse = tester(ordered_loader(test_x))
assert math.isfinite(validation_mse) and math.isfinite(test_mse)
print("Validation MSE:", validation_mse, "Test MSE:", test_mse)
```

For masked evaluation, choose `loss_region="masked"` and specify the mask or RNG
protocol. Dataset reductions aggregate selected errors across the loader;
empty selections fail. See [Tester](../training/tester.md). Reconstruction error
measures prediction accuracy for that objective, not downstream task quality.

## Optional: cluster directional features

This section uses section 4's features. It fits centers on training rows and
keeps them fixed for held-out prediction. For descriptive clustering, fitting
all rows being described is a different, valid use. CosineKMeans also accepts
compatible feature matrices from other models or preprocessing methods.

```python
from chemomae.clustering import CosineKMeans, silhouette_score_cosine_gpu

clusterer = CosineKMeans(n_components=3, max_iter=30, device=device, random_state=42)
clusterer.fit(train_features, chunk=32)
test_labels = clusterer.predict(test_features, chunk=32)
cluster_path = run_dir / "clusters.pt"
clusterer.save_centroids(cluster_path)
reloaded_clusterer = CosineKMeans(n_components=3, device=device)
reloaded_clusterer.load_centroids(cluster_path)
assert torch.equal(test_labels, reloaded_clusterer.predict(test_features, chunk=32))

used_classes = int(torch.unique(test_labels).numel())
silhouette = None
if 2 <= used_classes < len(test_labels):
    silhouette = silhouette_score_cosine_gpu(
        test_features, test_labels, device=device, chunk=8,
    )
print(clusterer.converged_, clusterer.stop_reason_, "Silhouette:", silhouette)
```

Choose K for the desired partition; numeric IDs do not provide semantic labels.
When evaluating held-out performance, choose settings without tuning on final
test scores. Silhouette describes compactness and separation in the supplied
representation. One-class predictions have undefined silhouette.

[VMFMixture](../clustering/vmf_mixture.md) provides a probabilistic alternative;
its `elbow_vmf` helper selects curve curvature, not the minimum BIC or an
application-specific optimum. Inspect its score curve and documented edge cases.
CPU CosineKMeans does not chunk its similarity matrix. CUDA fitting can stream
CPU feature chunks, while retaining the full CPU input. Requested distances
still occupy `(N, K)` memory. Silhouette chunks only its similarity tile.
See the [memory budget example](first_experiment.md#estimate-memory-by-operation)
before scaling these calls to a full image collection.

## Optional: evaluate a spatial label map

This section uses the templates from section 1 and the extractor and clusterer
above. For spatial analysis, retain actual coordinates. The example constructs
an `(H, W, L)` cube with an invalid edge and hole; unrelated rows cannot be
reshaped into a meaningful image. LLA can also evaluate an existing label map
without any model or clusterer from this tutorial.

```python
from chemomae.clustering import local_label_agreement

height, width = 12, 16
row, column = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
regions = (column // 6).clamp_max(2)
cube = templates[regions] + 0.03 * torch.randn(
    height, width, length, generator=data_stream,
)
valid_mask = torch.ones(height, width, dtype=torch.bool)
valid_mask[0] = False
valid_mask[3:5, 6:8] = False
spatial_features = extractor(ordered_loader(snv(cube[valid_mask], eps=snv_eps)))
label_map = torch.zeros(height, width, dtype=torch.int64)
label_map[valid_mask] = clusterer.predict(spatial_features, chunk=32)
lla = local_label_agreement(
    label_map, valid_mask, windows=(3, 5, 9), device=device, class_chunk=2,
)
assert label_map.shape == (12, 16) and lla.valid_pixels == 172
for window in lla.windows:
    print(window.window, window.score, window.raw_agreement, window.undefined_reasons)
```

Validity is determined by the mask, so label zero remains an ordinary class.
LLA counts valid directed neighbor pairs, omits the center and invalid/outside
neighbors, and applies occupancy-based chance correction. Window values 3/5/9
are widths. Negative scores are retained; undefined scores have explicit
reasons. Spatial agreement alone does not establish semantic correctness.
See the [definition, precision, and memory contract](../clustering/spatial.md).

To display the result with background masked:

```python
import numpy as np
import matplotlib.pyplot as plt

image = np.ma.array(label_map.numpy(), mask=~valid_mask.numpy())
plt.imshow(image, cmap="tab10", interpolation="nearest")
plt.title("Synthetic spatial cluster IDs")
plt.colorbar()
plt.show()
```

## Optional: save a workflow report

This final section assumes the evaluation, clustering, and spatial examples
were run. Adapt the recorded fields to the components you use. Model artifacts
contain architecture and weights; they do not contain your data identities,
preprocessing choices, split definitions, or row-to-coordinate mapping.

```python
import json
from dataclasses import asdict

def json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

report = {
    "chemomae_version": chemomae.__version__, "torch_version": str(torch.__version__),
    "device": str(device), "model_dtype": str(model_dtype),
    "global_seed": 42, "data_seed": 24,
    "model_config": inference_model.get_config(),
    "preprocessing": {"method": "snv", "eps": snv_eps},
    "selected_artifact": selected_artifact.name, "clustering": {"k": 3, "seed": 42},
    "synthetic_counts": {"train": len(train_x), "validation": len(validation_x), "test": len(test_x)},
    "validation_mse": validation_mse, "test_mse": test_mse,
    "silhouette": silhouette, "lla": asdict(lla),
}
(run_dir / "report.json").write_text(
    json.dumps(json_ready(report), indent=2, allow_nan=False), encoding="utf-8",
)
torch.save({"labels": label_map, "valid_mask": valid_mask}, run_dir / "spatial_map.pt")
assert (run_dir / "report.json").is_file()
assert (run_dir / "spatial_map.pt").is_file()
print("Workflow examples completed. Outputs:", run_dir)
```

Undefined metric values become JSON `null`; LLA reasons remain in the report.
Add the data and selection context required by your application. These checks
establish API behavior, not a claim about representation or clustering quality.

## Troubleshooting

- A patch-divisibility error means the spectrum length and patch count disagree.
- An existing-output error requires an explicit resume checkpoint or a fresh directory.
- A checkpoint mismatch requires the saved architecture and compatible training recipe.
- An input/model dtype error requires explicit conversion at the data boundary;
  for the FP32 model above, use `torch.as_tensor(array, dtype=model_dtype)`.
- Undefined LLA or silhouette should retain its reason, rather than become a zero score.
- Choose device, precision, and storage explicitly when moving these CPU examples to CUDA.

For custom masks, ordering, or random streams, use the
[Trainer hooks and plain PyTorch loop](../training/trainer.md).
For configuration comparisons and memory estimates, see
[planning a first experiment](first_experiment.md).
