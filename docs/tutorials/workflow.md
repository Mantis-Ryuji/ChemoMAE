# ChemoMAE v0.2.3 workflow tutorial

This tutorial explains preprocessing, reconstruction training, epoch-boundary
resume, evaluation, feature extraction, clustering, spatial LLA, and persistence.
The examples use small synthetic spectra and only ChemoMAE's runtime dependencies.
Read the code blocks in order in one Python session. They are written examples;
execution, timings, and CPU/CUDA equivalence have not yet been verified.

The chosen seeds, model size, augmentation strengths, two epochs, and K=3 are
illustrative API settings. They do not define a validated experimental recipe.

## 1. Install the matching version

These APIs target v0.2.3. In a development checkout, install from its root:

```bash
pip install -e .
```

After v0.2.3 is published, install that version with `pip install chemomae==0.2.3`.
Use an appropriate CPU or CUDA PyTorch installation for your environment. This
tutorial explicitly uses CPU and disables AMP/TF32.

```python
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import chemomae
from chemomae.preprocessing import snv, cosine_fps_downsample
from chemomae.models import ChemoMAE
from chemomae.training import (
    Trainer, TrainerConfig, Tester, TesterConfig, Extractor, ExtractorConfig,
    SpectraAugmenter, SpectraAugmenterConfig, build_optimizer, build_scheduler,
)
from chemomae.clustering import (
    CosineKMeans, local_label_agreement, silhouette_score_cosine_gpu,
)
from chemomae.utils import set_global_seed

assert chemomae.__version__ == "0.2.3"
device = torch.device("cpu")
set_global_seed(42)
run_dir = Path(tempfile.mkdtemp(prefix="chemomae-v023-"))
print("Outputs:", run_dir)
```

## 2. Create and standardize spectra

Every row represents one spectrum: shape `(N, L)`. This example uses L=64 to keep
the model small. Three simple spectral templates generate independent training,
validation, and test rows. Their template identities are not passed to training
or clustering. For measured data, the consuming project must define independent
specimen/group splits before sampling pixels or fitting any learned transform.

```python
length = 64
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

train_x = snv(sample_spectra(64))
validation_x = snv(sample_spectra(16))
test_x = snv(sample_spectra(16))

def ordered_loader(spectra: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(spectra), batch_size=16, shuffle=False, num_workers=0)

train_loader = DataLoader(
    TensorDataset(train_x), batch_size=16, shuffle=True, num_workers=0,
)
```

`snv` centers each row and divides by its sample standard deviation plus `eps`.
It acts independently on each row, so there are no fitted population statistics
to transfer between splits. Constant rows become zero. The default uses ddof=1;
read the [SNV contract](../preprocessing/snv.md) for short spectra and precision.
Choose preprocessing based on the measurement and purpose of your own data.

Optional FPS selects representative training rows and retains their row indices:

```python
fps_x, fps_indices = cosine_fps_downsample(
    train_x, ratio=0.5, seed=42, device="cpu",
    return_numpy=False, return_indices=True,
)
```

The rest of this example uses all training rows. If using FPS for fitting, record
the selected indices and build the training loader from `fps_x` explicitly.

## 3. Configure a model and its training recipe

The spectrum length must be divisible by the patch count: here 64 / 8 = 8 bands
per patch. Four patches are hidden during ordinary training. Attention heads
must divide `d_model`. The library validates these settings before construction.
It does not change the channel count of your data to match a model default.

```python
model_config = {
    "seq_len": length, "n_patches": 8, "n_mask": 4,
    "d_model": 16, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 32, "dropout": 0.0,
    "latent_dim": 8, "latent_normalize": True, "decoder_num_layers": 2,
}
augmentation_config = SpectraAugmenterConfig(
    shift_prob=0.3, shift_delta_range=(-0.5, 0.5),
    noise_prob=0.5, noise_angle_deg_range=(0.5, 1.0),
)

def make_trainer(resume_from: Path | None = None) -> Trainer:
    model = ChemoMAE(**model_config).to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = build_scheduler(
        optimizer, steps_per_epoch=len(train_loader), epochs=2,
        warmup_epochs=0, min_lr_scale=0.1,
    )
    return Trainer(
        model, optimizer, train_loader, scheduler=scheduler,
        augmenter=SpectraAugmenter(augmentation_config),
        cfg=TrainerConfig(
            out_dir=run_dir, device=device, resume_from=resume_from,
            amp=False, enable_tf32=False, use_ema=True, ema_decay=0.9,
            loss_type="mse", loss_region="masked", reduction="mean",
            progress=False, verbose=True,
        ),
    )
```

Move the model before creating its optimizer. Inspect `optimizer.param_groups`
when changing weight decay or the learning-rate recipe. The scheduler and EMA
advance only after successful optimizer updates; AMP overflow skips are counted
separately. See [optimizer/scheduler details](../training/optim.md).

The default augmenter receives clean targets from Trainer and perturbs only the
inputs. Here its randomness uses standard Torch streams, which checkpoints
capture. If supplying an independent `torch.Generator`, persist it through the
[public checkpoint hooks](../training/trainer.md#custom-ordering-masks-and-caller-state).

## 4. Train and resume at a completed epoch

```python
first = make_trainer()
first.fit(epochs=1)

resumed = make_trainer(run_dir / "checkpoints" / "last.pt")
result = resumed.fit(epochs=2)
print(result)
```

`epochs=2` is the final absolute epoch, so the second call performs epoch two.
An interrupted incomplete epoch is replayed from the last completed checkpoint.
Recreate the same model, optimizer, scheduler, augmentation, data, and ordering
recipe. The scheduler above uses the two-epoch budget in both constructions.

Checkpoints validate the schema, full ChemoMAE configuration, training settings,
component types, and saved tensors. They restore optimizer/scheduler/scaler/EMA
state and standard global RNG. Independent generators and worker/external state
need caller-owned hooks. Exact resumed trajectories require matching software,
device, data, and deterministic conditions; the corresponding tests remain unrun.

Use `progress=False` for batch progress and `verbose=False` for epoch summaries.
History/checkpoint/raw/EMA paths are configurable separately. Setting an output
path to `None` disables it; disabled checkpoints require `resume_from=None`.

## 5. Reload the selected inference artifact

Training leaves raw weights in memory. This recipe selects EMA-last in advance,
so reload its exported artifact before evaluating or extracting features:

```python
assert result["final_model"] is not None
selected_weights = run_dir / str(result["final_model"])
selected_artifact = selected_weights.with_name(
    selected_weights.stem + ".artifact" + selected_weights.suffix
)
inference_model = ChemoMAE.load(selected_artifact, device=device)
```

`ChemoMAE.load` reconstructs the full configuration, preserves the saved floating
dtype, and returns an eval-mode model. `*.artifact.pt` bundles configuration and
weights; ordinary `*.pt` weight files remain useful for caller-created models.
Training checkpoints serve a separate resume purpose. See [artifact formats](../models/persistence.md).

## 6. Evaluate reconstruction on clean held-out spectra

Use an all-visible mask and `loss_region="all"` for deterministic full-spectrum
MSE. No augmenter is supplied to this evaluation. Validation/test spectra do not
fit the model or the cluster centers in this example.

```python
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
print("Validation MSE:", validation_mse, "Test MSE:", test_mse)
```

For masked evaluation, explicitly define the visible-mask or masking RNG protocol
and use `loss_region="masked"`. Changing the evaluation region changes what the
reported error measures. Tester aggregates across the full loader independently
of batch partitioning; empty selections/loaders fail clearly.

## 7. Extract features in a chosen representation

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
```

Representations are `cls`, `raw_latent`, `normalized_latent`, and `latent`
(which follows the model's normalization setting). Extraction makes all patches
visible, uses eval mode, and restores original submodule modes. Zero projected
vectors remain zero; normalization does not invent a direction for them.

For larger inputs, consume one output batch at a time:

```python
stream = extractor.iter_transform(ordered_loader(test_x))
try:
    for features in stream:
        print(features.shape)
        # Pass this batch to your writer or downstream predictor.
finally:
    stream.close()
```

Streaming does not save implicitly or concatenate the full output. Calling
`extractor(loader)` does concatenate it. The caller controls loader order and
must preserve row-to-specimen/pixel indices when reconstructing maps.

## 8. Fit clustering on training features and keep centers fixed

```python
clusterer = CosineKMeans(n_components=3, max_iter=30, device=device, random_state=42)
clusterer.fit(train_features, chunk=32)
test_labels = clusterer.predict(test_features, chunk=32)
print(clusterer.n_iter_, clusterer.converged_, clusterer.stop_reason_)

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
print("Cosine silhouette:", silhouette, "used classes:", used_classes)
```

Cluster IDs are local numeric assignments. Choose K without tuning on final test
scores. A one-class prediction has undefined silhouette; preserve that fact.
The elbow helper is a heuristic and validates finite, ordered curves, including
nonuniform K spacing. [vMF mixtures](../clustering/vmf_mixture.md) provide an
optional probabilistic alternative with their own fit/selection costs.

Chunk settings bound particular work arrays, rather than every allocation. CUDA
CosineKMeans can stream CPU feature chunks to CUDA for fitting/prediction, while
retaining the full CPU input. Prediction labels stay on the centers' device;
requesting distances still allocates the full `(N, K)` output. CPU-only
CosineKMeans does not chunk its similarity matrix. Silhouette
retains full features, labels, class sums, within-class work arrays, and output;
its chunk bounds only the similarity tile. See each API's memory contract.

## 9. Build a spatial map using actual coordinates and a valid mask

The synthetic scene below explicitly defines an `(H, W, L)` cube. Invalid pixels
form an edge and an internal hole. This spatial example has coordinates by
construction; unrelated spectrum rows must not be reshaped into an image.

```python
height, width = 12, 16
row, column = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
regions = (column // 6).clamp_max(2)
cube = templates[regions] + 0.03 * torch.randn(
    height, width, length, generator=data_stream,
)
valid_mask = torch.ones(height, width, dtype=torch.bool)
valid_mask[0] = False
valid_mask[3:5, 6:8] = False

spatial_features = extractor(ordered_loader(snv(cube[valid_mask])))
label_map = torch.zeros(height, width, dtype=torch.int64)
label_map[valid_mask] = clusterer.predict(spatial_features, chunk=32)
lla = local_label_agreement(
    label_map, valid_mask, windows=(3, 5, 9), device=device, class_chunk=2,
)
for window in lla.windows:
    print(window.window, window.score, window.raw_agreement, window.undefined_reasons)
print("Occupancy:", lla.occupancy)
```

The validity mask, rather than label zero, defines excluded pixels. LLA counts
valid directed neighbor pairs, excludes the center and image/mask boundaries,
and applies the finite-sample chance correction in Thesis equation (11).
Neighborhood values 3/5/9 are widths, not radii. Negative corrected scores are
retained. Undefined scores are NaN with explicit reasons; do not replace them
with zero. High spatial agreement does not establish chemical correctness.

Optional visualization uses masked background so valid cluster zero stays visible:

```python
import numpy as np
import matplotlib.pyplot as plt

image = np.ma.array(label_map.numpy(), mask=~valid_mask.numpy())
plt.imshow(image, cmap="tab10", interpolation="nearest")
plt.title("Synthetic spatial cluster IDs")
plt.colorbar()
plt.show()
```

## 10. Save enough context to interpret outputs

Record versions, configurations, seeds, selected weights, sample/group splits,
coordinate mapping, and preprocessing whenever applying the workflow to research.
This example exports a small report and the synthetic map/mask:

```python
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
    "device": str(device), "global_seed": 42, "data_seed": 24,
    "model_config": inference_model.get_config(),
    "selected_artifact": selected_artifact.name, "clustering": {"k": 3, "seed": 42},
    "synthetic_counts": {"train": len(train_x), "validation": len(validation_x), "test": len(test_x)},
    "validation_mse": validation_mse, "test_mse": test_mse,
    "silhouette": silhouette, "lla": asdict(lla),
}
(run_dir / "report.json").write_text(
    json.dumps(json_ready(report), indent=2, allow_nan=False), encoding="utf-8",
)
torch.save({"labels": label_map, "valid_mask": valid_mask}, run_dir / "spatial_map.pt")
```

Undefined metric values become JSON `null`, and their reasons remain in the LLA
result. The report is illustrative; a research project must additionally record
its data identities, acquisition grouping, preprocessing settings, and selection
protocol. These small examples make no claims about performance or superiority.

## Custom loops and troubleshooting

Use the [Trainer hooks and plain PyTorch example](../training/trainer.md) when
controlling masks, batch ordering, LR timing, or independent generators. Hooks
reuse the optimizer/AMP loop. The model, augmenter, and loss primitives also work
without Trainer. Supervised fine-tuning remains a separate consuming workflow.

- A patch-divisibility error means your spectrum length and patch count disagree.
- An existing-output error requires an explicit resume checkpoint or a fresh directory.
- A checkpoint-config mismatch requires the original architecture/training recipe.
- Undefined LLA or silhouette needs its recorded reason, rather than a substituted score.
- Set device, precision, and output storage deliberately when moving this CPU example to CUDA.

Focused tests, documented examples, actual GitHub math rendering, and installed
package checks remain release gates. Do not treat source review as execution.
