# ChemoMAE documentation

These guides describe the v0.2.4 API being prepared in this checkout; v0.2.4 has
not been published to PyPI. Use the source installation in the
[workflow tutorial](tutorials/workflow.md#1-set-up-and-prepare-spectra) for these
examples. Relative links follow the repository revision you are viewing.
For the published v0.2.3 documentation, use the
[v0.2.3 snapshot](https://github.com/Mantis-Ryuji/ChemoMAE/tree/v0.2.3/docs).
[Release notes](../CHANGELOG.md) summarize changes and compatibility notes.

## Start with your task

| You want to… | Start here |
| --- | --- |
| Install the package and run a complete CPU example | [README quick start](../README.md#quick-start) |
| Learn representations, then choose optional downstream steps | [Staged workflow tutorial](tutorials/workflow.md) |
| Plan a first experiment and estimate its memory requirements | [First-experiment recipe](tutorials/first_experiment.md) |
| Normalize spectra or select a smaller set of rows | [SNV](preprocessing/snv.md), [cosine FPS](preprocessing/dowmsampling.md) |
| Build a model or use a custom PyTorch loop | [Model and representations](models/chemo_mae.md), [losses](models/losses.md), [Trainer hooks and plain loop](training/trainer.md) |
| Configure reconstruction training or resume a run | [Starting fresh or resuming](training/trainer.md#starting-fresh-or-resuming), [Trainer configuration](training/trainer.md#configuration-and-the-simple-path), [optimizer/scheduler](training/optim.md), [augmentation](training/augmenter.md) |
| Compute reconstruction errors or export features | [Tester](training/tester.md), [Extractor](training/extractor.md) |
| Cluster directional features from any compatible source | [CosineKMeans](clustering/cosine_kmeans.md), [vMF mixture](clustering/vmf_mixture.md) |
| Evaluate an existing partition or label map | [Cosine silhouette](clustering/metric.md), [spatial LLA](clustering/spatial.md) |
| Use cosine operations or inspect an inertia curve | [Clustering operations](clustering/ops.md) |
| Choose a saved file for inference or resume, or manage random streams | [Output-file guide](models/persistence.md#raw-and-ema-exports), [persistence](models/persistence.md), [seed and RNG state](utils/seed.md) |

## How the components fit together

The model consumes a matrix of spectra. Preprocessing, augmentation, and
downstream analysis are explicit choices made by the calling application:

- SNV removes each spectrum's mean and scale when relative shape is the desired
  input. It has no fitted population statistics. FPS optionally selects rows;
  it does not preserve their density or make data splits.
- ChemoMAE reconstructs spectra through a shared latent bottleneck. Trainer
  supplies a reconstruction loop; model, loss, and augmentation components can
  also be used independently in PyTorch.
- Extractor returns features in loader order. These can support clustering,
  visualization, or a downstream predictor. For gradients through the encoder,
  use `model.encode` directly.
- CosineKMeans and VMFMixture accept directional feature matrices independently
  of the encoder. Their guides explain hard assignments, responsibilities,
  numerical conventions, and resource costs.
- LLA accepts a two-dimensional label map and explicit valid mask. Coordinates
  matter for this spatial analysis, but are not required for spectral learning,
  feature extraction, or feature-space clustering.

## Contracts and interpretation

Each API guide gives a small CPU example, input/output contracts, and relevant
limits. Later snippets may extend that example; their prerequisites are stated.
CUDA examples require a compatible device and PyTorch build. `chunk` bounds
specific work arrays, not necessarily all memory; consult the relevant guide.
Trainer, Tester, and Extractor follow the model's device by default, while
CosineKMeans and VMFMixture default to CUDA. Pass `device="cpu"` explicitly to
clustering on CPU; see the [first-run choices](../README.md#choices-to-keep-explicit).

Choose preprocessing and perturbations for the information you intend to retain.
SNV and latent normalization have different constraints, and the encoder does
not preserve input cosine similarities by construction. Zero and very small
vectors follow each function's documented epsilon behavior.

Reconstruction error, feature-space separation, and spatial coherence answer
different questions. Cluster IDs do not acquire semantic labels automatically.
When claiming held-out generalization, separate the relevant independent groups
before fitting learned components and choosing settings. Fitting the dataset
being mapped is also a valid descriptive use, with a different interpretation.
Spatial maps require the original row-to-pixel association; arbitrary spectral
rows cannot be reshaped into a meaningful image.

## Checking examples and display

From a checkout with ChemoMAE and its runtime dependencies available, list the
selected examples or run them:

```bash
python tests/documentation_examples.py --list
python tests/documentation_examples.py
```

The runner reads actual Markdown code, uses CPU and small synthetic inputs, and
places outputs in temporary directories. It reports selected and skipped blocks;
API signatures and snippets requiring caller inputs are not standalone programs.
The CI workflow also runs these recipes against an installed package outside the
checkout. Neither this runner nor a local preview verifies GitHub math rendering.

Markdown math uses `$...$` inline and separate `$$...$$` display blocks, following
[GitHub's math documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
and the [MathJax supported commands](https://docs.mathjax.org/en/latest/input/tex/macros/index.html).
Inspect changed formulas and links in GitHub's rendered Markdown or unsaved
Preview before considering display verification complete.
