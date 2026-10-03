# ChemoMAE Documentation

These documents describe ChemoMAE v0.2.3. See the
[release notes](../CHANGELOG.md) for API, default, and artifact changes from
v0.2.2. Use the `v0.2.3` Git tag when browsing the documentation for this release.

## Research motivation and workflow

ChemoMAE supports exploration of the spatial distribution of spectral
differences when chemical states and their categories are not known in advance.
Self-supervised reconstruction learns a spectral representation, clustering
partitions that representation at a chosen observation granularity, and the
labels can be returned to their original image coordinates for spatial analysis.
Spectral summaries and local chemical measurements support interpretation of
the resulting regions.

The accompanying study uses SNV to compare relative spectral shapes and adds
mean- and norm-preserving perturbations for masked denoising. Visible patches
provide the model input; hidden channels of the spectrum before augmentation
provide the reconstruction target. After training, all-visible features without
augmentation are clustered with CosineKMeans. LLA evaluates local spatial
coherence using neighbor relations kept apart from model and cluster fitting,
while cosine silhouette describes separation within the representation.

The library exposes these steps as configurable components. Tutorial settings
illustrate their use on synthetic data; they are not the manuscript's experiment
configuration. See the [README](../README.md) for the manuscript status and link
placeholders.

## Tutorial and protocol

- [Step-by-step synthetic workflow tutorial](tutorials/workflow.md)
- [Model artifacts and training checkpoints](models/persistence.md)
- [README workflow](../README.md)

## API references

| Area | Documentation |
| --- | --- |
| Preprocessing | [SNV](preprocessing/snv.md), [cosine FPS](preprocessing/dowmsampling.md) |
| Model and representations | [ChemoMAE, encoder, decoder, encode](models/chemo_mae.md), [losses](models/losses.md) |
| Training | [Trainer customization and checkpoints](training/trainer.md), [optimizer/scheduler](training/optim.md), [spectral augmentation and RNG](training/augmenter.md) |
| Evaluation and extraction | [Tester reductions](training/tester.md), [batch-wise Extractor](training/extractor.md) |
| Clustering | [CosineKMeans](clustering/cosine_kmeans.md), [vMF mixture](clustering/vmf_mixture.md), [cosine operations](clustering/ops.md) |
| Cluster evaluation | [cosine silhouette](clustering/metric.md), [occupancy-corrected spatial LLA](clustering/spatial.md) |
| Persistence and reproducibility | [model/training artifacts](models/persistence.md), [global seed and RNG state](utils/seed.md), explicit generator contracts in model/augmentation docs |

## Numerical and experimental notes

SNV acts independently on each spectrum; choose it when removal of per-spectrum
mean and scale suits the measurement. The tutorial uses already aligned
synthetic spectra of length 64. Input and normalized latent representations can
both be compared by direction, but the encoder need not preserve input cosine
similarities, and latent vectors have no zero-mean constraint.

All-visible features are distinct from randomly masked training features. Fit
model weights and cluster centers only on training specimens when describing
held-out generalization. Fitting that includes the evaluated specimens serves
descriptive mapping of that dataset. LLA measures local spatial coherence
conditioned on occupancy; interpreting chemical states requires evidence beyond
the label map and its score.
