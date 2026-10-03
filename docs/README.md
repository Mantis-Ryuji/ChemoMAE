# ChemoMAE Documentation

These documents describe ChemoMAE v0.2.3 under development. PyPI v0.2.2 does
not include newly added APIs. Implementation status is recorded in
[ToDo](../ToDo.md).

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

SNV acts independently on each spectrum. The tutorial uses already aligned
synthetic spectra of length 64. All-visible features are distinct from randomly
masked training features. Fit model weights and cluster centers only on the training specimens
when describing held-out generalization. LLA measures spatial coherence and
occupancy rather than chemical correctness.

Math is authored for MathJax using inline `$...$` and display `$$...$$` with blank
lines. Actual GitHub rendering must be verified before release.
