# ChemoMAE Documentation

These documents describe ChemoMAE v0.2.3 under development. PyPI v0.2.2 does
not include newly added APIs. Implementation status and unperformed release checks
are recorded in [ToDo](../ToDo.md).

## Tutorial and protocol

- [Minerals in the Wild NIR-HSI tutorial](tutorials/nir_hsi.md)
- [Notebook](../notebooks/nir_hsi_tutorial.ipynb)
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
| Reproducibility | [global seed](utils/seed.md), explicit generator contracts in model/augmentation docs |

## Numerical and experimental notes

SNV acts independently on each spectrum, after the tutorial's wavelength
interpolation. All-visible features are distinct from randomly masked training
features. Fit model weights and cluster centers only on the training specimens
when describing held-out generalization. LLA measures spatial coherence and
occupancy rather than chemical or mineral annotation correctness.

Math is authored for MathJax using inline `$...$` and display `$$...$$` with blank
lines. Actual GitHub and Colab rendering must be verified before release.
