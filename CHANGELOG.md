# Changelog

## 0.2.3 — unreleased

This release improves the public spectral-learning workflow and adds spatial
Local Label Agreement. The source and focused tests are written; runtime,
CUDA, installed-package, and rendered-math validation remain pending.
Completed v0.2.2 research experiments and their artifacts are unchanged.

### Added

- Convolution-based, occupancy-corrected LLA following Thesis equation (11), with
  explicit masks, directed integer pair counts, class chunks, and undefined reasons.
- Full ChemoMAE configuration and versioned config-and-weights save/load artifacts.
- Public Trainer preparation, forward/loss, ordering, event, and extension-state hooks.
- Versioned training checkpoints with model/training config validation and standard
  global RNG snapshots; independent generators remain caller-owned through hooks.
- Configurable Trainer output paths/enablement and independent progress/summary controls.
- All-visible CLS/raw/normalized representations and streamed feature extraction.
- CosineKMeans convergence diagnostics and exact fitted-center persistence.
- A detailed Markdown workflow tutorial and a standalone installed-package smoke check.

### Changed

- Torch SNV runs natively, preserves autograd/device, and has explicit precision,
  statistics, constant/short-spectrum, and epsilon behavior.
- FPS, model masking, and augmentation expose explicit device/RNG controls.
- Trainer counts attempts, successful updates, and AMP skips; scheduler and EMA
  advance after successful updates. fit uses a final absolute epoch budget.
- Tester supports masked/full evaluation with dataset-level reductions; Extractor
  has explicit representation, type/device/dtype, ordering, and mode-restoration contracts.
- Cosine silhouette rejects invalid features/labels/class counts/chunks and bounds
  only its similarity temporary. Half inputs use float32 intermediates.
- Elbow curvature validates curves and smoothing parameters, bounds odd windows
  for even-length curves, and uses coordinate-aware gradients for nonuniform K.
- English source docstrings and API documentation use MathJax-compatible Markdown.
- PyTorch is declared as a runtime dependency.

### Migration and limits

v0.2.2 API, default, and artifact compatibility is not a release constraint.
Use the documented v0.2.3 signatures and save new model/training artifacts;
old raw weights do not supply missing constructor configuration automatically.
Trainer remains specific to reconstruction, with completed-epoch resume.
Direct step budgets, validation-based selection, arbitrary worker recovery,
MPS RNG restoration, and image tiling for LLA are not supplied by this release.
No benchmark result or cross-device/version reproducibility claim follows from
the implementation alone. GPU cost and supported environment checks are pending.
