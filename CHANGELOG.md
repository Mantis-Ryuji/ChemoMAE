# Changelog

## 0.2.4 (Unreleased)

This version is in preparation and has not been published to PyPI.

### Fixed

- Pass lower-is-better BIC/mean NLL scores directly to the nonincreasing-curve
  elbow helper. The previous sign reversal could erase the bend and return K=2
  with zero curvature for decreasing scores. Fixed-K fitting and raw score values
  are unchanged; the elbow remains a heuristic, distinct from minimum-score K.
- Reject fewer than three candidate K values before starting a vMF sweep.
- Explain input/model dtype mismatches before the standard ChemoMAE training
  forward pass, including the actual dtypes and an explicit conversion hint.
  Trainer still preserves input dtype; custom forward hooks retain their policy.

### Added

- `Trainer.fit()` returns `final_artifact`, an absolute path to the selected
  raw/EMA inference artifact that can be passed directly to `ChemoMAE.load`.
  It is `None` when artifact output is disabled, no weight export is selected,
  or the model does not support Trainer's ChemoMAE artifact export.
- Regression coverage for BIC/NLL score direction, selected artifact loading,
  and training dtype validation.

### Documentation

- Organize the README and documentation index around independently usable
  library components, with research background kept as an optional reference.
- Separate the tutorial's basic training/extraction path from optional
  augmentation, resume, clustering, spatial evaluation, and reporting.
- Correct loss/mask contracts, example imports and assertions, normalization
  qualifications, and clustering memory descriptions; add small CPU API examples.
- Use the selected artifact path in the public workflow and show explicit
  NumPy-to-model dtype conversion before training.
- Add a first-experiment guide covering caller-defined evaluation protocols,
  focused configuration comparisons, and operation-specific memory estimates.
- Link the associated WoodDegradationMap research repository from the README.

### Compatibility and preparation status

- `final_model` keeps its existing configured weight filename and raw/EMA
  selection rule. Model artifacts and training checkpoints keep their existing
  format versions. Default devices, automatic resume, and input conversion
  policies are unchanged.
- Corrected elbow K values may differ from v0.2.3. The shared curve helper still
  applies a cumulative minimum to nonmonotonic curves; flat curves still select
  the first interior point with zero curvature. Neither establishes a true K.
- Local validation is complete; same-commit CI gates and publication preparation
  remain pending. Publication is deferred; see `ToDo.md`.

### Local validation reported by the user

- In the `chemomae-test` environment, the targeted tests passed: 226 passed,
  69 warnings. The complete suite then passed: 603 passed, 137 warnings.
- After correcting the Trainer example selector to include its NumPy
  subsection under the parent level-two heading, the documentation runner
  passed 21 recipes containing 37 actual Python blocks. Unselected examples
  were not executed.
- The documentation run reported Python 3.11.17, NumPy 2.4.6, Torch
  2.14.1+cu126, and ChemoMAE 0.2.4 imported from the editable source checkout;
  the selected examples ran on CPU.
- The 0.2.4 wheel and source distribution were built; both passed `twine check`.
- In the separate `chemomae-min` environment, the documentation runner passed
  all 21 selected recipes and 37 Python blocks from outside the checkout with
  `python -I`. It reported Python 3.10.22, NumPy 1.26.4, Torch 2.1.0+cpu, and
  ChemoMAE 0.2.4 imported from that environment's `site-packages`.
- The dedicated `tests/installed_package_smoke.py` check also passed in
  `chemomae-min`, run with `python -I` from outside the checkout:
  `Installed ChemoMAE CPU workflow passed.`
- `python -m pip check` in `chemomae-min` reported
  `No broken requirements found.` These local results do not replace the
  supported CI matrix.

## 0.2.3

This release improves the public spectral-learning workflow and adds spatial
Local Label Agreement.

### Added

- Convolution-based, occupancy-corrected LLA with finite-sample chance correction,
  explicit masks, directed integer pair counts, class chunks, and undefined reasons.
- Full ChemoMAE configuration and versioned config-and-weights save/load artifacts.
- Public Trainer preparation, forward/loss, ordering, event, and extension-state hooks.
- Versioned training checkpoints with model/training config validation and standard
  global RNG snapshots; independent generators remain caller-owned through hooks.
- Configurable Trainer output paths/enablement and independent progress/summary controls.
- All-visible CLS/raw/normalized representations and streamed feature extraction.
- CosineKMeans convergence diagnostics and exact fitted-center persistence.
- A detailed Markdown workflow tutorial and a standalone installed-package smoke check.
- Direct execution checks for selected Markdown examples and a bounded synthetic
  CPU/CUDA LLA benchmark runner.
- Same-commit CI gates for publication, an explicit minimum-Torch CPU lane, and
  separation of production and RC tag triggers.

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

v0.2.3 does not guarantee API, default, or artifact compatibility with v0.2.2.
Use the documented v0.2.3 signatures and save new model/training artifacts;
old raw weights do not supply missing constructor configuration automatically.
For Extractor, replace `return_numpy=True` with
`ExtractorConfig(output_type="numpy")`; request `output_device="cpu"` explicitly
when CPU tensor output is needed. The default output is now a tensor on the
inference device with AMP disabled, and calls restore the original model modes.
Trainer remains specific to reconstruction, with completed-epoch resume.
Direct step budgets, validation-based selection, arbitrary worker recovery,
MPS RNG restoration, and image tiling for LLA are not supplied by this release.
