# ChemoMAE v0.2.3 Release ToDo

This v0.2.3 release plan was refined on 2026-10-03. Checked implementation items
mean that source, focused tests, and the associated documentation have been written.
They do not mean that tests, CUDA, data downloads, notebook execution, or rendered
math have passed. Those validation and publication gates remain unchecked.

## Current implementation status

- [x] Write convolution-based equation-(11) LLA with explicit masks, class chunks,
  integer pair diagnostics, finite-sample correction, and undefined-value reasons.
- [x] Add native differentiable Torch SNV, explicit FPS device/RNG controls,
  all-visible model representations, and owned masking/augmentation generators.
- [x] Add public Trainer preparation/event/checkpoint hooks, epoch-boundary resume,
  separate attempted/update/skip counters, and successful-update scheduler/EMA.
- [x] Add masked/full Tester with dataset-level reductions and streamed Extractor
  with explicit representation/output/device/dtype contracts and mode restoration.
- [x] Add CosineKMeans convergence diagnostics, final-center objective, and
  versioned exact-center fitted-state persistence.
- [x] Write the Minerals in the Wild notebook and protocol as an unexecuted draft:
  recorded wavelength interpolation 273→256 before SNV, specimen holdout, true
  spatial crops, selected EMA reload, train-only clustering, LLA, and artifacts.
- [x] Update MathJax authoring rules, API documentation, README examples, and
  move the existing Torch requirement into runtime dependencies.
- [x] Translate remaining source docstrings/comments into English and audit
  updated example signatures against their APIs; source-language search finds no Japanese
  in `src/chemomae`. Example execution remains unperformed.
- [ ] Run focused tests, installed-package checks, and fresh-Colab Run all.
- [ ] Publish matching source/notebook refs, pin tutorial dependencies and
  wavelength metadata, then verify and activate the Open in Colab badge.
- [ ] Validate actual GitHub/Colab math rendering and measure CPU/CUDA cost.

The quick notebook's small model, 8/2/2 specimen split, two epochs, K=4, and EMA
selection are labelled illustrative choices. The user selected the dataset and
273→256 interpolation; a scientific benchmark protocol is still a separate
decision. The archive transfer is part01, not a twelve-specimen-only download.

## Agreed direction and product scope

User decisions:

- Deliver the changes in this plan as ChemoMAE v0.2.3. The package metadata,
  public version string, version test, and development documentation use this
  target; release validation and publication remain pending.
- Finish a coherent production-ready library, documentation, and executable
  tutorials as one release effort.
- Backward compatibility with v0.2.2 is not a constraint. Public APIs, defaults,
  configuration, return types, and artifact formats may be redesigned. Do not add
  legacy aliases or parallel implementations solely to preserve compatibility.
- Keep completed v0.2.2 research experiments and their environments/artifacts
  unchanged. Historical experiments do not have to be migrated or rerun.
- Use openly licensed NIR data for the main end-to-end tutorial and add
  **Open in Colab** access from README.
- Use Minerals in the Wild v1.0.0 as the main tutorial dataset (reconfirmed by
  the user on 2026-10-03 after considering LeafVAE). Use its SWIR reflectance
  hyperspectral images; the separate specimen-level XRF table is outside this
  tutorial's workflow and receives no SNV. Linearly interpolate the images'
  273 bands on the recorded wavelength axis to 256 uniformly
  spaced wavelengths over the same range, before SNV. Use 16 patches of 16
  bands; record both wavelength grids and the interpolation method explicitly.
- Write README, user documentation, and public docstrings in English. Use
  MathJax-compatible mathematics across Markdown and notebook Markdown cells.

- [ ] Define one public workflow for prepare/train/evaluate/encode/stream/cluster/
  save/load, with consistent shapes, device/dtype rules, errors, and terminology.
- [ ] Review existing features against the tutorial and actual consuming workflows;
  simplify, redesign, or remove awkward/redundant interfaces where justified.
- [ ] Keep datasets, splits, augmentation strengths, model-selection rules, and
  sample-level aggregation in the consuming research project. Do not silently
  transfer WoodDegradationMap's experimental recipe into library defaults.
- [ ] Define the tutorial's dataset and experimental protocol explicitly before
  generating results. Unresolved specimen identities, masks, annotations, splits,
  and evaluation conditions must not be inferred from convenient file layouts.
- [ ] Implement focused changes within the integrated release plan, each with
  tests and English documentation. Keep Trainer focused on reconstruction rather
  than building a general-purpose training framework.
- [ ] Set the new defaults deliberately, including RNG ownership, scheduler/EMA
  behavior after an AMP skip, precision, and fresh-run/resume semantics. Explain
  changes in release notes; old behavior need not remain available.

## 0. Real NIR tutorial, experimental protocol, and Open in Colab

### Dataset selection and access

- [x] Choose the main openly licensed NIR dataset using its primary source,
  explicit data license, citation, spectral coverage, spatial coordinates, specimen
  metadata, download size, access requirements, and usable small-subset access.
  Public download availability alone is not a redistribution license. Selected:
  Minerals in the Wild v1.0.0 (user decision); download/subset execution is pending.
- [ ] Prefer multiple spatial NIR-HSI specimens with wavelength metadata and a
  documented valid-pixel definition. Verify repeated acquisitions and specimen/
  group identities before choosing a split. Tabular NIR spectra can support a
  secondary lightweight example, but arbitrary row reshaping cannot provide
  spatial LLA or a spatial evaluation protocol.
- [ ] Record the selected version/DOI, source URLs, data attribution, file hashes,
  selected sample IDs, and all subset/preprocessing steps. Keep data downloads
  outside the package/repository source. Verify a small distributable subset if
  the full archive is too large for the tutorial.
- [ ] Define reliable download/cache/verification cells with useful errors and
  no dataset-account credentials, local paths, or compulsory Drive mounting.
  Confirm real direct-download behavior; a landing-page link is insufficient.

Minerals in the Wild is the confirmed main dataset. SNV in this tutorial applies
to SWIR reflectance spectra at valid image pixels, after wavelength interpolation.
It does not apply to the separate XRF elemental-composition table. Direct Colab
downloads and small-subset feasibility have not been tested. The other datasets
below are reference alternatives, not pending replacements:

- [Minerals in the Wild](https://github.com/EleftheriaTtl/minerals-in-the-wild),
  [dataset release v1.0.0](https://github.com/EleftheriaTtl/minerals-in-the-wild/releases/tag/v1.0.0),
  is licensed CC BY 4.0. Its creator documents 1,132 identified rock specimens,
  spatial NumPy SWIR reflectance cubes with 273 bands from 996.34 to 2504.28 nm,
  wavelength metadata, and NaN-encoded background. Explicit specimen IDs support
  demonstrations of specimen holdout and masks. Its 273-to-256 wavelength
  interpolation is an explicit tutorial preprocessing decision, not an implicit
  library conversion. The full HSI volume is about
  4.1 GB before packaging; verify a bounded subset route, site/group metadata,
  and handling of short spatial crops. XRF references are specimen-level elemental
  compositions, not pixel-level mineral labels, and are not used in the tutorial.
- [Cryptogams-NIR-HSI, Mendeley Data V1](https://data.mendeley.com/datasets/p6pjkjpvxw/1),
  DOI `10.17632/p6pjkjpvxw.1`, is published under CC BY 4.0. The primary record
  reports 39 cubes, 370 x 520 pixels, 224 bands, and 900-1700 nm, with reflectance,
  supplied SNV data, semantic masks, and wavelengths. See the
  [creator's dataset guide](https://github.com/Shah433/Cryptogams-NIR-HSI-Dataset).
  Check download size/subset feasibility and independent specimen grouping.
  Annotation label 255 means unknown/unlabelled specimen pixels, so it must not
  automatically become an invalid-spectra mask for unsupervised LLA. Decide the
  valid-region contract separately from annotation availability.
- [Fraunhofer Hyperspectral NIR Dataset of Bulky Waste](https://fordatis.fraunhofer.de/handle/fordatis/375?locale=en)
  offers a materials-oriented alternative: 44 scenes with NIR cubes and registered
  RGB/class labels under CC BY 4.0. Its 9.26 GB archive is too large to assume a
  lightweight download path. Verify a licensed small subset, physical-item
  grouping across scenes, and RGB/label-to-HSI resolution alignment before use.

### Notebook and protocol design

- [x] Create the repository-owned English draft notebook
  `notebooks/nir_hsi_tutorial.ipynb` and protocol guide `docs/tutorials/nir_hsi.md`.
- [ ] Add README's **Open in Colab** badge only
  after the notebook exists at a reachable GitHub ref. Match the notebook's
  installed ChemoMAE version to its source/docs; use a release ref for reproducible
  release tutorials and label development notebooks distinctly.
- [ ] Make fresh-runtime **Run all** complete without manual code changes or
  private repository helpers. Include package installation, downloads, device
  detection, data inspection, and every required variable/import in the notebook.
- [ ] Provide a bounded quick tutorial and a separately identified fuller
  experimental protocol. Set sample/pixel budgets, epochs/steps, batch size,
  seeds, K values, precision, runtime, and memory targets explicitly; report
  measured cost rather than promising fixed Colab resources or hardware.
- [ ] Inspect reflectance versus supplied SNV arrays, wavelength order, invalid
  spectra, masks, sequence length, and patch divisibility. Demonstrate ChemoMAE
  SNV without double-normalizing already normalized data or silently interpolating
  channels to match a model default. The selected tutorial explicitly applies
  the user's 273-to-256 linear wavelength interpolation before SNV and records
  the original/target grids; this conversion is not a model or library default.
- [ ] Split at the verified specimen/group level for generalization examples;
  keep related acquisitions together. Fit representation transforms and cluster
  centers using training data, and apply them fixed to held-out data. Show
  full-dataset descriptive fitting as a separately labelled analysis.
- [ ] Define fair, bounded baseline/ablation examples, such as SNV/PCA features and
  ChemoMAE with/without augmentation. Record fitting pixels, budgets, seeds,
  selected weights, metrics, and repeat aggregation. Select these conditions
  explicitly; never claim superiority from a short quick-tutorial run.
- [ ] Define K/model selection before evaluation and keep held-out scores out of
  tuning. Explain why maximizing LLA alone does not establish chemically correct
  regions or preservation of small meaningful differences.
- [ ] State which labels are used for downstream evaluation and which are unused
  during unsupervised fitting. Do not infer pixel annotations from specimen-level
  targets or invent class thresholds for continuous reference properties.
- [x] Cover SNV, bounded FPS, TGN/FS, model construction, optimizer/scheduler,
  Trainer, reconstruction evaluation, selected raw/EMA reload, CLS/raw/normalized
  features, streaming extraction, clustering, spatial maps, and equation-(11)
  LLA. Use optional bounded cells for vMF, K exploration, or repeated experiments
  when including them in the quick path would be too expensive.
- [ ] Demonstrate occupancy and cosine-silhouette alongside LLA. Explain that
  reconstruction loss, spatial coherence, and annotation agreement measure
  different properties. Show LFR only with an explicit perturbation/RNG protocol;
  keep experiment orchestration separate from reusable metric primitives.
- [ ] Demonstrate save/load, prediction round trips, and an interruption/resume
  example. Export a compact config, split/subset manifest, metrics, versions,
  device information, elapsed time, and peak memory. Optional persistent storage
  must not be needed to complete the basic notebook.
- [x] Add a notebook feature-to-cell coverage table and a linked protocol/API
  index. Use public library functionality rather than hiding missing APIs inside
  notebook-only training or extraction implementations. Review notebook section
  links again when publishing the final documentation ref.

Acceptance: a new user can open the released notebook, run the documented path
in a fresh Colab runtime, and inspect meaningful NIR outputs, protocol metadata,
and saved/reloaded models. GPU availability is not assumed; document the CPU
quick path and measured GPU path. The dataset and interpolation are selected;
the development tutorial's illustrative protocol still requires execution and
review before release.

## 1. LLA following Thesis equation (11) — first priority

References:

- Thesis: `chapters/05_experimental_design.tex`, `eq:lla` (equation 11), and
  `chapters/appendix.tex`, undefined-value rules and spatial-association derivation.
- WoodDegradationMap: `src/wood_degradation_map/experiments/spatial_metrics.py`
  and `tests/experiments/test_spatial_metrics.py`.

- [x] Add a public Local Label Agreement API in `chemomae.clustering`.
  Accept NumPy arrays and Torch tensors with an explicit `valid_mask` and
  configurable odd neighborhood widths, including 3, 5, and 9.
- [x] Implement equation (11) directly: construct binary class maps and the valid
  mask, convolve with a center-zero/all-other-elements-one kernel, multiply by
  the corresponding binary map, and sum. Use `conv2d` as the main computation.
- [x] Count all valid directed neighbor pairs equally. Exclude the center,
  background, excluded pixels, and out-of-image positions; retain isolated valid
  pixels in the occupancy counts and total valid-pixel count.
- [x] Use the exact finite-sample chance agreement
  $P=\sum_k n_k(n_k-1)/(N(N-1))$ and return the corrected LLA as the primary score.
  Return raw agreement separately; do not confuse it with equation (11)'s LLA or
  silently rename WoodDegradationMap's existing saved keys.
- [x] Support zero-based and nonconsecutive valid class labels. Determine the
  evaluated region from `valid_mask`, not from a reserved label value or missing
  predictions. Remove the experiment-specific restriction on allowed K values.
- [x] Return window-specific scores, integer pair counts, occupancy, used-class
  count, chance agreement, coverage, and explicit undefined-value reasons.
  Do not clip negative LLA, replace undefined scores with zero, or automatically
  average the scores across windows. Reject invalid/empty-mask inputs clearly.
- [x] Bound binary-class-map temporary memory with class chunks. Avoid full neighborhood
  expansion with `unfold` and per-offset CPU synchronization. Add image tiles
  with a neighborhood halo only if class chunking is insufficient; count tile
  cores only and compute occupancy once for the full map. Image tiling is not
  implemented; full-image tensors and backend workspace still remain allocated.
- [x] Perform binary convolution outside AMP, verify local integer counts, and
  reduce counts as integers rather than summing millions of values in FP32.
  Finalize the correction from compact counts using overflow-safe integer
  arithmetic; do not form cubic-scale products in Torch `int64`.
- [x] Define the final score precision and CPU/GPU tolerance explicitly from the
  mathematical contract. Use the existing CPU implementation as a reference,
  including near-one chance agreement; its FP32 final ratio is not a required
  product default and matching its rounding is not a compatibility obligation.
- [ ] Validate exact counts against an independent pixel-pair oracle and the
  existing CPU implementation. Cover boundaries, holes, diagonal neighbors,
  isolated pixels, one pixel, one class, negative scores, extreme imbalance,
  label permutations, and class-chunk/tile invariance.
- [ ] Benchmark synthetic maps with different image sizes, class counts, and
  windows. Report kernel time, end-to-end time including transfer, and peak GPU
  memory separately. Retain a usable CPU path; do not assume CUDA is faster for
  every input size.

Acceptance: integer counts agree exactly with the reference definition; scores
agree under the documented precision contract; memory-control settings do not
change the result.

## 2. Make Trainer useful without replacing its whole loop

Current adoption evidence: WoodDegradationMap's `ExperimentTrainer` inherits
`Trainer` and reuses its fit/AMP/loss machinery, but supplies an empty base loader
and replaces the epoch loop, history handling, and checkpoint logic. Global
training and Raman pretraining use similar adapters. Trainer is therefore used
as a base class, but its public customization surface is too limited for these
workflows.

- [x] Add narrowly scoped, documented extension points for batch preparation,
  forward/loss computation, and step/epoch events. Support a caller-prepared
  model input, clean target, and explicit visible mask while retaining the
  simple Tensor/tuple DataLoader path.
- [x] Let callers provide epoch-dependent batch ordering and before/after-step
  LR updates without copying the AMP/backward/clipping/optimizer loop.
- [x] Provide supported hooks for additional checkpoint state and run metadata;
  avoid requiring overrides of private helpers such as `_checkpoint_state`,
  `_compute_loss`, `_autocast_ctx`, and `_save_history`.
- [ ] Make history/checkpoint/export locations and progress/logging controls
  configurable without introducing a mandatory external logging dependency.
- [x] Record attempted steps, successful optimizer updates, and AMP skips in
  history/checkpoints/results. Choose and document which event advances the
  scheduler and EMA; an AMP-skipped optimizer step must have a defined outcome.
- [x] Clarify fresh-run versus resume behavior and prevent unrelated histories
  from being mixed. Report incompatible/corrupt state and scaler-restoration
  failures clearly instead of silently ignoring them.
- [x] Audit the advertised fixed-step budget: the current `fit` API takes an epoch
  budget. Either document that limit accurately or add an explicit step
  budget with clearly defined attempted-step/successful-update semantics. This
  implementation keeps an absolute epoch budget; direct step budgets are pending.
- [x] Explain and expose optimizer parameter-group choices and LR indexing.
  The native helper excludes CLS/position embeddings from weight decay and its
  warmup differs from the experiment recipe. Define clear product defaults,
  caller-owned grouping/schedules, and inspectable parameter groups. Explain
  timing and exclusions rather than presenting one recipe as scientifically best.
- [x] Add a small synthetic customization example showing how to reuse Trainer
  while controlling masks, augmentation, ordering, LR timing, and extra state.
  Also document a plain PyTorch loop for users who only need the model and
  augmentation primitives. Keep supervised fine-tuning separate from the
  reconstruction-specific Trainer contract.

Acceptance: the standard and customization examples express the relevant
application needs through public APIs without copying the complete training loop
or using private methods.

## 3. Evaluation and feature extraction

- [x] Support reconstruction evaluation with `loss_region="masked" | "all"`.
  Masked evaluation uses `~visible_mask`; full evaluation uses every element.
  Added full-spectrum MSE cases and augmented-input/clean-target documentation;
  execution is pending.
- [x] Define and implement empty-mask and empty-loader behavior in evaluation
  helpers; do not let undefined evaluation appear to be a successful zero loss.
- [x] Add public all-visible `ChemoMAE.encode(...)` access to CLS features,
  pre-normalization latent vectors, and normalized latent vectors. Define one
  unambiguous representation and normalization contract across forward, encode,
  extraction, and clustering; return types and state-dict keys may be redesigned.
- [x] Replace the need for consuming applications to hook `encoder.to_latent`
  when inspecting raw latent norms or building downstream classification heads.
- [x] Add batch-wise feature iteration and an explicit output-device option to
  Extractor. Choose consistent output type/device/dtype defaults; let callers
  aggregate, stream into their own writers, or retain features on GPU without
  mandatory whole-dataset CPU accumulation.
- [ ] Specify dtype, input ordering, model/augmenter mode handling, and iterator
  cleanup, including early termination. Check that concatenated streamed output
  equals the aggregate output under deterministic settings.
- [x] Audit device/AMP resolution and validation across Trainer, Tester, and
  Extractor. Explain CPU use and unsupported precision/device combinations;
  avoid silently changing arithmetic or running stochastic augmentation.

## 4. Persistence and reproducibility

- [ ] Define standard versioned artifacts and public save/load APIs for model
  config/weights, training state, and fitted clustering state. Include settings
  not recoverable from weights, such as attention heads, dropout, normalization,
  and masking configuration. Old weight/checkpoint formats need not be supported.
- [x] Define generator ownership for masking and augmentation with explicit
  `torch.Generator` control. Support independent streams without temporarily
  replacing global RNG state; retaining the old global-RNG behavior is optional.
- [x] Support saving/restoring standard RNG state and caller-owned generator
  state through the checkpoint extension contract. Validate schema/config/state
  and fail clearly on unsupported or incomplete artifacts. The notebook owns its
  RNG schema through the extension hooks; generic model artifact versioning and
  complete model-config validation remain separate pending work.
- [ ] Define the supported resume boundary and test synthetic uninterrupted
  versus resumed preparation/update sequences under matching conditions.
  Do not promise restoration of arbitrary DataLoader worker or external state.
- [x] Document raw-last versus EMA-last selection. Ensure examples explicitly
  load the selected exported weights before downstream evaluation/extraction;
  the in-memory training model and the selected export may differ.

## 5. Clustering, preprocessing, and existing helper usability

- [x] Add CosineKMeans iteration count, convergence flag, and stop reason, and
  document initialization, stopping, empty-class, and RNG policies.
- [x] Define the reported objective against the final saved centers. Correct the
  current pre-final-update `inertia_` behavior rather than retaining parallel
  legacy diagnostics solely for compatibility.
- [x] Add a supported fitted-state persistence path that preserves centers
  without re-normalizing them on save/load. WoodDegradationMap currently assigns
  `centroids`, `latent_dim`, and `_fitted` directly to preserve its fitted predictor.
- [ ] Review chunking contracts across clustering and metrics: distinguish
  intermediate-memory bounds from full-input/full-output residency, avoid
  unnecessary device round trips, and document what remains allocated.
- [ ] Harden cosine-silhouette input validation, including positive chunk sizes,
  valid class counts, shapes, integer labels, and nonfinite values. Document
  singleton and zero-vector conventions and verify the stated metric definition.
- [ ] Audit elbow curvature inputs and smoothing: reject mismatched/nonfinite
  curves, define the spacing assumptions, and keep the odd S-G window within
  even-length inputs. Its return contract is a scalar maximum curvature, not a
  full curvature array; source/doc behavior must agree.
- [x] Make Torch SNV operate natively without CPU/NumPy conversion.
  Define `ddof`, `sd + eps`, length-one/constant inputs, statistics, dtype/device,
  and numerical differences. Focused tests are written; runtime verification is pending.
- [x] Audit FPS controls and validation: explicit computation device, ratio and
  initial-index validation, zero-vector handling, return types, and seed/generator
  behavior. Resolve documentation/implementation mismatches against the new API.

## 6. English documentation and README tutorial

Current language inventory: `docs/` and README are already predominantly English;
many source docstrings and comments contain Japanese. Complete the English
public documentation and docstring work rather than assuming every Markdown
document needs translation.

- [x] Translate Japanese public/module docstrings into English throughout
  `src/chemomae`, using NumPy-style sections for documented public interfaces.
  Translate explanatory
  implementation comments when maintaining the affected code. Keep docstrings
  aligned with the redesigned APIs and retain mathematical/numerical caveats.
- [ ] Audit `docs/` and README for remaining Japanese, outdated signatures,
  mismatched defaults, stale descriptions, and duplicated explanations. Update
  each API's documentation alongside its implementation.
- [ ] Expand English API documentation with input/output shapes, mask and label
  conventions, dtype/device behavior, numerical precision, memory cost,
  randomness, side effects, errors, persistence, and runnable examples.
- [ ] Write a complete English README tutorial based on the chosen real NIR
  dataset, with a visible **Open in Colab** entry point and links to the detailed
  notebook/protocol. Include understandable setup and core code examples;
  preserve a small synthetic example for offline smoke tests and debugging.
  Move the model to its computation device before constructing the optimizer;
  explain parameter groups, scheduler timing, and training/inference precision.
- [ ] Keep README, notebook, API docs, and package version aligned. Avoid copying
  complete divergent training pipelines into multiple documents; share public
  APIs and cross-check the concise README example against the full notebook.
- [x] Show both the simple Trainer workflow and customization/plain-PyTorch
  workflows. Distinguish self-supervised pretraining from downstream supervised
  fine-tuning; make clear that synthetic tutorial data are illustrative.
- [ ] Explain fit versus inference data and how to reconstruct a spatial label
  map with its valid mask. Do not suggest fitting preprocessing/representation/
  cluster centers on held-out evaluation data as a default evaluation procedure.
- [x] Declare the existing `torch>=2.1` requirement as a runtime dependency rather
  than only a development extra. Keep the declared minimum working with a
  legacy CUDA GradScaler fallback; actual minimum-version execution is pending.
- [ ] Validate supported Python/Torch combinations, decide tutorial extras, and
  document environment-specific CPU/CUDA installation against tested versions.
- [x] Add a documentation index linking tutorials, API references, customization,
  numerical notes, and troubleshooting. Make source docstrings and Markdown docs
  agree on terminology. Release-specific ref pinning remains a publication gate;
  current documents explicitly describe development source.
- [ ] Document new APIs as available only after implementation. Include migration
  or release notes explaining the new contracts without promising old-format/API
  support. Keep the v0.2.2 historical experiment reference separate.
- [ ] Use `$...$` and blank-line-separated `$$...$$` with MathJax-supported TeX
  in README, docs, and notebook Markdown. Avoid package/custom-macro dependencies,
  unsupported equation/reference commands, and non-rendering math code fences.
  Check rendered formulas on GitHub and Colab, including the full LLA definition.
  [GitHub's native renderer uses MathJax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions);
  use [MathJax TeX and LaTeX support](https://docs.mathjax.org/en/latest/input/tex/index.html)
  as the authoring reference. Verify host-specific support; do not assume every
  MathJax extension is enabled on GitHub or Colab.

## 7. Installed-package and production release gates

- [x] Set the release target to v0.2.3 and align package metadata, the public
  version string, the version test, and development documentation.
- [ ] Update the current alpha status, supported-version claims, and remaining
  release metadata only when they match the release's actual readiness.
- [ ] Verify wheel/sdist contents, public imports, `py.typed`, license/notice files,
  and dependency declarations. Install the built wheel in a clean environment and
  run a minimal public-API CPU workflow without editable installs, development
  extras, or access to the source checkout.
- [ ] Extend validation beyond the current CI's editable/development installation:
  focused units, installed-package smoke, public workflow integration,
  save/load/resume, and notebook execution. Define the supported Python/Torch
  matrix and record which CUDA environments were actually exercised.
- [ ] Validate versioned artifacts with round-trip outputs, configuration errors,
  missing/corrupt state, and unsupported schema versions. Saving/loading must not
  change fitted predictions through implicit re-normalization or reconstruction.
- [ ] Verify fresh-Colab notebook execution, badge targets, dataset downloads,
  package/version alignment, and rendered English Markdown/math before release.
  Local notebook execution alone does not establish Colab compatibility.
- [ ] Publish measured runtime/memory and known limitations. Treat tutorial
  results as examples of the documented protocol, not a replacement for the
  completed thesis experiments or a universal ranking of methods.

Reference for runtime planning:
[Colab FAQ](https://research.google.com/colaboratory/faq.html). GPU types,
availability, and runtime resources vary; use measured, bounded tutorial paths.

## Delivery order and validation

- [ ] First agree the real-data tutorial flow, dataset/access/mask/grouping
  contract, and product API shape. Use these to drive implementation priorities.
- [ ] Implement equation-(11) LLA, native preprocessing, and coherent
  train/evaluate/encode/stream/cluster primitives with their focused tests.
- [ ] Complete Trainer customization, RNG ownership, versioned persistence, and
  diagnostics, then exercise the public workflow through the NIR notebook.
- [ ] Develop the English README tutorial, Colab notebook, API docs, and
  docstrings alongside implementation. Complete every production gate before
  declaring the product release ready; intermediate changes are not separate
  compatibility-preserving release commitments.
- [ ] Start validation with focused synthetic unit tests, then related existing
  tests. Benchmark GPU behavior only when explicitly authorized, using bounded
  synthetic inputs and the explicitly selected tutorial subset rather than
  rerunning completed research experiments. This planning update does not
  authorize data downloads, GPU training, package installation, or publication.
- [x] Report performed and unperformed checks separately. Do not claim tutorial,
  CUDA, resume, or numerical equivalence validation from source review alone.

### Focused validation commands (not executed)

Run from the repository root in an already prepared development environment.
These tests use synthetic inputs; no dataset download or notebook training is
needed. CUDA-specific cases can run/skip according to available hardware.

```powershell
python -m pytest -q tests/clustering/test_spatial.py tests/clustering/test_cosine_kmeans.py
python -m pytest -q tests/models/test_chemo_mae_encode.py tests/models/test_chemo_mae_mask.py tests/preprocessing/test_snv.py tests/preprocessing/test_fps.py
python -m pytest -q tests/training/test_trainer_smoke.py tests/training/test_tester.py tests/training/test_extractor.py tests/training/test_augmenter.py
```

Check independent LLA oracle counts and edge/undefined cases; precision/device/RNG
contracts; aggregate-versus-stream features and restored modes; evaluation
invariance to batch partitioning; successful-update scheduler/EMA behavior and
checkpoint restoration; exact fitted-center persistence. Only after these pass,
broaden to related tests and installed-package/notebook release gates. Test and
notebook execution remain with the user under the repository execution policy.

Performed for this implementation update: read-only source/API review, independent
focused reviews, `git diff --check`, native JSON parsing of the notebook (27 cells,
13 code cells, no execution counts or outputs), source-language search, and
display-math delimiter/blank-line checks. No project code, tests, lint, builds,
package installs, dataset downloads, training, GPU benchmarks, fresh-Colab runs,
or actual GitHub/Colab math rendering were executed.
