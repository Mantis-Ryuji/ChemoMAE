# ChemoMAE v0.2.3 Release ToDo

This v0.2.3 release plan was refined on 2026-10-04. Checked implementation items
mean that source, focused tests, and associated documentation have been written.
Checked validation items indicate completed checks. Final-commit CI, GitHub math rendering,
final contents/metadata review, and publication remain pending.

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
- [x] Add full ChemoMAE config/weight artifacts, versioned training checkpoints,
  semantic config validation, and standard global RNG capture/restoration.
- [x] Add independently configurable Trainer output paths, output enablement,
  progress/summary controls, and selected raw/EMA inference bundles.
- [x] Harden silhouette and elbow inputs; document precision and chunk residency.
- [x] Write a synthetic Markdown workflow, concise README example, artifact guide,
  v0.2.3 release notes, and an isolated installed-wheel CI smoke check.
- [x] Update MathJax authoring rules, API documentation, README examples, and
  move the existing Torch requirement into runtime dependencies.
- [x] Translate remaining source docstrings/comments into English and audit
  updated example signatures against their APIs; source-language search finds no Japanese
  in `src/chemomae`.
- [x] Run focused/full regression tests and CUDA fp16/bf16 smoke checks.
- [x] Check wheel/sdist contents and execute an isolated runtime-only wheel workflow.
- [x] Execute the detailed synthetic tutorial and selected documentation recipes.
- [x] Prepare direct Markdown-example execution, bounded LLA measurements, and
  minimum-Torch/supported-Python CI; gate publication on CI at the tagged commit.
- [x] Run minimum-Torch CPU regression and bounded CPU/CUDA LLA measurements.
- [ ] Run final-commit CI and check actual GitHub math rendering.

## Agreed direction and product scope

User decisions:

- Deliver the changes in this plan as ChemoMAE v0.2.3. The package metadata,
  public version string, version test, and development documentation use this
  target; release validation and publication remain pending.
- Finish a coherent production-ready library, documentation, and document-based
  tutorials as one release effort.
- Backward compatibility with v0.2.2 is not a constraint. Public APIs, defaults,
  configuration, return types, and artifact formats may be redesigned. Do not add
  legacy aliases or parallel implementations solely to preserve compatibility.
- Keep completed v0.2.2 research experiments and their environments/artifacts
  unchanged. Historical experiments do not have to be migrated or rerun.
- Explain the public workflow carefully in English Markdown documentation,
  using small synthetic examples that require only the library's dependencies.
- Write README, user documentation, and public docstrings in English. Use
  MathJax-compatible mathematics across Markdown documentation.

- [x] Define one public workflow for prepare/train/evaluate/encode/stream/cluster/
  save/load, with consistent shapes, device/dtype rules, errors, and terminology.
- [x] Review existing features against the tutorial and actual consuming workflows;
  simplify, redesign, or remove awkward/redundant interfaces where justified.
- [x] Keep datasets, splits, augmentation strengths, model-selection rules, and
  sample-level aggregation in the consuming research project. Do not silently
  transfer WoodDegradationMap's experimental recipe into library defaults.
- [x] Define synthetic tutorial inputs, masks, train/validation/test separation,
  and illustrative settings explicitly; keep research decisions with the caller.
- [x] Implement focused changes within the integrated release plan, each with
  tests and English documentation. Keep Trainer focused on reconstruction rather
  than building a general-purpose training framework.
- [x] Set the new defaults deliberately, including RNG ownership, scheduler/EMA
  behavior after an AMP skip, precision, and fresh-run/resume semantics. Explain
  changes in release notes; old behavior need not remain available.

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
- [x] Validate exact counts against the independent pixel-pair oracle. Cover
  boundaries, holes, diagonal neighbors, isolated
  pixels, one pixel, one class, negative scores, extreme imbalance, label
  permutations, class-chunk invariance, and CPU/CUDA comparison. Historical CPU
  implementation results remain reference evidence, not a rounding contract;
  tile invariance is inapplicable because image tiling is not implemented.
- [x] Run bounded synthetic CPU/CUDA measurements with `tests/benchmark_lla.py`
  across image sizes, class counts, windows, and class chunks. Check exact
  diagnostics before measuring resident calls, transfer-inclusive calls, and
  allocator peaks. Keep a usable CPU path without a universal speed claim.

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
- [x] Make history/checkpoint/export locations and progress/logging controls
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
  Include full-spectrum MSE and augmented-input/clean-target tests and examples.
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
- [x] Specify dtype, input ordering, model/augmenter mode handling, and iterator
  cleanup, including early termination; write focused deterministic stream tests.
- [x] Execute streamed-versus-aggregate and iterator/mode restoration tests.
- [x] Audit device/AMP resolution and validation across Trainer, Tester, and
  Extractor. Explain CPU use and unsupported precision/device combinations;
  avoid silently changing arithmetic or running stochastic augmentation.

## 4. Persistence and reproducibility

- [x] Define standard versioned artifacts and public save/load APIs for model
  config/weights, training state, and fitted clustering state. Include settings
  not recoverable from weights, such as attention heads, dropout, normalization,
  and masking configuration. Old weight/checkpoint formats need not be supported.
- [x] Define generator ownership for masking and augmentation with explicit
  `torch.Generator` control. Support independent streams without temporarily
  replacing global RNG state; retaining the old global-RNG behavior is optional.
- [x] Support saving/restoring standard RNG state and caller-owned generator
  state through the checkpoint extension contract. Validate schema/config/state
  and fail clearly on unsupported or incomplete artifacts. Callers own their
  generator schema through the extension hooks. Core checkpoints capture standard
  global streams and validate the full ChemoMAE configuration.
- [x] Define completed-epoch resume and write synthetic uninterrupted-versus-resumed
  tests for owned streams and standard masking/dropout/loader randomness.
- [x] Execute resume trajectory tests under matching CPU conditions.
  Do not promise restoration of arbitrary DataLoader
  worker/external state or matching trajectories across devices/versions.
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
- [x] Review chunking contracts across clustering and metrics: distinguish
  intermediate-memory bounds from full-input/full-output residency, avoid
  unnecessary device round trips, and document what remains allocated.
- [x] Harden cosine-silhouette input validation, including positive chunk sizes,
  valid class counts, shapes, integer labels, and nonfinite values. Document
  singleton and zero-vector conventions; write reference/edge-case tests.
- [x] Audit elbow curvature inputs and smoothing: reject mismatched/nonfinite
  curves, define the spacing assumptions, and keep the odd S-G window within
  even-length inputs. Its return contract is a scalar maximum curvature, not a
  full curvature array; source/doc behavior must agree.
- [x] Make Torch SNV operate natively without CPU/NumPy conversion.
  Define `ddof`, `sd + eps`, length-one/constant inputs, statistics, dtype/device,
  and numerical differences; run focused regression tests.
- [x] Audit FPS controls and validation: explicit computation device, ratio and
  initial-index validation, zero-vector handling, return types, and seed/generator
  behavior. Resolve documentation/implementation mismatches against the new API.

## 6. English documentation and README tutorial

Source docstrings/comments are translated, and README/docs are English. The
updated API signatures/defaults, local links, and math delimiters have been
reviewed in source. Runnable tutorial/documentation checks are complete.
Final-commit CI and actual rendered-math checks remain gates.

- [x] Translate Japanese public/module docstrings into English throughout
  `src/chemomae`, using NumPy-style sections for documented public interfaces.
  Translate explanatory
  implementation comments when maintaining the affected code. Keep docstrings
  aligned with the redesigned APIs and retain mathematical/numerical caveats.
- [x] Audit `docs/` and README in source for remaining Japanese and updated
  signatures/defaults, fix stale output/precision descriptions and example
  imports, and verify local documentation links. Keep API references aligned
  with implementation and link the shared workflow rather than repeating it.
- [x] Expand English API documentation with input/output shapes, mask and label
  conventions, dtype/device behavior, numerical precision, memory cost,
  randomness, side effects, errors, persistence, and runnable examples.
- [x] Write a complete English README tutorial linked to a detailed Markdown
  workflow. Include understandable setup and core code examples;
  preserve a small synthetic example for offline smoke tests and debugging.
  Move the model to its computation device before constructing the optimizer;
  explain parameter groups, scheduler timing, and training/inference precision.
- [x] Keep README, tutorial, API docs, and package version aligned. Avoid copying
  complete divergent training pipelines into multiple documents; share public
  APIs and cross-check the concise README example against the detailed tutorial.
- [x] Show both the simple Trainer workflow and customization/plain-PyTorch
  workflows. Distinguish self-supervised pretraining from downstream supervised
  fine-tuning; make clear that synthetic tutorial data are illustrative.
- [x] Explain fit versus inference data and how to reconstruct a spatial label
  map with its valid mask. Do not suggest fitting preprocessing/representation/
  cluster centers on held-out evaluation data as a default evaluation procedure.
- [x] Declare the existing `torch>=2.1` requirement as a runtime dependency rather
  than only a development extra. Keep the declared minimum working with a
  legacy CUDA GradScaler fallback.
- [x] Run the declared minimum-Torch CPU regression suite.
- [x] Keep the synthetic workflow within runtime dependencies; no separate
  tutorial extras are required. Reference comparisons using scikit-learn are
  explicitly development-only examples.
- [ ] Validate supported Python/Torch combinations and document
  environment-specific CPU/CUDA installation against tested versions. CI now
  includes Python 3.10–3.13/current CPU Torch and Python 3.10/Torch 2.1.0 with
  NumPy `<2` as an environment constraint, without changing runtime metadata.
  These are selected pairs, not every Python/Torch combination. The Ubuntu CI
  matrix still requires execution.
- [x] Add a runtime-dependency-only runner that reads the actual selected README,
  workflow, Trainer customization/plain-loop, and optimizer Markdown blocks.
- [x] Execute the selected Markdown recipes against the installed package;
  catalog fragments, shell fences, and other API reference snippets are excluded.
- [x] Add a documentation index linking tutorials, API references, customization,
  numerical notes, and troubleshooting. Make source docstrings and Markdown docs
  agree on terminology. Release-specific ref pinning remains a publication gate;
  current documents explicitly describe development source.
- [x] Document new APIs as available only after implementation. Include migration
  or release notes explaining the new contracts without promising old-format/API
  support. Keep the v0.2.2 historical experiment reference separate.
- [x] Use `$...$` and blank-line-separated `$$...$$` with MathJax-supported TeX
  in README, docs, and tutorial Markdown. Avoid package/custom-macro dependencies,
  unsupported equation/reference commands, and non-rendering math code fences.
  Actual rendering remains unverified.
- [ ] Check rendered formulas on GitHub, including the full LLA definition.
  [GitHub's native renderer uses MathJax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions);
  use [MathJax TeX and LaTeX support](https://docs.mathjax.org/en/latest/input/tex/index.html)
  as the authoring reference. Verify host-specific support; do not assume every
  MathJax extension is enabled on GitHub.

## 7. Installed-package and production release gates

- [x] Set the release target to v0.2.3 and align package metadata, the public
  version string, the version test, and development documentation.
- [ ] Update the current alpha status, supported-version claims, and remaining
  release metadata only when they match the release's actual readiness.
- [x] Add a runtime-only installed-package CPU workflow and CI job that builds
  wheel/sdist, installs the wheel without dev extras, and runs outside the checkout.
- [x] Verify wheel/sdist contents, public imports, `py.typed`, license/notice files,
  and dependency declarations. Install the built wheel in a clean environment and
  run a minimal public-API CPU workflow without editable installs, development
  extras, or access to the source checkout.
- [ ] Rebuild distributions and repeat the installed-wheel check after final
  release changes.
- [x] Extend CI with runtime-only documentation execution, a minimum-Torch lane,
  and reusable validation at the same tagged commit. Exclude RC tags from PyPI
  production publication; retain the existing TestPyPI base-version convention.
- [ ] Verify the new CI jobs and publication dependency/filter behavior. Do not
  create/push a release tag before the final review and readiness decision.
- [x] Validate versioned artifacts with round-trip outputs, configuration errors,
  missing/corrupt state, and unsupported schema versions. Saving/loading must not
  change fitted predictions through implicit re-normalization or reconstruction.
- [x] Verify the selected documented recipes and installed package version;
  excluded catalog fragments remain outside the runner's scope.
- [ ] Verify final package/version alignment and rendered English Markdown/math
  before release.
- [x] Document numerical and memory limits. Keep tutorial results illustrative
  rather than presenting them as universal performance rankings.

## Delivery order and validation

- [x] Define the document-based tutorial flow and product API shape. Use these
  to drive implementation priorities.
- [x] Implement equation-(11) LLA, native preprocessing, and coherent
  train/evaluate/encode/stream/cluster primitives with their focused tests.
- [x] Complete Trainer customization, RNG ownership, versioned persistence, and
  diagnostics, and write synthetic public-workflow examples.
- [x] Develop the English README tutorial, detailed Markdown guide, API docs, and
  docstrings alongside implementation.
- [x] Execute the detailed synthetic workflow.
- [x] Execute the selected additional recipes and minimum-Torch CPU regression.
- [x] Execute the bounded LLA correctness/time/memory checks.
- [ ] Complete final-commit compatibility CI, GitHub math, and final contents/
  metadata review before declaring release ready.
- [x] Start validation with focused synthetic unit tests, then related existing
  tests. Benchmark GPU behavior only when explicitly authorized, using bounded
  synthetic inputs rather than rerunning completed research experiments.
  Execution remains user-controlled under the repository policy.
- [x] Report performed and unperformed checks separately. Do not claim tutorial,
  CUDA, resume, or numerical equivalence validation from source review alone.

### Development checks

Run from the repository root in an already prepared development environment.
These tests use synthetic inputs. CUDA-specific cases can run/skip according to
available hardware.

```powershell
python -m pytest -q tests/clustering/test_spatial.py tests/clustering/test_cosine_kmeans.py
python -m pytest -q tests/clustering/test_metric.py tests/clustering/test_ops.py
python -m pytest -q tests/models/test_chemo_mae_encode.py tests/models/test_chemo_mae_mask.py tests/preprocessing/test_snv.py tests/preprocessing/test_fps.py
python -m pytest -q tests/models/test_chemo_mae_persistence.py tests/utils/test_rng_state.py tests/test_import.py
python -m pytest -q tests/training/test_trainer_smoke.py tests/training/test_tester.py tests/training/test_extractor.py tests/training/test_augmenter.py
```

Check independent LLA oracle counts and edge/undefined cases; precision/device/RNG
contracts; aggregate-versus-stream features and restored modes; evaluation
invariance to batch partitioning; successful-update scheduler/EMA behavior and
checkpoint restoration; exact fitted-center persistence. Only after these pass,
broaden to related tests and installed-package/example release gates. Test and
example execution remain with the user under the repository execution policy.

The standalone documentation runner reads selected actual Markdown Python blocks;
the benchmark reports public-API intervals and allocator peaks. CUDA-event spans
are not pure kernel times, and allocator statistics exclude other processes.

```powershell
python -B tests/documentation_examples.py
python -B tests/benchmark_lla.py --device both --threads 1
```
