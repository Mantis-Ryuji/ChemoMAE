# Plan a first experiment

The [workflow tutorial](workflow.md) demonstrates the v0.2.4 APIs on synthetic
spectra. This guide covers configuration choices and memory estimates. The
tutorial's small model, epoch count, and K demonstrate APIs; choose settings
according to your application's requirements.

## Decide what a useful result would mean

Write down the intended use before choosing model settings: reconstructing
spectra, supplying features to a downstream predictor, describing directional
groups, or mapping a spatial partition. Choose the evaluation criterion and
comparison method for that purpose before examining final results. An existing
analysis can provide a reference when its inputs and evaluation conditions are
comparable; the library does not choose that reference for you.

Reconstruction error, silhouette, and LLA measure different properties. Lower
reconstruction error does not establish better downstream features; stronger
spatial agreement does not establish correct chemical identities. Record which
measure supports the intended claim and which measures are exploratory.

## Define the unit of separation

For held-out evaluation, identify which observations share information that
must stay within one partition. Depending on acquisition, the relevant unit
could be a specimen, subject, image, batch, or another caller-defined group.
Choose the unit from the data provenance; independent pixels are not an
automatic assumption. Keep row identities, group membership, and coordinates
when selecting or rearranging spectra.

Create train, validation, and final test partitions under that protocol before
fitting learned preprocessing, sampling training rows, or selecting settings.
Use validation results for the planned comparisons and reserve final test
results for the chosen procedure. For descriptive mapping of the supplied
dataset, fitting all mapped rows has a different interpretation; document that
choice instead of claiming held-out generalization.

## Fix a starting configuration

First check the complete train-save-load-extract path on a small, deliberately
selected input. Use the workflow's explicit NumPy dtype conversion, shared
`device`, new output directory, and `resume_from=None` for a new run. Distinguish
that functional check from an experiment used to assess scientific utility.

For the first comparison, record and hold fixed:

- Channel order, input length, dtype, preprocessing, and split membership. Use
  SNV only if removing each spectrum's mean and scale matches the intended use.
- Model dimensions, patch count, mask count, decoder depth, representation, and
  latent normalization. Check divisibility constraints before training.
- Optimizer, scheduler budget, batch size, stopping/selection rule, seeds, and
  the choice of raw or EMA weights for downstream work.
- Downstream fitting data, evaluation data, aggregation unit, and metrics.
  Keep these comparable across the chosen reference and candidate methods.

An initial run without augmentation isolates the input-to-representation path.
Record this as a chosen starting condition, not a claim that augmentation is
unnecessary. The synthetic tutorial's two epochs are only an API demonstration;
choose a training budget and stopping rule appropriate to the experiment.

## Compare a few settings deliberately

Choose a small set of candidate values and a selection rule before comparing
results. One possible sequence is to vary latent dimension while retaining the
starting setup, then vary mask count with patch count fixed. Record every
candidate, including runs that did not improve the selected criterion. This
sequence is a starting procedure, not an exhaustive search or evidence that
the selected configuration is globally optimal.

Consider decoder depth, normalization, or augmentation in subsequent comparisons
when the scientific question warrants them. Change one factor at a time for an
interpretable initial comparison; interactions may require a separate planned
experiment. For augmentation, define perturbations from measurement knowledge
and inspect their effect on representative spectra before assessing downstream
results. Choose repeated-run and uncertainty reporting policies explicitly;
one seed does not characterize run-to-run variability.

Use the selected artifact consistently for final evaluation. Record versions,
device, dtype, configuration, data/split identifiers, selection criteria, and
the exact artifact path. An artifact contains the model configuration and
weights; it does not preserve the full experiment protocol or data provenance.

## Estimate memory by operation

Consider **N = 1,000,000 rows, D = 64 feature dimensions, K = 20 clusters**, FP32
features, and a chunk of **B = 10,000 rows**. These are arithmetic storage
estimates, not recommended sizes, benchmarks, or peak-memory guarantees.
One MiB is 1,048,576 bytes. Retaining the original spectra requires additional
memory according to their channel count.

| One allocation | Dtype | Approximate size |
| --- | --- | ---: |
| Full feature matrix `(N, D)` | float32 | 244.1 MiB |
| Full responsibility/distance matrix `(N, K)` | float32 | 76.3 MiB |
| Full label vector `(N,)` | int64 | 7.6 MiB |
| Feature block `(B, D)` | float32 | 2.44 MiB |
| Similarity/responsibility block `(B, K)` | float32 | 0.76 MiB |
| Log-posterior block `(B, K)` | float64 | 1.53 MiB |
| Cluster directions `(K, D)` | float32 | 5 KiB |

Use those sizes to choose a calling pattern:

| Operation | Calling pattern | What remains resident or is additionally allocated |
| --- | --- | --- |
| Feature extraction | Consume `Extractor.iter_transform` batches and write or process them immediately | No full output concatenation is required. Retaining all yielded batches still retains the full output; model, activations, input data, and loader buffers are additional. |
| CosineKMeans on CUDA | Keep features on CPU and pass `chunk=10_000` | The 244.1 MiB CPU input remains; device assignment work is chunked, with full label and maximum-similarity vectors. CPU clustering does not chunk its similarity matrix. Requesting distances additionally retains 76.3 MiB for `(N, K)`. |
| VMFMixture fitting on CUDA | Keep features on CPU and pass `chunk=10_000` | The CPU input remains; EM uses blocks and cluster statistics. Log-posteriors use float64 even with FP32 model storage. Initialization has separate subset/permutation allocations. |
| VMFMixture probabilities or labels | Pass `chunk=10_000` and budget for the complete output | `predict_proba` allocates the full 76.3 MiB responsibility matrix on the model device. `predict` calls it before taking argmax, so labels alone still incur that allocation and then a 7.6 MiB label vector. |
| Cosine silhouette | Pass `chunk=10_000` only after budgeting full features and work arrays on `device` | Chunking reduces a `(B, K)` similarity tile to 0.76 MiB. The full input, normalized features, selected cluster sums, and within-class means each occupy about 244.1 MiB; those four arrays alone total about 976.6 MiB, before labels, row arrays, and arithmetic temporaries. |

These entries describe separate operations. Keeping intermediate features,
probabilities, models, or outputs alive across calls adds their storage together.
Transfers and dtype conversion can create extra copies, and library workspaces
and allocator reservations add overhead. Reducing `chunk` cannot remove a full
output or full input required by the selected API. In silhouette, even a very
small tile does not make the rest of the computation stream from CPU to GPU.

For label-only vMF inference on a large collection, an application can call
`predict` on successive input slices and consume the returned labels immediately.
The full responsibility allocation then applies to each slice, not the whole
collection. Preserve input order and coordinates yourself. This is different
from passing the full collection with a smaller internal `chunk`.

When sizing your own run, start with a size that fits the selected environment,
measure the actual peak, and retain headroom before increasing N or K.

See the detailed contracts for [Extractor](../training/extractor.md),
[CosineKMeans](../clustering/cosine_kmeans.md),
[VMFMixture](../clustering/vmf_mixture.md), and
[silhouette](../clustering/metric.md).
