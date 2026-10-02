# Real NIR-HSI development tutorial

Use the [development notebook](../../notebooks/nir_hsi_tutorial.ipynb) to follow
prepare, train, resume, evaluate, encode, stream, cluster, predict spatial maps,
compute LLA, and save/reload fitted state using public ChemoMAE APIs.

**Status: unexecuted draft.** The notebook installs
`git+https://github.com/Mantis-Ryuji/ChemoMAE.git@main`, whose matching new APIs
must be published before execution. It checks those APIs before downloading data.
The installed source commit is recorded when available. This tutorial targets
ChemoMAE v0.2.3 under development, separately from the completed v0.2.2 research
environment. Direct downloads, fresh-Colab Run all, source/ref
pinning, math rendering, and measured cost remain validation gates. A live Colab
badge must wait until the notebook exists at a reachable, verified source ref.

## Dataset and attribution

The selected dataset is [Minerals in the Wild](https://github.com/EleftheriaTtl/minerals-in-the-wild/blob/main/README.md),
[release v1.0.0](https://github.com/EleftheriaTtl/minerals-in-the-wild/releases/tag/v1.0.0),
DOI [10.5281/zenodo.21414930](https://doi.org/10.5281/zenodo.21414930). Attribute
the data to E. Tetoula-Tsonga, G. Arvanitakis, and T. Giannakas under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The dataset has individual specimen IDs and float32 SWIR (short-wave infrared)
reflectance cubes with shape `(height, width, 273)`, covering 996.34–2504.28 nm.
Each image pixel carries one reflectance spectrum. Background is all-band NaN.
Partially nonfinite spectra and infinities are rejected. The tutorial linearly
interpolates valid pixel spectra to 256 bands on the wavelength axis, then applies
SNV to those spectra. The separate specimen-level XRF elemental-composition table
is not loaded or SNV-normalized and is not a source of pixel mineral annotations.
The tutorial makes no mineral-taxonomy claims.

## Bounded illustrative protocol

| Setting | Notebook choice |
| --- | --- |
| Download | Part01 only, streamed and checked against release SHA256SUMS; 1600 MiB hard cap |
| Specimens | `sample_0` through `sample_11`; verify their flat ZIP entries |
| Split | Seed 42 permutation, 8 train / 2 validation / 2 held-out specimens |
| Selected pixels | At most 128 per specimen, native indices and hashes saved |
| Model spectrum | Recorded-grid linear interpolation 273→256, same endpoints, no extrapolation; SNV afterward |
| Spatial maps | Held-out specimens only, at most 48×48 native crops; masks/offsets/absolute coordinates saved |
| Model | 16 patches of length 16; 8 hidden patches; width 32, one layer, latent width 8 |
| Training | Two epochs, batch 64, AMP/TF32 off, explicit CPU default and optional CUDA |
| Final weights | Predetermined EMA export with decay 0.9; raw export also retained |
| Clustering | K=4, at most 30 updates; fit training pixels only and freeze held-out prediction |
| Metrics | Clean all-visible full-spectrum MSE, crop LLA/occupancy and at most 256-pixel cosine silhouette |

The archive is **not** a twelve-specimen transfer. Its actual byte count is
displayed; metadata and download behavior have not been executed here. The
creator's script caps each archive at 1400 MiB uncompressed, but that does not
replace measuring its actual transfer size. No Google Drive is needed: the
dataset uses an external runtime cache and each run gets a fresh temporary output
directory. Copy exported results before the runtime is discarded.

Both wavelength grids, metadata bytes/hash/URL, and interpolation settings are
saved. Wavelength metadata currently come from the dataset's moving `main` ref.
Pin a verified ref and verify its relationship to the selected data release
before publishing scientific results or a release tutorial.

## What the notebook demonstrates

The feature-to-cell table in the notebook covers SNV, bounded public FPS, TGN/FS
with owned generators, optimizer/scheduler, public Trainer preparation/checkpoint
hooks, epoch-boundary interruption/resume, selected raw/EMA reload, Tester, four
representations, streamed Extractor, cosine clustering persistence, a spectral SNV
baseline, true spatial maps, LLA, and optional bounded vMF.

Resume persists this example's masking, augmentation, loader, and standard RNG
state through public extension hooks. Loading uses one process and completed
epoch boundaries. Uninterrupted/resumed trajectory equivalence is not yet
validated, and arbitrary worker/external-state recovery is not promised.

All undefined LLA/silhouette values retain reasons and are exported as JSON
`null`; they are never substituted with zero. LLA is reported separately per
crop/window, with raw agreement, corrected score, occupancy, and pair counts.
Crop boundaries exclude neighbors outside the crop. Colors are local numeric
cluster IDs and do not imply correspondence between baseline and latent models.

## Before treating this as an experiment

The first twelve specimens are a convenience subset, not a representative
benchmark. Site/context grouping and related acquisitions remain unverified.
Choose representative specimens, group-aware separation, model/K/weight selection,
repeat seeds, fair budgets/baselines, neighborhood scale, and specimen aggregation
explicitly before held-out evaluation. XRF cannot substitute for pixel labels.

High LLA does not establish chemical correctness. The quick path does not claim
superiority, perform scientific tuning, or generate LFR results. Define the LFR
perturbation/RNG/repetition protocol separately.

Verify public source publication, direct asset downloads/checksums, installation,
fresh-Colab Run all, CPU/CUDA results, save/load/resume, and GitHub/Colab math
rendering. Report actual elapsed time and memory; the notebook records CUDA peak
allocated bytes and, where available, process-lifetime peak RSS with that limitation.

