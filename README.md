<h1 align="center">ChemoMAE</h1>

[![PyPI version](https://img.shields.io/pypi/v/chemomae.svg)](https://pypi.org/project/chemomae/)
[![CI](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/chemomae.svg)](https://pypi.org/project/chemomae/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)


> **ChemoMAE**: A Research-Oriented PyTorch Toolkit and Models for **1D Spectral Representation Learning and Spherical Clustering**.

---

## Why ChemoMAE ?

Traditional chemometrics has long relied on **linear methods** such as PCA and PLS. These remain foundational, but they can struggle with **nonlinear structure** and **high-dimensional variability** in modern spectral datasets.
<br>
**ChemoMAE is designed for the SNV-centric analysis flow**—the practical pipeline is **raw spectra → SNV preprocessing**, where information is effectively concentrated in **shape** (angles / cosine similarity). ChemoMAE learns representations suited to this geometry and uses them consistently in downstream tasks.

### 1) Extending Chemometrics with Deep Learning

A **Transformer-based Masked Autoencoder (MAE)** tailored to **1D spectra** enables flexible, data-driven representation learning while remaining compatible with standard chemometric preprocessing (including SNV).

### 2) Self-Supervised Learning (direction-aware)

We apply block masking to **SNV-preprocessed** spectra and optimize a masked MSE only on the hidden positions. The encoder outputs a **unit-norm embedding** `z`, capturing the directional  information. This design is consistent with the hyperspherical geometry induced by SNV and yields embeddings that are naturally suited for cosine similarity and hyperspherical clustering, while also transferring effectively to downstream tasks.

### 3) Spherical Geometry Toolkit (for downstream use)

Because embeddings are **L2-normalized**, they live on the **unit hypersphere** and are immediately compatible with **cosine similarity**. Built-in clustering modules (e.g., Cosine-KMeans, vMFMixture) operate natively in this geometry, enabling clustering that respects spectral shape after SNV.

---

## Quick Start

Install ChemoMAE (preparing)

```bash
pip install chemomae
```

### ChemoMAE example

```python

```

---

## Library Features

API、コード例 + docs_link

---

## License
