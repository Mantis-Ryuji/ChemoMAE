<h1 align="center">ChemoMAE</h1>

[![PyPI version](https://img.shields.io/pypi/v/chemomae.svg)](https://pypi.org/project/chemomae/)
[![CI](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/ChemoMAE/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/chemomae.svg)](https://pypi.org/project/chemomae/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)


> **ChemoMAE**: A research-oriented PyTorch toolkit and model for **1D spectral representation learning and clustering**.

---

## Why ChemoMAE ?

Traditional chemometrics has long relied on **linear methods** such as PCA or PLS for spectral analysis. While these approaches remain foundational, they can be limited when dealing with **nonlinear structure** and **high-dimensional variability** in modern datasets.

<br>

**ChemoMAE represents a new approach, introducing three key innovations beyond traditional linear chemometrics:**

### 1. Extending Chemometrics with Deep Learning

The **ChemoMAE model** leverages a **Transformer-based Masked Autoencoder (MAE)** adapted to 1D spectra. This enables more flexible, data-driven representation learning while remaining compatible with established preprocessing and scaling techniques.

### 2. Self-Supervised Representation Learning

By randomly masking large portions of the spectrum and reconstructing them, the **ChemoMAE model** learns from unlabeled data, producing **latent spectral embeddings** that generalize across downstream tasks.

### 3. Cosine Similarity Toolkit

Spectral data often carry more meaning in their **direction** than in their absolute magnitude — a perspective already implicit in traditional preprocessing methods such as **Standard Normal Variate (SNV)**, which normalize spectra to remove scale differences.

* All embeddings are **L2-normalized**, ensuring they lie on the unit hypersphere.  
* This design makes representations naturally compatible with **cosine similarity**, which is often more robust than Euclidean distance for spectral comparisons.  
* Built-in clustering modules (e.g., **Cosine-KMeans**, **vMF mixtures**) operate directly in this geometry, enabling analyses that respect the directional structure of spectral data.

<br>

By putting **cosine similarity** at its core, the **ChemoMAE package** provides a consistent framework where representation learning and clustering are fully aligned.

---

## Quick Start

Install ChemoMAE
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
