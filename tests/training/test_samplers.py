import numpy as np
import torch
from torch.utils.data import TensorDataset

from wavemae.training.samplers import (
    compute_reference_vector,
    cosine_to_reference,
    make_weighted_sampler_by_cosine,
)


def test_compute_reference_and_cosine_shapes_and_bounds():
    X = np.random.RandomState(0).randn(20, 16).astype(np.float32)
    ref = compute_reference_vector(X)
    assert ref.shape == (16,) and np.isfinite(ref).all() and abs(np.linalg.norm(ref) - 1.0) < 1e-5

    sims = cosine_to_reference(X, ref)
    assert sims.shape == (20,) and np.all(sims <= 1.0 + 1e-6) and np.all(sims >= -1.0 - 1e-6)


def test_make_weighted_sampler_runs_and_weights_reasonable(tmp_path):
    torch.manual_seed(0)
    X = torch.randn(50, 16)
    ds = TensorDataset(X)
    ref = compute_reference_vector(X)
    sampler = make_weighted_sampler_by_cosine(
        ds, ref, cos_mid=0.5, cos_beta=8.0, clip=(0.3, 3.0), replacement=True,
        plot_path=tmp_path / "weight_plot.png"  # also tests plotting branch
    )
    # smoke: can draw indices and plot exists
    idx = list(iter(sampler))
    assert len(idx) == len(ds)
    assert (tmp_path / "weight_plot.png").exists()
