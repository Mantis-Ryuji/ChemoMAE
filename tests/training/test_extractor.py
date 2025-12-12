import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.training.extractor import Extractor, ExtractorConfig
from chemomae.models.chemo_mae import ChemoMAE


def _make_tiny_model(seq_len=16, d_model=16, latent_dim=8):
    return ChemoMAE(
        seq_len=seq_len, d_model=d_model, nhead=4, num_layers=1, dim_feedforward=32,
        dropout=0.0, use_learnable_pos=True, latent_dim=latent_dim,
        n_patches=4, n_mask=1
    )


def test_extracter_returns_cpu_tensor_and_respects_return_numpy(tmp_path):
    B, L = 5, 16
    x = torch.randn(B, L)
    loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)

    model = _make_tiny_model(seq_len=L, d_model=16, latent_dim=8)
    cfg = ExtractorConfig(device="cpu", amp=False, save_path=tmp_path / "z.pt", return_numpy=False)
    ext = Extractor(model, cfg)

    Z = ext(loader)
    assert isinstance(Z, torch.Tensor)
    assert Z.shape[0] == B and Z.shape[1] == 8
    assert Z.device.type == "cpu"

    # saved file exists
    assert (tmp_path / "z.pt").exists()

    # numpy return path
    cfg2 = ExtractorConfig(device="cpu", amp=False, save_path=tmp_path / "z.npy", return_numpy=True)
    Znp = Extractor(model, cfg2)(loader)
    assert Znp.shape == (B, 8)
