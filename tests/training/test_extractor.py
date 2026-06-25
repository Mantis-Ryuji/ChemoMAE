import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.extractor import Extractor, ExtractorConfig


def _make_tiny_model(
    seq_len: int = 16,
    d_model: int = 16,
    latent_dim: int = 8,
) -> ChemoMAE:
    return ChemoMAE(
        seq_len=seq_len,
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        latent_dim=latent_dim,
        n_patches=4,
        n_mask=1,
    )


class _SpyAugmenter(SpectraAugmenter):
    """Extractor が augmenter を train mode で呼ぶことを検査するための deterministic augmenter。"""

    def __init__(self, delta: float = 0.01) -> None:
        super().__init__(
            SpectraAugmenterConfig(
                shift_prob=0.0,
                noise_prob=0.0,
            )
        )
        self.delta = delta
        self.calls = 0
        self.seen_training: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.seen_training.append(bool(self.training))
        if not self.training:
            return x
        return x + self.delta


def test_extractor_returns_cpu_tensor_and_respects_return_numpy(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 5
    seq_len = 16
    latent_dim = 8
    x = torch.randn(batch_size_total, seq_len)
    loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)

    model = _make_tiny_model(seq_len=seq_len, d_model=16, latent_dim=latent_dim)
    cfg = ExtractorConfig(
        device="cpu",
        amp=False,
        save_path=tmp_path / "z.pt",
        return_numpy=False,
    )
    extractor = Extractor(model, cfg)

    z = extractor(loader)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size_total, latent_dim)
    assert z.device.type == "cpu"
    assert (tmp_path / "z.pt").exists()

    loaded = torch.load(tmp_path / "z.pt", map_location="cpu", weights_only=False)
    assert isinstance(loaded, torch.Tensor)
    assert loaded.shape == (batch_size_total, latent_dim)

    cfg_numpy = ExtractorConfig(
        device="cpu",
        amp=False,
        save_path=tmp_path / "z.npy",
        return_numpy=True,
    )
    z_numpy = Extractor(model, cfg_numpy)(loader)

    assert isinstance(z_numpy, np.ndarray)
    assert z_numpy.shape == (batch_size_total, latent_dim)
    assert (tmp_path / "z.npy").exists()


def test_extractor_with_augmenter_applies_augmenter_and_restores_mode(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 6
    seq_len = 16
    latent_dim = 8
    x = torch.randn(batch_size_total, seq_len)
    loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)

    model = _make_tiny_model(seq_len=seq_len, d_model=16, latent_dim=latent_dim)
    cfg = ExtractorConfig(
        device="cpu",
        amp=False,
        save_path=tmp_path / "z_aug.pt",
        return_numpy=False,
    )

    augmenter = _SpyAugmenter()
    augmenter.eval()

    extractor = Extractor(
        model,
        cfg,
        augmenter=augmenter,
    )
    z = extractor(loader)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size_total, latent_dim)
    assert (tmp_path / "z_aug.pt").exists()

    assert augmenter.calls == len(loader)
    assert augmenter.seen_training == [True] * len(loader)

    # Extractor は評価後に augmenter の元の状態を復元する。
    assert augmenter.training is False
    assert model.training is False