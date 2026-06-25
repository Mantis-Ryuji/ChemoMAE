import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.tester import Tester, TesterConfig


def _tiny_model(seq_len: int = 16) -> ChemoMAE:
    return ChemoMAE(
        seq_len=seq_len,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        latent_dim=8,
        n_patches=4,
        n_mask=1,
    )


class _SpyAugmenter(SpectraAugmenter):
    """Tester が augmenter を train mode で呼ぶことを検査するための deterministic augmenter。"""

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


def test_tester_runs_and_writes_history_without_augmenter(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 6
    seq_len = 16
    x = torch.randn(batch_size_total, seq_len)
    data_loader = DataLoader(TensorDataset(x), batch_size=3, shuffle=False)
    model = _tiny_model(seq_len)

    cfg = TesterConfig(
        out_dir=tmp_path,
        device="cpu",
        amp=False,
        loss_type="sse",
        reduction="batch_mean",
        fixed_visible=None,
        log_history=True,
        history_filename="test_history.json",
    )
    tester = Tester(model, cfg)

    avg = tester(data_loader)

    assert isinstance(avg, float)

    hist_path = tmp_path / "test_history.json"
    assert hist_path.exists()

    items = json.loads(hist_path.read_text(encoding="utf-8"))
    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[-1]["phase"] == "test"
    assert items[-1]["loss_type"] == "sse"
    assert items[-1]["reduction"] == "batch_mean"
    assert items[-1]["augmented"] is False


def test_tester_fixed_visible_path_without_augmenter(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 4
    seq_len = 24
    x = torch.randn(batch_size_total, seq_len)
    data_loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    model = _tiny_model(seq_len)

    # 前半のみ可視にする。True = visible。
    visible = torch.zeros(seq_len, dtype=torch.bool)
    visible[: seq_len // 2] = True

    cfg = TesterConfig(
        out_dir=tmp_path,
        device="cpu",
        amp=False,
        loss_type="mse",
        reduction="batch_mean",
        fixed_visible=visible,
        log_history=False,
    )
    tester = Tester(model, cfg)

    avg = tester(data_loader)

    assert isinstance(avg, float)


def test_tester_with_augmenter_applies_augmenter_and_restores_mode(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 6
    seq_len = 16
    x = torch.randn(batch_size_total, seq_len)
    data_loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    model = _tiny_model(seq_len)

    cfg = TesterConfig(
        out_dir=tmp_path,
        device="cpu",
        amp=False,
        loss_type="mse",
        reduction="batch_mean",
        fixed_visible=None,
        log_history=True,
        history_filename="test_history_aug.json",
    )

    augmenter = _SpyAugmenter()
    augmenter.eval()

    tester = Tester(
        model,
        cfg,
        augmenter=augmenter,
    )
    avg = tester(data_loader)

    assert isinstance(avg, float)

    assert augmenter.calls == len(data_loader)
    assert augmenter.seen_training == [True] * len(data_loader)

    # Tester は評価後に augmenter の元の状態を復元する。
    assert augmenter.training is False
    assert model.training is False

    hist_path = tmp_path / "test_history_aug.json"
    assert hist_path.exists()

    items = json.loads(hist_path.read_text(encoding="utf-8"))
    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[-1]["phase"] == "test"
    assert items[-1]["augmented"] is True


def test_tester_fixed_visible_with_augmenter_path(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 4
    seq_len = 24
    x = torch.randn(batch_size_total, seq_len)
    data_loader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    model = _tiny_model(seq_len)

    visible = torch.zeros(1, seq_len, dtype=torch.bool)
    visible[:, : seq_len // 2] = True

    cfg = TesterConfig(
        out_dir=tmp_path,
        device="cpu",
        amp=False,
        loss_type="mse",
        reduction="batch_mean",
        fixed_visible=visible,
        log_history=False,
    )

    augmenter = _SpyAugmenter()
    augmenter.eval()

    tester = Tester(
        model,
        cfg,
        augmenter=augmenter,
    )
    avg = tester(data_loader)

    assert isinstance(avg, float)
    assert augmenter.calls == len(data_loader)
    assert augmenter.seen_training == [True] * len(data_loader)
    assert augmenter.training is False