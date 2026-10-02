import json

import pytest
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
    assert model.training is True  # Original caller mode is restored.

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


class _DeterministicModel(torch.nn.Module):
    def forward(self, x):
        # Row value chooses 1, 2, or 3 evaluated positions, independently of B.
        selected = torch.arange(x.shape[1], device=x.device)[None] < (x[:, :1] + 1)
        return torch.zeros_like(x), x[:, :1], ~selected


@pytest.mark.parametrize("reduction, expected", [
    ("sum", 14.0), ("mean", 14.0 / 6), ("batch_mean", 14.0 / 3),
])
def test_dataset_reduction_is_invariant_to_batches_and_mask_counts(tmp_path, reduction, expected):
    x = torch.arange(3, dtype=torch.float32)[:, None].expand(3, 4)
    tester = Tester(_DeterministicModel(), TesterConfig(
        device="cpu", reduction=reduction, out_dir=tmp_path,
        log_history=False, progress=False,
    ))
    assert tester([x]) == pytest.approx(expected)
    assert tester([x[:2], x[2:]]) == pytest.approx(expected)


def test_all_region_with_unmasked_model_matches_full_spectrum_mse(tmp_path):
    model = _tiny_model().eval()
    model.n_mask = 0
    x = torch.randn(3, 16)
    with torch.no_grad():
        expected = (model.reconstruct(x, n_mask=0) - x).square().double().mean().item()
    tester = Tester(model, TesterConfig(
        out_dir=tmp_path, loss_region="all", log_history=True, progress=False,
    ))
    actual = tester([x[:2], x[2:]])
    assert actual == pytest.approx(expected, rel=1e-6)
    record = json.loads((tmp_path / "test_history.json").read_text())[-1]
    assert record["loss_region"] == "all"
    assert record["samples"] == 3 and record["selected_elements"] == 48


def test_empty_mask_and_empty_loader_fail_without_history_and_restore_modes(tmp_path):
    model = _tiny_model().train()
    model.encoder.eval()  # Preserve a deliberately mixed module mode.
    model.n_mask = 0
    augmenter = _SpyAugmenter().eval()
    tester = Tester(model, TesterConfig(
        device="cpu", out_dir=tmp_path, progress=False,
    ), augmenter=augmenter)
    with pytest.raises(ValueError, match="no elements selected"):
        tester([torch.randn(2, 16)])
    assert model.training and not model.encoder.training and not augmenter.training
    with pytest.raises(ValueError, match="no spectra"):
        tester([])
    assert model.training and not model.encoder.training and not augmenter.training
    assert not (tmp_path / "test_history.json").exists()


def test_all_visible_fixed_mask_supports_all_region_only(tmp_path):
    model = _tiny_model().eval()
    cfg = TesterConfig(out_dir=tmp_path, log_history=False, progress=False,
                       fixed_visible=torch.ones(16, dtype=torch.bool), loss_region="all")
    x = torch.randn(2, 16)
    assert Tester(model, cfg)([x]) > 0
    cfg.loss_region = "masked"
    with pytest.raises(ValueError, match="loss_region='all'"):
        Tester(model, cfg)([x])


def test_no_logging_has_no_directory_side_effect_and_cpu_amp_is_rejected(tmp_path):
    out_dir = tmp_path / "unused"
    Tester(_tiny_model(), TesterConfig(out_dir=out_dir, log_history=False))
    assert not out_dir.exists()
    with pytest.raises(ValueError, match="CUDA"):
        Tester(_tiny_model(), TesterConfig(device="cpu", amp=True, out_dir=out_dir))
    assert not out_dir.exists()


def test_corrupt_history_is_reported(tmp_path):
    (tmp_path / "test_history.json").write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        Tester(_tiny_model(), TesterConfig(device="cpu", out_dir=tmp_path))


def test_half_error_arithmetic_does_not_overflow(tmp_path):
    class HalfModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))

        def forward(self, x):
            return x + 1000, x[:, :1], torch.zeros_like(x, dtype=torch.bool)

    tester = Tester(HalfModel(), TesterConfig(
        out_dir=tmp_path, log_history=False, progress=False,
    ))
    assert tester([torch.zeros(2, 4)]) == 1_000_000.0


def test_empty_tuple_and_broadcasted_reconstruction_are_rejected(tmp_path):
    class BroadcastModel(torch.nn.Module):
        def forward(self, x):
            return x[:, :1], x[:, :1], torch.zeros_like(x, dtype=torch.bool)

    tester = Tester(BroadcastModel(), TesterConfig(
        out_dir=tmp_path, log_history=False, progress=False,
    ))
    with pytest.raises(ValueError, match="first item"):
        tester([[]])
    with pytest.raises(ValueError, match="reconstruction"):
        tester([torch.zeros(2, 4)])


@pytest.mark.parametrize("dtype, value", [(torch.float16, 1e8), (torch.float64, 1e200)])
def test_nonfinite_cast_or_error_does_not_log_success(tmp_path, dtype, value):
    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros((), dtype=dtype))

        def forward(self, x):
            return torch.zeros_like(x), x[:, :1], torch.zeros_like(x, dtype=torch.bool)

    model = ZeroModel().train()
    tester = Tester(model, TesterConfig(out_dir=tmp_path, progress=False))
    with pytest.raises(ValueError, match="finitely|nonfinite"):
        tester([torch.full((2, 4), value, dtype=torch.float64)])
    assert model.training
    assert not (tmp_path / "test_history.json").exists()
