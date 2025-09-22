import json
import torch
from torch.utils.data import DataLoader, TensorDataset

from wavemae.training.tester import Tester
from wavemae.models.wave_mae import WaveMAE


def _tiny_model(L=16):
    return WaveMAE(
        seq_len=L, d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
        dropout=0.0, use_learnable_pos=True, latent_dim=8,
        dec_hidden=16, dec_dropout=0.0, n_blocks=4, n_mask=1
    )


def test_tester_runs_and_writes_history(tmp_path):
    B, L = 6, 16
    x = torch.randn(B, L)
    dl = DataLoader(TensorDataset(x), batch_size=3, shuffle=False)
    model = _tiny_model(L)

    t = Tester(model, device="cpu", out_dir=tmp_path)
    avg = t.run(dl, criterion="sse", reduction="batch_mean", fixed_mask=None)
    assert isinstance(avg, float)

    # history file appended
    hist_path = tmp_path / "training_history.json"
    assert hist_path.exists()
    items = json.loads(hist_path.read_text(encoding="utf-8"))
    assert isinstance(items, list) and len(items) >= 1
    assert items[-1]["phase"] == "test" and items[-1]["criterion"] == "sse"


def test_tester_fixed_mask_path(tmp_path):
    B, L = 4, 24
    x = torch.randn(B, L)
    dl = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    model = _tiny_model(L)

    # use a fixed half-mask
    mask = torch.zeros(L, dtype=torch.bool)
    mask[: L // 2] = True

    t = Tester(model, device="cpu", out_dir=tmp_path)
    avg = t.run(dl, criterion="mse", reduction="batch_mean", fixed_mask=mask)
    assert isinstance(avg, float)
