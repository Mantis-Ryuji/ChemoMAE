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
    # 新API: fixed_visible=None
    avg = t.run(dl, criterion="sse", reduction="batch_mean", fixed_visible=None)
    assert isinstance(avg, float)

    # history file appended
    hist_path = tmp_path / "training_history.json"
    assert hist_path.exists()
    items = json.loads(hist_path.read_text(encoding="utf-8"))
    assert isinstance(items, list) and len(items) >= 1
    assert items[-1]["phase"] == "test" and items[-1]["criterion"] == "sse"


def test_tester_fixed_visible_path(tmp_path):
    B, L = 4, 24
    x = torch.randn(B, L)
    dl = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
    model = _tiny_model(L)

    # 旧: fixed_mask（True=隠す） → 新: fixed_visible（True=使う）
    # ここでは「前半のみ可視」にしたいので、visible を直接作る
    visible = torch.zeros(L, dtype=torch.bool)
    visible[: L // 2] = True  # 前半 True=可視

    t = Tester(model, device="cpu", out_dir=tmp_path)
    avg = t.run(dl, criterion="mse", reduction="batch_mean", fixed_visible=visible)
    assert isinstance(avg, float)
