import json
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.training.tester import Tester, TesterConfig
from chemomae.models.chemo_mae import ChemoMAE


def _tiny_model(L=16):
    return ChemoMAE(
        seq_len=L, d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
        dropout=0.0, use_learnable_pos=True, latent_dim=8,
        dec_hidden=16, dec_dropout=0.0, n_blocks=4, n_mask=1
    )


def test_tester_runs_and_writes_history(tmp_path):
    B, L = 6, 16
    x = torch.randn(B, L)
    dl = DataLoader(TensorDataset(x), batch_size=3, shuffle=False)
    model = _tiny_model(L)

    cfg = TesterConfig(
        device="cpu",
        out_dir=tmp_path,
        amp=False,                    # CPU なのでAMPは無効
        criterion="sse",
        reduction="batch_mean",
        fixed_visible=None,
        log_history=True,
        history_filename="training_history.json",
    )
    t = Tester(model, cfg)

    with torch.inference_mode():
        avg = t(dl)
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

    # 前半のみ可視（True=visible）
    visible = torch.zeros(L, dtype=torch.bool)
    visible[: L // 2] = True

    cfg = TesterConfig(
        device="cpu",
        out_dir=tmp_path,
        amp=False,
        criterion="mse",
        reduction="batch_mean",
        fixed_visible=visible,
        log_history=False,        
    )
    t = Tester(model, cfg)

    with torch.inference_mode():
        avg = t(dl)        
    assert isinstance(avg, float)
