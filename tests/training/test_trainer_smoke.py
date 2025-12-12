import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.training.trainer import Trainer, TrainerConfig
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.models.chemo_mae import ChemoMAE


def _tiny_model(L=16):
    return ChemoMAE(
        seq_len=L, d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
        dropout=0.0, use_learnable_pos=True, latent_dim=8,
        n_patches=4, n_mask=1
    )


def test_trainer_fit_one_epoch_and_check_artifacts(tmp_path):
    torch.manual_seed(0)
    B, L = 12, 16
    x = torch.randn(B, L)
    train_dl = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)
    val_dl   = DataLoader(TensorDataset(x[:8]), batch_size=4, shuffle=False)

    model = _tiny_model(L)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    sch = build_scheduler(opt, steps_per_epoch=len(train_dl), epochs=2, warmup_epochs=1, min_lr_scale=0.1)

    # device は テストの再現性のため明示的に CPU 指定
    cfg = TrainerConfig(
        out_dir=str(tmp_path),
        device="cpu", 
        amp=False, enable_tf32=False,
        grad_clip=1.0, use_ema=True, ema_decay=0.9,
        loss_type="sse", reduction="batch_mean",
        early_stop_patience=5, early_stop_min_delta=0.0,
        resume_from=None,           # no resume in this test              
    )

    t = Trainer(model, opt, train_dl, val_dl, scheduler=sch, cfg=cfg)

    out = t.fit(epochs=2)
    # best updated within 2 epochs
    assert out["best"]["epoch"] > 0

    # artifacts created
    assert (tmp_path / "best_model.pt").exists()
    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert (tmp_path / "training_history.json").exists()
