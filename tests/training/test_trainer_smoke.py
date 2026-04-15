import json
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.training.trainer import Trainer, TrainerConfig


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


def _tiny_augmenter() -> SpectraAugmenter:
    cfg = SpectraAugmenterConfig(
        noise_prob=0.5,
        noise_cos_range=(0.995, 0.9995),
        tilt_prob=0.3,
        tilt_cos_range=(0.997, 0.9998),
    )
    return SpectraAugmenter(cfg)


def test_trainer_fit_with_val_and_augmenter_creates_best_and_last_artifacts(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 12
    seq_len = 16
    x = torch.randn(batch_size_total, seq_len)

    train_dl = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)
    val_dl = DataLoader(TensorDataset(x[:8]), batch_size=4, shuffle=False)

    model = _tiny_model(seq_len=seq_len)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    sch = build_scheduler(
        opt,
        steps_per_epoch=len(train_dl),
        epochs=2,
        warmup_epochs=1,
        min_lr_scale=0.1,
    )
    augmenter = _tiny_augmenter()

    cfg = TrainerConfig(
        out_dir=str(tmp_path),
        device="cpu",
        amp=False,
        enable_tf32=False,
        grad_clip=1.0,
        use_ema=True,
        ema_decay=0.9,
        loss_type="sse",
        reduction="batch_mean",
        early_stop_patience=5,
        early_stop_min_delta=0.0,
        resume_from=None,
    )

    trainer = Trainer(
        model,
        opt,
        train_dl,
        val_dl,
        scheduler=sch,
        augmenter=augmenter,
        cfg=cfg,
    )

    out = trainer.fit(epochs=2)

    assert out["best"]["epoch"] > 0
    assert out["epochs"] >= 1

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert (tmp_path / "training_history.json").exists()

    # Final exports are always produced.
    assert (tmp_path / "last_model.pt").exists()
    assert (tmp_path / "last_model_ema.pt").exists()

    # With validation + EMA enabled, best EMA export should exist.
    assert (tmp_path / "best_model_ema.pt").exists()

    # Legacy names should not be used anymore.
    assert not (tmp_path / "best_model.pt").exists()
    assert not (tmp_path / "ema_model.pt").exists()

    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert len(history) >= 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert "lr" in history[0]


def test_trainer_fit_without_val_and_with_augmenter_creates_last_exports(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 12
    seq_len = 16
    x = torch.randn(batch_size_total, seq_len)

    train_dl = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)

    model = _tiny_model(seq_len=seq_len)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    sch = build_scheduler(
        opt,
        steps_per_epoch=len(train_dl),
        epochs=2,
        warmup_epochs=1,
        min_lr_scale=0.1,
    )
    augmenter = _tiny_augmenter()

    cfg = TrainerConfig(
        out_dir=str(tmp_path),
        device="cpu",
        amp=False,
        enable_tf32=False,
        grad_clip=1.0,
        use_ema=True,
        ema_decay=0.9,
        loss_type="sse",
        reduction="batch_mean",
        early_stop_patience=5,
        early_stop_min_delta=0.0,
        resume_from=None,
    )

    trainer = Trainer(
        model,
        opt,
        train_dl,
        None,
        scheduler=sch,
        augmenter=augmenter,
        cfg=cfg,
    )

    out = trainer.fit(epochs=2)

    assert out["epochs"] >= 1
    assert out["best"]["epoch"] == -1

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert (tmp_path / "training_history.json").exists()

    # Final exports are always produced.
    assert (tmp_path / "last_model.pt").exists()
    assert (tmp_path / "last_model_ema.pt").exists()

    # Without validation, no best export should be produced.
    assert not (tmp_path / "best_model_ema.pt").exists()
    assert not (tmp_path / "best_model.pt").exists()

    # Legacy name should not be used anymore.
    assert not (tmp_path / "ema_model.pt").exists()

    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert len(history) >= 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert "lr" in history[0]

    # Validation-free run should record NaN val_loss in history.
    assert math.isnan(history[0]["val_loss"])