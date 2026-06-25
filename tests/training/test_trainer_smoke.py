import json

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
    cfg = SpectraAugmenterConfig()
    return SpectraAugmenter(cfg)


def test_trainer_fit_with_ema_and_augmenter_creates_last_artifacts(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 12
    seq_len = 16
    epochs = 2
    x = torch.randn(batch_size_total, seq_len)

    train_dl = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)

    model = _tiny_model(seq_len=seq_len)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    sch = build_scheduler(
        opt,
        steps_per_epoch=len(train_dl),
        epochs=epochs,
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
        resume_from=None,
    )

    trainer = Trainer(
        model,
        opt,
        train_dl,
        scheduler=sch,
        augmenter=augmenter,
        cfg=cfg,
    )

    out = trainer.fit(epochs=epochs)

    assert out == {
        "epochs": epochs,
        "completed": True,
        "final_model": "ema_last_model.pt",
    }

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert (tmp_path / "training_history.json").exists()

    # Final exports are always produced.
    assert (tmp_path / "last_model.pt").exists()
    assert (tmp_path / "ema_last_model.pt").exists()

    # Validation-based / legacy names should not be produced anymore.
    assert not (ckpt_dir / "best.pt").exists()
    assert not (tmp_path / "best_model.pt").exists()
    assert not (tmp_path / "best_model_ema.pt").exists()
    assert not (tmp_path / "last_model_ema.pt").exists()
    assert not (tmp_path / "ema_model.pt").exists()

    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert len(history) == epochs
    assert set(history[0]) == {"epoch", "train_loss", "lr", "time_sec"}
    assert "val_loss" not in history[0]
    assert history[0]["epoch"] == 1
    assert history[-1]["epoch"] == epochs

    # The Trainer calls scheduler.step() once per optimizer update.
    expected_update_steps = len(train_dl) * epochs
    assert sch.last_epoch == expected_update_steps

    ckpt = torch.load((ckpt_dir / "last.pt").as_posix(), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == epochs
    assert ckpt["selection_rule"] == "ema_last"
    assert ckpt["ema"] is not None
    assert "best" not in ckpt


def test_trainer_fit_without_ema_or_augmenter_creates_raw_last_only(tmp_path) -> None:
    torch.manual_seed(0)

    batch_size_total = 12
    seq_len = 16
    epochs = 2
    x = torch.randn(batch_size_total, seq_len)

    train_dl = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)

    model = _tiny_model(seq_len=seq_len)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    sch = build_scheduler(
        opt,
        steps_per_epoch=len(train_dl),
        epochs=epochs,
        warmup_epochs=1,
        min_lr_scale=0.1,
    )

    cfg = TrainerConfig(
        out_dir=str(tmp_path),
        device="cpu",
        amp=False,
        enable_tf32=False,
        grad_clip=1.0,
        use_ema=False,
        loss_type="sse",
        reduction="batch_mean",
        resume_from=None,
    )

    trainer = Trainer(
        model,
        opt,
        train_dl,
        scheduler=sch,
        cfg=cfg,
    )

    out = trainer.fit(epochs=epochs)

    assert out == {
        "epochs": epochs,
        "completed": True,
        "final_model": "last_model.pt",
    }

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert (tmp_path / "training_history.json").exists()
    assert (tmp_path / "last_model.pt").exists()

    # EMA and validation-based artifacts should not be produced.
    assert not (tmp_path / "ema_last_model.pt").exists()
    assert not (tmp_path / "last_model_ema.pt").exists()
    assert not (ckpt_dir / "best.pt").exists()
    assert not (tmp_path / "best_model.pt").exists()
    assert not (tmp_path / "best_model_ema.pt").exists()
    assert not (tmp_path / "ema_model.pt").exists()

    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert len(history) == epochs
    assert set(history[0]) == {"epoch", "train_loss", "lr", "time_sec"}
    assert "val_loss" not in history[0]

    expected_update_steps = len(train_dl) * epochs
    assert sch.last_epoch == expected_update_steps

    ckpt = torch.load((ckpt_dir / "last.pt").as_posix(), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == epochs
    assert ckpt["selection_rule"] == "raw_last"
    assert ckpt["ema"] is None
    assert "best" not in ckpt