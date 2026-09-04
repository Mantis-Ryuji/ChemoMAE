import json
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.training.trainer import Trainer, TrainerConfig


def _tiny_model(seq_len: int = 16, n_mask: int = 1) -> ChemoMAE:
    return ChemoMAE(
        seq_len=seq_len,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        latent_dim=8,
        n_patches=4,
        n_mask=n_mask,
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
    assert set(history[0]) == {
        "epoch",
        "train_loss",
        "lr",
        "time_sec",
        "loss_region",
        "n_mask",
    }
    assert "val_loss" not in history[0]
    assert history[0]["epoch"] == 1
    assert history[0]["loss_region"] == "masked"
    assert history[0]["n_mask"] == 1
    assert history[-1]["epoch"] == epochs

    # The Trainer calls scheduler.step() once per optimizer update.
    expected_update_steps = len(train_dl) * epochs
    assert sch.last_epoch == expected_update_steps

    ckpt = torch.load((ckpt_dir / "last.pt").as_posix(), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == epochs
    assert ckpt["selection_rule"] == "ema_last"
    assert ckpt["ema"] is not None
    assert ckpt["loss_region"] == "masked"
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
    assert set(history[0]) == {
        "epoch",
        "train_loss",
        "lr",
        "time_sec",
        "loss_region",
        "n_mask",
    }
    assert "val_loss" not in history[0]
    assert history[0]["loss_region"] == "masked"
    assert history[0]["n_mask"] == 1

    expected_update_steps = len(train_dl) * epochs
    assert sch.last_epoch == expected_update_steps

    ckpt = torch.load((ckpt_dir / "last.pt").as_posix(), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == epochs
    assert ckpt["selection_rule"] == "raw_last"
    assert ckpt["ema"] is None
    assert ckpt["loss_region"] == "masked"
    assert "best" not in ckpt


def _loss_trainer(
    tmp_path: Path,
    *,
    loss_type: str = "mse",
    loss_region: Literal["masked", "all"] = "masked",
    reduction: str = "mean",
    n_mask: int = 0,
) -> Trainer:
    model = _tiny_model(seq_len=4, n_mask=n_mask)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train_loader = DataLoader(TensorDataset(torch.zeros(1, 4)), batch_size=1)
    cfg = TrainerConfig(
        out_dir=tmp_path,
        device="cpu",
        amp=False,
        use_ema=False,
        loss_type=loss_type,
        loss_region=loss_region,
        reduction=reduction,
        resume_from=None,
    )
    return Trainer(model, optimizer, train_loader, cfg=cfg)


def test_trainer_config_defaults_to_masked_loss_region(tmp_path: Path) -> None:
    assert TrainerConfig().loss_region == "masked"

    trainer = _loss_trainer(tmp_path, n_mask=1)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_recon = torch.zeros_like(x)
    visible_mask = torch.tensor([[True, False], [False, True]])

    assert trainer._compute_loss(x_recon, x, visible_mask).item() == pytest.approx(6.5)


def test_trainer_config_rejects_unknown_loss_region() -> None:
    with pytest.raises(ValueError, match="loss_region must be 'masked' or 'all'"):
        TrainerConfig(loss_region="visible")  # type: ignore[arg-type]


@pytest.mark.parametrize("loss_type", ["mse", "sse"])
@pytest.mark.parametrize(
    ("reduction", "expected"),
    [("sum", 30.0), ("mean", 7.5), ("batch_mean", 15.0)],
)
def test_all_region_loss_matches_manual_value(
    tmp_path: Path,
    loss_type: str,
    reduction: str,
    expected: float,
) -> None:
    trainer = _loss_trainer(
        tmp_path,
        loss_type=loss_type,
        loss_region="all",
        reduction=reduction,
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_recon = torch.zeros_like(x)
    visible_mask = torch.ones_like(x, dtype=torch.bool)

    loss = trainer._compute_loss(x_recon, x, visible_mask)

    assert loss.item() == pytest.approx(expected)


def test_all_region_with_zero_mask_backpropagates_to_encoder_and_decoder(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    model = _tiny_model(seq_len=16, n_mask=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train_loader = DataLoader(TensorDataset(torch.randn(2, 16)), batch_size=2)
    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        cfg=TrainerConfig(
            out_dir=tmp_path,
            device="cpu",
            amp=False,
            use_ema=False,
            loss_region="all",
            resume_from=None,
        ),
    )
    x = torch.randn(2, 16)
    x_recon, _, visible_mask = model(x)

    loss = trainer._compute_loss(x_recon, x, visible_mask)
    loss.backward()

    encoder_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert visible_mask.all()
    assert encoder_grad > 0.0
    assert decoder_grad > 0.0


def test_masked_region_with_zero_mask_fails_fast(tmp_path: Path) -> None:
    trainer = _loss_trainer(tmp_path, loss_region="masked", n_mask=0)

    with pytest.raises(ValueError, match="requires at least one masked element"):
        trainer.train_one_epoch()


class _AddOneAugmenter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _AllVisibleEchoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.n_mask = 0

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visible_mask = torch.ones_like(x, dtype=torch.bool)
        return x * self.scale, x.mean(dim=1, keepdim=True), visible_mask


def test_all_region_with_augmentation_uses_clean_target(tmp_path: Path) -> None:
    x = torch.zeros(2, 4)
    train_loader = DataLoader(TensorDataset(x), batch_size=2)
    model = _AllVisibleEchoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        augmenter=_AddOneAugmenter(),  # type: ignore[arg-type]
        cfg=TrainerConfig(
            out_dir=tmp_path,
            device="cpu",
            amp=False,
            grad_clip=None,
            use_ema=False,
            loss_region="all",
            resume_from=None,
        ),
    )

    loss = trainer.train_one_epoch()

    assert loss == pytest.approx(1.0)


def test_all_region_history_records_zero_mask_configuration(tmp_path: Path) -> None:
    trainer = _loss_trainer(tmp_path, loss_region="all", n_mask=0)

    trainer.fit(epochs=1)

    history = json.loads(
        (tmp_path / "training_history.json").read_text(encoding="utf-8")
    )
    assert history[0]["loss_region"] == "all"
    assert history[0]["n_mask"] == 0


def test_checkpoint_retains_and_validates_loss_region(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "source" / "checkpoints" / "last.pt"
    source = _loss_trainer(tmp_path / "source", loss_region="all")
    source.save_checkpoint(epoch=3)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["loss_region"] == "all"

    matching = _loss_trainer(tmp_path / "matching", loss_region="all")
    assert matching.load_checkpoint(checkpoint_path) == 4

    mismatched = _loss_trainer(tmp_path / "mismatched", loss_region="masked")
    with pytest.raises(ValueError, match="checkpoint loss_region mismatch"):
        mismatched.load_checkpoint(checkpoint_path)
