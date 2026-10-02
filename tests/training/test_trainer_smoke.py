import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Literal

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.optim import build_optimizer, build_scheduler
from chemomae.training.trainer import PreparedBatch, Trainer, TrainerConfig


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
        "attempted_steps": len(train_dl) * epochs,
        "optimizer_updates": len(train_dl) * epochs,
        "amp_skips": 0,
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
        "attempted_steps",
        "optimizer_updates",
        "amp_skips",
        "cumulative_attempted_steps",
        "cumulative_optimizer_updates",
        "cumulative_amp_skips",
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
        "attempted_steps": len(train_dl) * epochs,
        "optimizer_updates": len(train_dl) * epochs,
        "amp_skips": 0,
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
        "attempted_steps",
        "optimizer_updates",
        "amp_skips",
        "cumulative_attempted_steps",
        "cumulative_optimizer_updates",
        "cumulative_amp_skips",
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
    assert TrainerConfig().device is None
    assert TrainerConfig().amp is False

    trainer = _loss_trainer(tmp_path, n_mask=1)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_recon = torch.zeros_like(x)
    visible_mask = torch.tensor([[True, False], [False, True]])

    assert trainer.compute_loss(x_recon, x, visible_mask).item() == pytest.approx(6.5)


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

    loss = trainer.compute_loss(x_recon, x, visible_mask)

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

    loss = trainer.compute_loss(x_recon, x, visible_mask)
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


class _PreparedEchoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, visible_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visible_mask is None:
            visible_mask = torch.zeros_like(x, dtype=torch.bool)
        return x * self.scale, x.mean(dim=1, keepdim=True), visible_mask


class _PublicHookTrainer(Trainer):
    def __init__(self, directory: Path, resume_from: Path | None = None) -> None:
        self.generator = torch.Generator().manual_seed(12)
        self.spectra = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        self.order: list[int] = []
        self.events: list[tuple[str, int]] = []
        model = _PreparedEchoModel()
        super().__init__(
            model, torch.optim.SGD(model.parameters(), lr=0.05), (),
            cfg=TrainerConfig(
                out_dir=directory, device="cpu", amp=False, use_ema=False,
                grad_clip=None, resume_from=resume_from,
            ),
        )

    def train_batches(self, epoch: int) -> Iterator[torch.Tensor]:
        self.events.append(("train_batches", epoch))
        order = torch.randperm(len(self.spectra), generator=self.generator)
        self.order.extend(order.tolist())
        for rows in order.split(2):
            yield self.spectra[rows]

    def prepare_batch(self, batch: object) -> PreparedBatch:
        assert isinstance(batch, torch.Tensor)
        self.events.append(("prepare_batch", self.current_epoch))
        visible = torch.ones_like(batch, dtype=torch.bool)
        visible[:, -2:] = False
        return PreparedBatch(batch + 1.0, batch, visible)

    def forward_batch(self, batch: PreparedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self.events.append(("forward_batch", self.current_epoch))
        return super().forward_batch(batch)

    def compute_loss(
        self, reconstructed: torch.Tensor, target: torch.Tensor, visible_mask: torch.Tensor
    ) -> torch.Tensor:
        self.events.append(("compute_loss", self.current_epoch))
        return super().compute_loss(reconstructed, target, visible_mask)

    def before_epoch(self, epoch: int) -> None:
        self.events.append(("before_epoch", epoch))

    def before_step(self, epoch: int, batch_index: int, batch: PreparedBatch) -> None:
        self.events.append(("before_step", epoch))
        assert batch.model_input.device == self.device
        assert batch.visible_mask is not None
        self.optimizer.param_groups[0]["lr"] = 0.05 / (epoch + batch_index)

    def after_step(
        self, epoch: int, batch_index: int, batch: PreparedBatch,
        *, loss: float, optimizer_updated: bool,
    ) -> None:
        self.events.append(("after_step", epoch))
        assert optimizer_updated and loss >= 0

    def after_epoch(self, epoch: int, record: dict[str, object]) -> None:
        self.events.append(("after_epoch", epoch))
        record["application_epoch"] = epoch

    def checkpoint_extra_state(self) -> dict[str, object]:
        return {
            "generator_state": self.generator.get_state(), "order": list(self.order),
            "model": "application metadata cannot replace the core model state",
        }

    def load_checkpoint_extra_state(self, state: dict[str, object]) -> None:
        generator_state = state["generator_state"]
        assert isinstance(generator_state, torch.Tensor)
        self.generator.set_state(generator_state)
        order = state["order"]
        assert isinstance(order, list)
        self.order = list(order)


def test_public_hooks_cover_preparation_order_events_and_namespaced_resume(tmp_path: Path) -> None:
    reference = _PublicHookTrainer(tmp_path / "reference")
    reference_result = reference.fit(epochs=2)
    first = _PublicHookTrainer(tmp_path / "resumed")
    first.fit(epochs=1)
    checkpoint = first.ckpt_dir / "last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(saved["model"], dict)
    assert isinstance(saved["extension_state"]["model"], str)
    restored = _PublicHookTrainer(tmp_path / "resumed", resume_from=checkpoint)
    resumed_result = restored.fit(epochs=2)
    assert restored.order == reference.order
    torch.testing.assert_close(restored.model.scale, reference.model.scale, rtol=0, atol=0)
    assert resumed_result == reference_result
    assert restored.history[-1]["application_epoch"] == 2
    assert restored.history[-1]["attempted_steps"] == 2
    assert restored.history[-1]["cumulative_optimizer_updates"] == 4
    assert restored.events == [
        ("before_epoch", 2), ("train_batches", 2),
        ("prepare_batch", 2), ("before_step", 2),
        ("forward_batch", 2), ("compute_loss", 2), ("after_step", 2),
        ("prepare_batch", 2), ("before_step", 2),
        ("forward_batch", 2), ("compute_loss", 2), ("after_step", 2),
        ("after_epoch", 2),
    ]


def _stochastic_trainer(directory: Path, *, resume_from: Path | None = None, nhead: int = 2) -> Trainer:
    model = ChemoMAE(
        seq_len=16, n_patches=4, n_mask=2, d_model=8, nhead=nhead,
        num_layers=1, dropout=0.25, latent_dim=4,
    )
    spectra = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 128
    loader = DataLoader(TensorDataset(spectra), batch_size=4, shuffle=True, num_workers=0)
    return Trainer(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3), loader,
        cfg=TrainerConfig(
            out_dir=directory, device="cpu", resume_from=resume_from,
            use_ema=True, ema_decay=0.9, progress=False, verbose=False,
        ),
    )


def test_global_rng_resume_matches_uninterrupted_mask_dropout_and_shuffle(tmp_path: Path) -> None:
    from chemomae.utils import capture_rng_state, restore_rng_state

    original = capture_rng_state()
    try:
        torch.manual_seed(123)
        reference = _stochastic_trainer(tmp_path / "reference")
        expected_result = reference.fit(epochs=2)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        partial = _stochastic_trainer(tmp_path / "resumed")
        partial.fit(epochs=1)
        checkpoint = partial.ckpt_dir / "last.pt"
        torch.randn(17)
        resumed = _stochastic_trainer(tmp_path / "resumed", resume_from=checkpoint)
        assert resumed.fit(epochs=2) == expected_result
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for name, value in reference.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
        for name, value in reference.ema.shadow.items():
            torch.testing.assert_close(value, resumed.ema.shadow[name], rtol=0, atol=0)
    finally:
        restore_rng_state(original, restore_cuda=False)


def test_resume_rejects_changed_head_count_before_loading_weights(tmp_path: Path) -> None:
    source = _stochastic_trainer(tmp_path / "source", nhead=2)
    source.save_checkpoint(epoch=0)
    target = _stochastic_trainer(tmp_path / "target", nhead=1)
    original = {name: value.clone() for name, value in target.model.state_dict().items()}
    with pytest.raises(ValueError, match="configuration mismatch"):
        target.load_checkpoint(source.ckpt_dir / "last.pt")
    for name, value in original.items():
        torch.testing.assert_close(target.model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["schema", "missing_optimizer", "rng", "ema"])
def test_checkpoint_schema_and_required_states_are_validated(tmp_path: Path, corruption: str) -> None:
    source = _stochastic_trainer(tmp_path / "source")
    source.save_checkpoint(epoch=0)
    path = source.ckpt_dir / "last.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if corruption == "schema":
        state["format_version"] = 99
    elif corruption == "missing_optimizer":
        del state["optimizer"]
    elif corruption == "rng":
        state["rng_state"]["torch_cpu"] = torch.zeros(3, dtype=torch.uint8)
    else:
        state["ema"]["shadow"] = {}
    torch.save(state, path)
    target = _stochastic_trainer(tmp_path / "target")
    with pytest.raises(ValueError):
        target.load_checkpoint(path)


def test_outputs_and_logging_can_be_disabled(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    model = _PreparedEchoModel()
    directory = tmp_path / "no_files"
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.1), [torch.ones(2, 4)],
        cfg=TrainerConfig(
            out_dir=directory, resume_from=None, use_ema=False,
            history_file=None, checkpoint_dir=None, raw_weights_file=None,
            ema_weights_file=None, progress=False, verbose=False,
        ),
    )
    result = trainer.fit(epochs=1)
    assert result["final_model"] is None
    assert len(trainer.history) == 1
    assert not directory.exists()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("use_ema,raw_export,ema_export,selected", [
    (False, True, True, "raw.pt"),
    (True, True, False, "raw.pt"),
    (True, False, True, "ema.pt"),
    (True, True, True, "ema.pt"),
    (True, False, False, None),
    (False, False, True, None),
])
def test_enabled_exports_define_selection_in_result_and_checkpoint(
    tmp_path: Path, use_ema: bool, raw_export: bool,
    ema_export: bool, selected: str | None,
) -> None:
    model = _PreparedEchoModel()
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.1), [torch.ones(2, 4)],
        cfg=TrainerConfig(
            out_dir=tmp_path, resume_from=None, use_ema=use_ema,
            raw_weights_file="raw.pt" if raw_export else None,
            ema_weights_file="ema.pt" if ema_export else None,
            progress=False, verbose=False,
        ),
    )
    result = trainer.fit(epochs=1)
    checkpoint = torch.load(tmp_path / "checkpoints/last.pt", weights_only=False)
    assert result["final_model"] == selected
    expected_rule = "ema_last" if selected == "ema.pt" else "raw_last" if selected else None
    assert checkpoint["selection_rule"] == expected_rule
    assert (tmp_path / "raw.pt").exists() == raw_export
    assert (tmp_path / "ema.pt").exists() == (use_ema and ema_export)


def test_custom_output_paths_and_model_bundle_reload(tmp_path: Path) -> None:
    model = _tiny_model()
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), [torch.ones(2, 16)],
        cfg=TrainerConfig(
            out_dir=tmp_path, resume_from=None, use_ema=False,
            history_file="logs/history.json", checkpoint_dir="state",
            raw_weights_file="exports/raw.pt", progress=False, verbose=False,
        ),
    )
    result = trainer.fit(epochs=1)
    assert result["final_model"] == "exports/raw.pt"
    assert (tmp_path / "logs/history.json").exists()
    assert (tmp_path / "state/last.pt").exists()
    restored = ChemoMAE.load(tmp_path / "exports/raw.artifact.pt")
    assert restored.get_config() == model.get_config()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


class _SkipFirstScaler:
    """CPU test double for the documented standard GradScaler overflow policy."""

    def __init__(self) -> None:
        self.current_scale = 8.0
        self.calls = 0

    def is_enabled(self) -> bool:
        return True

    def get_scale(self) -> float:
        return self.current_scale

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.calls > 0:
            optimizer.step()
        self.calls += 1

    def update(self) -> None:
        if self.calls == 1:
            self.current_scale *= 0.5

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.current_scale, "calls": self.calls}


def test_amp_skip_advances_neither_scheduler_nor_ema_and_records_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _PreparedEchoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batch = PreparedBatch(torch.ones(1, 4), torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
    trainer = Trainer(
        model, optimizer, [batch, batch], scheduler=scheduler,
        cfg=TrainerConfig(
            out_dir=tmp_path, device="cpu", amp=False, grad_clip=None,
            use_ema=True, ema_decay=0.5, resume_from=None,
        ),
    )
    trainer.scaler = _SkipFirstScaler()  # type: ignore[assignment]
    assert trainer.ema is not None
    ema_calls: list[int] = []
    ema_update = trainer.ema.update

    def record_ema_update(current_model: nn.Module) -> None:
        ema_calls.append(trainer.attempted_steps)
        ema_update(current_model)

    monkeypatch.setattr(trainer.ema, "update", record_ema_update)
    step_events: list[bool] = []

    def after_step(
        epoch: int, batch_index: int, prepared: PreparedBatch,
        *, loss: float, optimizer_updated: bool,
    ) -> None:
        step_events.append(optimizer_updated)

    monkeypatch.setattr(trainer, "after_step", after_step)
    result = trainer.fit(epochs=1)
    assert step_events == [False, True]
    assert scheduler.last_epoch == 1
    assert ema_calls == [2]
    assert result["attempted_steps"] == 2
    assert result["optimizer_updates"] == result["amp_skips"] == 1
    assert trainer.history[0]["amp_skips"] == 1
    checkpoint = torch.load(trainer.ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["progress"] == {"attempted_steps": 2, "optimizer_updates": 1, "amp_skips": 1}
    assert checkpoint["step_policy"] == "successful_optimizer_update"


def test_prepared_batches_bypass_augmentation(tmp_path: Path) -> None:
    class RejectAugmentation(nn.Module):
        def forward(self, spectra: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Prepared model inputs must not be augmented again.")

    model = _PreparedEchoModel()
    batch = PreparedBatch(torch.ones(2, 4), torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.bool))
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.0), [batch],
        augmenter=RejectAugmentation(),  # type: ignore[arg-type]
        cfg=TrainerConfig(out_dir=tmp_path, device="cpu", amp=False, use_ema=False, resume_from=None),
    )
    assert trainer.train_one_epoch() == pytest.approx(1.0)


def test_empty_epoch_and_invalid_prepared_batch_fail_fast(tmp_path: Path) -> None:
    model = _PreparedEchoModel()
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.1), [],
        cfg=TrainerConfig(out_dir=tmp_path, device="cpu", amp=False, use_ema=False, resume_from=None),
    )
    with pytest.raises(ValueError, match="No training samples"):
        trainer.fit(epochs=1)
    assert not (trainer.ckpt_dir / "last.pt").exists()
    with pytest.raises(ValueError, match="same shape"):
        PreparedBatch(torch.ones(2, 4), torch.ones(1, 4))
    with pytest.raises(TypeError, match="boolean"):
        PreparedBatch(torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4))
    with pytest.raises(ValueError, match="dense"):
        PreparedBatch(torch.ones(2, 4).to_sparse(), torch.ones(2, 4))
    with pytest.raises(ValueError, match="stored values"):
        PreparedBatch(torch.empty(2, 4, device="meta"), torch.ones(2, 4))


def test_fresh_run_does_not_mix_existing_artifacts(tmp_path: Path) -> None:
    model = _PreparedEchoModel()
    config = TrainerConfig(out_dir=tmp_path, device="cpu", amp=False, use_ema=False, resume_from=None)
    first = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), [torch.ones(1, 4)], cfg=config)
    first.fit(epochs=1)
    with pytest.raises(FileExistsError, match="resume explicitly"):
        Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), [], cfg=config)


@pytest.mark.parametrize("filename", ["training_history.json", "last_model.pt", "ema_last_model.pt"])
def test_auto_resume_rejects_artifacts_without_epoch_checkpoint(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).write_bytes(b"orphaned artifact")
    model = _PreparedEchoModel()
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.1), [],
        cfg=TrainerConfig(out_dir=tmp_path, device="cpu", amp=False, use_ema=False),
    )
    with pytest.raises(FileExistsError, match="without a resume checkpoint"):
        trainer.fit(epochs=1)


@pytest.mark.parametrize("parameter_free", [False, True])
def test_default_device_follows_model_without_gpu_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parameter_free: bool,
) -> None:
    model = nn.Identity() if parameter_free else _PreparedEchoModel()
    parameters = [nn.Parameter(torch.ones(()))] if parameter_free else list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.1)

    def reject_gpu_probe() -> bool:
        raise AssertionError("Default CPU models must not trigger GPU discovery.")

    monkeypatch.setattr(torch.cuda, "is_available", reject_gpu_probe)
    trainer = Trainer(
        model, optimizer, [],
        cfg=TrainerConfig(out_dir=tmp_path, use_ema=False, resume_from=None),
    )
    assert trainer.device == torch.device("cpu")
    assert not trainer.amp


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_amp_rejects_non_cuda_devices_before_moving_model(tmp_path: Path, device: str) -> None:
    model = _PreparedEchoModel()
    with pytest.raises(ValueError, match="AMP training requires a CUDA"):
        Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.1), [],
            cfg=TrainerConfig(out_dir=tmp_path, device=device, amp=True, resume_from=None),
        )


def test_explicit_cuda_and_bf16_require_device_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _PreparedEchoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested but is unavailable"):
        Trainer(model, optimizer, [], cfg=TrainerConfig(out_dir=tmp_path, device="cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="bf16 AMP is unsupported"):
        Trainer(
            model, optimizer, [],
            cfg=TrainerConfig(out_dir=tmp_path, device="cuda", amp=True, amp_dtype="bf16"),
        )


def test_fp16_amp_rejects_half_parameters_before_device_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = _PreparedEchoModel().half()
    with pytest.raises(ValueError, match="keep FP32 master parameters"):
        Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.1), [],
            cfg=TrainerConfig(out_dir=tmp_path, device="cuda", amp=True, amp_dtype="fp16"),
        )


def test_minimum_torch_scaler_api_without_torch_amp_grad_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyScaler:
        def __init__(self, *, enabled: bool) -> None:
            self.enabled = enabled

        def is_enabled(self) -> bool:
            return self.enabled

    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", LegacyScaler)
    model = _PreparedEchoModel()
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.1), [],
        cfg=TrainerConfig(out_dir=tmp_path, device="cpu", use_ema=False, resume_from=None),
    )
    assert isinstance(trainer.scaler, LegacyScaler)
    assert not trainer.scaler.is_enabled()


@pytest.mark.parametrize("field", ["model_input", "target"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_prepared_batch_rejects_nonfinite_values(field: str, value: float) -> None:
    model_input, target = torch.zeros(1, 4), torch.zeros(1, 4)
    invalid = model_input if field == "model_input" else target
    invalid[0, 0] = value
    with pytest.raises(ValueError, match=f"{field} contains NaN or infinity"):
        PreparedBatch(model_input, target)


@pytest.mark.parametrize("field", ["reconstruction", "target"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_loss_rejects_nonfinite_values_even_outside_selected_mask(
    tmp_path: Path, field: str, value: float,
) -> None:
    trainer = _loss_trainer(tmp_path, loss_region="masked")
    reconstructed, target = torch.zeros(1, 4), torch.zeros(1, 4)
    visible = torch.tensor([[True, False, True, False]])
    invalid = reconstructed if field == "reconstruction" else target
    invalid[0, 0] = value  # Visible and therefore excluded from the masked loss.
    with pytest.raises(ValueError, match="including positions outside the loss mask"):
        trainer.compute_loss(reconstructed, target, visible)


def test_amp_disabled_overrides_ambient_cpu_autocast_in_preparation_and_forward(tmp_path: Path) -> None:
    class PrecisionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.output_dtypes: list[torch.dtype] = []

        def forward(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            reconstructed = self.linear(spectra)
            self.output_dtypes.append(reconstructed.dtype)
            return reconstructed, reconstructed[:, :1], torch.ones_like(spectra, dtype=torch.bool)

    class PrecisionAugmenter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("identity", torch.eye(4))
            self.output_dtypes: list[torch.dtype] = []

        def forward(self, spectra: torch.Tensor) -> torch.Tensor:
            augmented = spectra @ self.identity
            self.output_dtypes.append(augmented.dtype)
            return augmented

    model, augmenter = PrecisionModel(), PrecisionAugmenter()
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.0), [torch.ones(2, 4)],
        augmenter=augmenter,  # type: ignore[arg-type]
        cfg=TrainerConfig(
            out_dir=tmp_path, device="cpu", loss_region="all",
            use_ema=False, grad_clip=None, resume_from=None,
        ),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        trainer.train_one_epoch()
        assert (torch.ones(2, 4) @ torch.ones(4, 2)).dtype == torch.bfloat16
    assert augmenter.output_dtypes == model.output_dtypes == [torch.float32]
