import torch
import torch.nn as nn

from chemomae.training.callbacks import EarlyStopping, EMACallback


def test_early_stopping_starts_after_ratio_and_triggers():
    es = EarlyStopping(patience=3, min_delta=0.0, start_epoch_ratio=0.5)
    es.setup(total_epochs=10)  # → start at epoch 5

    # before start: no count-up even if not improved
    assert es.step(1, 1.0) is False
    assert es.started is False

    # best update
    assert es.step(2, 0.9) is False
    assert es.best == 0.9

    # start_epoch 到達後の不改善カウント
    # epoch=5 → count=1（継続）
    assert es.step(5, 0.95) is False
    # epoch=6 → count=2（継続）
    assert es.step(6, 0.95) is False
    # epoch=7 → count=3 = patience 到達 → 停止(True)
    assert es.step(7, 0.95) is True

    assert es.started is True
    assert es.best_epoch == 2


def test_ema_register_update_and_apply_to():
    torch.manual_seed(0)

    # tiny model
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    ema = EMACallback(m, decay=0.5)  # setup with shadow = current params

    # mutate model params, then update shadow
    for p in m.parameters():
        with torch.no_grad():
            p.add_(1.0)
    ema.update(m)

    # apply_to should blend towards updated weights (not equal to old)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ema.apply_to(m)
    after = {k: v for k, v in m.state_dict().items()}

    # at least one floating param should change after apply_to
    changed = any(not torch.allclose(before[k], after[k]) for k in before if before[k].dtype.is_floating_point)
    assert changed, "EMA apply_to should alter model parameters"
