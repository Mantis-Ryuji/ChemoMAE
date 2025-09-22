import torch
import torch.nn as nn

from wavemae.training.callbacks import EarlyStopping, EMACallback


def test_early_stopping_starts_after_ratio_and_triggers():
    es = EarlyStopping(patience=3, min_delta=0.0, start_epoch_ratio=0.5)
    es.setup(total_epochs=10)  # → start at epoch 5

    # before start: no count-up even if not improved
    assert es.step(1, 1.0) is False
    assert es.started is False

    # best update
    assert es.step(2, 0.9) is False
    assert es.best == 0.9

    # from epoch >= 5, no-improve counts
    for e in [5, 6, 7]:
        assert es.step(e, 0.95) is False
    # next one (4th no-improve) should stop
    assert es.step(8, 0.96) is True
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
