import torch
import torch.nn as nn

from chemomae.training.callbacks import EMACallback


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