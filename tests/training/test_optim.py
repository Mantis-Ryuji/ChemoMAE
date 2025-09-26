import torch
import torch.nn as nn

from chemomae.training.optim import build_optimizer, build_warmup_cosine, build_scheduler


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 8))   # should be no_decay
        self.pos_embed = nn.Parameter(torch.zeros(1, 8))   # should be no_decay
        self.lin = nn.Linear(8, 8)                         # weight decay
        self.norm1 = nn.LayerNorm(8)                       # no_decay
        self.head = nn.Linear(8, 2)                        # weight decay


def test_build_optimizer_groups_exclusions():
    m = Toy()
    opt = build_optimizer(m, lr=1e-3, weight_decay=0.1)
    groups = opt.param_groups

    # two groups: decay and no_decay
    assert len(groups) == 2
    wd = sorted(set(g["weight_decay"] for g in groups))
    assert wd == [0.0, 0.1]

    # check some params go to expected groups
    names = {id(p): n for n, p in m.named_parameters()}
    for g in groups:
        if g["weight_decay"] == 0.0:  # no_decay
            for p in g["params"]:
                n = names[id(p)]
                assert (n.endswith(".bias")) or ("cls_token" in n) or ("pos_embed" in n) or ("norm1" in n)
        else:
            for p in g["params"]:
                n = names[id(p)]
                assert ("cls_token" not in n) and ("pos_embed" not in n) and ("norm1" not in n) and (not n.endswith(".bias"))


def test_warmup_cosine_and_wrapper_scheduler_steps():
    m = Toy()
    opt = build_optimizer(m, lr=1e-3, weight_decay=0.1)

    # direct builder
    sch = build_warmup_cosine(opt, warmup_steps=2, total_steps=10, min_lr_scale=0.2)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()
    assert min(lrs) >= 1e-3 * 0.2 and max(lrs) <= 1e-3

    # wrapper builder
    opt2 = build_optimizer(m, lr=1e-3, weight_decay=0.1)
    sch2 = build_scheduler(opt2, steps_per_epoch=5, epochs=2, warmup_epochs=1, min_lr_scale=0.1)
    lrs2 = []
    for _ in range(10):
        lrs2.append(opt2.param_groups[0]["lr"])
        sch2.step()
    assert min(lrs2) >= 1e-3 * 0.1
