# Optimizer and scheduler builders

`chemomae.training.optim` provides grouped AdamW and a per-update
linear-warmup/cosine scheduler. These defaults are an explicit recipe, not a
scientific recommendation for every spectrum, model, or dataset.

## AdamW parameter groups

```python
import torch
from chemomae.models import ChemoMAE
from chemomae.training import build_optimizer

device = torch.device("cpu")
model = ChemoMAE(seq_len=256).to(device)
optimizer = build_optimizer(model, lr=1.5e-4, weight_decay=0.05)

parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
for index, group in enumerate(optimizer.param_groups):
    names = [parameter_names[id(parameter)] for parameter in group["params"]]
    print(index, "weight_decay=", group["weight_decay"], "parameters=", names)
```

This inspection example uses CPU. Move the model to your chosen training device
and configure `requires_grad` before constructing the optimizer. Frozen
parameters are omitted.

| Argument | Default | Meaning |
| --- | --- | --- |
| `lr` | `1.5e-4` | Group base learning rate before scheduler scaling |
| `weight_decay` | `0.05` | AdamW decoupled decay for the decay group |
| `betas` | `(0.9, 0.95)` | AdamW moment coefficients |
| `eps` | `1e-8` | AdamW stability term |

No decay applies to parameter names ending in `.bias`, any parameter whose
module/ancestor walk includes `LayerNorm`, and names containing `cls_token` or
`pos_embed`. These are exact name/module rules; a top-level parameter simply
named `bias` does not match the `.bias` suffix rule. Other parameters receive
`weight_decay`. Only nonempty groups are created, with decay first and no-decay
second when both exist. Inspect `optimizer.param_groups` rather than assuming
every model yields two groups.

## Epoch-sized scheduler budgets

```python
from torch.utils.data import DataLoader, TensorDataset
from chemomae.training import build_scheduler

train_loader = DataLoader(
    TensorDataset(torch.randn(12, 256)), batch_size=4, shuffle=False,
)
scheduler = build_scheduler(
    optimizer,
    steps_per_epoch=len(train_loader),
    epochs=2,
    warmup_epochs=1,
    min_lr_scale=0.1,
)
```

The small synthetic `train_loader` above makes the budget example runnable in
the same session as the optimizer setup. Replace it with your training iterable
for a real application. The wrapper sets
`total_steps = steps_per_epoch * epochs` and
`warmup_steps = steps_per_epoch * warmup_epochs`, then delegates to
`build_warmup_cosine`. Each group receives the same multiplier relative to its
own base rate.

Use the lower-level builder for explicit update budgets:

```python
from chemomae.training.optim import build_warmup_cosine

scheduler = build_warmup_cosine(
    optimizer, warmup_steps=2, total_steps=6, min_lr_scale=0.1,
)
```

The lower-level default `min_lr_scale` is `0.0`; the epoch wrapper defaults to
`0.1`. Choose positive total steps, a meaningful warmup budget, and the intended
minimum scale explicitly. The builders do not validate arbitrary schedule
budgets. Their denominator guards prevent division by zero, not invalid research
or training configurations.

## Exact multiplier and update indexing

Let $S$ be `total_steps`, $W$ be `warmup_steps`, $\alpha$ be
`min_lr_scale`, and $s$ be the scheduler's zero-based index. The implementation is

$$
t(s) = \min\left(1,\frac{s-W}{\max(1,S-W)}\right),
$$

$$
\lambda(s) =
\begin{cases}
\max\left(10^{-8},\frac{s+1}{\max(1,W)}\right), & s\lt W, \cr
\alpha+\frac{1-\alpha}{2}\left(1+\cos(\pi t(s))\right), & s\ge W.
\end{cases}
$$

The rate at that index is the base learning rate multiplied by $\lambda(s)$.

`LambdaLR` applies index **0 during construction**. With positive warmup, the
first optimizer update therefore uses the positive multiplier
$\max(10^{-8},1/W)$, rather than starting at zero. With no warmup, index 0 uses
the full base rate.

Call `optimizer.step()` before `scheduler.step()`. If every update succeeds and
the scheduler is stepped once afterward, the $j$-th update uses index $j-1$;
the following scheduler call installs index $j$ for the next update.

| Point in a normal budget with $0<W<S$ | Scheduler index | Effect |
| --- | --- | --- |
| Before first update | $0$ | Positive initial warmup rate |
| Update $W$ | $W-1$ | First use of the full base rate |
| Update $W+1$ | $W$ | Cosine decay starts at the full base rate again |
| Last planned update $S$ | $S-1$ | Uses the penultimate cosine index |
| After that update and scheduler call | $S$ | Installs the minimum scale $\alpha$ |

The full base rate consequently appears at two adjacent indices around a
positive warmup boundary. With $W<S$ and $\alpha<1$, the final planned optimizer
update uses a rate above the minimum; the minimum is installed **after** that
update. Reading `get_last_lr()` after the scheduler step describes the next
update's rate. If $W\ge S$, the declared budget can end before cosine decay or
its minimum is used.

The following illustrative snippet exposes consumed versus next rates. It is
a scalar optimization example, not a spectral experimental recipe:

```python
import torch
from chemomae.training.optim import build_warmup_cosine

parameter = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.AdamW([parameter], lr=1e-3)
scheduler = build_warmup_cosine(
    optimizer, warmup_steps=2, total_steps=6, min_lr_scale=0.1,
)
for update in range(1, 7):
    optimizer.zero_grad(set_to_none=True)
    consumed_lr = optimizer.param_groups[0]["lr"]
    parameter.square().backward()
    optimizer.step()
    scheduler.step()
    print(update, "used=", consumed_lr, "next=", scheduler.get_last_lr()[0])
```

## Trainer, AMP, accumulation, and resume

Trainer advances the scheduler and EMA only after a successful optimizer update.
An AMP-skipped attempt increments attempted/skip counters without advancing the
scheduler. If an epoch-sized budget contains skipped updates, fewer scheduler
indices are consumed; reaching the nominal epoch count need not reach the
declared minimum rate.

For gradient accumulation in a caller-owned loop, advance the scheduler after
the actual successful optimizer update, rather than after every microbatch.
Specify whether the schedule budget counts attempts or successful updates.

Recreate the same optimizer/scheduler recipe before loading a training checkpoint.
The public Trainer restores their saved state at a completed epoch. Use its
checkpoint extension hooks for caller-owned state such as generators; do not
expect a scheduler constructor alone to restore an advanced stream or step index.

See [Trainer](trainer.md) for the public loop/customization contract.
