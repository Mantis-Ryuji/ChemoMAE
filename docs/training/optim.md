# Optimizer & Scheduler Builders for ChemoMAE

> Module: `chemomae.training.optim`

This module provides utilities to construct a **parameter-grouped AdamW optimizer** and a **linear-warmup + cosine-decay** learning-rate scheduler tailored for Transformer-based 1D spectral models such as ChemoMAE.

---

## Summary

* **`build_optimizer(model, lr=3e-4, weight_decay=1e-4, betas=(0.9,0.95), eps=1e-8)`**
  Returns an **AdamW** optimizer with **standard weight-decay exclusions**:

  * bias parameters (`.bias`)
  * LayerNorm weights
  * learned tokens and embeddings: `cls_token`, `pos_embed`

  Two parameter groups are created:
  `{weight_decay=wd}` and `{weight_decay=0.0}`.

* **`build_scheduler(optimizer, *, steps_per_epoch, epochs, warmup_epochs=1, min_lr_scale=0.1)`**
  Returns a `LambdaLR` implementing **linear warmup** (for `warmup_epochs`) followed by **cosine decay** down to `base_lr × min_lr_scale`.
  It’s a convenience wrapper around `build_warmup_cosine(...)`, which operates on global step counts.

---

## Rationale for Defaults

* **AdamW** is the de-facto optimizer for Vision Transformer–like architectures; its exclusions prevent over-regularization of normalization layers and special embeddings.
* **Linear warmup** stabilizes early training when activations and gradients are uncalibrated.
* **Cosine decay** yields a smooth, non-oscillatory LR schedule that converges gracefully.

---

## API Reference

### `build_optimizer(...) → torch.optim.AdamW`

#### Parameters

| Name           | Type                 | Default       | Description                                                   |
| -------------- | -------------------- | ------------- | ------------------------------------------------------------- |
| `model`        | `nn.Module`          | —             | Model whose parameters are grouped by decay / no-decay rules. |
| `lr`           | `float`              | `3e-4`        | Base learning rate.                                           |
| `weight_decay` | `float`              | `1e-4`        | L2 coefficient for the decay group.                           |
| `betas`        | `tuple[float,float]` | `(0.9, 0.95)` | AdamW β coefficients.                                         |
| `eps`          | `float`              | `1e-8`        | Numerical stability term for AdamW.                           |

#### Behavior

The function traverses the model hierarchy and assigns parameters to **no-decay** if they belong to:

* any bias parameter,
* LayerNorm weights, or
* special tokens/embeddings (`cls_token`, `pos_embed`).
  All others (typically Linear weights) belong to the **decay** group.

#### Example

```python
from chemomae.models import ChemoMAE
from chemomae.training.optim import build_optimizer

model = ChemoMAE(seq_len=256)
optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-4)

for i, g in enumerate(optimizer.param_groups):
    print(i, 'params=', len(g['params']), 'wd=', g['weight_decay'])
```

---

### `build_scheduler(...) → torch.optim.lr_scheduler.LambdaLR`

#### Parameters

| Name              | Type        | Default | Description                                            |
| ----------------- | ----------- | ------- | ------------------------------------------------------ |
| `optimizer`       | `Optimizer` | —       | Target optimizer.                                      |
| `steps_per_epoch` | `int`       | —       | Number of steps per epoch (e.g., `len(train_loader)`). |
| `epochs`          | `int`       | —       | Total number of training epochs.                       |
| `warmup_epochs`   | `int`       | `1`     | Number of epochs to linearly ramp LR from 0 → base LR. |
| `min_lr_scale`    | `float`     | `0.1`   | Final LR scale: `base_lr × min_lr_scale`.              |

#### Step-wise Definition

Let `S = steps_per_epoch × epochs`,
`W = steps_per_epoch × warmup_epochs`,
and global step index `s ∈ {0, 1, …}`.
The LR multiplier λ(s) is:

```math
\lambda(s) =
\begin{cases}
\max(10^{-8}, \tfrac{s+1}{\max(1,W)}) & s < W,\\[6pt]
\alpha + \tfrac{1-\alpha}{2}\bigl(1+\cos(\pi t)\bigr) & s \ge W, \quad
t=\tfrac{s-W}{\max(1, S-W)},\ \alpha=\text{min\_lr\_scale}.
\end{cases}
```

Actual LR: `lr(s) = base_lr × λ(s)`
If `warmup_epochs=0`, the function still behaves safely due to `max(1, W)`.

#### Example

```python
from chemomae.training.optim import build_optimizer, build_scheduler
optimizer = build_optimizer(model, lr=3e-4, weight_decay=1e-4)
scheduler = build_scheduler(
    optimizer,
    steps_per_epoch=len(train_loader),
    epochs=100,
    warmup_epochs=5,
    min_lr_scale=0.1
)

for epoch in range(100):
    for batch in train_loader:
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()     # one step per optimizer update
        optimizer.zero_grad(set_to_none=True)
```

---

### `build_warmup_cosine(optimizer, *, warmup_steps, total_steps, min_lr_scale=0.0)`

A lower-level variant that accepts explicit **global step** counts.
Useful when training with **gradient accumulation** or custom step accounting.

#### Parameters

| Name           | Type        | Default | Description                               |
| -------------- | ----------- | ------- | ----------------------------------------- |
| `optimizer`    | `Optimizer` | —       | Target optimizer.                         |
| `warmup_steps` | `int`       | —       | Number of warmup steps.                   |
| `total_steps`  | `int`       | —       | Total number of steps (including warmup). |
| `min_lr_scale` | `float`     | `0.0`   | Final LR scale at the end of training.    |

#### Example

```python
from chemomae.training.optim import build_warmup_cosine
opt = build_optimizer(model, lr=2e-4)
sched = build_warmup_cosine(opt, warmup_steps=1000, total_steps=50000, min_lr_scale=0.2)

for s in range(50000):
    loss = train_step()
    loss.backward()
    opt.step(); sched.step()
    opt.zero_grad(set_to_none=True)
```

---

## Practical Tips

* **Batch-size scaling:** When changing batch size significantly, scale `lr` linearly (and adjust `warmup_epochs` if needed).
* **Min LR:** `min_lr_scale=0.1` is robust; lower it for very long training or if late-epoch overfitting appears.
* **Frozen parameters:** Call `build_optimizer` *after* setting `requires_grad=False` to exclude frozen params.
* **EMA & AMP:** Compatible with both exponential moving average and mixed precision.
* **Step order:** Normally call `optimizer.step()` **before** `scheduler.step()`.
  If using gradient accumulation, step the scheduler only after the actual optimizer update.

---

## Minimal Tests

```python
opt = build_optimizer(model)
assert any(g['weight_decay'] > 0 for g in opt.param_groups)
assert any(g['weight_decay'] == 0 for g in opt.param_groups)

sched = build_scheduler(opt, steps_per_epoch=100, epochs=2, warmup_epochs=1)
lrs = []
for _ in range(200):
    opt.step(); sched.step()
    lrs.append(sched.get_last_lr()[0])
assert min(lrs[120:]) <= lrs[99]
```

---

## Version

* Introduced in `chemomae.training.optim` — initial public draft.