# Optimizer & Scheduler Builders for ChemoMAE

> Module: `chemomae.training.optim`

This document describes the utility functions that build a **parameter‑grouped AdamW optimizer** and a **linear‑warmup + cosine‑decay** learning‑rate scheduler suitable for Transformer‑based spectral models.

---

## Summary

* **`build_optimizer(model, lr=3e-4, weight_decay=0.05, betas=(0.9,0.95), eps=1e-8)`**

  * Returns **AdamW** with **standard weight‑decay exclusions**:

    * **bias** parameters (names ending with `.bias`)
    * **LayerNorm** weights
    * special tokens/embeddings: `cls_token`, `pos_embed`
  * Produces two param groups: `{weight_decay=wd}` and `{weight_decay=0.0}`.

* **`build_scheduler(optimizer, *, steps_per_epoch, epochs, warmup_epochs=1, min_lr_scale=0.1)`**

  * Returns `LambdaLR` implementing **linear warmup** (for `warmup_epochs`) then **cosine decay** down to `base_lr * min_lr_scale`.
  * Convenience wrapper over `build_warmup_cosine(...)` which accepts step counts directly.

---

## Why these defaults?

* **AdamW** is standard for ViT/Transformer regimes; the listed exclusions are widely used to avoid over‑regularizing normalization layers and learned tokens.
* **Linear warmup** stabilizes early training when gradients/activations are not yet calibrated.
* **Cosine decay** provides smooth, non‑oscillatory LR annealing with a principled minimum.

---

## API Reference

### `build_optimizer(...) → torch.optim.AdamW`

**Parameters**

* `model (nn.Module)`: model whose parameters will be split into decay/no‑decay groups.
* `lr (float)`: base learning rate.
* `weight_decay (float)`: L2 coefficient for the **decay** param group.
* `betas (Tuple[float,float])`: AdamW β parameters.
* `eps (float)`: numerical stability term for AdamW.

**Behavior**

* Walks the module hierarchy to detect whether a parameter belongs to a `LayerNorm` chain or to the special tokens `cls_token`/`pos_embed`.
* Builds two groups:

  1. `decay`: typical weights (e.g., Linear/GEMM weights) with `weight_decay`
  2. `no_decay`: biases, LayerNorm, special tokens with `weight_decay=0.0`

**Example**

```python
from chemomae.models import ChemoMAE
from chemomae.training.optim import build_optimizer

model = ChemoMAE(seq_len=256)
optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.05)

# Inspect groups
for i, g in enumerate(optimizer.param_groups):
    print(i, 'params=', len(g['params']), 'wd=', g['weight_decay'])
```

---

### `build_scheduler(...) → torch.optim.lr_scheduler.LambdaLR`

**Parameters**

* `optimizer (Optimizer)`: the optimizer to schedule.
* `steps_per_epoch (int)`: training steps per epoch (e.g., `len(train_loader)`).
* `epochs (int)`: total number of epochs.
* `warmup_epochs (int)`: number of epochs to linearly ramp from 0 → base LR.
* `min_lr_scale (float)`: final LR is `base_lr * min_lr_scale`.

**Equivalent step‑wise form**
Let `S = steps_per_epoch * epochs`, `W = steps_per_epoch * warmup_epochs`, and `s ∈ {0,1,...}` be the global step. The schedule multiplier `λ(s)` is

```math
\lambda(s)=\begin{cases}
\max(10^{-8}, \tfrac{s+1}{\max(1,W)}) & s < W,\\
\alpha + \tfrac{1-\alpha}{2}\bigl(1+\cos(\pi\,t)\bigr) & s \ge W,\quad t=\tfrac{s-W}{\max(1,S-W)},\ \alpha=\text{min\_lr\_scale}.
\end{cases}
```

Actual LR is `lr(s) = base_lr * λ(s)`.

**Example**

```python
from chemomae.training.optim import build_optimizer, build_scheduler
optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.05)
scheduler = build_scheduler(optimizer,
                            steps_per_epoch=len(train_loader),
                            epochs=100, warmup_epochs=5, min_lr_scale=0.1)

for epoch in range(100):
    for step, batch in enumerate(train_loader):
        loss = train_step(batch)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

---

### `build_warmup_cosine(optimizer, *, warmup_steps, total_steps, min_lr_scale=0.0)`

* Lower‑level builder if you already track **global step** counts explicitly.
* Use when you have atypical loops (e.g., gradient accumulation with custom step accounting).

**Example (custom loop)**

```python
from chemomae.training.optim import build_warmup_cosine
opt = build_optimizer(model, lr=2e-4)
sched = build_warmup_cosine(opt, warmup_steps=1_000, total_steps=50_000, min_lr_scale=0.2)

for s in range(50_000):
    loss = train_step()
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
```

---

## Practical Tips

* **Batch‑size scaling:** If you change batch size substantially, linearly scale `lr` (and optionally `warmup_epochs`).
* **Min LR:** `min_lr_scale=0.1` is a robust default; reduce it for longer training or if you observe late‑epoch overfitting.
* **Freeze stages:** When freezing parts of the model, call `build_optimizer` *after* setting `requires_grad=False` so frozen params are excluded.
* **EMA & AMP:** Exponential moving average and mixed precision are orthogonal; these builders are compatible with both.

---

## Minimal Tests

```python
# Optimizer groups exist
opt = build_optimizer(model)
assert any(g['weight_decay'] > 0 for g in opt.param_groups)
assert any(g['weight_decay'] == 0 for g in opt.param_groups)

# Scheduler decreases after warmup
sched = build_scheduler(opt, steps_per_epoch=100, epochs=2, warmup_epochs=1)
base_lr = opt.param_groups[0]['lr']
mult0 = sched.get_last_lr()[0]
for _ in range(150):
    opt.step(); sched.step()
mult1 = sched.get_last_lr()[0]
assert mult1 <= base_lr
```

---

## Version

* Introduced in `chemomae.training.optim` — initial public draft.
