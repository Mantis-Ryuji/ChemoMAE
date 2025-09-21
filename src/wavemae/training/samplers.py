from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt  # optional
import torch
from torch.utils.data import Dataset, WeightedRandomSampler



# -----------------------------------------------------------------------------
# Basic Dataset 
# -----------------------------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, data: np.ndarray | torch.Tensor):
        self.data = torch.as_tensor(data, dtype=torch.float32)
    def __len__(self) -> int:
        return int(self.data.shape[0])
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# -----------------------------------------------------------------------------
# Reference vector & cosine utilities
# -----------------------------------------------------------------------------

def compute_reference_vector(X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return L2-normalized mean spectrum as reference vector (np.float32)."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, np.float32)
    u = X.mean(axis=0)
    u /= (np.linalg.norm(u) + 1e-12)
    return u.astype(np.float32)


def cosine_to_reference(X: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray) -> np.ndarray:
    """Compute cosine similarity to reference vector for each row in X; returns (N,) np.float32."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    X = np.asarray(X, np.float32)
    ref = np.asarray(ref, np.float32)
    Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    sims = np.clip(Xu @ ref, -1.0, 1.0).astype(np.float32)
    return sims


# -----------------------------------------------------------------------------
# Weighted samplers
# -----------------------------------------------------------------------------

def make_weighted_sampler_by_cosine(
    train_ds: Dataset,
    ref_vec: np.ndarray | torch.Tensor,
    *,
    cos_mid: float = 0.50,       # どの類似度でS字の折れ目にするか
    cos_beta: float = 8.0,       # 勾配の鋭さ（大きいほど急）
    clip: tuple[float, float] = (0.3, 3.0),  # sim≈1も拾う下限 / 低cosの上限を抑える
    replacement: bool = True,
    plot_path: str | Path | None = None,
) -> WeightedRandomSampler:
    """w = sigmoid(beta * (cos_mid - cos)) で連続重み付け（単調シグモイド）。"""
    # 1) collect
    if hasattr(train_ds, "data") and isinstance(train_ds.data, torch.Tensor):
        X = train_ds.data
    else:
        X = torch.stack([train_ds[i] for i in range(len(train_ds))], dim=0)

    sims = cosine_to_reference(X, ref_vec).astype(np.float64)  # [-1, 1]

    # 2) weights: pure sigmoid (farther -> heavier)
    beta = float(cos_beta); mid = float(cos_mid)
    w = 1.0 / (1.0 + np.exp(-beta * (mid - sims)))   # ← 単調シグモイド
    w = w.astype(np.float32)

    # 3) normalize & clip
    w /= (w.mean() + 1e-12)
    w = np.clip(w, clip[0], clip[1])

    # 4) plot
    if plot_path and plt is not None:
        _plot_cosine_weights_sigmoid_cos(sims, cos_mid=mid, cos_beta=beta, path=plot_path)

    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=sims.shape[0], replacement=replacement)


def _plot_cosine_weights_sigmoid_cos(sims, *, cos_mid: float, cos_beta: float, path: str | Path):
    """cos ヒスト（左軸, log）+ 単一の重み曲線（右軸, cosベースの単調シグモイド）"""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax1.hist(sims, bins=max(40, int(np.sqrt(sims.size))), alpha=0.6, edgecolor="none",
             color="#4472C4", label="count")
    ax1.set_yscale("log"); ax1.set_xlabel("cosine similarity")
    ax1.set_ylabel("count (log)", color="#4472C4"); ax1.tick_params(axis="y", labelcolor="#4472C4")
    ax1.set_title("cos(x, ref) distribution with sigmoid weights")

    ax2 = ax1.twinx()
    cos_grid = np.linspace(-1.0, 1.0, 512)
    sig = 1.0 / (1.0 + np.exp(-float(cos_beta) * (float(cos_mid) - cos_grid)))
    sig = sig / (sig.mean() + 1e-12)
    ln, = ax2.plot(cos_grid, sig, color="#ED7D31", linewidth=2.0,
                   label=f"sigmoid (mid={cos_mid:.2f}, beta={cos_beta:.1f})")
    ax2.set_ylabel("weight", color="#ED7D31"); ax2.tick_params(axis="y", labelcolor="#ED7D31")

    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = [ln], [ln.get_label()]
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.savefig(path.as_posix(), dpi=150); 
    plt.close(fig)