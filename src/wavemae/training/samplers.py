from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt  # optional
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

__all__ = ["compute_reference_vector", "make_weighted_sampler_by_cosine",]

# -----------------------------------------------------------------------------
# Reference vector & cosine utilities
# -----------------------------------------------------------------------------

def compute_reference_vector(X: np.ndarray | torch.Tensor) -> np.ndarray:
    r"""
    Compute an L2-normalized mean spectrum as a reference vector.

    概要
    ----
    - 入力スペクトル集合 X の平均ベクトルを計算し、L2 正規化して返す。
    - 代表スペクトル（基準ベクトル）として、類似度計算や重み付けサンプリングなどに利用可能。

    Parameters
    ----------
    X : np.ndarray | torch.Tensor, shape (N, L)
        スペクトルデータ。N はサンプル数、L は波長チャネル数。
        - torch.Tensor の場合、自動的に CPU に移して numpy.float32 に変換する。

    Returns
    -------
    ref : np.ndarray, shape (L,), dtype=float32
        L2 正規化された基準ベクトル。

    Notes
    -----
    - 正規化は :math:`u \leftarrow u / \|u\|_2` として行う。
    - ゼロ除算を避けるため、分母に 1e-12 を加えている。

    例
    --
    >>> X = np.random.randn(100, 256).astype("float32")
    >>> ref = compute_reference_vector(X)
    >>> ref.shape
    (256,)
    >>> np.linalg.norm(ref)
    1.0
    """
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
    cos_mid: float = 0.50,
    cos_beta: float = 8.0,
    clip: tuple[float, float] = (0.3, 3.0),
    replacement: bool = True,
    plot_path: str | Path | None = None,
) -> WeightedRandomSampler:
    r"""
    Build a WeightedRandomSampler based on cosine similarity to a reference vector.

    概要
    ----
    - 各サンプルの系列ベクトルと基準ベクトル `ref_vec` のコサイン類似度を計算。
    - 類似度に基づきシグモイド関数で連続的な重みを割り当てる。
      - 基準に近いものは軽め、遠いものは重めにサンプリングされる。
    - これにより「多数派に埋もれるサンプル」を避けつつ、少数派・外れ値寄りも拾う。

    数式
    ----
    類似度 s ∈ [-1, 1] に対して重み w(s) を

        w(s) = 1 / (1 + exp( -β * (m - s) ))

    と定義する。ここで

    - m = cos_mid （折れ目の位置）
    - β = cos_beta （傾きの鋭さ）

    Parameters
    ----------
    train_ds : torch.utils.data.Dataset
        以下のいずれにも対応：
        - **TensorDataset**（1 つめのテンソルを特徴とみなす）
        - `.data` に `torch.Tensor` を持つ Dataset
        - `__getitem__` が `x` もしくは `(x, ...)` を返す一般的 Dataset（先頭要素を特徴とみなす）
    ref_vec : np.ndarray | torch.Tensor, shape (L,)
        基準ベクトル（例: `compute_reference_vector(X)` の出力）。
    cos_mid : float, default=0.50
        シグモイドの折れ目（しきい値 m）。
    cos_beta : float, default=8.0
        シグモイドの傾き（β）。大きいほど遷移が急。
    clip : (float, float), default=(0.3, 3.0)
        平均を 1 に正規化後、重みをこの範囲にクリップ。
    replacement : bool, default=True
        WeightedRandomSampler の復元抽出フラグ。
    plot_path : str | Path | None, default=None
        指定すると重み関数のプロットを保存。

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
        コサイン類似度に応じて重み付けされたサンプラー。

    Notes
    -----
    - `train_ds` が **TensorDataset** の場合、`train_ds.tensors[0]` を特徴ベクトルとみなします。
    - それ以外は、`__getitem__` が `(x, label, ...)` を返すと仮定し **先頭要素 x** を集めます。
    - 重みは平均を 1 に正規化した後、`clip` 範囲でクリップされる。
    - `replacement=True` の場合、1エポックで `num_samples=len(train_ds)` サンプルが得られる。
    - 典型的な使い方は DataLoader に渡すこと：
    
    ```python
      >>> sampler = make_weighted_sampler_by_cosine(train_ds, ref_vec)
      >>> loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    ```
    
    適用データ分布
    --------------
    この重み付けは、サンプルのコサイン類似度 s=cos(x, ref) の実測分布 p(s) に依存する。
    次のような状況で有効性が高い：

    - ロングテール・不均衡
    多数が ref 近傍（s 高め）に密集し、少数が遠方（s 低め）に長い裾を持つ。
    → 低 s 側に重みが乗るため、少数・珍しいサンプルも学習に取り込める。

    - 単峰＋裾（unimodal with tails）
    分布のモードは 1 側に寄るが、外れ・難例が一定数存在する。
    → `cos_mid` をモード少し手前に置くと “難例の相対的増量” が素直に効く。

    - 擬似ハードネガティブの強調（自己教師/再構成系）
    ラベルが無くても “ref から遠い＝難しい/珍しい” とみなしてサンプリングできる。

    一方、以下では効果が薄い／不適：

    - 強い多峰性（複数クラスタが互いに遠い）
    単一の ref ではどれかのクラスタを過度に重くしがち。
    → クラスタごとに ref を用意して「混合サンプラー」にするか、`CosineKMeans` などで
        代表ベクトルを複数持たせるのが望ましい。

    - 類似度の狭帯域（ほぼ全サンプルが同じ s に密集）
    `w(s)` がほぼ一定になり、重み付け効果が出ない。
    → 前処理（SNVなど）や ref の見直し、`cos_beta` を上げるなど調整。

    - ref が代表性を欠く（平均が非代表、汚染や大きなドリフト）
    → `compute_reference_vector` ではなく、ロバスト平均/メディアン、
        あるいはクラス別 ref・クラスタ中心を使う。

    - コサインが意味を持たない表現空間（スケールや直交性が崩壊）
    → 正規化・前処理（SNV/標準化）や表現の再学習を検討。

    実務的チューニング
    ------------------
    - まず `s = cosine_to_reference(X, ref)` のヒストグラムを見て、モードと裾の長さを把握。
    - `cos_mid` はヒストグラムのモード少し手前（例: 0.6–0.8 近辺）から試す。
    - `cos_beta` は 5–12 程度を探索（大きいほど “難例” への重み遷移が急）。
    - `clip=(a,b)` はエクストリームな重みを抑える安全弁。a を下げると類似例も拾える、
    b を下げると遠方の過強調を抑えられる。
    - 学習が不安定なら：`replacement=False` も検討（ただし 1epoch の有効サンプルが減る）。

    用途の目安
    ----------
    - 自己教師／再構成（MAE系）で“珍しさ”を均すサンプリング。
    - クラスラベルが粗い/不均衡な場面でのバランス取りの前段階。
    - ハードサンプルマイニングの軽量近似として。

    Examples
    --------
    >>> X = np.random.randn(100, 256).astype("float32")
    >>> ds = SimpleDataset(X)
    >>> ref = compute_reference_vector(X)
    >>> sampler = make_weighted_sampler_by_cosine(ds, ref, cos_mid=0.5, cos_beta=10)
    >>> len(list(iter(sampler)))
    100
    """
    # 1) collect features X robustly (TensorDataset / .data / generic Dataset)
    if hasattr(train_ds, "tensors"):  # TensorDataset
        X = train_ds.tensors[0]
    elif hasattr(train_ds, "data") and isinstance(getattr(train_ds, "data"), torch.Tensor):
        X = train_ds.data
    else:
        first = train_ds[0]
        if isinstance(first, (tuple, list)):
            X = torch.stack([train_ds[i][0] for i in range(len(train_ds))], dim=0)
        else:
            X = torch.stack([train_ds[i] for i in range(len(train_ds))], dim=0)
    X = torch.as_tensor(X, dtype=torch.float32)

    # 2) cosine similarity to reference in [-1, 1]
    sims = cosine_to_reference(X, ref_vec).astype(np.float64)

    # 3) weights: pure sigmoid (farther -> heavier)
    beta = float(cos_beta)
    mid = float(cos_mid)
    w = 1.0 / (1.0 + np.exp(-beta * (mid - sims)))
    w = w.astype(np.float32)

    # 4) normalize & clip
    w /= (w.mean() + 1e-12)
    w = np.clip(w, clip[0], clip[1])

    # 5) optional plot
    if plot_path and plt is not None:
        _plot_cosine_weights_sigmoid_cos(sims, cos_mid=mid, cos_beta=beta, path=plot_path)

    return WeightedRandomSampler(
        weights=torch.from_numpy(w),
        num_samples=int(sims.shape[0]),
        replacement=replacement,
    )


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