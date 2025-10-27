from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ChemoMAE",
    "ChemoEncoder",
    "ChemoDecoderLP",
    "make_block_mask",
    "sinusoidal_positional_encoding",
]


def sinusoidal_positional_encoding(L: int, d_model: int, device: torch.device) -> torch.Tensor:
    r"""
    Sinusoidal positional encoding for 1D sequences.

    概要
    ----
    - Vaswani et al. (2017) の Transformer で導入された固定位置埋め込みを実装。
    - 各次元に異なる周波数の sin/cos を割り当て、系列長 L にわたって周期的な位置信号を付与する。
    - 返り値は (1, L, d_model) で、バッチにブロードキャストして利用する。

    Parameters
    ----------
    L : int
        系列長（位置数）。
    d_model : int
        埋め込み次元。偶数・奇数どちらでもよいが、偶数次元で sin/cos が対になる。
    device : torch.device
        出力テンソルを置くデバイス。

    Returns
    -------
    pe : torch.Tensor, shape (1, L, d_model)
        サイン波位置埋め込み。第1次元=1なので (B, L, d_model) へ自動的にブロードキャスト可能。

    Notes
    -----
    - 学習可能ではなく固定値。軽量で汎用的だが、学習可能埋め込みに比べて柔軟性は低い。

    例
    --
    >>> pe = sinusoidal_positional_encoding(L=8, d_model=4, device=torch.device("cpu"))
    >>> pe.shape
    torch.Size([1, 8, 4])
    >>> pe[0, 0]   # 位置=0 の埋め込み
    tensor([0., 1., 0., 1.])
    """
    position = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(L, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, L, d_model)


def make_block_mask(
    batch_size: int,
    seq_len: int,
    n_blocks: int,
    n_mask: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""
    Generate a random block-wise boolean mask.

    概要
    ----
    - 入力系列を `n_blocks` 個の連続ブロックに等分し、そのうち `n_mask` 個をランダムに選んで隠す。
    - 出力は (B, L) の bool テンソルで、**True = masked (隠す部分)**、False = visible。

    Parameters
    ----------
    batch_size : int
        バッチサイズ B。
    seq_len : int
        系列長 L。必ず `n_blocks` で割り切れる必要がある。
    n_blocks : int
        系列を分割するブロック数。
    n_mask : int
        隠すブロック数。`0 <= n_mask <= n_blocks` を満たす必要がある。
    device : torch.device, optional
        出力テンソルを配置するデバイス (default="cpu")。

    Returns
    -------
    mask : torch.Tensor, shape (B, L), dtype=bool
        - True = 隠す領域（masked）
        - False = 可視領域（visible）

    Notes
    -----
    - ブロック長 = `seq_len // n_blocks`。
    - サンプルごとに `n_mask` 個のブロックがランダムに選ばれる。
    - `ChemoMAE.make_visible` では `~mask` を取って **visible=True=使う** の可視マスクに変換する。

    例
    --
    >>> mask = make_block_mask(batch_size=2, seq_len=16, n_blocks=4, n_mask=2)
    >>> mask.shape
    torch.Size([2, 16])
    >>> mask.sum(dim=1)
    tensor([8, 8])  # 各サンプルで 2 ブロック (16/4*2=8) がマスクされている
    """
    assert 0 <= n_mask <= n_blocks, "n_mask must be within [0, n_blocks]"
    assert seq_len % n_blocks == 0, "seq_len must be divisible by n_blocks"
    device = device or torch.device("cpu")

    B, L = int(batch_size), int(seq_len)
    block_size = L // n_blocks
    block_ids = torch.arange(n_blocks, device=device).expand(B, -1)               # (B, n_blocks)
    perm = torch.rand(B, n_blocks, device=device).argsort(dim=1)[:, :n_mask]      # (B, n_mask)
    chosen = torch.gather(block_ids, 1, perm)                                     # (B, n_mask)
    within = torch.arange(block_size, device=device).view(1, 1, -1)               # (1,1,bs)
    ids_mask = (chosen.unsqueeze(-1) * block_size + within).reshape(B, -1)        # (B, n_mask*bs)

    mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    mask.scatter_(1, ids_mask, True)
    return mask


class ChemoEncoder(nn.Module):
    r"""
    ChemoEncoder: Transformer encoder for 1D spectra.

    概要
    ----
    - 入力系列 x (B, L) をトークン化し、**可視マスク visible_mask (B, L)** に基づいて
      「可視トークン + CLS トークン」のみを Transformer Encoder に通す。
    - 出力は CLS トークンの埋め込みを latent_dim 次元に線形変換し、L2 正規化した潜在表現 z。

    設計思想
    --------
    - **visible_mask=True の部分だけを使う** → マスク部は完全に除外される。
    - 系列長 L は固定（seq_len）、n_blocks によるマスク分割は外部で制御。
    - pos_embed は学習可能 or サイン波固定を選択可能。

    Parameters
    ----------
    seq_len : int
        入力系列長 L（スペクトルの波長チャンネル数など）。
    latent_dim : int, default=64
        出力潜在表現の次元 D。
    d_model : int, default=256
        Transformer モデル幅。
    nhead : int, default=4
        Multi-Head Attention のヘッド数。
    num_layers : int, default=4
        Transformer Encoder 層の数。
    dim_feedforward : int, default=1024
        各層の FFN の隠れ次元。
    dropout : float, default=0.1
        Dropout 率。
    use_learnable_pos : bool, default=True
        True=学習可能位置埋め込み, False=サイン波埋め込み。

    Shapes
    ------
    Input
        x : torch.Tensor, shape (B, L)
            入力スペクトル系列。
        visible_mask : torch.Tensor, shape (B, L), dtype=bool
            True=可視トークン。False=マスクされて除外。
    Output
        z : torch.Tensor, shape (B, latent_dim)
            L2 正規化された潜在表現。

    Notes
    -----
    - CLS トークンを常に先頭に追加し、その出力を潜在ベクトル z として返す。
    - 可視トークン数がサンプルごとに異なる場合も、
      Transformer には padding mask (src_key_padding_mask) を与えて整合性を確保。
    - 出力 z は L2 正規化済みなので、コサイン類似度に直ちに利用可能。
    - `ChemoMAE` の `forward` からは visible_mask を自動生成するため、
      通常の利用者は `encoder` を直接呼び出す場合のみ visible_mask を指定すればよい。

    例
    --
    >>> enc = ChemoEncoder(seq_len=128, latent_dim=32, d_model=64, nhead=4, num_layers=2)
    >>> x = torch.randn(8, 128)
    >>> visible = torch.ones(8, 128, dtype=torch.bool)   # 全可視
    >>> z = enc(x, visible)
    >>> z.shape
    torch.Size([8, 32])
    >>> z.norm(dim=1).mean().item()   # ≈ 1.0 （L2 正規化済み）
    """

    def __init__(
        self,
        *,
        seq_len: int,
        latent_dim: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_learnable_pos: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

        self.token_proj = nn.Linear(1, d_model, bias=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if use_learnable_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # buffer（学習しない）として登録。実デバイスへは forward 時に移す。
            self.register_buffer(
                "pos_embed",
                sinusoidal_positional_encoding(self.seq_len, d_model, device=torch.device("cpu")),
                persistent=False,
            )

        self.to_latent = nn.Linear(d_model, latent_dim, bias=False)

    def forward(self, x: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "x must be (B, L)"
        B, L = x.shape
        assert L == self.seq_len, "seq_len mismatch"
        assert visible_mask.shape == (B, L), "visible_mask shape mismatch"

        tok = self.token_proj(x.unsqueeze(-1))  # (B, L, d_model)

        # 可視(True)を前方へ寄せるインデックスを一括生成
        order = torch.argsort(visible_mask.int(), dim=1, descending=True)  # (B, L)
        vis_counts = visible_mask.sum(dim=1)                                # (B,)
        max_vis = int(vis_counts.max().item())
        idx = order[:, :max_vis]  # (B, max_vis)

        # 有効長（短いサンプルの後半はパディング）
        pos_idx = torch.arange(max_vis, device=x.device).unsqueeze(0).expand(B, -1)
        valid = pos_idx < vis_counts.unsqueeze(1)  # (B, max_vis) bool

        gathered_tok = tok.gather(1, idx.unsqueeze(-1).expand(-1, -1, tok.size(-1)))  # (B, max_vis, d)

        # pos_embed を (B,L,d) に expand → gather（Parameter/Tensor どちらでもOK）
        pe_full = self.pos_embed.to(x.device)
        if pe_full.dim() == 2:
            pe_full = pe_full.unsqueeze(0)
        if pe_full.size(0) != B:
            pe_full = pe_full.expand(B, -1, -1)
        gathered_pe = pe_full.gather(1, idx.unsqueeze(-1).expand(-1, -1, pe_full.size(-1)))  # (B, max_vis, d)

        enc_in = torch.cat([self.cls_token.expand(B, 1, -1), gathered_tok + gathered_pe], dim=1)  # (B, 1+V, d)

        # Transformer への padding 指定（True=無視）
        pad = ~valid
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)

        h = self.encoder(enc_in, src_key_padding_mask=key_pad)
        cls_out = h[:, 0, :]
        z = F.normalize(self.to_latent(cls_out), p=2, dim=1)
        return z


class ChemoDecoderLP(nn.Module):
    r"""
    ChemoDecoderLP: 潜在表現 z から 1D 系列 x を再構成する **線形デコーダ**。

    概要
    ----
    - 入力: 潜在表現 z (B, D) ※ D = latent_dim
    - 出力: 再構成系列 x_recon (B, L) ※ L = seq_len
    - 構成: Linear（単層線形写像）

    特徴
    ----
    - デコーダは非線形変換を行わず、元の前処理空間（例: SNV空間）へ射影する。
    
    Parameters
    ----------
    seq_len : int
        出力系列長 L（波長チャンネル数など）。
    latent_dim : int, default=64
        潜在表現の次元 D。エンコーダ出力と一致させること。

    Shapes
    ------
    z : torch.Tensor, shape (B, latent_dim)
        潜在表現。
    return : torch.Tensor, shape (B, seq_len)
        再構成系列 x_recon。

    例
    --
    >>> dec = ChemoDecoderLP(seq_len=256, latent_dim=64)
    >>> z = torch.randn(8, 64)
    >>> x_rec = dec(z)   # 出力形状: (8, 256)
    """

    def __init__(self, *, seq_len: int, latent_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Linear(latent_dim, seq_len, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """潜在表現 z から再構成系列 x_recon を返す。"""
        return self.net(z)


class ChemoMAE(nn.Module):
    r"""
    ChemoMAE: Masked Autoencoder for 1D spectra.

    1D スペクトル（例: 近赤外スペクトル, HSI バンド列）に特化した MAE 実装です。

    Overview
    --------
    - **forward(x, visible_mask=None, *, n_mask=None)**  
      再構成と潜在表現を返す。visible_mask が None の場合、n_mask に基づき内部で可視マスクを生成。
    - **reconstruct(x, visible_mask=None, *, n_mask=None)**  
      可視マスクを与えて再構成を返す。visible_mask=None の場合は自動生成。
    - **make_visible(batch_size, *, n_mask=None)**  
      ブロックマスクから可視マスク (True=使う) を生成。

    Shapes
    ------
    - x: (B, L)  
      B=batch size, L=seq_len  
    - visible_mask: (B, L) bool  
      True=可視（使う）、False=マスク（隠す）  
    - z: (B, D)  
      D=latent_dim  
    - x_recon: (B, L)  

    Typical Usage
    -------------
    学習時:
    >>> mae = ChemoMAE(seq_len=256, latent_dim=64, n_blocks=16, n_mask=4)
    >>> x = torch.randn(8, 256)
    >>> x_rec, z, visible_mask = mae(x)   # visible_mask=None なので内部で生成
    >>> loss = ((x_rec - x)**2)[~visible_mask].sum() / x.size(0)
    >>> loss.backward()

    特徴抽出（全可視）:
    >>> visible_mask = torch.ones(8, 256, dtype=torch.bool)
    >>> z_all = mae.encoder(x, visible_mask)

    下流タスク:
    - CosineKMeans, vMF Mixture などコサイン距離ベースのクラスタリング
    - UMAP/TSNE などの次元削減 (metric="cosine" 推奨)

    Parameters
    ----------
    seq_len : int
        入力系列長 L。
    d_model : int, default=256
        Transformer Encoder のモデル幅。
    nhead : int, default=4
        Multi-Head Attention のヘッド数 (d_model % nhead == 0 が必要)。
    num_layers : int, default=4
        Encoder 層数。
    dim_feedforward : int, default=1024
        各 Encoder 層の MLP の中間次元。
    dropout : float, default=0.1
        Encoder 内のドロップアウト率。
    use_learnable_pos : bool, default=True
        True=学習可能な位置埋め込み, False=サイン波埋め込み。
    latent_dim : int, default=64
        潜在表現の次元 D。
    dec_hidden : int, default=256
        Decoder(MLP) の隠れ次元。
    dec_dropout : float, default=0.1
        Decoder 内のドロップアウト率。
    n_blocks : int, default=32
        系列を等分するブロック数。
    n_mask : int, default=16
        デフォルトでマスクするブロック数。

    Notes
    -----
    - 損失は内部で計算しない設計。外部で自由に選択 (SSE, MSE, Huber など)。
    - SNV 等の前処理でデータを正規化すると、潜在表現のコサイン幾何が意味を持ちやすい。
    - AMP (bf16/fp16) は外部 Trainer 側で `torch.autocast` を使って問題なく利用可能。
    - 再現性を担保する場合は torch.manual_seed(...) を利用してください。
    """

    def __init__(
        self,
        *,
        seq_len: int,
        # encoder
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_learnable_pos: bool = True,
        latent_dim: int = 64,
        # masking
        n_blocks: int = 32,
        n_mask: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_blocks = int(n_blocks)
        self.n_mask = int(n_mask)

        self.encoder = ChemoEncoder(
            seq_len=self.seq_len,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_learnable_pos=use_learnable_pos,
        )
        self.decoder = ChemoDecoderLP(
            seq_len=self.seq_len, latent_dim=latent_dim
        )

    def make_visible(
        self,
        batch_size: int,
        *,
        n_mask: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        可視マスク (True=使う) を生成する。

        Parameters
        ----------
        batch_size : int
            バッチサイズ。
        n_mask : int, optional
            マスクするブロック数 (デフォルトは self.n_mask)。
        device : torch.device, optional
            出力テンソルを配置するデバイス。

        Returns
        -------
        visible_mask : torch.Tensor, shape (B, L), dtype=bool
            True=使う / False=隠す
        """
        if n_mask is None:
            n_mask = self.n_mask
        masked = make_block_mask(
            batch_size=batch_size,
            seq_len=self.seq_len,
            n_blocks=self.n_blocks,
            n_mask=n_mask,
            device=device,
        )
        return ~masked

    def reconstruct(
        self,
        x: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
        *,
        n_mask: Optional[int] = None,
    ) -> torch.Tensor:
        """可視マスクから再構成を返す。visible_mask=None の場合は make_visible(...) で生成。"""
        if visible_mask is None:
            visible_mask = self.make_visible(x.size(0), n_mask=n_mask, device=x.device)
        self._check_shapes(x, visible_mask)
        z = self.encoder(x, visible_mask)
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
        *,
        n_mask: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_recon, z, visible_mask)。"""
        if visible_mask is None:
            visible_mask = self.make_visible(x.size(0), n_mask=n_mask, device=x.device)
        self._check_shapes(x, visible_mask)
        z = self.encoder(x, visible_mask)
        x_recon = self.decoder(z)
        return x_recon, z, visible_mask

    def _check_shapes(self, x: torch.Tensor, visible_mask: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B,L), got shape={tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"seq_len mismatch: expected {self.seq_len}, got {x.size(1)}")
        if visible_mask.shape != x.shape or visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool tensor with shape equal to x (B,L)")

