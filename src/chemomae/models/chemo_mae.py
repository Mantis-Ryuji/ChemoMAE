from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ChemoMAE",
    "ChemoEncoder",
    "ChemoDecoder",
    "make_patch_mask",
]


def make_patch_mask(
    batch_size: int,
    seq_len: int,
    n_patches: int,
    n_mask: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""
    Make a random **patch-aligned boolean mask** for 1D MAE (per-sample diverse).

    概要
    ----
    1D スペクトル系列（長さ `seq_len=L`）を `n_patches=P` 個の等長パッチに分割し、
    **パッチ単位**で `n_mask` 個をランダムに「隠す」マスクを生成する。

    本実装では **サンプルごとに独立なマスクパターン**を生成することで、
    MAE 学習におけるマスク多様性（diversity）を確保する。

    マスク仕様
    ---------
    - `mask[b, t] = True`  : masked（隠す / 入力に見せない領域）
    - `mask[b, t] = False` : visible（可視 / 入力に見せる領域）
    - 形状は `(B, L)`、dtype は `torch.bool`
    - パッチ整合性（patch-aligned）は常に保証される

    制約
    ----
    - `seq_len % n_patches == 0`（等長パッチ）
    - `0 <= n_mask <= n_patches`

    Parameters
    ----------
    batch_size : int
        バッチサイズ `B`。
    seq_len : int
        系列長 `L`。
    n_patches : int
        パッチ数 `P`（`patch_size = L // P`）。
    n_mask : int
        マスクするパッチ数（`0..P`）。
    device : torch.device, optional
        出力テンソルのデバイス。None の場合は CPU。

    Returns
    -------
    mask : torch.Tensor, shape (B, L), dtype=bool
        パッチ整合マスク。True=masked, False=visible。

    Notes
    -----
    - 各サンプル `b` について独立に `n_mask` 個のパッチを選択する。
    - マスクは **呼び出しごとにランダム**に生成され、
      学習中のマスク多様性を最大化する設計となっている。
    - 特徴抽出時は全可視（mask 無し）で行うことを想定しており、
      学習時のマスク再現性を厳密に管理する必要はない。
    - MAE における情報リーク防止のため、パッチ内で
      True / False が混在することはない。
    """
    if seq_len % n_patches != 0:
        raise ValueError("seq_len must be divisible by n_patches")
    if not (0 <= n_mask <= n_patches):
        raise ValueError("n_mask must be in [0, n_patches]")

    if device is None:
        device = torch.device("cpu")

    patch_size = seq_len // n_patches

    # -------------------------------------------------
    # パッチ単位マスク (B, P) : per-sample
    # -------------------------------------------------
    patch_mask = torch.zeros(batch_size, n_patches, device=device, dtype=torch.bool)

    if n_mask > 0:
        # (B, P) の乱数 → 各行で n_mask 個選択
        r = torch.rand(batch_size, n_patches, device=device)
        idx = torch.argsort(r, dim=1)[:, :n_mask]  # (B, n_mask)
        patch_mask.scatter_(1, idx, True)

    # (B, P) → (B, P, S) → (B, L)
    return (
        patch_mask
        .unsqueeze(-1)
        .expand(-1, -1, patch_size)
        .reshape(batch_size, seq_len)
    )


class ChemoEncoder(nn.Module):
    r"""
    Transformer encoder for 1D spectra with **patch-aligned visible masking**.

    概要
    ----
    入力スペクトル `x ∈ R^{B×L}` を `P` 個のパッチ（長さ `S=L/P`）へ分割し、
    **可視パッチのみ**を Transformer Encoder に投入して潜在表現
    `z ∈ R^{B×latent_dim}` を得る。

    本 Encoder は MAE の encoder 側に相当し、
    「隠した領域が入力へ漏れない」ために `visible_mask` に強い制約を課す。

    入出力（テンソル形状）
    --------------------
    - Input:
      - `x` : `(B, L)` 連続値スペクトル
      - `visible_mask` : `(B, L)` bool, **True=可視 / False=隠す**
    - Output:
      - `z` : `(B, latent_dim)` **L2 正規化済み**潜在

    マスク制約（重要）
    ----------------
    `visible_mask` は **パッチ整合 (patch-aligned) のみ許容**：
    各パッチ内で True/False が混在するマスクは MAE の前提（隠した情報を見せない）を破壊し得るため、
    forward 内で
    `vm.all(dim=2)` と `vm.any(dim=2)` の一致性チェックにより検出し、例外を投げる。

    アーキテクチャ
    --------------
    1. Patchify: `x -> x_patches` shape `(B,P,S)`
    2. Patch projection: `Linear(S -> d_model)`（bias=False）
    3. `CLS` token + learned positional embedding（長さ `1+P`）
    4. Visible compaction:
       - 可視パッチを前に詰め、最大可視数 `max_vis` に揃えて PAD
       - `src_key_padding_mask` で PAD を無効化
    5. TransformerEncoder（`norm_first=True`, activation="gelu"）
    6. `to_latent: Linear(d_model -> latent_dim)` → `F.normalize`

    Parameters
    ----------
    seq_len : int, default=256
        系列長 `L`。
    n_patches : int, default=16
        パッチ数 `P`。`L % P == 0` が必要。
    d_model : int, default=256
        Transformer の埋め込み次元。
    nhead : int, default=4
        Multi-head attention のヘッド数。
    num_layers : int, default=2
        Transformer Encoder 層数。
    dim_feedforward : int | None, default=None
        FFN 隠れ次元。None の場合 `4*d_model`。
    dropout : float, default=0.0
        Transformer 内部 dropout。
    latent_dim : int, default=16
        出力潜在次元（最後に L2 正規化）。

    Attributes
    ----------
    seq_len : int
        `L`。
    n_patches : int
        `P`。
    patch_size : int
        `S = L // P`。
    patch_proj : nn.Linear
        `(S -> d_model)` のパッチ埋め込み。
    cls_token : nn.Parameter, shape (1,1,d_model)
        CLS トークン。
    pos_embed : nn.Parameter, shape (1,1+P,d_model)
        学習可能な位置埋め込み。
    encoder : nn.TransformerEncoder
        Encoder 本体。
    to_latent : nn.Linear
        `(d_model -> latent_dim)`。

    Notes
    -----
    - `max_vis == 0`（全パッチ不可視）は通常起こらないが、数値崩壊回避のため
      「先頭パッチのみ可視」に置き換える安全ガードが入っている。
    - 出力 `z` は球面上（L2 正規化）であり、後段の CosineKMeans / vMF mixture と整合する設計。
    - `visible_mask` の True/False の意味は **ChemoMAE 側と一致（True=可視）**させること。
      逆の意味で渡すと学習が破綻する（本実装は dtype/shape のみをチェックする）。
    """

    def __init__(
        self,
        *,
        seq_len: int = 256,
        n_patches: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_patches = int(n_patches)
        if self.seq_len % self.n_patches != 0:
            raise ValueError("seq_len must be divisible by n_patches")
        self.patch_size = self.seq_len // self.n_patches

        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)

        if dim_feedforward is None:
            dim_feedforward = 4 * self.d_model

        # パッチ埋め込み
        self.patch_proj = nn.Linear(self.patch_size, self.d_model, bias=False)

        # CLS + 位置埋め込み（learned）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.to_latent = nn.Linear(self.d_model, self.latent_dim)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B,L), got shape={tuple(x.shape)}")
        B, L = x.shape
        if L != self.seq_len:
            raise ValueError(f"seq_len mismatch: expected {self.seq_len}, got {L}")
        if visible_mask.shape != (B, L):
            raise ValueError(f"visible_mask shape mismatch: expected {(B, L)}, got {tuple(visible_mask.shape)}")

        visible_mask = visible_mask.to(device=x.device, dtype=torch.bool)

        # (B, L) -> (B, P, S)
        x_patches = x.view(B, self.n_patches, self.patch_size)
        vm = visible_mask.view(B, self.n_patches, self.patch_size)

        # パッチ整合性チェック: any==all であること
        patch_all = vm.all(dim=2)  # (B,P)
        patch_any = vm.any(dim=2)  # (B,P)
        if not torch.equal(patch_all, patch_any):
            raise ValueError("visible_mask must be patch-aligned: each patch must be all True or all False.")
        patch_visible = patch_all  # (B,P)

        # パッチ埋め込み
        tok = self.patch_proj(x_patches)  # (B,P,d)

        # 可視パッチを前に詰める order
        order = torch.argsort(patch_visible.int(), dim=1, descending=True)  # (B,P)
        vis_counts = patch_visible.sum(dim=1)  # (B,)
        max_vis = int(vis_counts.max().item())

        if max_vis == 0:
            # 全パッチ不可視は通常起きないが、数値崩壊回避の安全ガード:
            # 先頭パッチのみ可視にして処理継続
            patch_visible = torch.zeros((B, self.n_patches), device=x.device, dtype=torch.bool)
            patch_visible[:, 0] = True
            order = torch.arange(self.n_patches, device=x.device).unsqueeze(0).expand(B, -1)
            vis_counts = torch.ones((B,), device=x.device, dtype=torch.long)
            max_vis = 1

        idx = order[:, :max_vis]  # (B,max_vis)

        # 有効長（短いサンプルは後半を PAD）
        pos_idx = torch.arange(max_vis, device=x.device).unsqueeze(0).expand(B, -1)
        valid = pos_idx < vis_counts.unsqueeze(1)  # (B,max_vis)

        gathered_tok = tok.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.d_model))  # (B,max_vis,d)

        # CLS + pos
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,d)
        enc_in = torch.cat([cls, gathered_tok], dim=1)  # (B,1+max_vis,d)

        pos_cls = self.pos_embed[:, :1, :].expand(B, -1, -1)
        pos_patch = self.pos_embed[:, 1:, :].expand(B, -1, -1).gather(
            1, idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        pos = torch.cat([pos_cls, pos_patch], dim=1)
        enc_in = enc_in + pos

        # key padding: True が無効（PAD）
        key_pad = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), ~valid], dim=1)

        h = self.encoder(enc_in, src_key_padding_mask=key_pad)  # (B,1+max_vis,d)
        cls_out = h[:, 0, :]
        z = self.to_latent(cls_out)  # (B,latent_dim)
        return F.normalize(z, dim=1)


class ChemoDecoder(nn.Module):
    r"""
    Lightweight MLP decoder for 1D spectral reconstruction (configurable depth).

    概要
    ----
    潜在表現 `z ∈ R^{B×latent_dim}` から元系列 `x̂ ∈ R^{B×L}` を復元する
    **低パラメータ復元ヘッド**。

    本デコーダは ViT-MAE のように「マスクトークン + Transformer デコーダ」で
    パッチ系列を復元する方式ではなく、`z` から **全系列へ直接写像**する。

    層数
    ----
    `num_layers` により深さを切り替える。

    - `num_layers = 1` : **Linear projection**（`Linear(latent_dim, L)`）
    - `num_layers >= 2`: MLP（隠れ次元 `hidden_dim`、GELU）

    既定値（`num_layers=2, hidden_dim=L`）は従来実装と等価：
    `z -> Linear(latent_dim, L) -> GELU -> Linear(L, L)`

    Parameters
    ----------
    seq_len : int
        出力系列長 `L`。
    latent_dim : int
        入力潜在次元。
    num_layers : int, default=2
        デコーダの層数（1 以上）。1 の場合は線形写像として振る舞う。
    hidden_dim : int | None, default=None
        MLP の隠れ次元。None の場合 `seq_len`。

    Returns (forward)
    -----------------
    x_recon : torch.Tensor, shape (B, L)
        再構成系列。

    Notes
    -----
    - “MAE事前学習を最大化する高容量デコーダ” ではなく、
      **潜在表現の質（クラスタリング等）を崩しにくい**軽量デコーダを志向した設計。
    - 入力 `z` は `(B, latent_dim)` を厳密に要求し、形状不一致は例外とする。
    """

    def __init__(
        self,
        *,
        seq_len: int,
        latent_dim: int,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)

        nl = int(num_layers)
        if nl < 1:
            raise ValueError("num_layers must be >= 1")

        if nl == 1:
            self.net = nn.Linear(self.latent_dim, self.seq_len)
        else:
            if hidden_dim is None:
                hidden_dim = self.seq_len
            hd = int(hidden_dim)
            layers = [nn.Linear(self.latent_dim, hd), nn.GELU()]
            for _ in range(nl - 2):
                layers += [nn.Linear(hd, hd), nn.GELU()]
            layers += [nn.Linear(hd, self.seq_len)]
            self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must be (B,{self.latent_dim}), got shape={tuple(z.shape)}")
        return self.net(z)


class ChemoMAE(nn.Module):
    r"""
    Masked Autoencoder for 1D spectra with **spherical latent**.

    概要
    ----
    `ChemoEncoder` と `ChemoDecoder` を組み合わせた 1D スペクトル向け MAE。
    学習では「パッチ単位のマスク」を用いて一部のスペクトル情報を隠し、
    そこから潜在表現 `z` を学習しつつ系列全体を再構成する。

    本クラスは **潜在 `z` の L2 正規化**を前提にしており、
    そのまま CosineKMeans / vMF mixture 等の球面クラスタリングへ接続できる。

    返り値の契約（重要）
    ------------------
    `forward()` は常に **(x_recon, z, visible_mask)** を返す。

    - `x_recon` : `(B, L)` 再構成
    - `z` : `(B, latent_dim)` L2 正規化潜在
    - `visible_mask` : `(B, L)` bool, **True=可視 / False=隠す**（パッチ整合）

    マスク生成
    ---------
    `visible_mask=None` の場合、内部で `make_visible()` により
    **パッチ単位の可視マスク**を生成する。

    Parameters
    ----------
    seq_len : int, default=256
        系列長 `L`。
    n_patches : int, default=16
        パッチ数 `P`。`L % P == 0` が必要。
    d_model : int, default=256
        Encoder の埋め込み次元。
    nhead : int, default=4
        Encoder attention のヘッド数。
    num_layers : int, default=2
        Encoder 層数。
    dim_feedforward : int | None, default=None
        Encoder FFN 隠れ次元（None は `4*d_model`）。
    dropout : float, default=0.0
        Encoder dropout。
    latent_dim : int, default=16
        潜在次元。
    decoder_num_layers : int, default=2
        Decoder 層数。1 の場合は `Linear(latent_dim -> L)`。
    n_mask : int, default=4
        デフォルトで隠すパッチ数（`make_visible()`/`forward()` の `n_mask` 未指定時に使用）。

    Attributes
    ----------
    encoder : ChemoEncoder
        可視パッチのみで潜在 `z` を出力する Encoder。
    decoder : ChemoDecoder
        `z -> x_recon` の復元ヘッド。
    seq_len, n_patches, n_mask : int
        入力仕様と既定マスク設定。

    Notes
    -----
    - `visible_mask` の意味は **True=可視**で統一。
      `make_patch_mask()` は True=masked を返すため、`make_visible()` はその反転を返す。
    - `reconstruct()` は `forward()` の「x_recon だけ欲しい」用途の薄いラッパ。
    - shape/dtype の整合性は `_check_shapes()` が保証し、異常入力は例外とする。
    """
    
    def __init__(
        self,
        *,
        seq_len: int = 256,
        n_patches: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        latent_dim: int = 16,
        decoder_num_layers: int = 2,
        n_mask: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_patches = int(n_patches)
        self.n_mask = int(n_mask)

        self.encoder = ChemoEncoder(
            seq_len=self.seq_len,
            n_patches=self.n_patches,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            latent_dim=latent_dim,
        )
        self.decoder = ChemoDecoder(seq_len=self.seq_len, latent_dim=latent_dim, num_layers=decoder_num_layers)

    def make_visible(
        self,
        batch_size: int,
        *,
        n_mask: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""
        Generate a **patch-aligned visible mask** for MAE.

        概要
        ----
        `n_patches=P` のうち `n_mask` 個を masked として選び、可視マスク
        `visible_mask = ~masked_mask` を返す。

        マスク仕様
        ---------
        - `visible_mask[b, t] = True`  : visible（Encoder へ入力されるパッチ）
        - `visible_mask[b, t] = False` : masked（隠すパッチ）

        Parameters
        ----------
        batch_size : int
            バッチサイズ `B`。
        n_mask : int | None, default=None
            隠すパッチ数。None の場合は `self.n_mask`。
        device : torch.device | None, default=None
            出力デバイス。None の場合は CPU。

        Returns
        -------
        visible_mask : torch.Tensor, shape (B, L), dtype=bool
            True=visible / False=masked（パッチ整合）。

        Notes
        -----
        - 内部では `make_patch_mask(...)->masked` を作り、`~masked` を返す。
        """
        if n_mask is None:
            n_mask = self.n_mask
        if device is None:
            device = torch.device("cpu")
        masked = make_patch_mask(
            batch_size=batch_size,
            seq_len=self.seq_len,
            n_patches=self.n_patches,
            n_mask=int(n_mask),
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
        r"""
        Reconstruct the input spectrum from (optionally generated) visible mask.

        概要
        ----
        `visible_mask` を与えて `z = encoder(x, visible_mask)` を計算し、
        `decoder(z)` により `x_recon` を返す。

        `visible_mask=None` の場合は `make_visible()` でマスクを生成する
        （再現性やマスク制御が必要な場合は明示的に `visible_mask` を渡す）。

        Parameters
        ----------
        x : torch.Tensor, shape (B, L)
            入力スペクトル。
        visible_mask : torch.Tensor | None, shape (B, L), dtype=bool, default=None
            True=visible / False=masked の可視マスク。
        n_mask : int | None, default=None
            `visible_mask=None` のときにのみ使用する「隠すパッチ数」。
            None の場合は `self.n_mask`。

        Returns
        -------
        x_recon : torch.Tensor, shape (B, L)
            再構成系列。

        Raises
        ------
        ValueError
            `x` / `visible_mask` の shape/dtype が不正な場合。

        Notes
        -----
        - `reconstruct()` は **x_recon のみ**返す（潜在 `z` が不要な推論用途向け）。
        潜在も欲しい場合は `forward()` を使用する。
        """
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
        r"""
        Forward pass returning reconstruction, spherical latent, and mask.

        概要
        ----
        MAE の標準計算を実行し、学習・評価で必要になる 3 つ組
        `(x_recon, z, visible_mask)` を返す。

        Parameters
        ----------
        x : torch.Tensor, shape (B, L)
            入力スペクトル。
        visible_mask : torch.Tensor | None, shape (B, L), dtype=bool, default=None
            True=visible / False=masked の可視マスク。
            None の場合は `make_visible()` により生成する。
        n_mask : int | None, default=None
            `visible_mask=None` のときに使用する「隠すパッチ数」。

        Returns
        -------
        x_recon : torch.Tensor, shape (B, L)
            再構成系列。
        z : torch.Tensor, shape (B, latent_dim)
            L2 正規化済み潜在（球面潜在）。
        visible_mask : torch.Tensor, shape (B, L), dtype=bool
            True=visible / False=masked（パッチ整合）。

        Raises
        ------
        ValueError
            入力 shape/dtype が不正な場合。
        """
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
        if visible_mask.shape != x.shape:
            raise ValueError(f"visible_mask must have same shape as x, got {tuple(visible_mask.shape)}")
        if visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool dtype")