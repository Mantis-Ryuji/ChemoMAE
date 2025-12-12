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
    **パッチ単位のランダムマスク**を生成する（MAE 用）。

    出力マスクの仕様
    -----------------
    - True  = **隠す（masked）**
    - False = 可視（visible）
    - 形状は (B, L)

    注意
    ----
    - seq_len は n_patches で割り切れる必要がある。
    - n_mask は [0, n_patches] の範囲にあるべき。

    Parameters
    ----------
    batch_size : int
        バッチサイズ B
    seq_len : int
        系列長 L
    n_patches : int
        パッチ数 P
    n_mask : int
        マスクするパッチ数
    device : torch.device, optional
        出力テンソル device

    Returns
    -------
    mask : torch.Tensor, shape (B, seq_len), dtype=bool
        True=mask（隠す）, False=visible
    """
    if seq_len % n_patches != 0:
        raise ValueError("seq_len must be divisible by n_patches")
    if not (0 <= n_mask <= n_patches):
        raise ValueError("n_mask must be in [0, n_patches]")

    if device is None:
        device = torch.device("cpu")

    patch_size = seq_len // n_patches

    # パッチ単位マスク (B, P)
    patch_mask = torch.zeros(batch_size, n_patches, device=device, dtype=torch.bool)
    if n_mask > 0:
        idx = torch.randperm(n_patches, device=device)[:n_mask]
        patch_mask[:, idx] = True

    # (B, P) → (B, P, S) → (B, L)
    return patch_mask.unsqueeze(-1).expand(-1, -1, patch_size).reshape(batch_size, seq_len)


class ChemoEncoder(nn.Module):
    r"""
    ChemoEncoder: **1D スペクトル用 Transformer Encoder（Patch Mask 対応）**

    概要
    ----
    1D スペクトル x (B, L) を patch 化し、可視パッチのみを Transformer Encoder に入力して
    潜在表現 z (B, latent_dim) を得る。

    重要: visible_mask の制約
    -------------------------
    visible_mask は **パッチ整合マスクのみ許容**：
    各パッチ内で True/False が混在するマスクは、MAE 目的（隠した情報を見せない）を破壊し得るため
    forward 内で検出して例外を投げる。

    入出力
    ------
    - x:            (B, L)           SNV 済みスペクトル
    - visible_mask: (B, L) bool      True=可視, False=隠す（パッチ整合のみ）
    - z:            (B, latent_dim)  L2 正規化済み潜在
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
    ChemoDecoder: **2 層 MLP デコーダ（軽量な復元ヘッド）**

    潜在 z (B, latent_dim) を元の 1D スペクトル長 seq_len に復元する。

    - Linear(latent_dim -> seq_len) -> GELU -> Linear(seq_len -> seq_len)
    - パッチ単位の ViT-MAE デコーダではなく、**全系列へ直接写像**する設計。
    """

    def __init__(self, *, seq_len: int, latent_dim: int) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, self.seq_len),
            nn.GELU(),
            nn.Linear(self.seq_len, self.seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must be (B,{self.latent_dim}), got shape={tuple(z.shape)}")
        return self.net(z)


class ChemoMAE(nn.Module):
    r"""
    ChemoMAE: **1D スペクトル用 Masked AutoEncoder + 球面潜在**

    返り値の設計
    ------------
    forward は **(x_recon, z, visible_mask)** を返す（返り値は変更しない）。

    - x_recon: (B,L) 再構成
    - z:       (B,latent_dim) L2 正規化潜在
    - visible_mask: (B,L) True=可視 / False=隠す（パッチ整合）

    visible_mask が None の場合
    ---------------------------
    forward 内で make_visible(...) を用いて **パッチ単位の可視マスク**を生成する。
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
        self.decoder = ChemoDecoder(seq_len=self.seq_len, latent_dim=latent_dim)

    def make_visible(
        self,
        batch_size: int,
        *,
        n_mask: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        可視マスク (True=使う) を生成する。

        Returns
        -------
        visible_mask : torch.Tensor, shape (B, L), dtype=bool
            True=使う / False=隠す
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
        if visible_mask.shape != x.shape:
            raise ValueError(f"visible_mask must have same shape as x, got {tuple(visible_mask.shape)}")
        if visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool dtype")