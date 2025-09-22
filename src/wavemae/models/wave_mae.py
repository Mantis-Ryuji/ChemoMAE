from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WaveMAE",
    "WaveEncoder",
    "WaveDecoderMLP",
    "make_block_mask",
    "sinusoidal_positional_encoding",
]

# -----------------------------------------------------------------------------
# Positional encoding & mask
# -----------------------------------------------------------------------------
def sinusoidal_positional_encoding(L: int, d_model: int, device: torch.device) -> torch.Tensor:
    """(1, L, d_model) sinusoidal PE for 1D sequences."""
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
    """Boolean mask (B, L) with True for *masked* positions. Vectorized under L % n_blocks == 0."""
    assert 0 <= n_mask <= n_blocks, "n_mask must be within [0, n_blocks]"
    assert seq_len % n_blocks == 0, "seq_len must be divisible by n_blocks"
    device = device or torch.device("cpu")

    B, L = int(batch_size), int(seq_len)
    block_size = L // n_blocks
    block_ids = torch.arange(n_blocks, device=device).expand(B, -1)               # (B, n_blocks)
    perm = torch.rand(B, n_blocks, device=device).argsort(dim=1)[:, :n_mask]     # (B, n_mask)
    chosen = torch.gather(block_ids, 1, perm)                                     # (B, n_mask)
    within = torch.arange(block_size, device=device).view(1, 1, -1)               # (1,1,bs)
    ids_mask = (chosen.unsqueeze(-1) * block_size + within).reshape(B, -1)        # (B, n_mask*bs)

    mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    mask.scatter_(1, ids_mask, True)
    return mask  # True=masked, False=visible


# -----------------------------------------------------------------------------
# Encoder (visible tokens + CLS)
# -----------------------------------------------------------------------------
class WaveEncoder(nn.Module):
    """Transformer encoder that consumes only *visible* tokens + CLS.

    Input:
        x : (B, L) spectra
        visible_mask : (B, L) bool, True for visible (unmasked) positions
    Output:
        z : (B, latent_dim), L2-normalized
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

        self.token_proj = nn.Linear(1, d_model)
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

        self.to_latent = nn.Linear(d_model, latent_dim)

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


# -----------------------------------------------------------------------------
# Decoder (MLP: z -> R^L)
# -----------------------------------------------------------------------------
class WaveDecoderMLP(nn.Module):
    def __init__(self, *, seq_len: int, latent_dim: int = 64, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -----------------------------------------------------------------------------
# WaveMAE (model only; no loss inside)
# -----------------------------------------------------------------------------
class WaveMAE(nn.Module):
    """Masked Autoencoder for 1D spectra (model-only; no loss inside).

    Public API (no dict configs):
      - constructor arguments explicitly specify encoder/decoder/mask settings.
      - forward(x, mask=None) -> (x_recon, z, mask)
      - encode(x, visible_mask) -> z
      - reconstruct(x, mask=None) -> x_recon
      - make_mask(batch_size) -> mask (True=masked)

    Notes:
      - Loss should be computed *outside* (e.g., masked SSE).
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
        # decoder
        dec_hidden: int = 256,
        dec_dropout: float = 0.1,
        # mask
        n_blocks: int = 16,
        n_mask: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_blocks = int(n_blocks)
        self.n_mask = int(n_mask)

        self.encoder = WaveEncoder(
            seq_len=self.seq_len,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_learnable_pos=use_learnable_pos,
        )
        self.decoder = WaveDecoderMLP(
            seq_len=self.seq_len, latent_dim=latent_dim, hidden_dim=dec_hidden, dropout=dec_dropout
        )

    def make_mask(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return make_block_mask(batch_size, self.seq_len, self.n_blocks, self.n_mask, device=device)

    def encode(self, x: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, visible_mask)

    def reconstruct(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = self.make_mask(x.size(0), device=x.device)
        z = self.encoder(x, ~mask)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_recon, z, mask). No loss computed inside."""
        if mask is None:
            mask = self.make_mask(x.size(0), device=x.device)
        z = self.encoder(x, ~mask)
        x_recon = self.decoder(z)
        return x_recon, z, mask
