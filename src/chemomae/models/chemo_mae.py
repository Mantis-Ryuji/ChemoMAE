from __future__ import annotations

from typing import Literal, Optional, Tuple

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
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""
    Generate independent patch-aligned random masks for each spectrum.

    Parameters
    ----------
    batch_size : int
        Number of spectra B.
    seq_len : int
        Spectrum length L, divisible by n_patches.
    n_patches : int
        Number of contiguous equal-length patches P.
    n_mask : int
        Number of hidden patches per spectrum, from zero through P.
    device : torch.device, optional
        Output and random-number device; defaults to CPU.
    generator : torch.Generator, optional
        Caller-owned random stream on the selected device. None uses Torch's
        global stream. Calls with n_mask=0 consume no random numbers.

    Returns
    -------
    torch.Tensor, shape (B, L), dtype=bool
        True means hidden; False means visible. Every element within a patch
        shares its mask value. Different spectra receive independent draws.

    Notes
    -----
    This function's mask convention is the inverse of the encoder's visible
    mask. Pass its logical inverse to ChemoEncoder.
    """
    if seq_len % n_patches != 0:
        raise ValueError("seq_len must be divisible by n_patches")
    if not (0 <= n_mask <= n_patches):
        raise ValueError("n_mask must be in [0, n_patches]")

    if device is None:
        device = torch.device("cpu")

    patch_size = seq_len // n_patches

    # -------------------------------------------------
    # Independent patch masks, shape (B, P).
    # -------------------------------------------------
    patch_mask = torch.zeros(batch_size, n_patches, device=device, dtype=torch.bool)

    if n_mask > 0:
        # Select n_mask patches independently in each row.
        r = torch.rand(batch_size, n_patches, device=device, generator=generator)
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
    Transformer encoder for visible patches of one-dimensional spectra.

    Parameters
    ----------
    seq_len : int, default=256
        Spectrum length L, divisible by n_patches.
    n_patches : int, default=16
        Number of contiguous patches.
    d_model : int, default=256
        Token and CLS dimension.
    nhead : int, default=4
        Attention heads; must divide d_model.
    num_layers : int, default=4
        Transformer encoder depth.
    dim_feedforward : int, optional
        Feed-forward width; defaults to 4*d_model.
    dropout : float, default=0.0
        Transformer dropout probability.
    latent_dim : int, default=16
        Dimension of the linear CLS projection.
    latent_normalize : bool, default=True
        Whether the default latent representation is L2-normalized.

    Notes
    -----
    Input spectra have shape (B, L). The visible mask has the same shape with
    True for visible positions. Each patch must be entirely visible or hidden.
    Visible tokens are compacted, padded, and joined with a learned CLS token
    and learned positional embeddings. Padding is excluded from attention.

    Forward returns shape (B, latent_dim), or (B, d_model) for representation
    "cls". A normalized zero projection remains zero. The current all-hidden
    batch guard exposes the first patch; callers requiring strict masked-input
    isolation must provide at least one visible patch per spectrum.

    Training/evaluation mode and autograd are controlled by the caller.
    """

    def __init__(
        self,
        *,
        seq_len: int = 256,
        n_patches: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        latent_dim: int = 16,
        latent_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_patches = int(n_patches)
        if self.seq_len % self.n_patches != 0:
            raise ValueError("seq_len must be divisible by n_patches")
        self.patch_size = self.seq_len // self.n_patches

        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.latent_normalize = bool(latent_normalize)

        if dim_feedforward is None:
            dim_feedforward = 4 * self.d_model

        # Embed contiguous spectral patches.
        self.patch_proj = nn.Linear(self.patch_size, self.d_model, bias=True)

        # Learned CLS token and positional embeddings.
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

    def forward(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor,
        *,
        representation: Literal["latent", "raw_latent", "normalized_latent", "cls"] = "latent",
    ) -> torch.Tensor:
        """Encode visible patches, returning the selected representation.

        ``latent`` follows ``latent_normalize``; ``raw_latent`` returns the
        projected CLS vector before normalization; ``normalized_latent`` always
        L2-normalizes that projection; ``cls`` returns the Transformer CLS output.
        This method preserves the module's mode and gradient recording.
        """
        if representation not in {"latent", "raw_latent", "normalized_latent", "cls"}:
            raise ValueError(f"unknown representation: {representation!r}")
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

        # Reject masks that partially expose a patch.
        patch_all = vm.all(dim=2)  # (B,P)
        patch_any = vm.any(dim=2)  # (B,P)
        if not torch.equal(patch_all, patch_any):
            raise ValueError("visible_mask must be patch-aligned: each patch must be all True or all False.")
        patch_visible = patch_all  # (B,P)

        # Embed contiguous spectral patches.
        tok = self.patch_proj(x_patches)  # (B,P,d)

        # Compact visible patches before padding.
        order = torch.argsort(patch_visible.int(), dim=1, descending=True)  # (B,P)
        vis_counts = patch_visible.sum(dim=1)  # (B,)
        max_vis = int(vis_counts.max().item())

        if max_vis == 0:
            # Guard the legacy all-hidden-batch case.
            # Continue with the first patch exposed.
            patch_visible = torch.zeros((B, self.n_patches), device=x.device, dtype=torch.bool)
            patch_visible[:, 0] = True
            order = torch.arange(self.n_patches, device=x.device).unsqueeze(0).expand(B, -1)
            vis_counts = torch.ones((B,), device=x.device, dtype=torch.long)
            max_vis = 1

        idx = order[:, :max_vis]  # (B,max_vis)

        # Pad samples to the maximum visible length in this batch.
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

        # True marks padding positions excluded from attention.
        key_pad = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), ~valid], dim=1)

        h = self.encoder(enc_in, src_key_padding_mask=key_pad)  # (B,1+max_vis,d)
        cls_out = h[:, 0, :]
        if representation == "cls":
            return cls_out
        z = self.to_latent(cls_out)  # (B,latent_dim)
        if representation == "normalized_latent" or (
            representation == "latent" and self.latent_normalize
        ):
            z = F.normalize(z, dim=1)
        return z


class ChemoDecoder(nn.Module):
    r"""
    Decode a latent vector directly into a complete spectrum.

    Parameters
    ----------
    seq_len : int
        Output spectrum length.
    latent_dim : int
        Input latent dimension.
    num_layers : int, default=2
        Number of linear layers. One gives a linear projection; larger values
        use an MLP with GELU between linear layers.
    hidden_dim : int, optional
        MLP hidden width; defaults to seq_len.

    Notes
    -----
    Forward accepts shape (B, latent_dim) and returns shape (B, seq_len).
    The decoder has no patch mask tokens and performs no normalization.
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
    Masked autoencoder for spectra with an optional spherical latent space.

    Parameters
    ----------
    seq_len : int, default=256
        Input spectrum length L.
    n_patches : int, default=16
        Equal-length patch count; must divide seq_len.
    d_model : int, default=256
        Encoder token dimension.
    nhead : int, default=4
        Attention head count.
    num_layers : int, default=4
        Encoder depth.
    dim_feedforward : int, optional
        Encoder feed-forward width; defaults to 4*d_model.
    dropout : float, default=0.0
        Encoder dropout probability.
    latent_dim : int, default=16
        Projected CLS dimension.
    latent_normalize : bool, default=True
        Apply L2 normalization to the latent used by forward and the decoder.
    decoder_num_layers : int, default=2
        Decoder linear-layer count.
    n_mask : int, default=4
        Hidden patches per spectrum when no explicit visible mask is supplied.

    Notes
    -----
    Forward returns (reconstruction, latent, visible_mask), with shapes
    (B, L), (B, latent_dim), and (B, L). True always means visible for model
    masks. Reconstruction covers the full spectrum; the loss region is chosen
    by the training/evaluation caller.

    Use encode for all-visible CLS, raw projected, or normalized features.
    It bypasses the decoder and random masking, preserving autograd and mode.
    Use reconstruct when only the decoded spectrum is needed.
    """

    def __init__(
        self,
        *,
        seq_len: int = 256,
        n_patches: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        latent_dim: int = 16,
        latent_normalize: bool = True,
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
            latent_normalize=latent_normalize,
        )
        self.decoder = ChemoDecoder(seq_len=self.seq_len, latent_dim=latent_dim, num_layers=decoder_num_layers)

    def encode(
        self,
        x: torch.Tensor,
        *,
        representation: Literal["latent", "raw_latent", "normalized_latent", "cls"] = "latent",
    ) -> torch.Tensor:
        """Extract a representation with every spectral patch visible.

        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len)
            Floating-point spectra on the model's device.
        representation : {"latent", "raw_latent", "normalized_latent", "cls"}
            ``latent`` uses the configured normalization; ``raw_latent`` is the
            linear CLS projection; ``normalized_latent`` always L2-normalizes
            that projection; ``cls`` returns the CLS vector before projection.

        Returns
        -------
        torch.Tensor
            Shape (B, latent_dim), or (B, d_model) for ``cls``. The output remains
            on the input device and follows the model/autocast arithmetic.

        Notes
        -----
        No random mask is generated, and the decoder is not called. The method
        preserves training/evaluation mode and autograd so it can be used for
        supervised fine-tuning. For inference without dropout, call ``eval()``
        and use ``torch.inference_mode()`` or the streaming ``Extractor``.
        ``F.normalize`` leaves an exactly zero projection at zero.
        """
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating-point torch.Tensor")
        if x.ndim != 2 or x.size(0) == 0:
            raise ValueError("x must be a nonempty 2D tensor (B, seq_len)")
        visible = torch.ones_like(x, dtype=torch.bool)
        self._check_shapes(x, visible)
        return self.encoder(x, visible, representation=representation)

    def make_visible(
        self,
        batch_size: int,
        *,
        n_mask: Optional[int] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Generate a patch-aligned visible mask.

        Parameters
        ----------
        batch_size : int
            Number of spectra B.
        n_mask : int, optional
            Hidden patches per spectrum; defaults to self.n_mask.
        device : torch.device, optional
            Output and random-number device; defaults to CPU.
        generator : torch.Generator, optional
            Caller-owned stream on that device; None uses the global stream.

        Returns
        -------
        torch.Tensor, shape (B, seq_len), dtype=bool
            True means visible and False means hidden.
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
            generator=generator,
        )
        return ~masked

    def reconstruct(
        self,
        x: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
        *,
        n_mask: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Reconstruct spectra from visible patches.

        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len)
            Input spectra.
        visible_mask : torch.Tensor, optional
            Boolean mask with the same shape as x; True means visible.
        n_mask : int, optional
            Hidden patch count when generating a mask.
        generator : torch.Generator, optional
            Stream for generated masks. Ignored when visible_mask is supplied.

        Returns
        -------
        torch.Tensor, shape (B, seq_len)
            Reconstruction. Use forward to receive latent and mask as well.

        Notes
        -----
        This method preserves training/evaluation mode and gradient recording.
        """
        if visible_mask is None:
            visible_mask = self.make_visible(
                x.size(0), n_mask=n_mask, device=x.device, generator=generator,
            )
        self._check_shapes(x, visible_mask)
        z = self.encoder(x, visible_mask)
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
        *,
        n_mask: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Return the reconstructed spectrum, configured latent, and visible mask.

        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len)
            Input spectra.
        visible_mask : torch.Tensor, optional
            Patch-aligned boolean mask with the same shape as x. True means
            visible. None generates a random mask.
        n_mask : int, optional
            Hidden patch count for a generated mask; defaults to self.n_mask.
        generator : torch.Generator, optional
            Stream for generated masks. Ignored when visible_mask is supplied.

        Returns
        -------
        x_recon : torch.Tensor, shape (B, seq_len)
            Full-spectrum reconstruction.
        z : torch.Tensor, shape (B, latent_dim)
            Linear CLS projection, normalized if latent_normalize is enabled.
        visible_mask : torch.Tensor, shape (B, seq_len), dtype=bool
            True means visible; False means hidden.

        Raises
        ------
        ValueError
            Input dimensions or visible-mask dtype/patch alignment are invalid.
        """
        if visible_mask is None:
            visible_mask = self.make_visible(
                x.size(0), n_mask=n_mask, device=x.device, generator=generator,
            )
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
