"""
Preprocessing utilities for spectral data.

Currently includes:
- SNV (Standard Normal Variate) as functional API and sklearn-style transformer.
- FPS downsampling (Farthest-Point Sampling)
"""

from .snv import SNVScaler, snv
from .downsampling import cosine_fps_downsample

__all__ = [
    "SNVScaler",
    "snv",
    "cosine_fps_downsample"
]
