"""
Preprocessing utilities for spectral data.

Currently includes:
- SNV (Standard Normal Variate) as functional API and sklearn-style transformer.
- FPS downsampling (Farthest-Point Sampling)
"""

from .snv import SNVScaler
from .downsampling import fps_downsample

__all__ = [
    "SNVScaler",
    "fps_downsample"
]
