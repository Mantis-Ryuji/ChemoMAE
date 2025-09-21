"""
Preprocessing utilities for spectral data.

Currently includes:
- SNV (Standard Normal Variate) as functional API and sklearn-style transformer.
"""

from .snv import snv, SNVScaler

__all__ = [
    "snv",
    "SNVScaler",
]
