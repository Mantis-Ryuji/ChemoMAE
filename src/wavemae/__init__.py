"""
WaveMAE: Transformer-based Masked AutoEncoder for 1D spectral data.
"""

from ._version import __version__

# 公開サブパッケージをインポート
from . import preprocessing
from . import models
from . import training
from . import clustering
from . import utils

__all__ = [
    "__version__",
    "preprocessing",
    "models",
    "training",
    "clustering",
    "utils",
]
