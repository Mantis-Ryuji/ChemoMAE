"""
ChemoMAE: 1D Spectral Masked Autoencoder + Hyperspherical Clustering Toolkit
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

from packaging.version import Version
import torch

_MIN_TORCH = Version("2.1.0")
_CUR_TORCH = Version(torch.__version__.split("+")[0])

if _CUR_TORCH < _MIN_TORCH:
    raise ImportError(
        f"ChemoMAE requires PyTorch >= {_MIN_TORCH}, "
        f"but detected torch=={torch.__version__}. "
        "Please upgrade PyTorch (e.g., pip install torch>=2.1.0)."
    )