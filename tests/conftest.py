import random
import numpy as np
import torch
import pytest

@pytest.fixture(scope="session", autouse=True)
def _seed_everywhere():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
