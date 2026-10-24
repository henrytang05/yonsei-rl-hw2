import pytest
import numpy as np
import torch
from cas4160.infrastructure import pytorch_util as ptu


@pytest.fixture(autouse=True)
def init_device():
    """Ensure ptu.device is set to CPU for all tests."""
    ptu.init_gpu(use_gpu=False)


@pytest.fixture
def discrete_env_dims():
    return {"ob_dim": 4, "ac_dim": 2, "discrete": True}


@pytest.fixture
def continuous_env_dims():
    return {"ob_dim": 8, "ac_dim": 3, "discrete": False}


@pytest.fixture
def network_params():
    return {"n_layers": 2, "layer_size": 64, "learning_rate": 5e-3}
