"""
conftest.py
"""
import pytest
import numpy as np
import torch
from sentinel_flood_mapper.data.preprocess import normalise_sen1floods11
from sentinel_flood_mapper.models.unet import get_model
from sentinel_flood_mapper.config import ModelConfig

@pytest.fixture
def sample_sar_array():
    """2-band SAR array of shape (2, 128, 128) in dB scale with some NaN values."""
    arr = np.random.uniform(low=-30, high=10, size=(2, 128, 128)).astype(np.float32)
    arr[0, :10, :10] = np.nan       # Introduce NaN values
    return arr

@pytest.fixture
def sample_label_array():
    """Binary label array of shape (128, 128) with values 0, 1 and 255."""
    arr = np.random.randint(low=0, high=2, size=(128, 128)).astype(np.int64)
    arr[:5, :5] = 255               # Introduce nodata pixels
    return arr

@pytest.fixture
def sample_predictions():
    """Raw model output tensor of shape (B, 1, H, W)."""
    return torch.randn(4, 1, 128, 128)

@pytest.fixture
def sample_targets():
    """Label tensor of shape (B, H, W) with values 0, 1 and 255."""
    targets = torch.randint(low=0, high=2, size=(4, 128, 128))
    targets[0, :5, :5] = 255            # Introduce nodata pixels
    return targets

@pytest.fixture
def normalised_sar_array(sample_sar_array):
    """Normalised SAR array produced by normalise_sen1floods11."""
    arr = sample_sar_array.copy()
    arr = np.where(np.isnan(arr), 0.0, arr)
    return normalise_sen1floods11(
        arr=arr,
        nan_fill=0.0,
        clip_min=-30.0,
        clip_max=10.0,
        apply_lee_filter=False,
    )

@pytest.fixture
def perfect_predictions():
    """High confidence predictions perfectly matching a half-flood target."""
    targets = torch.zeros(4, 128, 128, dtype=torch.long)
    targets[:, :64, :] = 1
    predictions = torch.full((4, 1, 128, 128), -10.0)
    predictions[:, :, :64, :] = 10.0
    return predictions, targets

@pytest.fixture
def all_wrong_predictions():
    """Predictions that are wrong for every valid pixel."""
    targets = torch.zeros(4, 128, 128, dtype=torch.long)
    targets[:, :64, :] = 1
    predictions = torch.full((4, 1, 128, 128), 10.0)
    predictions[:, :, :64, :] = -10.0
    return predictions, targets

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")

@pytest.fixture
def trained_model(device):
    """Initialised U-Net model with random weights for inference testing."""
    model_config = ModelConfig(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        classes=1,
    )
    model = get_model(model_config=model_config, device=device)
    model.eval()
    return model
