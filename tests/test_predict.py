"""
test_predict.py

Tests for sentinel_flood_mapper.models.predict.
"""
import pytest
import numpy as np
from sentinel_flood_mapper.models.predict import (
    _pad_to_multiple,
    predict
)

# --- Fixtures ---

@pytest.fixture
def sample_sar_norm():
    """Normalised 2-band SAR array of shape (2, 512, 512)."""
    return np.random.uniform(low=0.0, high=1.0, size=(2, 512, 512)).astype(np.float32)

@pytest.fixture
def uneven_sar_array():
    """SAR array with dimensions not divisible by 32."""
    return np.random.uniform(low=0.0, high=1.0, size=(2, 100, 150)).astype(np.float32)

@pytest.fixture
def predict_outputs(trained_model, sample_sar_norm, device):
    """Probability map and binary mask from a single forward pass."""
    return predict(
        model=trained_model,
        arr=sample_sar_norm,
        device=device,
        use_tiling=False,
    )

# --- _pad_to_multiple ---

def test_pad_output_divisible_by_multiple(uneven_sar_array):
    """Padded dimensions should be divisible by the target multiple."""
    padded, _ = _pad_to_multiple(uneven_sar_array, multiple=32)
    _, h, w = padded.shape
    assert h % 32 == 0
    assert w % 32 == 0

def test_pad_output_larger_than_input(uneven_sar_array):
    """Padded array should be at least as large as the input array."""
    _, h_in, w_in = uneven_sar_array.shape
    padded, _ = _pad_to_multiple(uneven_sar_array, multiple=32)
    _, h_out, w_out = padded.shape
    assert h_out >= h_in
    assert w_out >= w_in

def test_no_padding_needed():
    """Arrays already divisible by multiple should be returned unchanged."""
    arr = np.random.rand(2, 128, 128).astype(np.float32)
    padded, padding = _pad_to_multiple(arr, multiple=32)
    assert padded.shape == arr.shape
    assert padding == (0, 0, 0, 0)

def test_pad_preserves_original_content(uneven_sar_array):
    """Original content should be preserved in top-left of padded array."""
    _, h, w = uneven_sar_array.shape
    padded, _ = _pad_to_multiple(uneven_sar_array, multiple=32)
    np.testing.assert_array_equal(padded[:, :h, :w], uneven_sar_array)

def test_pad_fills_with_zeros(uneven_sar_array):
    """Padded regions should be filled with zeros."""
    _, h, w = uneven_sar_array.shape
    padded, (_, pad_bottom, _, pad_right) = _pad_to_multiple(
        uneven_sar_array, multiple=32
    )
    if pad_bottom > 0:
        assert (padded[:, h:, :] == 0.0).all()
    if pad_right > 0:
        assert (padded[:, :, w:] == 0.0).all()

def test_pad_channel_dimension_unchanged(uneven_sar_array):
    """Number of channels should not change after padding."""
    c_in = uneven_sar_array.shape[0]
    padded, _ = _pad_to_multiple(uneven_sar_array, multiple=32)
    assert padded.shape[0] == c_in

# --- predict ---

def test_predict_output_shape_matches_input(predict_outputs, sample_sar_norm):
    """Output probability map shape should match input spatial dimensions."""
    _, h, w = sample_sar_norm.shape
    prob_map, binary_mask = predict_outputs
    assert prob_map.shape == (h, w)
    assert binary_mask.shape == (h, w)

def test_predict_prob_map_range(predict_outputs):
    """Probability map values should be in [0, 1]."""
    prob_map, _ = predict_outputs
    assert prob_map.min() >= 0.0
    assert prob_map.max() <= 1.0

def test_predict_binary_mask_values(predict_outputs):
    """Binary mask should contain only 0 and 1."""
    _, binary_mask = predict_outputs
    unique_values = set(np.unique(binary_mask).tolist())
    assert unique_values.issubset({0, 1})

def test_predict_tiling_output_shape(trained_model, sample_sar_norm, device):
    """Tiled inference output shape should match input spatial dimensions."""
    _, h, w = sample_sar_norm.shape
    prob_map, binary_mask = predict(
        model=trained_model,
        arr=sample_sar_norm,
        device=device,
        use_tiling=True,
        tile_size=128,
        stride=64,
    )
    assert prob_map.shape == (h, w)
    assert binary_mask.shape == (h, w)

def test_predict_uneven_input_shape(trained_model, uneven_sar_array, device):
    """Predict should handle images not divisible by 32 via padding."""
    _, h, w = uneven_sar_array.shape
    prob_map, binary_mask = predict(
        model=trained_model,
        arr=uneven_sar_array,
        device=device,
        use_tiling=False,
    )
    assert prob_map.shape == (h, w)
    assert binary_mask.shape == (h, w)
