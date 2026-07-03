"""
test_preprocess.py

Tests for sentinel_flood_mapper.data.preprocess
"""
import pytest
import numpy as np
from sentinel_flood_mapper.data.preprocess import (
    normalise_sen1floods11,
    remap_labels,
    lee_filter
)

# --- normalise_sen1floods11 ---

def test_normalise_output_range(normalised_sar_array):
    """Normalised output should be in [0, 1] after NaN replacement and clipping."""
    assert normalised_sar_array.min() >= 0.0, "Normalised values should not be below 0.0."
    assert normalised_sar_array.max() <= 1.0, "Normalised values should not exceed 1.0."

def test_normalise_nan_replacement(sample_sar_array):
    """NaN values should be replaced and no NaN should remain in output."""
    assert np.isnan(sample_sar_array).any(), "Fixture should contain NaN values"

    result = normalise_sen1floods11(
        arr=sample_sar_array,
        nan_fill=0.0,
        clip_min=-30.0,
        clip_max=10.0,
        apply_lee_filter=False,
    )
    assert not np.isnan(result).any(), "Output should contain no NaN values"

def test_normalise_output_shape(sample_sar_array):
    """Output shape should match input shape."""
    result = normalise_sen1floods11(
        arr=sample_sar_array,
        nan_fill=0.0,
        clip_min=-30.0,
        clip_max=10.0,
        apply_lee_filter=False,
    )
    assert result.shape == sample_sar_array.shape

def test_normalise_output_dtype(sample_sar_array):
    """Output dtype should be float32."""
    result = normalise_sen1floods11(
        arr=sample_sar_array,
        nan_fill=0.0,
        clip_min=-30.0,
        clip_max=10.0,
        apply_lee_filter=False,
    )
    assert result.dtype == np.float32

def test_normalise_nan_fill_value(sample_sar_array):
    """NaN pixels should be replaced with the normalised equivalent of nan_fill."""
    nan_fill = -30.0        # Corresponds to 0.0 after normalisation to [-30, 10]
    result = normalise_sen1floods11(
        arr=sample_sar_array,
        nan_fill=nan_fill,
        clip_min=-30.0,
        clip_max=10.0,
        apply_lee_filter=False,
    )
    # Pixels that were NaN should now be 0.0 in normalised space
    nan_mask = np.isnan(sample_sar_array[0, :10, :10])
    assert (result[0, :10, :10][nan_mask] == pytest.approx(0.0)), (
        "NaN pixels should map to 0.0 when nan_fill equals clip_min"
    )

def test_normalise_with_lee_filter(sample_sar_array):
    """Applying Lee filter should still produce valid normalised output."""
    # Replace NaN first since lee_filter expects no NaN
    arr = sample_sar_array.copy()
    arr = np.where(np.isnan(arr), 0.0, arr)

    result = normalise_sen1floods11(
        arr=arr,
        nan_fill=0.0,
        clip_min=-30.0,
        clip_max=10.0,
        apply_lee_filter=True,
        window_size=7,
    )
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert not np.isnan(result).any()

# --- remap_labels ---

def test_remap_labels_values():
    """All output values should in the set {0, 1, 255}."""
    arr = np.array([[0, 1, -1], [0, 1, 0]], dtype=np.int64)
    label_map = {-1: 255, 0: 0, 1: 1}
    result = remap_labels(arr, label_map)
    unique_values = set(np.unique(result).tolist())
    assert unique_values.issubset({0, 1, 255}), (
        f"Unexpected label values: {unique_values - {0, 1, 255}}"
    )

def test_remap_labels_nodata():
    """-1 values should be remapped to 255"""
    arr = np.full((4, 4), -1, dtype=np.int64)
    label_map = {-1: 255, 0: 0, 1: 1}
    result = remap_labels(arr, label_map)
    assert (result == 255).all(), "All -1 values shoudl map to 255"

def test_remap_labels_water():
    """1 values should remain 1"""
    arr = np.ones((4, 4), dtype=np.int64)
    label_map = {-1: 255, 0: 0, 1: 1}
    result = remap_labels(arr, label_map)
    assert (result == 1).all()

def test_remap_labels_non_water():
    """0 values should remain 0"""
    arr = np.zeros((4, 4), dtype=np.int64)
    label_map = {-1: 255, 0: 0, 1: 1}
    result = remap_labels(arr, label_map)
    assert (result == 0).all()

def test_remap_labels_shape():
    """Output shape should match input shape."""
    arr = np.random.randint(low=-1, high=2, size=(64, 64)).astype(np.int64)
    label_map = {-1: 255, 0: 0, 1: 1}
    result = remap_labels(arr, label_map)
    assert result.shape == arr.shape

def test_remap_labels_sturm():
    """STURM multi-class labels should all remap to 0, 1 or 255."""
    arr = np.array([[0, 1, 2, 3, 4, 5, 99]], dtype=np.int64)
    label_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 99: 255}
    result = remap_labels(arr, label_map)
    unique_values = set(np.unique(result).tolist())
    assert unique_values.issubset({0, 1, 255})

# --- lee_filter ---

def test_lee_filter_output_shape():
    """Output shape should match input shape."""
    arr = np.random.uniform(low=-30, high=10, size=(2, 128, 128)).astype(np.float32)
    result = lee_filter(arr, window_size=7)
    assert result.shape == arr.shape

def test_lee_filter_no_nan_introduced():
    """Lee filter should not introduce NaN values."""
    arr = np.random.uniform(low=-30, high=10, size=(2, 128, 128)).astype(np.float32)
    result = lee_filter(arr, window_size=7)
    assert not np.isnan(result).any()

def test_lee_filter_output_dtype():
    """Output dtype should be float32."""
    arr = np.random.uniform(low=-30, high=10, size=(2, 128, 128)).astype(np.float32)
    result = lee_filter(arr, window_size=7)
    assert result.dtype == np.float32

def test_lee_filter_smoothing():
    """Filtered output should have lower variance than input, confirming smoothing."""
    arr = np.random.uniform(low=-30, high=10, size=(2, 128, 128)).astype(np.float32)
    result = lee_filter(arr, window_size=7)
    assert result.var() < arr.var(), (
        "Lee filter should reduce variance by smoothing speckle noise"
    )








