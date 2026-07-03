"""
test_metrics.py

Tests for sentinel_flood_mapper.models.metrics.
"""
import pytest
import torch
from sentinel_flood_mapper.models.metrics import SegmentationMetrics

IGNORE_INDEX = 255

# --- Fixtures ---

@pytest.fixture
def metrics(device):
    return SegmentationMetrics(device=device)

# --- Reset ---

def test_reset_zeros_counts(metrics, sample_predictions, sample_targets):
    """After reset all, counts should be zero."""
    metrics.update(sample_predictions, sample_targets)
    metrics.reset()
    assert metrics.tp.item() == 0
    assert metrics.fp.item() == 0
    assert metrics.fn.item() == 0
    assert metrics.tn.item() == 0

# --- Update ---

def test_update_accumulates_counts(metrics, sample_predictions, sample_targets):
    """Calling update twice should accumulate counts."""
    metrics.update(sample_predictions, sample_targets)
    tp_after_one = metrics.tp.item()

    metrics.update(sample_predictions, sample_targets)
    tp_after_two = metrics.tp.item()

    assert tp_after_two == 2 * tp_after_one

def test_update_ignores_nodata(metrics, device):
    """Nodata pixels should not contribute to any count."""
    predictions = torch.full((4, 1, 128, 128), 10.0)
    targets_nodata = torch.full((4, 128, 128), IGNORE_INDEX, dtype=torch.long)

    metrics.update(predictions, targets_nodata)

    assert metrics.tp.item() == 0
    assert metrics.fp.item() == 0
    assert metrics.fn.item() == 0
    assert metrics.tn.item() == 0

# --- Compute ---

def test_compute_returns_all_keys(metrics, sample_predictions, sample_targets):
    """Computed metrics dict shoudl contain all expected keys."""
    metrics.update(sample_predictions, sample_targets)
    result = metrics.compute()
    assert set(result.keys()) == {"iou", "f1", "precision", "recall", "accuracy"}

def test_compute_returns_floats(metrics, sample_predictions, sample_targets):
    """All metric values should be floats."""
    metrics.update(sample_predictions, sample_targets)
    result = metrics.compute()
    for key, value in result.items():
        assert isinstance(value, float), f"{key}: should be a float"

def test_compute_values_in_range(metrics, sample_predictions, sample_targets):
    """All metric values should be in [0, 1]."""
    metrics.update(sample_predictions, sample_targets)
    result = metrics.compute()
    for key, value in result.items():
        assert 0.0 <= value <= 1.0, f"{key} value {value} is outside [0, 1]"

def test_perfect_predictions_give_iou_one(metrics, perfect_predictions):
    """Perfect predictions should give IoU of 1.0"""
    predictions, targets = perfect_predictions
    metrics.update(predictions, targets)
    result = metrics.compute()
    assert result["iou"] == pytest.approx(1.0, abs=0.01)

def test_perfect_predictions_give_f1_one(metrics, perfect_predictions):
    """Perfect predictions should give F1 of 1.0"""
    predictions, targets = perfect_predictions
    metrics.update(predictions, targets)
    result = metrics.compute()
    assert result["f1"] == pytest.approx(1.0, abs=0.01)

def test_all_wrong_predictions_give_low_iou(metrics, all_wrong_predictions):
    """Completely wrong predictions should give IoU close to 0."""
    predictions, targets = all_wrong_predictions
    metrics.update(predictions, targets)
    result = metrics.compute()
    assert result["iou"] < 0.01

def test_metrics_reset_between_epochs(metrics, perfect_predictions, all_wrong_predictions):
    """Resetting between epochs should give independent results."""
    predictions_perfect, targets = perfect_predictions
    metrics.update(predictions_perfect, targets)
    result_perfect = metrics.compute()

    metrics.reset()

    predictions_wrong, targets = all_wrong_predictions
    metrics.update(predictions_wrong, targets)
    result_wrong = metrics.compute()

    assert result_perfect["iou"] > result_wrong["iou"]
