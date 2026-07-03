"""
test_losses.py

Tests for sentinel_flood_mapper.models.losses.
"""
import pytest
import torch
from sentinel_flood_mapper.models.losses import DiceLoss, BCEDiceLoss

IGNORE_INDEX = 255

# --- Fixtures ---

@pytest.fixture
def dice_loss():
    return DiceLoss()

@pytest.fixture
def bce_dice_loss():
    return BCEDiceLoss()

@pytest.fixture
def all_nodata_targets():
    return torch.full((4, 128, 128), IGNORE_INDEX, dtype=torch.long)

@pytest.fixture
def all_valid_targets():
    return torch.zeros(4, 128, 128, dtype=torch.long)

# --- DiceLoss ---

def test_dice_loss_returns_scalar(sample_predictions, sample_targets, dice_loss):
    """Loss should return a scalar tensor"""
    loss = dice_loss(sample_predictions, sample_targets)
    assert loss.ndim == 0

def test_dice_loss_non_negative(sample_predictions, sample_targets, dice_loss):
    """Loss should always be non-negative"""
    loss = dice_loss(sample_predictions, sample_targets)
    assert loss.item() >= 0.0

def test_dice_loss_perfect_prediction(perfect_predictions, dice_loss):
    """Perfect predictions should produce a loss close to 0."""
    predictions, targets = perfect_predictions
    loss = dice_loss(predictions, targets)
    assert loss.item() < 0.05

def test_dice_loss_ignores_nodata(sample_predictions, all_nodata_targets, all_valid_targets, dice_loss):
    """Nodata pixels (255) should be excluded from loss computation."""
    loss_with_nodata = dice_loss(sample_predictions, all_nodata_targets)
    loss_without_nodata = dice_loss(sample_predictions, all_valid_targets)
    assert loss_with_nodata.item() != loss_without_nodata.item()

# --- BCEDiceLoss ---

def test_bce_dice_loss_returns_scalar(sample_predictions, sample_targets, bce_dice_loss):
    """Loss should return a scalar tensor"""
    loss = bce_dice_loss(sample_predictions, sample_targets)
    assert loss.ndim == 0

def test_bce_dice_loss_non_negative(sample_predictions, sample_targets, bce_dice_loss):
    """Loss should always be non-negative"""
    loss = bce_dice_loss(sample_predictions, sample_targets)
    assert loss.item() >= 0.0

def test_bce_dice_loss_perfect_prediction(perfect_predictions, bce_dice_loss):
    """Perfect predictions should produce a loss close to 0."""
    predictions, targets = perfect_predictions
    loss = bce_dice_loss(predictions, targets)
    assert loss.item() < 0.05

def test_bce_dice_loss_ignores_nodata(sample_predictions, all_nodata_targets, all_valid_targets, bce_dice_loss):
    """Nodata pixels (255) should not contribute to loss computation."""
    loss_valid = bce_dice_loss(sample_predictions, all_valid_targets)
    loss_nodata = bce_dice_loss(sample_predictions, all_nodata_targets)
    assert loss_valid.item() != loss_nodata.item()

def test_bce_dice_loss_weights():
    """Combined loss should reflect bce and dice weights."""
    predictions = torch.randn(2, 1, 128, 128)
    targets = torch.randint(low=0, high=2, size=(2, 128, 128))

    criterion_equal = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    criterion_bce = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
    criterion_dice = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)

    loss_equal = criterion_equal(predictions, targets).item()
    loss_bce = criterion_bce(predictions, targets).item()
    loss_dice = criterion_dice(predictions, targets).item()

    assert loss_equal != loss_bce
    assert loss_equal != loss_dice
    assert loss_bce != loss_dice

def test_bce_dice_loss_moves_to_device(sample_predictions, sample_targets):
    """Loss function should work correctly on CPU"""
    criterion = BCEDiceLoss().to(torch.device("cpu"))
    loss = criterion(
        sample_predictions.to("cpu"),
        sample_targets.to("cpu"),
    )
    assert loss.item() >= 0.0
