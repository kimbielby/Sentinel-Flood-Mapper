"""
losses.py

Loss functions for binary flood extent segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Computes 1 - Dice coefficient where the Dice coefficient measures the
    overlap between predicted and ground truth flood pixels. A smooth
    term is added to the numerator and denominator to prevent division by
    zero in tiles with no flood pixels.

    Nodata pixels (label value 255) are excluded from the computation via a
    binary weight mask.
    """
    def __init__(
            self,
            smooth: float = 1.0
    ) -> None:
        """
        Args:
            smooth: Smoothing term added to numerator and denominator. Default 1.0
        """
        super().__init__()
        self.smooth = smooth

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            predictions: Raw model output of shape (B, 1, H, W)
            targets: Ground truth labels of shape (B, H, W) with values 0, 1 or
                255 (ignore)

        Returns:
            Scalar Dice loss tensor
        """
        # Build mask - exclude nodata pixels
        mask = (targets != IGNORE_INDEX).float()

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions.squeeze(1))

        # Mask out nodata pixels
        probs = probs * mask
        targets_masked = targets.float() * mask

        intersection = (probs * targets_masked).sum()
        denominator = probs.sum() + targets_masked.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice loss for binary segmentation.

    BCE provides stable gradients and penalises individual pixel
    misclassifications. Dice directly optimises the overlap between predicted
    and actual flood extent, handling class imbalance naturally.

    Nodata pixels (label value 255) are excluded from both loss components
    via a weight mask.
    """

    def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            pos_weight: float = 3.0,
            smooth: float = 1.0
    ) -> None:
        """
        Args:
            bce_weight: Weight applied to the BCE component. Default 0.5
            dice_weight: Weight applied to the Dice component. Default 0.5
            pos_weight: Weight applied to positive (flood) class in BCE to
                address class imbalance. A value of 3.0 reflects the approximate
                1:3 water to non-water ratio in the combined dataset. Default 3.0
            smooth: Smoothing term for Dice loss. Default 1.0
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

        # pos_weight as tensor - registered as buffer so it moves to the
        # correct device automatically with .to(device)
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight])
        )

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined BCE and Dice loss.

        Args:
            predictions: Raw model output of shape (B, 1, H, W)
            targets: Ground truth labels of shape (B, H, W) with values 0, 1 or 255 (ignore)

        Returns:
            Scalar combined loss tensor
        """
        # Build weight mask - zero out nodata pixels for BCE
        mask = (targets != IGNORE_INDEX).float()

        # Clamp targets to 0/1 for BCE - nodata pixels are masked out so
        # their clamped value does not matter
        targets_clamped = targets.clone().float()
        targets_clamped[targets == IGNORE_INDEX] = 0.0

        # BCE with logits - reduction=none so we can apply mask
        bce = F.binary_cross_entropy_with_logits(
            predictions.squeeze(1),
            targets_clamped,
            pos_weight=self.pos_weight,
            reduction="none"
        )

        # Apply mask and compute mean over valid pixels only
        n_valid = mask.sum().clamp(min=1.0)
        bce = (bce * mask).sum() / n_valid

        dice = self.dice_loss(predictions, targets)

        return self.bce_weight * bce + self.dice_weight * dice
