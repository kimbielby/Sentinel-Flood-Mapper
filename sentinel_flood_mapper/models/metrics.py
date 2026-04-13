"""
metrics.py

Pixel-wise segmentation metrics for binary flood extent mapping.
"""
import torch

IGNORE_INDEX = 255

class SegmentationMetrics:
    """
    Accumulates pixel-wise segmentation statistics across batches and
    computes binary classification metrics at epoch end.

    Tracks true positives, false positives, false negatives and true negatives
    for the flood (positive) class, excluding nodata pixels.
    """
    def __init__(
            self,
            device: torch.device,
            threshold: float = 0.5
    ) -> None:
        """
        Args:
            device: Device to store accumulated tensors on
            threshold: Probability threshold above which a pixel is classified
                as flood. Default 0.5
        """
        self.device = device
        self.threshold = threshold

        # Initialise true/false positive/negative counts
        self.tp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fn = torch.tensor(0, dtype=torch.long, device=self.device)
        self.tn = torch.tensor(0, dtype=torch.long, device=self.device)

    def reset(
            self
    ) -> None:
        """
        Reset all accumulated counts to zero.
        Call at start of each epoch.
        """
        self.tp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fn = torch.tensor(0, dtype=torch.long, device=self.device)
        self.tn = torch.tensor(0, dtype=torch.long, device=self.device)

    def update(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            is_probability: bool = False,
    ) -> None:
        """
        Update accumulated counts with a batch of predictions and targets.

        Args:
            predictions: Raw model output of shape (B, 1, H, W). Sigmoid is
                applied internally
            targets: Ground truth labels of shape (B, H, W) with values 0, 1 or
                255 (ignore)
            is_probability: If True, probabilities are passed in. If False, logits
                are being passed in. Default False
        """
        #  Apply sigmoid and threshold to get binary predictions
        if is_probability:
            probs = predictions.squeeze(1)
        else:
            probs = torch.sigmoid(predictions.squeeze(1))

        preds = (probs >= self.threshold).long()

        # Build mask to exclude nodata pixels
        mask = (targets != IGNORE_INDEX)

        preds_masked = preds[mask]
        targets_masked = targets[mask].long()

        self.tp += (preds_masked * targets_masked).sum()
        self.fp += (preds_masked * (1 - targets_masked)).sum()
        self.fn += ((1 - preds_masked) * targets_masked).sum()
        self.tn += ((1 - preds_masked) * (1 - targets_masked)).sum()

    def compute(
            self
    ) -> dict[str, float]:
        """
        Compute all metrics from accumulated counts.

        Returns:
            Dictionary containing:
                - iou: Intersection over Union
                - f1: F1 score (Dice coefficient)
                - precision: Precision
                - recall: Recall
                - accuracy: Overall pixel accuracy
        """
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()
        tn = self.tn.float()

        epsilon = 1e-7

        iou = tp / (tp + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)

        return {
            "iou": iou.item(),
            "f1": f1.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "accuracy": accuracy.item()
        }

