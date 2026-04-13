"""
evaluate.py

Core evaluation loop.

Runs a trained model over a dataloader, accumulates predictions and
computes pixel-wise segmentation metrics. Optionally collects sample
predictions for visualisation.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentinel_flood_mapper.models import SegmentationMetrics

def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_vis_samples: int = 4,
) -> tuple[dict, Optional[dict]]:
    """
    Run evaluation on a dataloader and compute segmentation metrics.

    Args:
        model: Trained model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        num_vis_samples: Number of sample batches to collect for visualisation.
            Set to 0 to disable collection. Default 4

    Returns:
        Tuple of (metrics, vis_samples) where:
            metrics: Dictionary with keys 'iou', 'f1', 'precision', 'recall', 'accuracy'
            vis_samples: Dictionary with keys 'images', 'labels', 'predictions'
                containing tensors from the first num_vis_samples batches, or
                None if num_vis_samples is 0
    """
    model.eval()

    metrics = SegmentationMetrics(device=device)
    metrics.reset()

    vis_images = []
    vis_labels = []
    vis_predictions = []

    eval_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Evaluating",
        unit="batch",
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in eval_bar:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            metrics.update(predictions, labels)

            # Only collect visualisation samples from batches containing flood pixels
            has_flood = (labels == 1).any()
            if len(vis_images) < num_vis_samples and has_flood:
                vis_images.append(images.cpu())
                vis_labels.append(labels.cpu())
                vis_predictions.append(predictions.cpu())

    results = metrics.compute()

    vis_samples = None
    if num_vis_samples > 0 and vis_images:
        vis_samples = {
            "images": torch.cat(vis_images, dim=0),
            "labels": torch.cat(vis_labels, dim=0),
            "predictions": torch.cat(vis_predictions, dim=0)
        }

    return results, vis_samples
