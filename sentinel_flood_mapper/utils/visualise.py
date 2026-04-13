"""
visualise.py

Visualise functions for the sentinel flood mapper project.

All functions accept an optional save_path and show flag:
    - save_path: If provided, the figure is saved to this path
    - show: If True, the figure is displayed with plt.show()

Figures are returned in all cases so the caller can perform additional
operations if needed.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

def _save_and_show(
        fig: plt.Figure,
        save_path: Optional[Path],
        show: bool
) -> None:
    """
    Save and/or display a matplotlib figure.

    Args:
        fig: Figure to save or display
        save_path: If provided, figure is saved to this path. Parent directories
            are created if needed
        show: If True, figure is displayed with plt.show()
    """
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

def _ensure_2d_axes(
        axes: np.ndarray,
        n_samples: int
) -> np.ndarray:
    """
    Ensure axes array is always 2D regardless of n_samples.

    When n_samples=1, plt.subplots returns a 1D axes array. This function
    reshapes it to 2D so downstream indexing with axes[i, j] works consistently

    Args:
        axes: Axes array returned by plt.subplots
        n_samples: Number of sample rows in plot

    Returns:
        2D axes array of shape (n_samples, n_cols)
    """
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    return axes

def plot_training_curves(
        history: dict,
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Plot training and validation loss and metrics across epochs.

    Args:
        history: Dictionary with keys 'epoch', 'train_loss', 'val_loss',
            'train_iou', 'val_iou', 'train_f1', 'val_f1'
        save_path: If provided, the figure is saved to this path
        show: If True, the figure is displayed. Default True

    Returns:
        Matplotlib figure object
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Training curves", fontsize=14)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="darkorange")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU
    axes[1].plot(epochs, history["train_iou"], label="Train", color="steelblue")
    axes[1].plot(epochs, history["val_iou"], label="Val", color="darkorange")
    axes[1].set_title("IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, history["train_f1"], label="Train", color="steelblue")
    axes[2].plot(epochs, history["val_f1"], label="Val", color="darkorange")
    axes[2].set_title("F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    _save_and_show(fig, save_path, show)

    return fig

def plot_sample_tiles(
        images: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 4,
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Plot a sample of SAR tiles alongside their flood label masks.

    Displays the VV band of the SAR image next to the binary label mask
    for each sample.

    Args:
        images: SAR image tensor of shape (B, 2, H, W), normalised to [0, 1]
        labels: Label tensor of shape (B, H, W) with values 0, 1 or 255
        n_samples: Number of tile samples to display. Default 4
        save_path: If provided, the figure is saved to this path
        show: If True, the figure is displayed. Default True

    Returns:
        Matplotlib figure object
    """
    n_samples = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n_samples, 2, figsize=(6, 3 * n_samples))
    fig.suptitle("Sample tiles - VV band and flood label", fontsize=12)

    axes = _ensure_2d_axes(axes, n_samples)

    for i in range(n_samples):
        vv = images[i, 0].cpu().numpy()
        label = labels[i].cpu().numpy().astype(float)

        # Mask nodata pixels
        label_masked = np.ma.masked_where(label == 255, label)

        axes[i, 0].imshow(vv, cmap="gray")
        axes[i, 0].set_title(f"VV band (sample {i + 1})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(vv, cmap="gray")
        axes[i, 1].imshow(label_masked, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
        axes[i, 1].set_title(f"Flood label (sample {i + 1})")
        axes[i, 1].axis("off")

    plt.tight_layout()

    _save_and_show(fig, save_path, show)

    return fig

def plot_predictions(
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        n_samples: int = 4,
        threshold: float = 0.5,
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Plot SAR image, ground truth label and predicted flood mask side by
    side for a batch of samples.

    Args:
        images: SAR image tensor of shape (B, 2, H, W)
        labels: Ground truth label tensor of shape (B, H, W)
        predictions: Raw model output tensor of shape (B, 1, H, W). Sigmoid
            and threshold are applied internally
        n_samples: Number of samples to display. Default 4
        threshold: Probability threshold for flood classification. Default 0.5
        save_path: If provided, the figure is saved to this path
        show: If True, the figure is displayed. Default True

    Returns:
        Matplotlib figure object
    """
    n_samples = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    fig.suptitle("SAR image - ground truth - prediction", fontsize=12)

    axes = _ensure_2d_axes(axes, n_samples)

    probs = torch.sigmoid(predictions.squeeze(1)).cpu().numpy()

    for i in range(n_samples):
        vv = images[i, 0].cpu().numpy()
        label = labels[i].cpu().numpy().astype(float)
        pred = (probs[i] >= threshold).astype(float)

        label_masked = np.ma.masked_where(label == 255, label)

        axes[i, 0].imshow(vv, cmap="gray")
        axes[i, 0].set_title("VV band")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(vv, cmap="gray")
        axes[i, 1].imshow(label_masked, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(vv, cmap="gray")
        axes[i, 2].imshow(pred, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()

    _save_and_show(fig, save_path, show)

    return fig

def plot_contingency_map(
        image: np.ndarray,
        label: np.ndarray,
        prediction: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Plot a contingency map showing true positives, false positives and false
    negatives overlaid on the SAR image.

    Follows the STURM paper convention:
        - True positives (correctly detected flood) - blue
        - False positives (predicted flood, actually non-flood) - red
        - False negatives (missed flood) - yellow

    Args:
        image: SAR image array of shape (2, H, W), normalised to [0, 1]. VV
            band is used for the background
        label: Ground truth label array of shape (H, W) with values 0, 1 or 255
        prediction: Model probability output of shape (H, W) with values in
            [0, 1]. Threshold is applied internally
        threshold: Probability threshold for flood classification. Default 0.5
        save_path: If provided, the figure is saved to this path
        show: If True, the figure is displayed. Default True

    Returns:
        Matplotlib figure object
    """
    vv = image[0]
    pred = (prediction >= threshold).astype(np.uint8)

    # Build mask to exclude nodata pixels
    valid = (label != 255)

    tp = (pred == 1) & (label == 1) & valid
    fp = (pred == 1) & (label == 0) & valid
    fn = (pred == 0) & (label == 1) & valid

    # Build RGBA contingency overlay
    overlay = np.zeros((*vv.shape, 4), dtype=np.float32)
    overlay[tp] = [0.0, 0.4, 0.8, 0.7]      # Blue - true positive
    overlay[fp] = [0.8, 0.0, 0.0, 0.7]      # Red - false positive
    overlay[fn] = [1.0, 0.9, 0.0, 0.7]      # Yellow - false negative

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Contingency map", fontsize=12)

    # SAR image
    axes[0].imshow(vv, cmap="gray")
    axes[0].set_title("VV band")
    axes[0].axis("off")

    # Contingency map
    axes[1].imshow(vv, cmap="gray")
    axes[1].imshow(overlay)
    axes[1].set_title("Contingency map")
    axes[1].axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=[0.0, 0.4, 0.8], label="True positive"),
        mpatches.Patch(facecolor=[0.8, 0.0, 0.0], label="False positive"),
        mpatches.Patch(facecolor=[1.0, 0.9, 0.0], label="False negative")
    ]
    axes[1].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9
    )

    plt.tight_layout()

    _save_and_show(fig, save_path, show)

    return fig

def plot_metrics_summary(
        metrics: dict,
        title: str = "Evaluation metrics",
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Plot a bar chart summarising segmentation metrics.

    Args:
        metrics: Dictionary with keys 'iou', 'f1', 'precision', 'recall', 'accuracy'.
            Values are floats in [0, 1]
        title: Plot title. Default 'Evaluation metrics'
        save_path: If provided, the figure is saved to this path
        show: If True, the figure is displayed. Default True

    Returns:
        Matplotlib figure object
    """
    names = ["IoU", "F1", "Precision", "Recall", "Accuracy"]
    values = [
        metrics["iou"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["accuracy"]
    ]
    colours = ["steelblue", "darkorange", "seagreen", "mediumpurple", "firebrick"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colours, edgecolor="white", linewidth=0.5)

    # Add value labels on top of each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    _save_and_show(fig, save_path, show)

    return fig
