"""
checkpoint.py

Utilities for saving and loading model checkpoints.
"""
from pathlib import Path
import torch
import torch.nn as nn

def save_checkpoint(
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        epoch: int,
        val_iou: float,
        val_loss: float,
        checkpoint_path: Path
) -> None:
    """
    Save a model checkpoint to disk.

    Args:
        model: Model to save
        optimiser: Optimiser to save. Allows training to be resumed from this checkpoint
        epoch: Current epoch number
        val_iou: Validation IoU at this checkpoint
        val_loss: Validation loss at this checkpoint
        checkpoint_path: Full path to save the checkpoint file
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "val_iou": val_iou,
            "val_loss": val_loss
        },
        checkpoint_path
    )

def load_checkpoint(
        model: nn.Module,
        checkpoint_path: Path,
        device: torch.device
) -> tuple[nn.Module, dict]:
    """
    Load a saved model checkpoint.

    Args:
        model: Model instance with the same architecture as the saved checkpoint
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains
        epoch, val_iou and val_loss from the saved checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Checkpoint loaded from .../{checkpoint_path.name}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Val IoU: {checkpoint['val_iou']:.4f}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")

    return model, checkpoint









