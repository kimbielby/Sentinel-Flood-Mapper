"""
train.py

Training loop for binary flood extent segmentation.

Handles one complete training run including per-epoch train and validation
phases, metric tracking, early stopping and checkpoint saving.
"""
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentinel_flood_mapper import TrainConfig
from sentinel_flood_mapper.models import BCEDiceLoss, SegmentationMetrics
from sentinel_flood_mapper.utils import save_checkpoint

def _get_optimiser(
        model: nn.Module,
        config: TrainConfig,
) -> torch.optim.Optimizer:
    """
    Build and return an optimiser from the training config.

    Args:
        model: Model whose parameters will be optimised
        config: TrainConfig containing optimiser settings

    Returns:
        Configured optimiser instance

    Raises:
        ValueError: If the optimiser name is not recognised
    """
    name = config.optimiser.lower()

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(
            f"Unrecognised optimiser '{config.optimiser}'. "
            f"Supported options: 'adam', 'adamw', 'sgd'."
        )

def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        checkpoint_dir: Path
) -> dict:
    """
    Run a complete training loop with validation, early stopping and checkpoint saving.

    Saves the best model checkpoint based on validation IoU. Training stops
    early if validation IoU does not improve for config.patience consecutive epochs.

    Args:
        model: Initialised model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: TrainConfig object containing training settings
        device: Device to train on
        checkpoint_dir: Directory to save model checkpoints and training
            history. Created if it does not exist

    Returns:
        History dictionary with keys:
            - epoch: List of epoch numbers
            - train_loss: List of per-epoch training loss values
            - val_loss: List of per-epoch validation loss values
            - train_iou: List of per-epoch training IoU values
            - val_iou: List of per-epoch validation IoU values
            - train_f1: List of per-epoch training F1 values
            - val_f1: List of per-epoch validation F1 values
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialise loss, optimiser and metrics
    criterion = BCEDiceLoss().to(device)
    optimiser = _get_optimiser(model, config)
    train_metrics = SegmentationMetrics(device=device)
    val_metrics = SegmentationMetrics(device=device)

    # Scheduler
    scheduler = None
    if config.lr_scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="max",         # Maximise val IoU
            factor=config.lr_scheduler.factor,
            patience=config.lr_scheduler.patience,
            min_lr=config.lr_scheduler.min_lr,
        )

    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_f1": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_iou = 0.0
    num_epochs_no_improvement = 0

    print(f"Training for up to {config.epochs} epochs (early stopping patience: {config.patience})")
    print("-" * 70)

    for epoch in range(config.epochs):

        # --- Training ---
        model.train()
        train_metrics.reset()
        train_loss_sum = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d} train",
            leave=False,
            unit="batch"
        )

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimiser.step()

            train_loss_sum += loss.item()
            train_metrics.update(predictions.detach(), labels)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / len(train_loader)
        train_results = train_metrics.compute()

        # --- Validation ---
        model.eval()
        val_metrics.reset()
        val_loss_sum = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch:03d} val",
            leave=False,
            unit="batch",
        )

        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = criterion(predictions, labels)
                val_loss_sum += loss.item()
                val_metrics.update(predictions, labels)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_loss_sum / len(val_loader)
        val_results = val_metrics.compute()

        if scheduler is not None:
            scheduler.step(val_results["iou"])
            current_lr = optimiser.param_groups[0]["lr"]
            print(f"Learning rate: {current_lr:.2e}")

        # --- Record history ---
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_results["iou"])
        history["val_iou"].append(val_results["iou"])
        history["train_f1"].append(train_results["f1"])
        history["val_f1"].append(val_results["f1"])
        history["lr"].append(optimiser.param_groups[0]["lr"])

        # --- Print epoch summary ---
        print(
            f"Epoch {epoch:03d} / {config.epochs}  |  "
            f"Train loss: {train_loss:.4f}  |  "
            f"Val loss: {val_loss:.4f}  |  "
            f"Train IoU: {train_results['iou']:.4f}  |  "
            f"Val IoU: {val_results['iou']:.4f}  |  "
            f"Train F1: {train_results['f1']:.4f}  |  "
            f"Val F1: {val_results['f1']:.4f}"
        )

        # --- Checkpoint saving ---
        if val_results["iou"] > best_val_iou:
            best_val_iou = val_results["iou"]
            num_epochs_no_improvement = 0
            checkpoint_path = checkpoint_dir / "best_model.pt"

            save_checkpoint(
                model=model,
                optimiser=optimiser,
                epoch=epoch,
                val_iou=best_val_iou,
                val_loss=val_loss,
                checkpoint_path=checkpoint_path
            )
            print(f"New best val IoU: {best_val_iou:.4f} - checkpoint saved ({checkpoint_path.name}).")

        else:
            num_epochs_no_improvement += 1
            print(f"No improvement for {num_epochs_no_improvement}/{config.patience} epochs.")

        # --- Early stopping ---
        if num_epochs_no_improvement >= config.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print("-" * 70)
    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path.name}")

    return history 

