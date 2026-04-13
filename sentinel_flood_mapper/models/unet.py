"""
unet.py
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from sentinel_flood_mapper import ModelConfig

def get_model(
        model_config: ModelConfig,
        device: torch.device
) -> nn.Module:
    """
    Build and return a U-Net model configured from the model config.

    Uses a pretrained EfficientNet-B0 encoder backbone from
    segmentation-models-pytorch. The first convolutional layer is adapted
    to accept 2-channel Sentinel-1 SAR input (VV and VH) rather than the
    standard 3-channel RGB input.

    Args:
        model_config: ModelConfig object containing architecture settings
        device: Device to move the model to after construction

    Returns:
        Configured U-Net model moved to the specified device
    """
    # Sentinel-1 SAR imagery has two polarisation bands: VV and VH
    IN_CHANNELS = 2

    model = smp.Unet(
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        in_channels=IN_CHANNELS,
        classes=model_config.classes,
        activation=None
    )

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: U-Net with {model_config.encoder_name} encoder")
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    return model
