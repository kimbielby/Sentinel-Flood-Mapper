"""
predict.py

Core inference functions for flood extent prediction from Sentinel-1 SAR imagery.

Handles two inference modes:
    - Single forward pass with optional padding (use_tiling=False)
    - Tiled inference with reassembly (use_tiling=True)

Both modes produce a probability map and binary flood mask as numpy arrays,
preserving the original image spatial dimensions.
"""
import numpy as np
import torch
import torch.nn as nn

def _pad_to_multiple(
        arr: np.ndarray,
        multiple: int = 32
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pad a SAR image array to the nearest multiple of a given value.

    Padding is added to the right and bottom edges only, so the top-left
    origin of the image is preserved. This ensures the geospatial transform
    of the output remains valid without adjustment.

    Args:
        arr: SAR image array of shape (C, H, W)
        multiple: Target multiple for H and W dimensions. Default 32

    Returns:
        Tuple of (padded_arr, padding) where padding is (pad_top, pad_bottom,
        pad_left, pad_right) - all values needed to crop the padding off after inference
    """
    _, h, w = arr.shape

    pad_bottom = (multiple - h % multiple) % multiple
    pad_right = (multiple - w % multiple) % multiple

    if pad_bottom == 0 and pad_right == 0:
        return arr, (0, 0, 0, 0)

    padded_arr = np.pad(
        arr,
        pad_width=((0, 0), (0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=0.0
    )

    return padded_arr, (0, pad_bottom, 0, pad_right)

def _predict_single(
        model: nn.Module,
        arr: np.ndarray,
        device: torch.device,
) -> np.ndarray:
    """
    Run a single forward pass on a SAR image array.

    Args:
        model: Trained model in eval mode
        arr: SAR image array of shape (C, H, W), normalised to [0, 1]
        device: Device to run inference on

    Returns:
        Probability map of shape (H, W), float32, values in [0, 1]
    """
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)                              # (1, 1, H, W)
        probs = torch.sigmoid(logits)                   # (1, 1, H, W)

    return probs.squeeze(0).squeeze(0).cpu().numpy()       # (H, W)

def _make_gaussian_weight_map(
        tile_size: int,
) -> np.ndarray:
    """
    Create a 2D Gaussian weight map for blending overlapping tile predictions.

    Pixels near the centre of a tile receive higher weight than pixels near
    the edges, reflecting the model's higher confidence in predictions where
    full spatial context is available in all directions.

    Args:
        tile_size: Size of the tile in pixels

    Returns:
        Float32 array of shape (tile_size, tile_size) with values in [0, 1]
    """
    sigma = tile_size / 4.0
    centre = tile_size / 2.0

    y = np.arange(tile_size, dtype=np.float32) - centre
    x = np.arange(tile_size, dtype=np.float32) - centre

    yy, xx = np.meshgrid(y, x, indexing="ij")
    weight_map = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    return weight_map.astype(np.float32)

def _predict_tiled(
        model: nn.Module,
        arr: np.ndarray,
        device: torch.device,
        tile_size: int = 128,
        stride: int = 64,
) -> np.ndarray:
    """
    Run overlapping tiled inference on a SAR image array and reassemble
    predictions using Gaussian-weighted blended.

    Splits the image into overlapping tiles using the given stride, runs inference
    on each tile and blends the probability maps back into a single array
    matching the input dimensions using a 2D Gaussian weight map.

    Args:
        model: Trained model in eval mode
        arr: SAR image array of shape (C, H, W), normalised to [0, 1].
            H and W must be divisible by stride after padding
        device: Device to run inference on
        tile_size: Size of each tile in pixels. Default 128
        stride: Step size between tiles in pixels. Default 64 (50% overlap).

    Returns:
        Probability map of shape (H, W), float32, values in [0, 1]

    Raises:
        ValueError: If H or W are not divisible by tile_size
    """
    _, h, w = arr.shape
    weight_map = _make_gaussian_weight_map(tile_size)

    # Accumulators for weighted predictions and weights
    prob_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    # Generate all tile top-left corners
    row_starts = list(range(0, h - tile_size + 1, stride))
    col_starts = list(range(0, w - tile_size + 1, stride))

    # Ensure the last tile covers the bottom/right edge
    if row_starts[-1] + tile_size < h:
        row_starts.append(h - tile_size)
    if col_starts[-1] + tile_size < w:
        col_starts.append(w - tile_size)

    for r0 in row_starts:
        for c0 in col_starts:
            r1 = r0 + tile_size
            c1 = c0 + tile_size

            tile = arr[:, r0:r1, c0:c1]
            probs = _predict_single(model, tile, device)

            prob_acc[r0:r1, c0:c1] += probs * weight_map
            weight_acc[r0:r1, c0:c1] += weight_map

    # Normalise by accumulated weights
    prob_map = prob_acc / np.maximum(weight_acc, 1e-8)

    return prob_map

def predict(
        model: nn.Module,
        arr: np.ndarray,
        device: torch.device,
        use_tiling: bool = False,
        tile_size: int = 128,
        stride: int = 64,
        threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run flood extent prediction on a preprocessed SAR image array.

    Handles padding, inference and cropping transparently. The output
    arrays always match the spatial dimensions of the input array.

    Args:
        model: Trained model in eval mode
        arr: Preprocessed SAR image array of shape (C, H, W), normalised to [0, 1]
        device: Device to run inference on
        use_tiling: If True, split into tile_size tiles for inference. If False,
            pad to nearest multiple of 32 and run as a single forward pass
        tile_size: Tile size for tiled inference. Default 128
        stride: Step size between tiles in pixels. Default 64 (50% overlap).
        threshold: Probability threshold for binary mask. Default 0.5

    Returns:
        Tuple of (prob_map, binary_mask) where:
            prob_map: Float32 array of shape (H, W), values in [0, 1]
            binary_mask: Uint8 array of shape (H, W), values 0 or 1
    """
    _, h, w = arr.shape

    if use_tiling:
        prob_map = _predict_tiled(model, arr, device, tile_size=tile_size, stride=stride)
    else:
        prob_map = _predict_single(model, arr, device)

    # Crop padding off - restore original spatial dimensions
    prob_map = prob_map[:h, :w]

    binary_mask = (prob_map >= threshold).astype(np.uint8)

    return prob_map, binary_mask
















