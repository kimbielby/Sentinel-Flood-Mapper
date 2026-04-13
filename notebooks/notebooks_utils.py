import numpy as np
import rasterio
from pathlib import Path

def check_image_nodata(directory: str | Path, glob_pattern: str, n_sample: int = 50) -> None:
    """Check a sample of images for nan, inf, and nodata characteristics."""
    files = sorted(Path(directory).glob(glob_pattern))

    has_nan = 0
    has_inf = 0
    has_nodata = 0
    val_mins = []
    val_maxs = []
    val_means = []

    for f in files[:n_sample]:
        with rasterio.open(f) as src:
            arr = src.read().astype(np.float32)
            nodata = src.nodata

        has_nan += int(np.isnan(arr).any())
        has_inf += int(np.isinf(arr).any())

        if nodata is not None:
            has_nodata += int((arr == nodata).any())

        # Stats ignoring nan
        val_mins.append(float(np.nanmin(arr)))
        val_maxs.append(float(np.nanmax(arr)))
        val_means.append(float(np.nanmean(arr)))

    print(f"  Files sampled:       {n_sample}")
    print(f"  Files with nan:      {has_nan}")
    print(f"  Files with inf:      {has_inf}")
    print(f"  Files with nodata:   {has_nodata}")
    print(f"  Value range:         {min(val_mins):.4f}  to  {max(val_maxs):.4f}")
    print(f"  Mean value range:    {min(val_means):.4f}  to  {max(val_means):.4f}")

def validate_sturm_normalisation(
        arr: np.ndarray,
        tolerance: float = 0.01,
) -> bool:
    """
    Validate that a STURM Sentinel-1 tile is within the expected
    normalised range of [0, 1].

    A small tolerance is applied to account for any floating point edge
    cases at the boundaries.

    Args:
         arr: SAR image array of shape (2, H, W), expected to be normalised to [0, 1].
         tolerance: Allowable margin outside [0, 1]. Default 0.01.

    Returns:
        True if all values are within [0 - tolerance, 1 + tolerance], False otherwise
    """
    return bool(
        np.nanmin(arr) >= -tolerance and
        np.nanmax(arr) <= 1.0 + tolerance
    )
