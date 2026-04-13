"""
preprocess.py

Functions for preprocessing Sen1Floods11 and STURM-Flood datasets.
"""
import numpy as np
from pathlib import Path
import rasterio
import shutil
from scipy.ndimage import uniform_filter
from tqdm import tqdm

def remap_labels(
        arr: np.ndarray,
        label_map: dict
) -> np.ndarray:
    """
    Remap label values in a 2D array according to a mapping dictionary.
    Any values not present in the mapping are set to 255 (ignore index).

    Args:
        arr: 2D numpy array of label values
        label_map: Dictionary mapping source values to target values.

    Returns:
        Remapped array of same shape and dtype uint8.
    """
    out = np.full(arr.shape, 255, dtype=np.uint8)
    for src_val, dst_val in label_map.items():
        out[arr == src_val] = dst_val
    return out

def normalise_sen1floods11(
        arr: np.ndarray,
        nan_fill: float,
        clip_min: float,
        clip_max: float,
        apply_lee_filter: bool = True,
        window_size: int = 7
) -> np.ndarray:
    """
    Normalise a Sen1Floods11 Sentinel-1 SAR image to [0, 1].

    Steps:
        1. Replace nan values with nan_fill (in dB scale)
        2. Optionally apply Lee speckle filter (in dB scale)
        3. Clip values to [clip_min, clip_max] dB
        4. Min-max normalise to [0, 1]

    Args:
        arr: SAR image array of shape (2, H, W) in dB scale, may contain nan values
        nan_fill: Value to replace nan pixels with, in dB scale. Should be within [clip_min, clip_max].
        clip_min: Lower clip bound in dB. Values below this are set to clip_min before normalisation.
        clip_max: Upper clip bound in dB. Values above this are set to clip_max before normalisation.
        apply_lee_filter: Whether to apply Lee speckle filtering before normalisation. Default True
        window_size: Lee filter window size in pixels. Only used if apply_lee_filter is True. Default 7

    Returns:
        Normalised array of shape (2, H, W), dtype float32, with values in [0, 1]
    """
    arr = arr.astype(np.float32)
    arr = np.where(np.isnan(arr), nan_fill, arr)

    if apply_lee_filter:
        arr = lee_filter(arr, window_size=window_size)

    arr = np.clip(arr, clip_min, clip_max)
    arr = (arr - clip_min) / (clip_max - clip_min)
    return arr

def tile_sen1floods11_chip(
        s1_arr: np.ndarray,
        label_arr: np.ndarray,
        chip_id: str,
        nan_fill: float,
        clip_min: float,
        clip_max: float,
        label_map: dict,
        tile_size: int = 128,
        save_path: Path = None,
        src_meta: dict = None,
        apply_lee_filter: bool = True,
        lee_window_size: int = 7
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Tile a single Sen1Floods11 512x512 chip into 128x128 tiles, applying
    normalisation to the SAR image and remapping the labels.

    Args:
        s1_arr: SAR array of shape (2, 512, 512) in dB scale
        label_arr: Label array of shape (512, 512)
        chip_id: Chip identifier string e.g. 'Ghana_103272'.
                    Used for output filenames if save_path is provided
        nan_fill: Value to replace nan pixels with, in dB scale
        clip_min: Lower clip bound in dB
        clip_max: Upper clip bound in dB
        label_map: Dictionary mapping source label values to target values
        tile_size: Size of output tiles in pixels. Default 128
        save_path: If provided, tiles are saved as GeoTIFFs to this directory.
                    Directory is created if it does not exist.
        src_meta: Rasterio metadata from the source file. Required if
                    save_path is provided, to preserve geospatial metadata in output files
        apply_lee_filter: Whether to apply Lee speckle filtering before
                    normalisation. Default True
        lee_window_size: Lee filter window size in pixels. Only used if
                    apply_lee_filter is True. Default 7

    Returns:
        Tuple of (s1_tiles, label_tiles) where each is a list of numpy arrays.
        s1_tiles contains float32 arrays of shape (2, tile_size, tile_size).
        label_tiles contains uint8 arrays of shape (tile_size, tile_size)

    Raises:
        ValueError: If save_path is provided but src_meta is None
        ValueError: If s1_arr spatial dimensions are not divisible by tile_size
    """
    _, h, w = s1_arr.shape
    if h % tile_size != 0 or w % tile_size != 0:
        raise ValueError(
            f"SAR array spatial dimensions ({h}, {w}) are not divisible by tile_size {tile_size}."
        )

    if save_path is not None and src_meta is None:
        raise ValueError(
            "src_meta must be provided when save_path is specified in order "
            "to preserve geospatial metadata."
        )

    # Normalise SAR and remap labels
    s1_norm = normalise_sen1floods11(
        arr=s1_arr,
        nan_fill=nan_fill,
        clip_min=clip_min,
        clip_max=clip_max,
        apply_lee_filter=apply_lee_filter,
        window_size=lee_window_size
    )
    label_remapped = remap_labels(
        arr=label_arr,
        label_map=label_map
    )

    n_rows = h // tile_size
    n_cols = w // tile_size

    s1_tiles = []
    label_tiles = []

    if save_path is not None:
        save_path = Path(save_path)
        s1_dir = save_path / "S1"
        label_dir = save_path / "LabelHand"
        s1_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    for row in range(n_rows):
        for col in range(n_cols):
            r0, r1 = row * tile_size, (row + 1) * tile_size
            c0, c1 = col * tile_size, (col + 1) * tile_size

            s1_tile = s1_norm[:, r0:r1, c0:c1]
            label_tile = label_remapped[r0:r1, c0:c1]

            s1_tiles.append(s1_tile)
            label_tiles.append(label_tile)

            if save_path is not None:
                tile_id = f"{chip_id}_{row}_{col}"
                transform = src_meta["transform"]

                # Compute the tile's transform from the parent chip transform
                tile_transform = rasterio.transform.from_origin(
                    transform.c + c0 * transform.a,
                    transform.f + r0 * transform.e,
                    transform.a,
                    abs(transform.e)
                )

                # Save SAR tile
                s1_meta = {
                    **src_meta,
                    "count": 2,
                    "dtype": "float32",
                    "height": tile_size,
                    "width": tile_size,
                    "transform": tile_transform,
                    "nodata": None
                }
                with rasterio.open(
                    s1_dir / f"{tile_id}_S1Hand.tif", "w", **s1_meta
                ) as dst:
                    dst.write(s1_tile)

                # Save label tile
                label_meta = {
                    **src_meta,
                    "count": 1,
                    "dtype": "uint8",
                    "height": tile_size,
                    "width": tile_size,
                    "transform": tile_transform,
                    "nodata": 255
                }
                with rasterio.open(
                    label_dir / f"{tile_id}_LabelHand.tif", "w", **label_meta
                ) as dst:
                    dst.write(label_tile[np.newaxis, :, :])

    return s1_tiles, label_tiles

def _check_preprocessing_complete(
        output_dir: Path,
        source_dir: Path,
        tile_size: int
) -> bool:
    """
    Check whether preprocessing has already been completed for a dataset.

    Compares the number of existing processed tiles against the expected
    count derived from the source files and tile size. Warns if the directory
    exists but is incomplete, suggesting interrupted processing.

    Args:
        output_dir: Directory where processed tiles are saved
        source_dir: Directory containing the original source files.
                    Used to derive the expected number of output tiles.
        tile_size: Tile size used during preprocessing, in pixels.

    Returns:
        True if preprocessing is complete, False otherwise.
    """
    source_files = sorted(Path(source_dir).glob("*.tif"))

    if len(source_files) == 0:
        raise FileNotFoundError(
            f"No source files found in {source_dir}"
        )

    with rasterio.open(source_files[0]) as src:
        chip_h, chip_w = src.height, src.width

    tiles_per_chip = (chip_h // tile_size) * (chip_w // tile_size)
    expected_tiles = len(source_files) * tiles_per_chip
    existing_tiles = len(list(Path(output_dir).glob("*.tif"))) \
        if Path(output_dir).exists() else 0

    if existing_tiles >= expected_tiles:
        print(f"Preprocessing already complete ({existing_tiles} tiles found).")
        return True
    elif existing_tiles > 0:
        print(f"Warning: incomplete preprocessing detected.")
        print(f"Expected {expected_tiles} tiles, found {existing_tiles}.")
        print(f"Re-running preprocessing from scratch.")
        return False
    else:
        return False

def preprocess_sen1floods11(
        s1_dir: Path,
        label_dir: Path,
        output_dir: Path,
        nan_fill: float,
        clip_min: float,
        clip_max: float,
        label_map: dict,
        tile_size: int = 128,
        apply_lee_filter: bool = True,
        lee_window_size: int = 7
) -> None:
    """
    Preprocess all Sen1Floods11 hand-labelled chips into 128x128 tiles,
    saving normalised SAR images and remapped labels to output_dir.

    Skips processing if output tiles already exist in output_dir.

    Args:
        s1_dir: Directory containing S1Hand GeoTIFF files
        label_dir: Directory containing LabelHand GeoTIFF files
        output_dir: Root directory to save processed tiles.
                Subdirectories S1/ and LabelHand/ are created inside
        nan_fill: Value to replace nan pixels with, in dB scale
        clip_min: Lower clip bound in dB
        clip_max: Upper clip bound in dB
        label_map: Dictionary mapping source label values to target values
        tile_size: Size of output tiles in pixels. Default 128
        apply_lee_filter: Whether to apply Lee speckle filtering before
                normalisation. Default True
        lee_window_size: Lee filter window size in pixels. Only used if
                apply_lee_filter is True. Default 7

    Raises:
        FileNotFoundError: If no S1Hand files are found in s1_dir
    """
    s1_out = Path(output_dir) / "S1"

    if _check_preprocessing_complete(s1_out, s1_dir, tile_size):
        return

    s1_files = sorted(Path(s1_dir).glob("*.tif"))

    print(f"Processing {len(s1_files)} Sen1Floods11 chips into "
          f"{tile_size}x{tile_size} tiles...")

    total_tiles = 0
    skipped = 0

    for s1_path in tqdm(s1_files, desc="Processing Sen1Floods11 chips", unit="chip"):
        chip_id = s1_path.name.replace("_S1Hand.tif", "")
        label_path = Path(label_dir) / f"{chip_id}_LabelHand.tif"

        if not label_path.exists():
            print(f"Warning: no label found for {chip_id}, skipping.")
            skipped += 1
            continue

        with rasterio.open(s1_path) as src:
            s1_arr = src.read().astype(np.float32)
            src_meta = src.meta.copy()

        with rasterio.open(label_path) as src:
            label_arr = src.read(1)

        tile_sen1floods11_chip(
            s1_arr=s1_arr,
            label_arr=label_arr,
            chip_id=chip_id,
            nan_fill=nan_fill,
            clip_min=clip_min,
            clip_max=clip_max,
            label_map=label_map,
            tile_size=tile_size,
            save_path=output_dir,
            src_meta=src_meta,
            apply_lee_filter=apply_lee_filter,
            lee_window_size=lee_window_size
        )

        n_tiles = (s1_arr.shape[1] // tile_size) * (s1_arr.shape[2] // tile_size)
        total_tiles += n_tiles

    print(f"Done. {total_tiles} tiles saved to {output_dir}.")
    if skipped > 0:
        print(f"Warning: {skipped} chips skipped due to missing labels.")

def preprocess_sturm(
        s1_dir: Path,
        label_dir: Path,
        output_dir: Path,
        label_map: dict
) -> None:
    """
    Preprocess all STURM Sentinel-1 floodmap labels by remapping class
    values to binary water/non-water scheme and saving to output_dir.

    STURM SAR images are already normalised and tiled at 128x128 pixels
    so are copied to output_dir without modification.

    Skips processing if output labels already exist in output_dir.

    Args:
        s1_dir: Directory containing STURM Sentinel-1 GeoTIFF files
        label_dir: Directory containing STURM floodmap GeoTIFF files
        output_dir: Root directory to save processed files.
                Subdirectories S1/ and Floodmaps/ are created inside
        label_map: Dictionary mapping source label values to target values

    Raises:
        FileNotFoundError: If no floodmap files are found in label_dir
    """
    s1_out = Path(output_dir) / "S1"
    label_out = Path(output_dir) / "Floodmaps"

    if _check_preprocessing_complete(label_out, label_dir, tile_size=128):
        return

    label_files = sorted(Path(label_dir).glob("*.tif"))

    s1_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(label_files)} STURM floodmap labels...")

    skipped = 0

    for label_path in tqdm(label_files, desc="Processing STURM labels", unit="tile"):
        s1_path = Path(s1_dir) / label_path.name

        if not s1_path.exists():
            print(f"Warning: no S1 image found for {label_path.name}, skipping")
            skipped += 1
            continue

        # Remap label
        with rasterio.open(label_path) as src:
            label_arr = src.read(1)
            label_meta = src.meta.copy()

        label_remapped = remap_labels(arr=label_arr, label_map=label_map)

        # Save remapped label
        label_meta.update(dtype="uint8", nodata=255)
        with rasterio.open(label_out / label_path.name, "w", **label_meta) as dst:
            dst.write(label_remapped[np.newaxis, :, :])

        # Copy S1 image unchanged
        shutil.copy2(s1_path, s1_out / s1_path.name)

    print(f"Done. {len(label_files) - skipped} STURM tiles processed to {output_dir}.")
    if skipped > 0:
        print(f"Warning: {skipped} tiles skipped due to missing S1 images.")

def lee_filter(
        arr: np.ndarray,
        window_size: int = 7
) -> np.ndarray:
    """
    Apply a Lee speckle filter to a SAR image array.

    The Lee filter reduces multiplicative speckle noise by estimating the
    local mean and variance within a sliding window, and using them to compute
    a weighted average between the local mean and the observed pixel value.
    Edges and high-contrast features are preserved because areas with
    high local variance (i.e. real structure) are weighted towards the observed
    value, while homogeneous areas are weighted towards the local mean.

    Applied to Sen1Floods11 SAR imagery to match the Refined Lee speckle
    filtering applied to STURM tiles during their preprocessing pipeline.

    Args:
        arr: SAR image array of shape (2, H, W) in dB scale. Should have
                nan values replaced before calling.
        window_size: Size of the local neighbourhood window in pixels.
                Larger values produce stronger smoothing but may blur fine
                features. Default 7, consistent with common SAR speckle filtering
                practice.

    Returns:
        Speckle filtered array of same shape and dtype as input
    """
    arr = arr.astype(np.float32)
    result = np.zeros_like(arr)

    for band in range(arr.shape[0]):
        img = arr[band]
        local_mean = uniform_filter(img, size=window_size)
        local_sq = uniform_filter(img ** 2, size=window_size)
        local_var = local_sq - local_mean ** 2

        # Noise variance estimated from the whole image
        noise_var = np.mean(local_var) / (local_mean ** 2 + 1e-10)

        # Lee filter weights
        weights = local_var / (local_var + noise_var * local_mean ** 2 + 1e-10)
        result[band] = local_mean + weights * (img - local_mean)

    return result











