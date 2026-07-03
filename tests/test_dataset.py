"""
test_dataset.py

Tests for sentinel_flood_mapper.data.dataset.
"""
import pytest
import torch
from sentinel_flood_mapper.data.dataset import (
    FloodDataset,
    get_dataloader
)

# --- Fixtures ---

@pytest.fixture
def file_pairs(tmp_path, sample_sar_array, sample_label_array):
    """
    Write sample SAR and label arrays to temporary GeoTIFFs and return
    a list of (s1_path, label_path) tuples.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(0, 0, 1, 1, 128, 128)
    crs = "EPSG:4326"

    pairs = []
    for i in range(4):
        s1_path = tmp_path / f"tile_{i}_S1.tif"
        label_path = tmp_path / f"tile_{i}_label.tif"

        with rasterio.open(
            s1_path,
            mode="w",
            driver="GTiff",
            height=128,
            width=128,
            count=2,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(sample_sar_array[:, :128, :128])

        with rasterio.open(
            label_path,
            mode="w",
            driver="GTiff",
            height=128,
            width=128,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(
                sample_label_array[:128, :128].astype("uint8")[np.newaxis, :, :]
            )

        pairs.append((s1_path, label_path))

    return pairs

@pytest.fixture
def dataset(file_pairs):
    return FloodDataset(file_pairs=file_pairs)

@pytest.fixture
def dataloader(file_pairs):
    return get_dataloader(
        file_pairs=file_pairs,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

# --- FloodDataset ---

def test_dataset_length(dataset, file_pairs):
    """Dataset length should match number of file pairs."""
    assert len(dataset) == len(file_pairs)

def test_dataset_image_shape(dataset):
    """SAR image tensor should have shape (2, H, W)."""
    image, _ = dataset[0]
    assert image.shape == (2, 128, 128)

def test_dataset_label_shape(dataset):
    """Label tensor should have shape (H, W)."""
    _, label = dataset[0]
    assert label.shape == (128, 128)

def test_dataset_image_dtype(dataset):
    """SAR image tensor should be float32."""
    image, _ = dataset[0]
    assert image.dtype == torch.float32

def test_dataset_label_dtype(dataset):
    """Label tensor should be int64"""
    _, label = dataset[0]
    assert label.dtype == torch.int64

def test_dataset_label_values(dataset):
    """Label values should only be 0, 1 or 255"""
    _, label = dataset[0]
    unique_values = set(label.unique().tolist())
    assert unique_values.issubset({0, 1, 255})

def test_dataset_raises_on_empty_pairs():
    """FloodDataset should raise ValueError when given empty file pairs."""
    with pytest.raises(ValueError):
        FloodDataset(file_pairs=[])

def test_dataset_raises_on_missing_file(tmp_path):
    """FloodDataset should raise FileNotFoundError for missing files."""
    from pathlib import Path
    pairs = [(Path(tmp_path / "missing_s1.tif"), Path(tmp_path / "missing_label.tif"))]
    dataset = FloodDataset(file_pairs=pairs)
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]

def test_dataset_with_metadata(file_pairs):
    """Dataset with return_metadata=True should return three items."""
    dataset = FloodDataset(file_pairs=file_pairs, return_metadata=True)
    result = dataset[0]
    assert len(result) == 3

def test_dataset_metadata_contains_crs(file_pairs):
    """Metadata should contain crs key."""
    dataset = FloodDataset(file_pairs=file_pairs, return_metadata=True)
    _, _, metadata = dataset[0]
    assert "crs" in metadata

# --- get_dataloader ---

def test_dataloader_batch_shape(dataloader):
    """Batched images should have shape (batch_size, 2, H, W)."""
    images, labels = next(iter(dataloader))
    assert images.shape == (2, 2, 128, 128)
    assert labels.shape == (2, 128, 128)

def test_dataloader_length(dataloader, file_pairs):
    """Dataloader should have correct number of batches."""
    expected_batches = len(file_pairs) // 2
    assert len(dataloader) == expected_batches
