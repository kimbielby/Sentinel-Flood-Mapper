"""
dataset.py
"""
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

class FloodDataset(Dataset):
    def __init__(
            self,
            file_pairs: list[tuple[Path, Path]],
            transform: Optional[Callable] = None,
            return_metadata: bool = False
    ) -> None:
        """
        Dataset for loading preprocessed Sentinel-1 SAR tiles and binary flood label masks.

        Handles both Sen1Floods11 and STURM-Flood tiles, which share the
        same format after preprocessing - 128x128 pixel GeoTIFF files with
        2-band float32 SAR images and uint8 binary labels.

        Args:
            file_pairs: List of (image_path, label_path) tuples
            transform: Optional transform applied to the SAR image tensor after loading. Should accept and return a float32 tensor of shape (2, H, W)
            return_metadata: If True, __getitem__ returns a third element
                    containing the rasterio metadata dict for the SAR tile. Used
                    during inference to preserve geospatial information for
                    georeferenced output. Default False
        """
        if len(file_pairs) == 0:
            raise ValueError("file_pairs must not be empty.")

        self.file_pairs = file_pairs
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Load and return a SAR image and label mask pair.

        Args:
            idx: Index of the sample to load.

        Returns:
            If return_metadata is False:
                Tuple of (image, label) where image is a float32 tensor of shape
                (2, H, W) and label is a long tensor of shape (H, W) with values
                0, 1 or 255.
            If return_metadata is True:
                Tuple of (image, label, metadata) where metadata is a dict
                containing rasterio profile information including CRS and affine
                transform.

        Raises:
            FileNotFoundError: If the image or label file does not exist.
        """
        s1_path, label_path = self.file_pairs[idx]

        if not Path(s1_path).exists():
            raise FileNotFoundError(f"SAR image not found: {s1_path}")
        if not Path(label_path).exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load SAR image
        with rasterio.open(s1_path) as src:
            s1_arr = src.read().astype(np.float32)      # (2, H, W)
            metadata = src.meta.copy()

        # Load label
        with rasterio.open(label_path) as src:
            label_arr = src.read(1).astype(np.int64)    # (H, W)

        # Convert to tensors
        image = torch.from_numpy(s1_arr)
        label = torch.from_numpy(label_arr)

        # Apply transforms to image if provided
        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.return_metadata:
            return image, label, metadata

        return image, label

def get_dataloader(
        file_pairs: list[tuple[Path, Path]],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
) -> DataLoader:
    """
    Construct a DataLoader from a list of file pairs.

    Args:
        file_pairs: List of (image_path, label_path) tuples, typically produced
                by build_file_pairs()
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples each epoch. Should be True
                for training, False for validation and test. Default True
        num_workers: Number of subprocesses for data loading. 0 means
                data is loaded in the main process. Default 0
        transform: Optional transform applied to SAR image tensors
        return_metadata: If True, each sample includes rasterio metadata.
                Should only be True during inference. Default False

    Returns:
        Configured DataLoader instance
    """
    dataset = FloodDataset(
        file_pairs=file_pairs,
        transform=transform,
        return_metadata=return_metadata
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )








