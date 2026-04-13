"""
transforms.py

Paired image and label transforms for SAR flood detection training.

All transforms accept and return a (image, label) tuple to ensure spatial
consistency between SAR tiles and their corresponding masks. Geometric
transforms are applied identically to both image and label. Radiometric
transforms are applied to the image only.
"""
import torch
import torch.nn.functional as F

class Compose:
    def __init__(
            self,
            transforms: list
    ) -> None:
        """
        Sequentially apply a list of paired transforms to an image and label.

        Args:
            transforms: List of transform objects. Each must accept
                    (image, label) and return (image, label)
        """
        self.transforms = transforms

    def __call__(
            self,
            image: torch.Tensor,
            label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class RandomHFlip:
    def __init__(
            self,
            p: float = 0.5
    ) -> None:
        """
        Randomly flip image and label horizontally with probability p.

        Args:
            p: Probability of applying the flip. Default 0.5
        """
        self.p = p

    def __call__(
            self,
            image: torch.Tensor,
            label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        return image, label

class RandomVFlip:
    def __init__(
            self,
            p: float = 0.5
    ) -> None:
        """
        Randomly flip image and label vertically with probability p.

        Args:
            p: Probability of applying the flip. Default 0.5
        """
        self.p = p

    def __call__(
            self,
            image: torch.Tensor,
            label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])
        return image, label

class RandomRotation90:
    def __init__(
            self,
            p: float = 0.5
    ) -> None:
        """
        Randomly rotate image and label by a multiple of 90 degrees.

        90 degree increments are used rather than arbitrary angles to avoid
                interpolation artefacts in SAR imagery. Each of the four possible
                rotations (0, 90, 180, 270 degrees) is equally likely.

        Args:
            p: Probability of applying a rotation. Default 0.5
        """
        self.p = p

    def __call__(
            self,
            image: torch.Tensor,
            label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            k = torch.randint(1, 4, (1,)).item()        # 1, 2 or 3 rotations
            image = torch.rot90(image, k=k, dims=[1, 2])
            label = torch.rot90(label, k=k, dims=[0, 1])
        return image, label

class RandomBrightnessContrast:
    def __init__(
            self,
            brightness_limit: float = 0.1,
            contrast_limit: float = 0.1,
            p: float = 0.5
    ) -> None:
        """
        Randomly adjust brightness and contrast of the SAR image.

        Simulates variation in acquisition conditions across different events
        and seasons. Applied to the image only - label is unchanged.

        Brightness adjustment adds a random offset drawn from [-brightness_limit, +brightness_limit].
        Contrast adjustment multiplies by a random factor drawn from [1 - contrast_limit, 1 + contrast_limit].

        Args:
            brightness_limit: Max absolute brightness offset. Default 0.1
            contrast_limit:  Max contrast scaling deviation from 1. Default 0.1
            p: Probability of apply the transform. Default 0.5
        """
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(
            self,
            image: torch.Tensor,
            label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            brightness = (torch.rand(1).item() * 2 - 1) * self.brightness_limit
            contrast = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast_limit
            image = image * contrast + brightness
            image = torch.clamp(image, 0.0, 1.0)
        return image, label


