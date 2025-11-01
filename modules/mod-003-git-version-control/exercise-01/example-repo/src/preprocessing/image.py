"""
Image Preprocessing Module

Handles image preprocessing for ML models including resizing, normalization,
and data augmentation.
"""

import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, Union
import numpy as np

from utils.logging import get_logger


logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing pipeline for classification models

    Handles standard preprocessing operations including:
    - Resizing
    - Normalization (ImageNet statistics)
    - Tensor conversion
    - Data augmentation (optional)
    """

    # ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize preprocessor

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize with ImageNet stats
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # Build transformation pipeline
        self.transform = self._build_transform()

        logger.info(
            f"Initialized preprocessor: size={target_size}, "
            f"normalize={normalize}, augment={augment}"
        )

    def _build_transform(self) -> transforms.Compose:
        """
        Build transformation pipeline

        Returns:
            Composed transformation pipeline
        """
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize(self.target_size))

        # Optional augmentation
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            ])

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalization
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            )

        return transforms.Compose(transform_list)

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, str]
    ) -> torch.Tensor:
        """
        Preprocess a single image

        Args:
            image: PIL Image, numpy array, or path to image file

        Returns:
            Preprocessed image tensor [C, H, W]

        Raises:
            ValueError: If image format is invalid
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image)
                logger.debug(f"Loaded image from: {image}")

            # Convert numpy to PIL
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.debug(f"Converted image to RGB")

            # Apply transformations
            tensor = self.transform(image)

            logger.debug(
                f"Preprocessed image: shape={tensor.shape}, "
                f"dtype={tensor.dtype}"
            )

            return tensor

        except Exception as e:
            logger.error(f"Preprocessing error: {e}", exc_info=True)
            raise ValueError(f"Failed to preprocess image: {e}")

    def preprocess_batch(
        self,
        images: list
    ) -> torch.Tensor:
        """
        Preprocess multiple images into a batch

        Args:
            images: List of images (PIL, numpy, or paths)

        Returns:
            Batch tensor [B, C, H, W]
        """
        tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(tensors)

        logger.debug(f"Preprocessed batch: shape={batch.shape}")

        return batch

    def denormalize(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse normalization for visualization

        Args:
            tensor: Normalized tensor [C, H, W] or [B, C, H, W]

        Returns:
            Denormalized tensor
        """
        if not self.normalize:
            return tensor

        mean = torch.tensor(self.IMAGENET_MEAN).view(-1, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(-1, 1, 1)

        # Handle batch dimension
        if tensor.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        denormalized = tensor * std + mean
        return torch.clamp(denormalized, 0, 1)

    def to_pil(
        self,
        tensor: torch.Tensor
    ) -> Image.Image:
        """
        Convert tensor back to PIL Image

        Args:
            tensor: Image tensor [C, H, W]

        Returns:
            PIL Image
        """
        # Denormalize if needed
        tensor = self.denormalize(tensor)

        # Convert to numpy
        array = tensor.cpu().numpy()

        # Transpose from [C, H, W] to [H, W, C]
        array = np.transpose(array, (1, 2, 0))

        # Scale to [0, 255]
        array = (array * 255).astype(np.uint8)

        # Convert to PIL
        image = Image.fromarray(array)

        return image

    def get_transform(self) -> transforms.Compose:
        """
        Get the transformation pipeline

        Returns:
            Transformation pipeline
        """
        return self.transform

    def __repr__(self) -> str:
        return (
            f"ImagePreprocessor(size={self.target_size}, "
            f"normalize={self.normalize}, augment={self.augment})"
        )


class DataAugmentor:
    """
    Advanced data augmentation pipeline for training

    Provides additional augmentation techniques beyond basic preprocessing.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize augmentor

        Args:
            target_size: Target image size
        """
        self.target_size = target_size
        self.transform = self._build_augmentation()

    def _build_augmentation(self) -> transforms.Compose:
        """
        Build augmentation pipeline

        Returns:
            Composed augmentation pipeline
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.target_size,
                scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImagePreprocessor.IMAGENET_MEAN,
                std=ImagePreprocessor.IMAGENET_STD
            ),
            transforms.RandomErasing(p=0.1)
        ])

    def augment(self, image: Image.Image) -> torch.Tensor:
        """
        Apply augmentation to image

        Args:
            image: PIL Image

        Returns:
            Augmented tensor
        """
        return self.transform(image)


def resize_and_pad(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio using padding

    Args:
        image: Input PIL Image
        target_size: Target size (height, width)
        fill_color: Color for padding (R, G, B)

    Returns:
        Resized and padded image
    """
    # Calculate scaling factor
    width, height = image.size
    target_height, target_width = target_size

    scale = min(target_width / width, target_height / height)

    # Resize maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create padded image
    padded = Image.new("RGB", (target_width, target_height), fill_color)

    # Paste resized image centered
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded.paste(image, (paste_x, paste_y))

    return padded


def center_crop(
    image: Image.Image,
    target_size: Tuple[int, int]
) -> Image.Image:
    """
    Center crop image to target size

    Args:
        image: Input PIL Image
        target_size: Target size (height, width)

    Returns:
        Center-cropped image
    """
    width, height = image.size
    target_height, target_width = target_size

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))
