#!/usr/bin/env python3
"""
Model Loader for Image Classification

Handles loading pre-trained models, preprocessing images, and running inference.

Usage:
    loader = ModelLoader(model_name="resnet50", device="cpu")
    predictions = loader.predict(image_path)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
import json
import time

from config import get_settings


logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage image classification models"""

    def __init__(self, model_name: str = "resnet50", device: str = "cpu"):
        """
        Initialize model loader

        Args:
            model_name: Name of the model (resnet50, mobilenet_v2)
            device: Device to run inference on (cpu, cuda)
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.classes: List[str] = []

        logger.info(f"Initializing ModelLoader with model={model_name}, device={device}")

        # Load model and preprocessing
        self._load_model()
        self._load_classes()
        self._create_transform()

        logger.info(f"ModelLoader initialized successfully")

    def _load_model(self):
        """Load pre-trained model"""
        logger.info(f"Loading {self.model_name} model...")

        try:
            if self.model_name == "resnet50":
                # Load ResNet-50 with pre-trained ImageNet weights
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

            elif self.model_name == "mobilenet_v2":
                # Load MobileNetV2 with pre-trained ImageNet weights
                self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_classes(self):
        """Load ImageNet class labels"""
        # ImageNet class labels (1000 classes)
        # For production, these should be loaded from a file
        # Here we use a simplified version with category IDs
        try:
            # Load from torchvision
            if self.model_name == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            elif self.model_name == "mobilenet_v2":
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            self.classes = weights.meta["categories"]
            logger.info(f"Loaded {len(self.classes)} class labels")

        except Exception as e:
            logger.warning(f"Failed to load class labels: {e}. Using category IDs instead.")
            # Fallback to numeric labels
            self.classes = [f"class_{i}" for i in range(1000)]

    def _create_transform(self):
        """Create image preprocessing pipeline"""
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.debug("Image preprocessing pipeline created")

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference

        Args:
            image: Image path or PIL Image object

        Returns:
            Preprocessed image tensor

        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Apply transformations
            img_tensor = self.transform(img)

            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)

            # Move to device
            img_tensor = img_tensor.to(self.device)

            return img_tensor

        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise ValueError(f"Image preprocessing failed: {e}")

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Run inference on image

        Args:
            image: Image path or PIL Image object
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class names and confidence scores

        Raises:
            ValueError: If prediction fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            start_time = time.time()

            # Preprocess image
            img_tensor = self.preprocess_image(image)

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)

                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # Get top-k predictions
                top_prob, top_indices = torch.topk(probabilities, top_k)

            # Convert to Python types and format results
            predictions = []
            for i in range(top_k):
                class_idx = top_indices[i].item()
                confidence = top_prob[i].item()

                predictions.append({
                    "class": self.classes[class_idx],
                    "confidence": float(confidence),
                    "class_id": int(class_idx)
                })

            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.3f}s")

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")

    def batch_predict(
        self,
        images: List[Union[str, Path, Image.Image]],
        top_k: int = 5
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Run batch inference on multiple images

        Args:
            images: List of image paths or PIL Image objects
            top_k: Number of top predictions to return per image

        Returns:
            List of prediction lists (one per image)
        """
        results = []

        for image in images:
            try:
                predictions = self.predict(image, top_k=top_k)
                results.append(predictions)
            except Exception as e:
                logger.error(f"Failed to predict image: {e}")
                results.append([])

        return results

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get model metadata

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "name": self.model_name,
            "device": str(self.device),
            "num_classes": len(self.classes),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": (224, 224),
            "framework": "pytorch",
            "version": torch.__version__
        }

    def warmup(self, num_iterations: int = 3):
        """
        Warmup model with dummy inputs

        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model with {num_iterations} iterations...")

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        for i in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)

        logger.info("Model warmup completed")

    def __repr__(self) -> str:
        """String representation"""
        return f"ModelLoader(model={self.model_name}, device={self.device})"
