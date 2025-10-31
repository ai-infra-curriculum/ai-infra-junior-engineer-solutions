"""
PyTorch ResNet-50 model inference with observability.
"""

import io
import time
from typing import Dict, Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from app.core.config import settings
from app.core.exceptions import (
    ModelNotLoadedException,
    InvalidInputException,
    ModelInferenceException,
)
from app.instrumentation.logging import get_logger
from app.instrumentation.metrics import (
    model_loaded,
    set_model_loaded,
    set_model_memory,
)
from app.instrumentation.tracing import TracedOperation

logger = get_logger(__name__)


# ImageNet class labels (top 10 for demo)
IMAGENET_CLASSES = {
    0: "tench",
    1: "goldfish",
    2: "great_white_shark",
    3: "tiger_shark",
    207: "golden_retriever",
    208: "labrador_retriever",
    281: "tabby_cat",
    282: "tiger_cat",
    283: "persian_cat",
    284: "siamese_cat",
}


class ModelInferenceService:
    """
    Model inference service with observability.

    Handles loading, inference, and lifecycle of PyTorch models.
    """

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device(settings.model_device)
        self.transform = self._get_transform()
        self._is_loaded = False

    def _get_transform(self) -> transforms.Compose:
        """
        Get image preprocessing transform.

        Returns:
            Composed transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def load_model(self):
        """
        Load PyTorch model with observability tracking.

        Raises:
            ModelInferenceException: If model loading fails
        """
        with TracedOperation("load_model", {"model_name": settings.model_name}):
            try:
                start_time = time.time()
                logger.info("Loading model", model_name=settings.model_name)

                # Load pretrained ResNet-50
                if settings.model_name == "resnet50":
                    self.model = models.resnet50(pretrained=True)
                else:
                    raise ValueError(f"Unknown model: {settings.model_name}")

                self.model.to(self.device)
                self.model.eval()

                load_duration = time.time() - start_time

                # Update metrics
                set_model_loaded(settings.model_name, True)
                self._is_loaded = True

                # Estimate model memory (rough approximation)
                param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
                total_size = param_size + buffer_size
                set_model_memory(settings.model_name, total_size)

                logger.info(
                    "Model loaded successfully",
                    model_name=settings.model_name,
                    load_duration_seconds=round(load_duration, 2),
                    memory_bytes=total_size,
                    device=str(self.device)
                )

            except Exception as e:
                logger.error(
                    "Failed to load model",
                    model_name=settings.model_name,
                    error=str(e),
                    exc_info=True
                )
                set_model_loaded(settings.model_name, False)
                raise ModelInferenceException(f"Failed to load model: {str(e)}")

    def is_ready(self) -> bool:
        """
        Check if model is loaded and ready for inference.

        Returns:
            True if model is ready, False otherwise
        """
        return self._is_loaded and self.model is not None

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image bytes for model input.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed tensor

        Raises:
            InvalidInputException: If image is invalid
        """
        with TracedOperation("preprocess_image"):
            try:
                # Open image
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # Apply transforms
                tensor = self.transform(image)

                # Add batch dimension
                tensor = tensor.unsqueeze(0)

                return tensor.to(self.device)

            except Exception as e:
                logger.error("Failed to preprocess image", error=str(e))
                raise InvalidInputException(f"Invalid image: {str(e)}")

    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Run inference on image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Prediction dictionary with class, confidence, and top 5 predictions

        Raises:
            ModelNotLoadedException: If model is not loaded
            InvalidInputException: If image is invalid
            ModelInferenceException: If inference fails
        """
        if not self.is_ready():
            raise ModelNotLoadedException("Model is not loaded")

        with TracedOperation(
            "model_inference",
            {
                "model_name": settings.model_name,
                "device": str(self.device)
            }
        ):
            try:
                start_time = time.time()

                # Preprocess image
                input_tensor = self.preprocess_image(image_bytes)

                # Run inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)

                # Get top 5 predictions
                top5_prob, top5_idx = torch.topk(probabilities, 5)

                # Convert to list
                top5_predictions = [
                    {
                        "class_id": int(idx),
                        "class_name": IMAGENET_CLASSES.get(int(idx), f"class_{int(idx)}"),
                        "confidence": float(prob)
                    }
                    for prob, idx in zip(top5_prob, top5_idx)
                ]

                # Get top prediction
                top_prediction = top5_predictions[0]

                inference_duration = time.time() - start_time

                logger.info(
                    "Inference completed",
                    model_name=settings.model_name,
                    prediction_class=top_prediction["class_name"],
                    confidence=round(top_prediction["confidence"], 4),
                    inference_duration_ms=round(inference_duration * 1000, 2)
                )

                return {
                    "model_name": settings.model_name,
                    "prediction": top_prediction,
                    "top5": top5_predictions,
                    "inference_time_ms": round(inference_duration * 1000, 2)
                }

            except InvalidInputException:
                raise
            except Exception as e:
                logger.error(
                    "Inference failed",
                    model_name=settings.model_name,
                    error=str(e),
                    exc_info=True
                )
                raise ModelInferenceException(f"Inference failed: {str(e)}")

    def warmup(self):
        """
        Warmup model with dummy inference.

        This pre-allocates memory and compiles kernels.
        """
        if not self.is_ready():
            return

        logger.info("Warming up model", model_name=settings.model_name)

        try:
            # Create dummy image (random noise)
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_bytes = io.BytesIO()
            dummy_image.save(dummy_bytes, format='JPEG')
            dummy_bytes.seek(0)

            # Run inference
            self.predict(dummy_bytes.getvalue())

            logger.info("Model warmup complete", model_name=settings.model_name)

        except Exception as e:
            logger.warning(
                "Model warmup failed",
                model_name=settings.model_name,
                error=str(e)
            )


# Global model instance
_model_service: Optional[ModelInferenceService] = None


def get_model_service() -> ModelInferenceService:
    """
    Get or create the global model service instance.

    Returns:
        ModelInferenceService instance
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelInferenceService()
    return _model_service
