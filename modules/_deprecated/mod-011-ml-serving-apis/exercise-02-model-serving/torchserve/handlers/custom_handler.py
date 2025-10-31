"""Custom TorchServe handler for ML model inference."""

import json
import logging
import os
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class CustomMLHandler(BaseHandler):
    """
    Custom handler for ML model serving with TorchServe.

    This handler demonstrates:
    - Custom preprocessing
    - Model inference
    - Custom postprocessing
    - Error handling
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Initialize model.

        Args:
            context: TorchServe context
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Load model
        try:
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)

            # Load TorchScript model
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
            self.model.eval()

            logger.info(f"Model loaded from {model_pt_path}")
            self.initialized = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess(self, requests):
        """
        Preprocess input data.

        Args:
            requests: List of request data

        Returns:
            Preprocessed tensor
        """
        inputs = []

        for request in requests:
            # Get data from request
            data = request.get("data") or request.get("body")

            # Parse JSON if string
            if isinstance(data, str):
                data = json.loads(data)

            # Extract features
            if isinstance(data, dict):
                features = data.get("features", data.get("data"))
            else:
                features = data

            # Convert to tensor
            tensor = torch.tensor(features, dtype=torch.float32)

            # Handle single vs batch
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)

            inputs.append(tensor)

        # Stack batch
        batch_tensor = torch.cat(inputs, dim=0).to(self.device)

        logger.info(f"Preprocessed batch shape: {batch_tensor.shape}")

        return batch_tensor

    def inference(self, data):
        """
        Run model inference.

        Args:
            data: Preprocessed input tensor

        Returns:
            Model predictions
        """
        try:
            with torch.no_grad():
                predictions = self.model(data)

            logger.info(f"Inference completed. Output shape: {predictions.shape}")

            return predictions

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def postprocess(self, inference_output):
        """
        Postprocess model predictions.

        Args:
            inference_output: Raw model output

        Returns:
            Formatted predictions
        """
        # Move to CPU and convert to numpy
        predictions = inference_output.cpu().numpy()

        # Format output
        results = []
        for pred in predictions:
            # Assuming binary classification
            if len(pred) == 2:
                predicted_class = int(pred.argmax())
                probability = pred.tolist()

                results.append({
                    "prediction": predicted_class,
                    "probability": probability,
                    "confidence": float(max(probability))
                })
            else:
                results.append({
                    "prediction": pred.tolist()
                })

        logger.info(f"Postprocessed {len(results)} predictions")

        return results
