"""
Image Classification Model

Wrapper for PyTorch image classification models with consistent interface
for inference.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Dict, Optional
import numpy as np

from utils.logging import get_logger


logger = get_logger(__name__)


class ImageClassifier:
    """
    Image classification model wrapper

    Provides a simple interface for loading pre-trained models and
    performing inference with confidence scores.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 1000,
        device: Optional[str] = None
    ):
        """
        Initialize the image classifier

        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_name = model_name
        self.num_classes = num_classes

        # Determine device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Load class labels
        self.class_labels = self._load_imagenet_labels()

        logger.info(
            f"Initialized {model_name} with {num_classes} classes"
        )

    def _load_model(self) -> nn.Module:
        """
        Load pre-trained model

        Returns:
            PyTorch model in eval mode

        Raises:
            ValueError: If model name is not supported
        """
        try:
            if self.model_name == "resnet50":
                model = models.resnet50(pretrained=True)
            elif self.model_name == "resnet18":
                model = models.resnet18(pretrained=True)
            elif self.model_name == "mobilenet_v2":
                model = models.mobilenet_v2(pretrained=True)
            elif self.model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=True)
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_name}. "
                    f"Supported: resnet50, resnet18, mobilenet_v2, efficientnet_b0"
                )

            logger.info(f"Loaded pre-trained {self.model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_imagenet_labels(self) -> List[str]:
        """
        Load ImageNet class labels

        Returns:
            List of class label strings

        Note:
            In production, load from a JSON file. This is a subset for demo.
        """
        # Subset of ImageNet classes for demonstration
        # In production, load full list from file
        return [
            "tench", "goldfish", "great_white_shark", "tiger_shark",
            "hammerhead", "electric_ray", "stingray", "cock", "hen",
            "ostrich", "brambling", "goldfinch", "house_finch", "junco",
            "indigo_bunting", "robin", "bulbul", "jay", "magpie",
            "chickadee", "water_ouzel", "kite", "bald_eagle",
            "vulture", "great_grey_owl", "european_fire_salamander",
            "common_newt", "eft", "spotted_salamander", "axolotl",
            "bullfrog", "tree_frog", "tailed_frog", "loggerhead",
            "leatherback_turtle", "mud_turtle", "terrapin", "box_turtle",
            "banded_gecko", "common_iguana", "American_chameleon",
            "whiptail", "agama", "frilled_lizard", "alligator_lizard",
            "Gila_monster", "green_lizard", "African_chameleon",
            "Komodo_dragon", "African_crocodile", "American_alligator",
            "triceratops", "thunder_snake", "ringneck_snake",
            "hognose_snake", "green_snake", "king_snake", "garter_snake",
            "water_snake", "vine_snake", "night_snake", "boa_constrictor",
            "rock_python", "Indian_cobra", "green_mamba", "sea_snake",
            "horned_viper", "diamondback", "sidewinder", "trilobite",
            "harvestman", "scorpion", "black_and_gold_garden_spider",
            "barn_spider", "garden_spider", "black_widow", "tarantula",
            "wolf_spider", "tick", "centipede", "black_grouse",
            "ptarmigan", "ruffed_grouse", "prairie_chicken", "peacock",
            "quail", "partridge", "African_grey", "macaw",
            "sulphur-crested_cockatoo", "lorikeet", "coucal", "bee_eater",
            "hornbill", "hummingbird", "jacamar", "toucan", "drake",
            "red-breasted_merganser", "goose", "black_swan", "tusker",
            "echidna", "platypus", "wallaby", "koala", "wombat",
            "jellyfish", "sea_anemone", "brain_coral", "flatworm",
            "nematode", "conch", "snail", "slug", "sea_slug", "chiton",
            "chambered_nautilus", "Dungeness_crab", "rock_crab",
            "fiddler_crab", "king_crab", "American_lobster",
            "spiny_lobster", "crayfish", "hermit_crab", "isopod",
            "white_stork", "black_stork", "spoonbill", "flamingo",
            "little_blue_heron", "American_egret", "bittern", "crane",
            "limpkin", "European_gallinule", "American_coot", "bustard",
            "ruddy_turnstone", "red-backed_sandpiper", "redshank",
            "dowitcher", "oystercatcher", "pelican", "king_penguin"
        ] + ["class_" + str(i) for i in range(150, 1000)]  # Placeholder

    def predict(
        self,
        image_tensor: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Perform inference on a single image

        Args:
            image_tensor: Preprocessed image tensor [C, H, W]
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class names and confidence scores
        """
        try:
            # Add batch dimension if needed
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # Move to device
            image_tensor = image_tensor.to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)

            # Format results
            predictions = []
            for prob, idx in zip(
                top_probs[0].cpu().numpy(),
                top_indices[0].cpu().numpy()
            ):
                predictions.append({
                    "class": self.class_labels[idx],
                    "confidence": float(prob),
                    "class_id": int(idx)
                })

            return predictions

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise

    def predict_batch(
        self,
        image_tensors: List[torch.Tensor],
        top_k: int = 5
    ) -> List[List[Dict[str, any]]]:
        """
        Perform inference on multiple images

        Args:
            image_tensors: List of preprocessed image tensors
            top_k: Number of top predictions per image

        Returns:
            List of prediction lists, one per image
        """
        # Stack into batch
        batch = torch.stack(image_tensors).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get top k for each image
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Format results for each image
        all_predictions = []
        for probs, indices in zip(top_probs, top_indices):
            predictions = []
            for prob, idx in zip(probs.cpu().numpy(), indices.cpu().numpy()):
                predictions.append({
                    "class": self.class_labels[idx],
                    "confidence": float(prob),
                    "class_id": int(idx)
                })
            all_predictions.append(predictions)

        return all_predictions

    def get_classes(self) -> List[str]:
        """
        Get list of all class labels

        Returns:
            List of class names
        """
        return self.class_labels

    def get_num_parameters(self) -> int:
        """
        Count total number of model parameters

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameters(self) -> int:
        """
        Count trainable parameters

        Returns:
            Trainable parameter count
        """
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def get_model_size_mb(self) -> float:
        """
        Estimate model size in megabytes

        Returns:
            Model size in MB
        """
        param_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        )
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def __repr__(self) -> str:
        return (
            f"ImageClassifier(model={self.model_name}, "
            f"classes={self.num_classes}, device={self.device})"
        )
