"""Simple image preprocessing."""

import torch
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image with RGB conversion."""
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed (handles grayscale and RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image
