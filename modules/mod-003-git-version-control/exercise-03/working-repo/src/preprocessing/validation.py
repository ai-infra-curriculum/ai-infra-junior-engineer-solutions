"""
Input validation for image preprocessing.

Fixes null pointer issues when processing invalid images.
"""

from PIL import Image
from typing import Optional
import io


def validate_image(image_data: bytes) -> Optional[Image.Image]:
    """
    Validate and load image data safely.

    Args:
        image_data: Raw image bytes

    Returns:
        PIL Image or None if invalid
    """
    if not image_data:
        return None

    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as e:
        print(f"Invalid image data: {e}")
        return None
