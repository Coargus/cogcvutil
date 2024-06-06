"""Color Converter Module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


def hex_to_bgr(hex_color: str) -> tuple:
    """Convert a hex color to BGR (Blue, Green, Red).

    Args:
        hex_color (str): The color in hex format.

    Returns:
        tuple: The color in BGR format.
    """
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    return tuple(
        int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)
    )[::-1]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from BGR to RGB format.

    Args:
        image (np.ndarray): The image in BGR format.

    Returns:
        np.ndarray: The image in RGB format.
    """
    # Check if the image is in BGR format
    num_channel = 3
    if len(image.shape) == num_channel and image.shape[2] == num_channel:
        # Assuming it's BGR if it's a 3-channel image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Return the image as is if it's not BGR
    return image
