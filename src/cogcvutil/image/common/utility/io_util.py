"""Image Input / Output Utility Module."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import numpy as np


def numeric_sort_key(s: str) -> list:
    """Natural sort key function for sorting filenames.

    Args:
    - s (str): The filename to be sorted.

    Extract numeric parts as integers and non-numeric parts
    as text from the filename. This allows for natural sorting
    based on numerical values.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_image(
    path: str, format_type: np.ndarray = "numpy"
) -> np.ndarray | Image.Image | any:
    """Reads an image and returns it in the specified format.

    Args:
        path (str): The path to the image file.
        format_type (str): The format to return the image in.

    Supported formats: 'numpy', 'torch', 'PIL'.


    Return:
        if numpy of torch, shape of image is (H, W, C)

    Raises FileNotFoundError if the image can't be found,
           ValueError for unsupported formats.
    """
    try:
        # Load the image with PIL
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        msg = "The image file was not found at the specified path."
        raise FileNotFoundError(msg)  # noqa: B904

    img_array = np.asarray(img)

    if format_type == "numpy":
        return img_array

    elif format_type == "torch":  # noqa: RET505
        import torch

        # Convert HWC to CHW format and ensure it is a float tensor normalized between 0 and 1
        return torch.from_numpy(img_array)
    elif format_type == "PIL":
        return img
    else:
        msg = (
            "Unsupported format type. Choose 'numpy', 'torch', 'PIL', or 'cv'."
        )
        raise ValueError(msg)


def read_images_sorted(directory: str) -> list[np.ndarray]:
    """Reads images and sort them in natural numeric order.

    Only reads files with the extensions .png, .jpg, and .jpeg.

    Args:
    - directory (str): The directory path containing the images.

    Returns:
    - List[np.ndarray]: A list of images as numpy arrays in RGB format.
    """
    files = os.listdir(directory)
    # Sort files using the numeric_sort_key function
    sorted_files = sorted(files, key=numeric_sort_key)
    images = []
    for file in sorted_files:
        if file.endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Extend or modify as needed
            path = os.path.join(directory, file)
            image = cv2.imread(path)
            # Convert from BGR to RGB
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images
