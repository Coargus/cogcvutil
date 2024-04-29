from __future__ import annotations

import cv2
import numpy as np

from cogcvutil.common.converter.color import hex_to_bgr

"""Bounding Box Annotator."""


def visualize_bbox(
    image: np.ndarray,
    bboxes: list[list[int]],
    bbox_border_thickness: int = 1,
    bbox_border_color: str = "#FF0000",
) -> np.ndarray:
    """Draw rectangles around bounding boxes within an image.

    Args:
    - image (np.ndarray): A numpy array of the image.
    - bboxes (list[list[int]]): The bounding boxes in format [[x1, y1, x2, y2], ...].
    - bbox_border_thickness (int): The thickness of the border drawn around the bboxes.
    - bbox_border_color (str): The color of the border drawn around the bboxes, as a hex code.

    Returns:
    - np.ndarray: The image with bounding boxes annotated.
    """
    bbox_border_color_bgr = hex_to_bgr(bbox_border_color)
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integer
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            bbox_border_color_bgr,
            bbox_border_thickness,
        )

    return image
