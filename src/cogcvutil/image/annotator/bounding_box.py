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


def visualize_bbox_with_annotations(
    image: np.ndarray,
    bboxes: list[list[int]],
    labels: list[str],
    confs: list[float],
    bbox_border_thickness: int = 1,
    bbox_border_color: str = "#FF0000",
    font_scale: float = 0.5,
    font_thickness: int = 1,
    text_color: str = "#FFFFFF",
    text_loc: str = "up",
) -> np.ndarray:
    """Draw rectangles around bounding boxes within an image and annotate with labels and confidence scores.

    Args:
    - image (np.ndarray): A numpy array of the image.
    - bboxes (list[dict]): The bounding boxes with format [{"bbox": [x1, y1, x2, y2], "label": str, "conf": float}, ...].
    - bbox_border_thickness (int): The thickness of the border drawn around the bboxes.
    - bbox_border_color (str): The color of the border drawn around the bboxes, as a hex code.
    - font_scale (float): The scale of the font used for annotations.
    - font_thickness (int): The thickness of the font used for annotations.
    - text_color (str): The color of the text annotations, as a hex code.
    - text_loc (str): The location of the text annotations, either "up" or "down".

    Returns:
    - np.ndarray: The image with bounding boxes and annotations.
    """
    bbox_border_color_bgr = hex_to_bgr(bbox_border_color)
    text_color_bgr = hex_to_bgr(text_color)

    for idx, bbox_info in enumerate(bboxes):
        bbox = bboxes[idx]
        label = labels[idx]
        conf = confs[idx]

        x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integer
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            bbox_border_color_bgr,
            bbox_border_thickness,
        )

        text = f"{label}: {conf:.2f}"
        ((text_width, text_height), _) = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        if text_loc == "up":
            rec_x = (x1, y1 - text_height - 10)
            rec_y = (x1 + text_width, y1)
            text_xy = (x1, y1 - 5)
        else:
            rec_x = (x1, y2 - text_height - 10)
            rec_y = (x1 + text_width, y2)
            text_xy = (x1, y2 - 5)

        cv2.rectangle(
            image,
            rec_x,
            rec_y,
            bbox_border_color_bgr,
            -1,
        )
        cv2.putText(
            image,
            text,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color_bgr,
            font_thickness,
        )

    return image
