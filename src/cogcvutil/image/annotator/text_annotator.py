from __future__ import annotations

import cv2
import numpy as np

from cogcvutil import save_image

"""Text Annotator."""


class TextAnnotator:
    def __init__(
        self,
        font: any = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1.0,
        font_color: tuple = (57, 255, 20),
        line_spacing: int = 10,
    ):
        """Initialize the image annotator.

        Args:
            image_path (str): Path to the image file.
            font_scale (float): Scale of the font size. Default is 3.0.
            font_color (tuple): Font color in BGR format. Default is neon green.

        """
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = max(
            2, int(font_scale)
        )  # Adjust thickness based on font scale
        self.line_spacing = line_spacing

    def insert_annotation(
        self,
        image: np.ndarray,
        text_annotations: list[str],
        position: str = "upper_left",
        save_path: str | None = None,
        auto_indexing: bool = False,
    ) -> np.ndarray:
        """Insert text_annotations to the current frame at the pre-specified position.

        Args:
            text_annotations (list[str]): List of text_annotations.
            position (str): The default position for text_annotations.
                            One of 'upper_left', 'upper_right', 'bottom_left', 'bottom_right'.
        """
        # Calculate the bounding box for the annotations
        text_height_total = 0
        max_text_width = 0
        for annotation in text_annotations:
            (text_width, text_height), _ = cv2.getTextSize(
                annotation, self.font, self.font_scale, self.font_thickness
            )
            text_height_total += text_height + self.line_spacing
            max_text_width = max(max_text_width, text_width)

        # Adjust starting position based on alignment
        if position.startswith("upper"):
            y = 0
        else:
            y = image.shape[0] - text_height_total

        for annotation in text_annotations:
            # Recalculate x position for right alignments
            (text_width, text_height), _ = cv2.getTextSize(
                annotation, self.font, self.font_scale, self.font_thickness
            )
            if position.endswith("left"):
                x = 10
            else:
                x = image.shape[1] - text_width - 10

            y += (
                text_height + self.line_spacing
            )  # Update y position for each annotation

            # Adding the text to image
            cv2.putText(
                image,
                annotation,
                (x, y),
                self.font,
                self.font_scale,
                self.font_color,
                self.font_thickness,
                cv2.LINE_AA,
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if save_path:
            save_image(image=image, path=save_path, auto_indexing=auto_indexing)
        return image
