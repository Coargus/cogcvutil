"""Image Filter Module."""

from __future__ import annotations

import cv2
import numpy as np

from cogcvutil.image.annotator.bounding_box import visualize_bbox

"""Image Filter Module."""


class ImageFilter:
    """Image Filter."""

    def gaussian_blur(self, image: np.ndarray, blur_radius: int) -> np.ndarray:
        """Blur an entire image.

        Args:
            image (np.ndarray): A numpy array of the image
            blur_radius (int): The radius (in pixels) to use in gaussian blurring

        Returns:
            np.ndarray: The fully blurred image
        """
        return cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

    def create_black_image(self, image: np.ndarray) -> np.ndarray:
        """Create a completely black image of the same size as the input image.

        Args:
            image (np.ndarray): A numpy array of the image.

        Returns:
            A completely black image of the same size.
        """
        shape_threshold = 3
        if len(image.shape) == shape_threshold:
            height, width, channels = image.shape
            black_image = np.zeros((height, width, channels), dtype=np.uint8)
        else:
            height, width = image.shape
            black_image = np.zeros((height, width), dtype=np.uint8)
        return black_image

    def apply_filter_to_bbox(  # noqa: PLR0913
        self,
        image: np.ndarray,
        bboxes: list[list],
        filter_type: str = "black",
        blur_radius: int = 31,
        bbox_border_thickness: int = 0,
        bbox_border_color: str = "#FF0000",
    ) -> np.ndarray:
        """Apply gaussian blur to bounding boxes within an image.

        Args:
            image (np.ndarray): A numpy array of the image
            bboxes (list[list]): The bounding boxes in format [[x1, y1, x2, y2], ...]
            filter_type (str): Types of filter to apply - either "black" or "blur"
            blur_radius (int): The radius (in pixels) to use in gaussian blurring
            bbox_border_thickness (int): The thickness of the border drawn around the bboxes - defaults to 0 (no border)
            bbox_border_color (str): The color of the border drawn around the bboxes, as a hex code - defaults to Blue

        Returns:
            np.ndarray: The final image, with bounding boxes blurred.
        """  # noqa: E501
        if filter_type == "black":
            filter_image = self.create_black_image(image)
        elif filter_type == "blur":
            if blur_radius % 2 == 0:
                blur_radius += 1
            filter_image = self.gaussian_blur(image, blur_radius)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for bbox in bboxes:
            x1, y1, x2, y2 = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        final_image = np.where(mask_3ch == (0, 0, 0), image, filter_image)

        if bbox_border_thickness > 0:
            final_image = visualize_bbox(
                final_image, bboxes, bbox_border_thickness, bbox_border_color
            )

        return final_image
