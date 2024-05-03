from pathlib import Path

import cv2
from cogcvutil.image.filter.image_filter import ImageFilter

ROOT_DIR = Path(__file__).parent.parent.parent.parent
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "sample_data"

if __name__ == "__main__":
    image_filter = ImageFilter()
    image_path = SAMPLE_DATA_DIR / "titanic.png"
    input_image = cv2.imread(str(image_path))

    # Blur the image
    blurred_image = image_filter.apply_filter_to_bbox(
        image=input_image,
        bboxes=[
            [300, 300, 900, 900],
            [1500, 0, 2700, 1350],
        ],
        filter_type="blur",
        blur_radius=int(max(input_image.shape[0], input_image.shape[1]) / 40),
    )
    save_path = ROOT_DIR / "test_image_blurred.png"
    cv2.imwrite(str(save_path), blurred_image)

    # Blur the image and visualize bounding boxes
    blurred_and_bordered_image = image_filter.apply_filter_to_bbox(
        image=input_image,
        bboxes=[
            [300, 300, 900, 900],
            [1500, 0, 2700, 1350],
        ],
        blur_radius=int(max(input_image.shape[0], input_image.shape[1]) / 40),
        filter_type="blur",
        bbox_border_thickness=10,
        bbox_border_color="#FF0000",
    )
    save_path = ROOT_DIR / "test_image_blurred_and_bordered.png"
    cv2.imwrite(str(save_path), blurred_and_bordered_image)

    # Black out the image
    blacked_out = image_filter.apply_filter_to_bbox(
        image=input_image,
        bboxes=[
            [300, 300, 900, 900],
            [1500, 0, 2700, 1350],
        ],
        filter_type="black",
    )
    save_path = ROOT_DIR / "test_image_blacked_out.png"
    cv2.imwrite(str(save_path), blacked_out)
