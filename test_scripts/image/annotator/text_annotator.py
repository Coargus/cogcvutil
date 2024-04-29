from pathlib import Path

import cv2
from swarm_cv.image.annotator.text_annotator import TextAnnotator

ROOT_DIR = Path(__file__).parent.parent.parent.parent
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "sample_data"

if __name__ == "__main__":
    text_annotator = TextAnnotator()
    image_path = SAMPLE_DATA_DIR / "titanic.png"
    save_path = ROOT_DIR / "test_image_annotated.png"
    text_annotator.insert_annotation(
        image=cv2.imread(str(image_path)),
        text_annotations=["Hello, World!", "Hello, World!", "Hello, World!"],
        position="upper_right",
        save_path=str(save_path),
    )
    print("Text annotation added to the image.")
