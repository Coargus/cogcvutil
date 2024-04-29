from __future__ import annotations

from pathlib import Path
from typing import Optional

import imageio
import numpy as np

from cogcvutil.image.common.utility.io_util import read_images_sorted


class VideoWriter:
    def __init__(
        self,
        save_dir: str,
        file_name: str,
        output_extension: str = "mp4",  # gif, mp4
        frame_rate: int = 20,
        read_image_from_dir: Optional[str | Path] = None,
        frame_sequence: Optional[np.ndarray] = None,
        codec: str = "libx264",  # Default codec for mp4
    ) -> None:
        """Initialize VideoWriter.

        Args:
            save_dir (str): Directory to save video.
            file_name (str): Name of the video file.
            output_extension (str, optional): Output video extension. Defaults to "mp4".
            frame_rate (int, optional): Frame rate of the video. Defaults to 20.
            read_image_from_dir (Optional[str | Path], optional): Directory to read images from. Defaults to None.
            frame_sequence (Optional[np.ndarray], optional): Frame sequence to write to video. Defaults to None.
            codec (str, optional): Video codec for encoding. Defaults to 'libx264'.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if not file_name.endswith((".mp4", ".gif")):
            file_name += f".{output_extension}"
        self.output_path = self.save_dir / file_name
        self.frame_rate = frame_rate
        if frame_sequence:
            self.frame_sequence = frame_sequence
        elif read_image_from_dir:
            self.frame_sequence = read_images_sorted(read_image_from_dir)
        else:
            self.frame_sequence = []

        self.codec = codec

        # Adjusting writer initialization based on output format
        if output_extension == "gif":
            self.video_writer = imageio.get_writer(
                str(self.output_path), fps=frame_rate
            )
        else:
            self.video_writer = imageio.get_writer(
                str(self.output_path), fps=frame_rate, codec=self.codec
            )

    def add_frame(self, frame: np.ndarray) -> None:
        self.frame_sequence.append(frame)

    def write(self, frame_sequence: Optional[np.ndarray] = None) -> None:
        """Write frame image to video."""
        if frame_sequence is not None:
            self.frame_sequence = frame_sequence
        assert self.frame_sequence, "Frame sequence is empty."

        for frame in self.frame_sequence:
            frame = frame.astype(np.uint8)
            self.video_writer.append_data(frame)

        self.video_writer.close()
