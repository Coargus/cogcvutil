import numpy as np
from cogcvutil.video.writer.images_to_video import VideoWriter


def main():
    save_dir = "save_data_path"  # Update this to a valid directory
    file_name = "random_video"
    output_extension = "gif"  # Can be "mp4" or "gif"
    frame_rate = 20
    num_frames = 50  # Number of frames in the video

    # Initialize VideoWriter
    video_writer = VideoWriter(
        save_dir, file_name, output_extension, frame_rate
    )

    # Generate and add random frames
    for _ in range(num_frames):
        # Generate a random grayscale frame with shape (100, 100)
        frame = np.random.rand(100, 100) * 255
        # Convert the grayscale frame to a 3-channel (RGB) frame
        frame_rgb = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        # Ensure the frame is in the correct data type
        frame_rgb = frame_rgb.astype(np.uint8)
        # Add the frame
        assert frame_rgb.shape == (
            100,
            100,
            3,
        ), f"Incorrect frame shape: {frame_rgb.shape}"
        video_writer.add_frame(frame_rgb)

    # Write the video
    video_writer.write()


if __name__ == "__main__":
    main()
