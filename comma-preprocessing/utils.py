import cv2
import numpy as np
import os

def convert_video_to_frames(video_path, output_path):
    """Convert video to frames and save them in the output path."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame to output directory
        frame_path = os.path.join(output_path, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    # Release video capture object
    cap.release()

    print(f"Converted video to {frame_count} frames")

#TODO add utility functions to convert angles from numpy array into csv format