import os
import cv2
import numpy as np
import pandas as pd

def convert_video_to_frames_with_data(video_path, output_path, npy_file_path, csv_file_path):
    """Convert video to frames, save them, and create a CSV with frame count and corresponding NumPy data."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the Numpy array with corresponding data
    data = np.load(npy_file_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Initialize frame counter and prepare an empty list for CSV data
    frame_count = 0
    csv_data = []

    # Read video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame to output directory
        frame_path = os.path.join(output_path, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Extract corresponding data from the Numpy array
        if frame_count < len(data):
            corresponding_data = data[frame_count]
        else:
            print(f"Warning: More frames than data in {npy_file_path}. Stopping.")
            break

        # If corresponding_data is a scalar, wrap it in a list
        if np.isscalar(corresponding_data):
            corresponding_data = [corresponding_data]  # Wrap the scalar in a list

        # Add frame count and corresponding data to CSV data
        csv_data.append([frame_count] + list(corresponding_data))

        # Increment frame counter
        frame_count += 1

    # Release video capture object
    cap.release()

    # Save frame count and corresponding data to CSV
    column_names = ['frame_count'] + [f"data_{i}" for i in range(len(corresponding_data))]  # Adjust based on data shape
    df = pd.DataFrame(csv_data, columns=column_names)
    df.to_csv(csv_file_path, index=False)

    print(f"Converted video to {frame_count} frames and saved corresponding data to {csv_file_path}")

# Example usage:
video_file = '/home/norhan/comma_pre/b0c9d2329ad1606b|2018-07-27--06-03-57/3/video.hevc'
output_dir = '/home/norhan/comma_pre/output3'
npy_file = '/home/norhan/comma_pre/b0c9d2329ad1606b|2018-07-27--06-03-57/3/processed_log/CAN/steering_angle/value'
csv_file = 'output_with_frame_data.csv'

convert_video_to_frames_with_data(video_file, output_dir, npy_file, csv_file)
