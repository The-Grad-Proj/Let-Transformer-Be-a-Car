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

def convert_npy_to_csv(npy_file_path, csv_file_path):
    # Load the Numpy array from the .npy file
    data = np.load(npy_file_path)

    # Check if the data is a structured array (with named columns) or a regular array
    if isinstance(data, np.recarray) or data.dtype.names is not None:
        # Structured array: Convert to pandas DataFrame for column handling
        df = pd.DataFrame(data)
    else:
        # Regular Numpy array: Convert to DataFrame without column names
        df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")