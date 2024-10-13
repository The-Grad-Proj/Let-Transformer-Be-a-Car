import cv2
import numpy as np
import os
import csv

# Paths to the processed log directory and video file
processed_log_dir = r'C:\Grad Proj\Let-Transformer-Be-a-Car-cc03bd2a4ac404d9883aad5f3f1da73cb67ef026\Example\b0c9d2329ad1606b_2018-07-27--06-03-57\4\processed_log'
video_file = r'C:\Grad Proj\Let-Transformer-Be-a-Car-cc03bd2a4ac404d9883aad5f3f1da73cb67ef026\Example\b0c9d2329ad1606b_2018-07-27--06-03-57\4\video.hevc'

# Load steering angle and timestamp arrays
steering_angle = np.load(os.path.join(processed_log_dir, 'CAN', 'steering_angle', 'value'))
steering_time = np.load(os.path.join(processed_log_dir, 'CAN', 'steering_angle', 't'))

# Load video
cap = cv2.VideoCapture(video_file)

# Get frame rate of the video (frames per second)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Path to save preprocessed frames
os.makedirs('preprocessed_frames', exist_ok=True)

frame_count = 0
processed_frame_count = 0

# Prepare CSV file to write
csv_file_path = 'steering_data.csv'
with open(csv_file_path, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['frame_id', 'steering_angle'])  # CSV header

# Calculate the number of angles we can use
max_steering_angles = len(steering_angle) // 4

# Loop through video and extract frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # If we've processed enough frames, stop
    if processed_frame_count >= max_steering_angles:
        break

    # Save frame with the same ID as the processed frame count
    frame_id = processed_frame_count
    frame_path = f'preprocessed_frames/frame_{frame_id}.jpg'
    cv2.imwrite(frame_path, frame)

    # Get the corresponding steering angle based on the 1:4 sampling
    angle_index = processed_frame_count * 4
    angle = steering_angle[angle_index]

    # Write frame ID and steering angle to CSV
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([frame_id, angle])  # Only frame_id and steering_angle

    processed_frame_count += 1  # Increment processed frame count

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

print(f"Data has been written to {csv_file_path}.")
