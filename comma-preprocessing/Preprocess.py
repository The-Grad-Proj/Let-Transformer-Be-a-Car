import cv2
import numpy as np
import os
import csv

# Path to the parent directory where the main folders are located
parent_dir = r'D:\Mechatronics\Graduation Project\Let-Transformer-Be-a-Car\Example'

# Ask the user how many folders to iterate through
num_folders_to_process = 4

# Path to save all preprocessed frames directly in this directory
output_frames_dir = 'preprocessed_frames'
os.makedirs(output_frames_dir, exist_ok=True)

# Path for the combined CSV file
csv_file_path = 'steering_data_combined.csv'

# Open the combined CSV file for writing
with open(csv_file_path, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['frame_id', 'steering_angle'])  # CSV header

# Global frame counter to ensure unique frame IDs across all folders
global_frame_id = 0

# Iterate through each subfolder in the parent directory, limited by user input
folders_processed = 0
for main_folder in os.listdir(parent_dir):
    main_folder_path = os.path.join(parent_dir, main_folder)
    
    # Check if it's a directory
    if not os.path.isdir(main_folder_path):
        continue

    # Iterate through the nested folders (like '3')
    for nested_folder in os.listdir(main_folder_path):
        nested_folder_path = os.path.join(main_folder_path, nested_folder, 'processed_log')

        # Define paths for current folder
        video_file = os.path.join(main_folder_path, nested_folder, 'video.hevc')  # Assuming videos are named 'video.hevc'

        # Check if the required processed_log directory and video file exist
        if not os.path.exists(nested_folder_path) or not os.path.isfile(video_file):
            print(f"Skipping {nested_folder}, missing processed log directory or video file.")
            continue

        # Load steering angle and timestamp arrays
        steering_angle_path = os.path.join(nested_folder_path, 'CAN', 'steering_angle', 'value')
        steering_time_path = os.path.join(nested_folder_path, 'CAN', 'steering_angle', 't')

        if not os.path.isfile(steering_angle_path) or not os.path.isfile(steering_time_path):
            print(f"Skipping {nested_folder}, missing steering angle data.")
            continue

        steering_angle = np.load(steering_angle_path)
        steering_time = np.load(steering_time_path)

        # Load video
        cap = cv2.VideoCapture(video_file)

        # Create a variable to keep track of the maximum number of angles we can use
        max_steering_angles = len(steering_angle) // 4

        processed_frame_count = 0

        # Loop through video and extract frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # If we've processed enough frames, stop
            if processed_frame_count >= max_steering_angles:
                break

            # Save frame with the global frame ID directly in the output directory
            frame_path = os.path.join(output_frames_dir, f'frame_{global_frame_id}.jpg')
            cv2.imwrite(frame_path, frame)

            # Get the corresponding steering angle based on the 1:4 sampling
            angle_index = processed_frame_count * 4
            angle = steering_angle[angle_index]

            # Write the global frame ID and steering angle to the combined CSV
            with open(csv_file_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([global_frame_id, angle])

            # Increment the global frame ID and processed frame count
            global_frame_id += 1
            processed_frame_count += 1

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

        folders_processed += 1

        # Stop processing if the specified number of folders has been reached
        if folders_processed >= num_folders_to_process:
            break

print("All folders processed. Combined data written to", csv_file_path)
