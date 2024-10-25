import cv2
import numpy as np
import os
import csv

def create_output_directories(output_frames_dir):
    """Create the output directory for frames if it doesn't exist."""
    os.makedirs(output_frames_dir, exist_ok=True)

def initialize_csv(csv_file_path):
    """Initialize the CSV file with headers."""
    with open(csv_file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['frame_ID', 'steering_angle'])  # CSV header

def get_processed_log_paths(main_folder, nested_folder):
    """Get paths for the processed log directory and video file."""
    nested_folder_path = os.path.join(main_folder, nested_folder, 'processed_log')
    video_file = os.path.join(main_folder, nested_folder, 'video.hevc')  # Assuming videos are named 'video.hevc'
    return nested_folder_path, video_file

def load_steering_angles(nested_folder_path):
    """Load steering angles from the processed log directory."""
    steering_angle_path = os.path.join(nested_folder_path, 'CAN', 'steering_angle', 'value')
    
    if not os.path.isfile(steering_angle_path):
        print(f"Missing steering angle data: {steering_angle_path}")
        return None
    
    return np.load(steering_angle_path)

def extract_frames_from_video(video_file, steering_angle, output_frames_dir, main_folder, nested_folder):
    """Extract frames from video and save them with steering angles."""
    cap = cv2.VideoCapture(video_file)

    # Calculate the maximum number of angles we can use
    max_steering_angles = len(steering_angle) // 4
    local_frame_count = 0

    # Loop through video and extract frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # If we've processed enough frames, stop
        if local_frame_count >= max_steering_angles:
            break

        # Resizing the frame
        frame = cv2.resize(frame, (224, 224))

        # Create the frame name
        frame_ID = f"{os.path.basename(main_folder)}_{nested_folder}_{local_frame_count}.jpg"
        frame_path = os.path.join(output_frames_dir, frame_ID)
        cv2.imwrite(frame_path, frame)

        # Get the corresponding steering angle based on the 1:4 sampling
        angle_index = local_frame_count * 4
        angle = steering_angle[angle_index]

        yield frame_ID, angle  # Yielding frame name and angle to be written to CSV

        local_frame_count += 1

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

def write_to_csv(csv_file_path, frame_ID, angle):
    """Write the frame name and steering angle to the CSV file."""
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([frame_ID, angle])

def process_video(main_folder, nested_folder, output_frames_dir, csv_file_path):
    """Process the video to extract frames and save them along with steering angles."""
    nested_folder_path, video_file = get_processed_log_paths(main_folder, nested_folder)

    # Check if the required processed_log directory and video file exist
    if not os.path.exists(nested_folder_path) or not os.path.isfile(video_file):
        print(f"Skipping {nested_folder}, missing processed log directory or video file.")
        return None, None

    steering_angle = load_steering_angles(nested_folder_path)
    if steering_angle is None:
        return None, None

    min_angle = float('inf')
    max_angle = float('-inf')

    for frame_ID, angle in extract_frames_from_video(video_file, steering_angle, output_frames_dir, main_folder, nested_folder):
        write_to_csv(csv_file_path, frame_ID, angle)
        min_angle = min(min_angle, angle)
        max_angle = max(max_angle, angle)

    return min_angle, max_angle

def main():
    # Path to the parent directory where the main folders are located
    parent_dir = r'D:\Mechatronics\Graduation Project\Let-Transformer-Be-a-Car\Example'

    # User input for the starting and ending index of nested folders to process
    start_index = int(input("Enter the starting index of the nested folders to process: "))
    end_index = int(input("Enter the ending index of the nested folders to process: "))

    # Path to save all preprocessed frames directly in this directory
    output_frames_dir = 'preprocessed_frames'
    create_output_directories(output_frames_dir)

    # Path for the combined CSV file
    csv_file_path = 'steering_data_combined.csv'
    initialize_csv(csv_file_path)

    # Variables to keep track of global min and max steering angles
    global_min_angle = float('inf')
    global_max_angle = float('-inf')
    folders_processed = 0
    total_nested_folders = []

    # Gather all nested folders across main folders
    for main_folder in os.listdir(parent_dir):
        main_folder_path = os.path.join(parent_dir, main_folder)
        if not os.path.isdir(main_folder_path):
            continue

        nested_folders = os.listdir(main_folder_path)
        for nested_folder in nested_folders:
            total_nested_folders.append((main_folder_path, nested_folder))

    # Process the nested folders within the specified range
    for i in range(start_index, min(end_index, len(total_nested_folders))):
        main_folder_path, nested_folder = total_nested_folders[i]
        min_angle, max_angle = process_video(main_folder_path, nested_folder, output_frames_dir, csv_file_path)

        if min_angle is not None and max_angle is not None:
            global_min_angle = min(global_min_angle, min_angle)
            global_max_angle = max(global_max_angle, max_angle)

        folders_processed += 1

    # Print the global minimum and maximum steering angles
    if global_min_angle != float('inf') and global_max_angle != float('-inf'):
        print(f"Global minimum steering angle: {global_min_angle}")
        print(f"Global maximum steering angle: {global_max_angle}")
    else:
        print("No valid steering angle data found in the specified range.")

if __name__ == "__main__":
    main()
