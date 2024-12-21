import cv2
import numpy as np
import os
import csv
from process_video import process_video

def create_output_directories(output_frames_dir: str) -> None:
    """
    Create the output directory for frames if it doesn't exist.

    Args:
        output_frames_dir (str): Path to the directory where output frames will be saved.

    Returns:
        None
    """
    os.makedirs(output_frames_dir, exist_ok=True)


def initialize_csv(csv_file_path: str) -> None:
    """
    Initialize the CSV file with headers.

    Args:
        csv_file_path (str): Path to the CSV file to initialize.

    Returns:
        None
    """
    with open(csv_file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['frame_id', 'steering_angle'])  # CSV header


def get_processed_log_paths(main_folder: str, nested_folder: str) -> tuple[str, str]:
    """
    Get paths for the processed log directory and video file.

    Args:
        main_folder (str): Path to the main folder containing subdirectories.
        nested_folder (str): Name of the nested folder to process.

    Returns:
        tuple[str, str]: Paths to the processed log directory and video file.
    """
    nested_folder_path = os.path.join(main_folder, nested_folder, 'processed_log')
    video_file = os.path.join(main_folder, nested_folder, 'video.hevc')  # Assuming videos are named 'video.hevc'
    return nested_folder_path, video_file


def load_steering_angles(nested_folder_path: str) -> np.ndarray | None:
    """
    Load steering angles from the processed log directory.

    Args:
        nested_folder_path (str): Path to the processed log directory.

    Returns:
        np.ndarray | None: Array of steering angles if the file exists, otherwise None.
    """
    steering_angle_path = os.path.join(nested_folder_path, 'CAN', 'steering_angle', 'value')

    if not os.path.isfile(steering_angle_path):
        print(f"Missing steering angle data: {steering_angle_path}")
        return None
    return np.load(steering_angle_path)


def extract_frames_from_video(video_file: str, steering_angle: np.ndarray, output_frames_dir: str, main_folder: str, nested_folder: str) -> iter:
    """
    Extract frames from a video and yield them along with their corresponding steering angles.

    Args:
        video_file (str): Path to the video file.
        steering_angle (np.ndarray): Array of steering angles.
        output_frames_dir (str): Path to the directory where frames will be saved.
        main_folder (str): Path to the main folder containing the video.
        nested_folder (str): Name of the nested folder containing the video.

    Yields:
        tuple[str, float]: Frame ID and its corresponding steering angle.
    """
    sample_rate = 4  # Extract every 4th frame
    cap = cv2.VideoCapture(video_file)
    max_steering_angles = len(steering_angle) // sample_rate
    local_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if local_frame_count >= max_steering_angles:
            break

        frame = cv2.resize(frame, (224, 224))
        frame_ID = f"{os.path.basename(main_folder)}_{nested_folder}_{local_frame_count}.jpg"
        frame_path = os.path.join(output_frames_dir, frame_ID)
        cv2.imwrite(frame_path, frame)

        angle_index = local_frame_count * sample_rate
        angle = steering_angle[angle_index]

        yield frame_ID, angle
        local_frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def write_to_csv(csv_file_path: str, frame_ID: str, angle: float) -> None:
    """
    Write the frame name and steering angle to the CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.
        frame_ID (str): Name of the frame file.
        angle (float): Steering angle corresponding to the frame.

    Returns:
        None
    """
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([frame_ID, angle])

# def process_video(main_folder: str, nested_folder: str, output_frames_dir: str, csv_file_path: str) -> tuple[float, float] | tuple[None, None]:
#     """
#     Process the video to extract frames and save them along with steering angles.

#     Args:
#         main_folder (str): Path to the main folder containing the video and logs.
#         nested_folder (str): Name of the nested folder to process.
#         output_frames_dir (str): Path to the directory where frames will be saved.
#         csv_file_path (str): Path to the CSV file where data will be logged.

#     Returns:
#         tuple[float, float] | tuple[None, None]: Minimum and maximum steering angles,
#         or (None, None) if processing fails.
#     """
#     nested_folder_path, video_file = get_processed_log_paths(main_folder, nested_folder)

#     if not os.path.exists(nested_folder_path) or not os.path.isfile(video_file):
#         print(f"Skipping {nested_folder}, missing processed log directory or video file.")
#         return None, None

#     steering_angle = load_steering_angles(nested_folder_path)
#     if steering_angle is None:
#         return None, None

#     min_angle = float('inf')
#     max_angle = float('-inf')

#     for frame_ID, angle in extract_frames_from_video(video_file, steering_angle, output_frames_dir, main_folder, nested_folder):
#         write_to_csv(csv_file_path, frame_ID, angle)
#         min_angle = min(min_angle, angle)
#         max_angle = max(max_angle, angle)

#     return min_angle, max_angle

def main():
    # Path to the parent directory where the main folders are located
    chunks_dir = r'D:\Mechatronics\Graduation Project\Let-Transformer-Be-a-Car\Example'

    # User input for the starting and ending index of nested folders to process
    start_index = int(input("Enter the starting index of the nested folders to process: "))
    end_index = int(input("Enter the ending index of the nested folders to process: "))

    # Path to save all preprocessed frames directly in this directory
    output_frames_dir = 'preprocessed_frames'
    # create_output_directories(output_frames_dir)

    # Path for the combined CSV file
    csv_file_path = 'steering_data_combined.csv'
    # initialize_csv(csv_file_path)

    # Variables to keep track of global min and max steering angles
    global_min_angle = float('inf')
    global_max_angle = float('-inf')
    folders_processed = 0
    total_nested_folders = []

    # Gather all nested folders across main folders
    for chunk in os.listdir(chunks_dir):
        parent_dir = os.path.join(chunks_dir, chunk)
        if not os.path.isdir(parent_dir):
            continue
        if chunk == "Chunk_1":
            continue

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
        print(f"{i} folders from {min(end_index, len(total_nested_folders))} are processed.")

    # Print the global minimum and maximum steering angles
    if global_min_angle != float('inf') and global_max_angle != float('-inf'):
        print(f"Global minimum steering angle: {global_min_angle}")
        print(f"Global maximum steering angle: {global_max_angle}")
    else:
        print("No valid steering angle data found in the specified range.")

if __name__ == "__main__":
    main()
