import numpy as np
import os
import cv2
import csv
from scipy.interpolate import interp1d


def process_video(main_folder: str, nested_folder: str, output_frames_dir: str, csv_file_path: str) -> None:
    """
    Process the video to extract frames and interpolate steering angles and speeds to match frame timestamps.

    Args:
        main_folder (str): Path to the main folder containing the video and logs.
        nested_folder (str): Name of the nested folder to process.
        output_frames_dir (str): Path to the directory where frames will be saved.
        csv_file_path (str): Path to the CSV file where data will be logged.

    Returns:
        None
    """
    # Define paths
    nested_folder_path = os.path.join(main_folder, nested_folder, 'processed_log')
    video_file = os.path.join(main_folder, nested_folder, 'video.hevc')
    
    # Define paths for values and their timestamps
    steering_value_path = os.path.join(nested_folder_path, 'CAN', 'steering_angle', 'value')
    steering_time_path = os.path.join(nested_folder_path, 'CAN', 'steering_angle', 't')
    speed_value_path = os.path.join(nested_folder_path, 'CAN', 'speed', 'value')
    speed_time_path = os.path.join(nested_folder_path, 'CAN', 'speed', 't')
    frame_times_path = os.path.join(main_folder, nested_folder, 'global_pose', 'frame_times')

    # Check if required files and directories exist
    required_files = [video_file, steering_value_path, steering_time_path, 
                     speed_value_path, speed_time_path, frame_times_path]
    
    if not all(os.path.isfile(f) for f in required_files):
        print(f"Missing required files in {nested_folder}")
        return

    try:
        # Load data and timestamps
        steering_angles = np.load(steering_value_path)
        steering_times = np.load(steering_time_path)
        speeds = np.load(speed_value_path).flatten()  # Flatten in case of 2D array
        speed_times = np.load(speed_time_path)
        frame_times = np.load(frame_times_path)

        # Create interpolation functions
        steering_interpolator = interp1d(steering_times, steering_angles, 
                                       kind='linear', fill_value='extrapolate')
        speed_interpolator = interp1d(speed_times, speeds, 
                                    kind='linear', fill_value='extrapolate')

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Unable to open video file: {video_file}")
            return

        # Create output frames directory if not exists
        os.makedirs(output_frames_dir, exist_ok=True)

        # Process frames
        rows = []
        frame_idx = 0
        min_angle = float('inf')
        max_angle = float('-inf')

        while cap.isOpened() and frame_idx < len(frame_times):
            ret, frame = cap.read()
            if not ret:
                break

            # Get timestamp for current frame
            timestamp = frame_times[frame_idx]
            
            try:
                # Interpolate steering angle and speed for this timestamp
                interpolated_angle = float(steering_interpolator(timestamp))
                interpolated_speed = float(speed_interpolator(timestamp))
                
                # Update min and max angles
                min_angle = min(min_angle, interpolated_angle)
                max_angle = max(max_angle, interpolated_angle)
                
                # Resize and save frame
                frame = cv2.resize(frame, (224, 224))
                frame_id = f"{nested_folder}_{timestamp}"
                frame_path = os.path.join(output_frames_dir, f"{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Add to rows for CSV
                rows.append([
                    timestamp,
                    "center_camera",
                    frame_path,
                    interpolated_angle,
                    interpolated_speed
                ])
                
            except ValueError as e:
                print(f"Warning: Interpolation failed for frame {frame_idx}: {e}")
                
            frame_idx += 1

        cap.release()

        # Write all data to the CSV
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(rows)

        print(f"Processed {frame_idx} frames from {nested_folder}")
        print(f"Steering angle range: {min_angle:.2f} to {max_angle:.2f}")
        print(f"Original data lengths:")
        print(f"  Steering angles: {len(steering_angles)}")
        print(f"  Speeds: {len(speeds)}")
        print(f"  Frame times: {len(frame_times)}")
        print("-------------------------------------------------------------")

        return min_angle, max_angle

    except Exception as e:
        print(f"Error processing {nested_folder}: {e}")
        return None, None