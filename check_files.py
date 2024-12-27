import pandas as pd
import os
import cv2
from OpticalFlow import calculate_opticalFlow

DATASET_PATH = "/home/ibraa04/grad_project/output"
OPTICAL_FLOW_FOLDER = os.path.join(DATASET_PATH, "optical_flow")

# Load the CSV file
csv_file = os.path.join(DATASET_PATH, "interpolated.csv")
camera_csv = pd.read_csv(csv_file)
camera_csv = camera_csv[camera_csv['frame_id'] == "center_camera"]
camera_csv = camera_csv.reset_index(drop=True)

# Initialize counter
renamed_count = 0

# Iterate over the CSV and rename optical flow images
for index in range(1, len(camera_csv)):
    current_file_name = camera_csv.loc[index, 'filename']
    previous_file_name = camera_csv.loc[index - 1, 'filename']

    current_optical_file = f"{os.path.splitext(current_file_name)[0]}_optical.jpg"
    previous_optical_file = f"{os.path.splitext(previous_file_name)[0]}_optical.jpg"

    current_optical_path = os.path.join(OPTICAL_FLOW_FOLDER, current_optical_file)
    previous_optical_path = os.path.join(OPTICAL_FLOW_FOLDER, previous_optical_file)

    if os.path.exists(current_optical_path):
        os.rename(current_optical_path, previous_optical_path)
        print(f"Renamed {current_optical_path} to {previous_optical_path}")
        renamed_count += 1
    else:
        print(f"File not found: {current_optical_path}")

print(f"Renaming completed. Total renamed images: {renamed_count}")