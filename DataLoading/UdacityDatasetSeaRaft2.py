import os
import cv2
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
import sys
sys.path.append('/home/norhan/SEA-RAFT/core')
sys.path.append('/home/norhan/Let-Transformer-Be-a-Car/DataLoading/aug_utils.py')
sys.path.append('/home/norhan/SEA-RAFT/config')
import argparse
from raft import RAFT
from .aug_utils import apply_augs
from utils.utils import load_ckpt
from parser import parse_args
from torchvision import transforms
from utils.flow_viz import flow_to_image

class UdacityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, select_camera=None, slice_frames=None,
                 select_ratio=1.0, select_range=None, optical_flow=True, seq_len=0, img_size=(224, 224),
                 model_path='/home/norhan/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth', config_path='/home/norhan/SEA-RAFT/config/eval/spring-M.json'):
        
        # Ensure valid selection ratio
        assert -1.0 <= select_ratio <= 1.0, "select_ratio must be between -1.0 and 1.0"
        
        # Dataset attributes
        self.seq_len = seq_len
        self.img_size = img_size
        self.root_dir = root_dir
        self.transform = transform
        self.optical_flow = optical_flow

        # Load CSV and apply selection filters
        camera_csv = pd.read_csv(csv_file)
        if select_camera:
            assert select_camera in ['left_camera', 'right_camera', 'center_camera'], "Invalid camera selection"
            camera_csv = camera_csv[camera_csv['frame_id'] == select_camera]
        
        csv_len = len(camera_csv)
        if slice_frames:
            csv_selected = camera_csv[0:0]
            for start_idx in range(0, csv_len, slice_frames):
                if select_ratio > 0:
                    end_idx = int(start_idx + slice_frames * select_ratio)
                else:
                    start_idx, end_idx = int(start_idx + slice_frames * (1 + select_ratio)), start_idx + slice_frames
                end_idx = min(end_idx, csv_len)
                start_idx = min(start_idx, csv_len)
                csv_selected = csv_selected.append(camera_csv[start_idx:end_idx])
            self.camera_csv = csv_selected
        elif select_range:
            self.camera_csv = camera_csv.iloc[select_range[0]: select_range[1]]
        else:
            self.camera_csv = camera_csv

        # Load SEA-RAFT model
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)        
        args = parse_args(parser)       # Parse SEA-RAFT configuration file
        self.model = RAFT(args)                # Initialize the SEA-RAFT model with parsed configuration
        load_ckpt(self.model, model_path)      # Load pretrained model weights
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.eval()                      # Set model to evaluation mode

        # Store mean and std for certain fields
        self.mean = {key: np.mean(camera_csv[key]) for key in ['angle', 'torque', 'speed']}
        self.std = {key: np.std(camera_csv[key]) for key in ['angle', 'torque', 'speed']}

    def __len__(self):
        return len(self.camera_csv)

    def forward_flow(self, image1, image2):
        """Compute optical flow between two images using SEA-RAFT."""
        output = self.model(image1, image2, iters=20, test_mode=True)  # Configurable iterations
        flow_final = output['flow'][-1]  # Get final optical flow output
        return flow_final

    def read_data_single(self, idx, augs):
        path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25, :, :]  # Crop the image as needed
        original_img = image.copy()

        if self.optical_flow:
            # Load previous image for optical flow computation
            if idx != 0:
                path_prev = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx - 1])
                prev = cv2.imread(path_prev)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25, :, :]
            else:
                prev = original_img.copy()

            # Prepare images as tensors for SEA-RAFT
            image1 = torch.tensor(prev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.model.device)
            image2 = torch.tensor(original_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.model.device)

            # Run SEA-RAFT to get flow
            with torch.no_grad():
                flow = self.forward_flow(image1, image2)

            # Convert SEA-RAFT flow to HSV format for visualization
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
            mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
            hsv = np.zeros_like(original_img, dtype=np.uint8)
            hsv[..., 1] = 255  # Set saturation to maximum
            hsv[..., 0] = ang * 180 / np.pi / 2  # Set angle in HSV
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude
            optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            # Apply augmentations
            optical_rgb, _ = apply_augs(optical_rgb, 0, augs, optical=True)

            # Resize and transform if required
            optical_rgb = self.transform(cv2.resize(optical_rgb, tuple(self.img_size)))

            # Flip image and angle if flip augmentation is set
            if augs['flip']:
                image_transformed = torch.fliplr(image_transformed)
                angle_t = angle_t * -1.0
                optical_rgb = torch.fliplr(optical_rgb)

            # Extract speed and angle
            speed = self.camera_csv['speed'].iloc[idx]
            speed_t = torch.tensor(speed)

            return image_transformed, angle_t, optical_rgb, speed_t
        else:
            # If optical flow is not enabled, return the image only
            if self.transform:
                image = self.transform(cv2.resize(image, tuple(self.img_size)))
            return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        augs = dict(flip=False, trans=None, rot=None)  # Default augmentations
        data = self.read_data_single(idx, augs)
        sample = {'image': data[0]}

        if self.optical_flow:
            sample['optical'] = data[1]

        return sample
