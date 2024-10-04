#!/usr/bin/env python
# coding: utf-8

import os
import sys
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
sys.path.append('/home/norhan/SEA-RAFT/core')
from raft import RAFT  # Importing the RAFT model
from utils.utils import load_ckpt  # Utility to load model checkpoint
from utils.flow_viz import flow_to_image  # Utility to visualize optical flow
import pandas as pd

class EvalDataset(Dataset):
    def _init_(self, csv_file, root_dir, transform=None, optical_flow=True, public=False, private=False, img_size=(224, 224), raft_model_path=None, device='cuda', args=None):
        assert False in (public, private)
        mode = -1
        if public:
            mode = 1
        elif private:
            mode = 0
        camera_csv = pd.read_csv(csv_file)

        self.optical_flow = optical_flow
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.args = args

        # Initialize SEA-RAFT model
        self.device = torch.device(device)
        self.raft_model = RAFT(self.args)  # Initialize RAFT model with arguments

        if raft_model_path:
            load_ckpt(self.raft_model, raft_model_path)  # Load pretrained weights

        self.raft_model = self.raft_model.to(self.device)
        self.raft_model.eval()  # Set RAFT model to evaluation mode

        csv_len = len(camera_csv)
        camera_csv['og_idx'] = range(len(camera_csv))
        self.camera_csv = camera_csv if mode == -1 else camera_csv[camera_csv['public'] == mode]
        self.original_camera = camera_csv

    def _len_(self):
        return len(self.camera_csv)

    def compute_sea_raft_flow(self, img1, img2):
        """
        Computes optical flow using the SEA-RAFT model.
        :param img1: The first image in the sequence (previous frame)
        :param img2: The second image in the sequence (current frame)
        :return: Optical flow between the two images
        """
        model_output = self.raft_model(img1, img2, iters=self.args.iters, test_mode=True)
        flow = model_output['flow'][-1]  # Get the last flow prediction
        return flow

    def read_data_single(self, idx):
        # Read and process the current image
        path = os.path.join(self.root_dir, str(self.camera_csv['frame_id'].iloc[idx]) + '.jpg')
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25, :, :]  # Cropping the image as per original logic
        original_img = image.copy()

        # Get the steering angle and index from the CSV
        angle = self.camera_csv['steering_angle'].iloc[idx]
        angle_t = torch.tensor(angle, dtype=torch.float32)  # Create tensor for the angle
        idx_t = torch.tensor(idx, dtype=torch.long)  # Create tensor for the index

        # Transform the image
        if self.transform:
            image_transformed = self.transform(cv2.resize(image, tuple(self.img_size)))

        if self.optical_flow:
            # Handle optical flow calculation using SEA-RAFT
            og_idx = self.camera_csv['og_idx'].iloc[idx]
            if og_idx != 0:
                prev_path = os.path.join(self.root_dir, str(self.original_camera['frame_id'].iloc[og_idx - 1]) + '.jpg')
                prev = cv2.imread(prev_path)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25, :, :]  # Cropping the previous frame
            else:
                prev = original_img  # Use current image as previous if it's the first frame

            # Prepare the images for SEA-RAFT model
            prev_tensor = torch.tensor(prev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            cur_tensor = torch.tensor(original_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Compute optical flow using SEA-RAFT
            flow = self.compute_sea_raft_flow(prev_tensor, cur_tensor)

            # Convert flow to a visual representation (RGB image)
            optical_rgb = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy())

            # Apply transformation to the optical flow data
            if self.transform:
                optical_rgb = self.transform(cv2.resize(optical_rgb, tuple(self.img_size)))

            return image_transformed, angle_t, idx_t, optical_rgb

        return image_transformed, angle_t, idx_t

    def _getitem_(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.read_data_single(idx)
        sample = {
            'image': data[0],
            'angle': data[1],
            'idx': data[2],
        }

        if self.optical_flow:
            sample['optical'] = data[3]

        return sample
