#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import pandas as pd
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import sys
sys.path.append('/home/norhan/SEA-RAFT/core')
from raft import RAFT  # Import SEA RAFT
from utils.utils import InputPadder  # Helper function for RAFT
import torch.nn.functional as F
from aug_utils import apply_augs  # Import augmentations from aug_utils

class UdacityDatasetSeaRaft(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, select_camera=None, slice_frames=None, 
                 select_ratio=1.0, select_range=None, optical_flow=True, seq_len=0, img_size=(224, 224), raft_weights=None):
        
        assert select_ratio >= -1.0 and select_ratio <= 1.0
        self.seq_len = seq_len
        camera_csv = pd.read_csv(csv_file)
        if select_camera:
            assert select_camera in ['left_camera', 'right_camera', 'center_camera'], "Invalid camera: {}".format(select_camera)
            camera_csv = camera_csv[camera_csv['frame_id'] == select_camera]
        self.img_size = img_size
        csv_len = len(camera_csv)
        if slice_frames:
            csv_selected = camera_csv[0:0]
            for start_idx in range(0, csv_len, slice_frames):
                if select_ratio > 0:
                    end_idx = int(start_idx + slice_frames * select_ratio)
                else:
                    start_idx, end_idx = int(start_idx + slice_frames * (1 + select_ratio)), start_idx + slice_frames

                if end_idx > csv_len:
                    end_idx = csv_len
                if start_idx > csv_len:
                    start_idx = csv_len
                csv_selected = csv_selected.append(camera_csv[start_idx:end_idx])
            self.camera_csv = csv_selected
        elif select_range:
            csv_selected = camera_csv.iloc[select_range[0]: select_range[1]]
            self.camera_csv = csv_selected
        else:
            self.camera_csv = camera_csv
            
        self.root_dir = root_dir
        self.transform = transform
        self.optical_flow = optical_flow
        
        # Load SEA RAFT model
        self.raft_model = RAFT()
        if raft_weights:
            self.raft_model.load_state_dict(torch.load(raft_weights))
        self.raft_model = self.raft_model.to('cuda').eval()
        
        # Keep track of mean and cov value in each channel
        self.mean = {}
        self.std = {}
        for key in ['angle', 'torque', 'speed']:
            self.mean[key] = np.mean(camera_csv[key])
            self.std[key] = np.std(camera_csv[key])

    def __len__(self):
        return len(self.camera_csv)

    def read_data_single(self, idx, augs):
        path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25, :, :]  # Crop the image as needed
        original_img = image.copy()

        # Angle independent augmentations
        augs['random_brightness'] = random.uniform(0, 1) > 0.5
        augs['random_shadow'] = random.uniform(0, 1) > 0.5
        augs['random_blur'] = random.uniform(0, 1) > 0.5

        angle = self.camera_csv['angle'].iloc[idx]

        # Apply augmentations
        image, angle = apply_augs(image, angle, augs)
        angle_t = torch.tensor(angle)

        if self.transform:
            image_transformed = self.transform(cv2.resize(image, tuple(self.img_size)))

        if self.optical_flow:
            if idx != 0:
                prev_path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx - 1])
                prev = cv2.imread(prev_path)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25, :, :]
                prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            else:
                prev = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            cur = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

            # Optical flow using SEA RAFT
            prev_tensor = transforms.ToTensor()(prev).unsqueeze(0).cuda()
            cur_tensor = transforms.ToTensor()(cur).unsqueeze(0).cuda()
            padder = InputPadder(prev_tensor.shape)
            prev_tensor, cur_tensor = padder.pad(prev_tensor, cur_tensor)
            flow_low, flow_up = self.raft_model(prev_tensor, cur_tensor, iters=12, test_mode=True)
            flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

            # Create HSV image from optical flow
            hsv = np.zeros(original_img.shape, dtype=np.uint8)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            optical_rgb, _ = apply_augs(optical_rgb, 0, augs, optical=True)
            optical_rgb = self.transform(cv2.resize(optical_rgb, tuple(self.img_size)))

            del original_img
            if augs['flip']:
                image_transformed = torch.fliplr(image_transformed)
                angle_t = angle_t * -1.0
                optical_rgb = torch.fliplr(optical_rgb)
            speed = self.camera_csv['speed'].iloc[idx]
            speed_t = torch.tensor(speed)
            return image_transformed, angle_t, optical_rgb, speed_t

        if self.transform:
            del image
            image = image_transformed
        if augs['flip']:
            image = torch.fliplr(image)
            angle_t = angle_t * -1.0

        return image, angle_t

    def read_data(self, idx, augs):
        """
        Parameters
        ----------
        idx : list or int
        augs : dict of augmentations
        Returns
        -------
        image(s), angle(s), (optical_flow: optional)
        """
        if (isinstance(idx, int) and self.seq_len == 0) or (isinstance(idx, list) and len(idx) == self.seq_len):
            flip_horizontally = random.uniform(0, 1) > 0.5
            translate = None
            if random.uniform(0, 1) > 0.65:
                translation_x = np.random.randint(-10, 10)
                translation_y = np.random.randint(-10, 10)
                translate = (translation_x, translation_y, 0.35 / 100.0)
            rotate = None
            if random.uniform(0, 1) > 0.5:
                random_rot = random.uniform(-1, 1)
                rotate = (random_rot,)
            augs = dict(
                flip=flip_horizontally,
                trans=translate,
                rot=rotate,
            )
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i, augs)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
            if self.optical_flow:
                for stack_idx in [0, 1, 2, 3]:
                    data[stack_idx] = torch.stack(data[stack_idx])
            else:
                for stack_idx in [0, 1]:
                    data[stack_idx] = torch.stack(data[stack_idx])

            return data

        else:
            return self.read_data_single(idx, augs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        augs = dict(
            flip=False,
            trans=None,
            rot=None,
        )
        data = self.read_data(idx, augs)

        sample = {'image': data[0],
                  'angle': data[1]}
        if self.optical_flow:
            sample['optical'] = data[2]
            sample['speed'] = data[3]

        del data

        return sample
