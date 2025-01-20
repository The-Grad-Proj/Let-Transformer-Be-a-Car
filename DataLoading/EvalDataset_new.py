#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:57:21 2021

@author: chingis
"""

#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('core')
from core.raft import RAFT
from core.utils.utils import load_ckpt

import torch
import torch.nn.functional as F


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# defining customized Dataset class for Udacity

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    name = 'kitti-S'
    dataset = 'kitti'
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    use_var = True
    var_min = 0
    var_max = 10
    pretrain = 'resnet18'
    initial_dim = 64
    block_dims = [64, 128, 256]
    radius = 4
    dim = 128
    num_blocks = 2
    iters = 4
    image_size = [432, 960]
    scale = 0
    batch_size = 16
    epsilon = 1e-8
    lr = 0.0001
    wdecay = 1e-5
    dropout = 0
    clip = 1.0
    gamma = 0.85
    num_steps = 10000
    restore_ckpt = None
    coarse_config = None
    cfg = 'raft/kitti-S.json'
    path = 'raft/Tartan480x640-S.pth'
    url = None
    device = device

def load_pretrained_model():
    args = Args()
    model = RAFT(args)
    load_ckpt(model, args.path)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    return model



def calculate_opticalFlow(prev, cur, model):
    prev = torch.tensor(prev, dtype=torch.float32).permute(2, 0, 1)
    cur = torch.tensor(cur, dtype=torch.float32).permute(2, 0, 1)
    prev = prev[None].to(device)
    cur = cur[None].to(device)

    output = model(prev, cur, iters=4)
    flow_final = output['flow'][-1]
    flow_2d = flow_final[0]
    # Create a magnitude channel
    flow_mag = torch.sqrt(flow_2d[0]**2 + flow_2d[1]**2).unsqueeze(0)  # [1, H, W]
    flow_3d = torch.cat([flow_2d, flow_mag], dim=0)                    # [3, H, W]
    flow_3d = F.interpolate(flow_3d.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    flow_3d = flow_3d.squeeze(0)  # final shape: [3, 224, 224]
    # Convert to numpy
    optical_rgb = flow_3d.detach().cpu().numpy().transpose(1, 2, 0)  # [224, 224, 3]
    optical_rgb = cv2.normalize(optical_rgb, None, 0, 255, cv2.NORM_MINMAX)
    optical_rgb = optical_rgb.astype(np.uint8)
    optical_rgb = cv2.cvtColor(optical_rgb, cv2.COLOR_BGR2RGB)

    return optical_rgb


class EvalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, optical_flow=True, public=False, private=False, img_size=(224, 224)):
        self.model = load_pretrained_model()

        assert False in (public, private)
        mode = -1
        if public:
            mode = 1
        elif private:
            mode = 0
        camera_csv = pd.read_csv(csv_file)

        self.optical_flow = optical_flow
        csv_len = len(camera_csv)
        camera_csv['og_idx'] = range(len(camera_csv))
        self.camera_csv = camera_csv if mode == -1 else camera_csv[camera_csv['public'] == mode]
        self.original_camera = camera_csv
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.camera_csv)
    
    def read_data_single(self, idx):
        #print(str(self.camera_csv['frame_id'].iloc[idx]) + '.jpg')
        path = os.path.join(self.root_dir, str(self.camera_csv['frame_id'].iloc[idx]) + '.jpg')
        #print(path)
        image = cv2.imread(path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25,:,:]

        angle = self.camera_csv['steering_angle'].iloc[idx]
        if self.transform:
            image_transformed = self.transform(cv2.resize(image, tuple(self.img_size)))
        angle_t = torch.tensor(angle)
        idx_t = torch.tensor(idx)
        if self.optical_flow:
            og_idx = self.camera_csv['og_idx'].iloc[idx]
            if og_idx != 0:
                path = os.path.join(self.root_dir, str(self.original_camera['frame_id'].iloc[og_idx - 1]) + '.jpg')
                prev = cv2.imread(path)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25,:,:]
            else:
                prev = image

            optical_rgb = calculate_opticalFlow(prev, image, self.model)
            # cur = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # # Use Hue, Saturation, Value colour model
            # flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # hsv = np.zeros(image.shape, dtype=np.uint8)
            # hsv[..., 1] = 255
            
            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            optical_rgb = self.transform(cv2.resize(optical_rgb, tuple(self.img_size)))
            del image
            return image_transformed, angle_t, idx_t, optical_rgb
               
        if self.transform:
            del image
            image = image_transformed
        return image, angle_t, idx_t
    
    def read_data(self, idx):
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
            indeces = [0, 1, 2] if not self.optical_flow else [0, 1, 2, 3]
            for stack_idx in indeces: # we don't stack timestamp and frame_id since those are string data
                data[stack_idx] = [torch.tensor(item) if isinstance(item, np.ndarray) else item for item in data[stack_idx]]
                data[stack_idx] = torch.stack(data[stack_idx])
            
            return data
        
        else:
            return self.read_data_single(idx)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.read_data(idx)
        
        sample = {'image': data[0],
                  'angle': data[1],
                    'idx':data[2]}
        if self.optical_flow:
            sample['optical'] = data[3]
        del data
        
        return sample

