#!/usr/bin/env python
# coding: utf-8


from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import sys
sys.path.append('core')
from core.raft import RAFT
from core.utils.utils import load_ckpt

# defining customized Dataset class for Udacity
from .aug_utils import apply_augs
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random
import torch.nn.functional as F

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


class UdacityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, select_camera=None, slice_frames=None, select_ratio=1.0, select_range=None, optical_flow=True, seq_len=0, img_size=(224, 224)):
        
        # Load raft model
        self.model = load_pretrained_model()

        assert select_ratio >= -1.0 and select_ratio <= 1.0 # positive: select to ratio from beginning, negative: select to ration counting from the end
        self.seq_len = seq_len
        camera_csv = pd.read_csv(csv_file)
        if select_camera:
            assert select_camera in ['left_camera', 'right_camera', 'center_camera'], "Invalid camera: {}".format(select_camera)
            camera_csv = camera_csv[camera_csv['frame_id']==select_camera]
        self.img_size = img_size
        csv_len = len(camera_csv)
        if slice_frames:
            csv_selected = camera_csv[0:0] # empty dataframe
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
        image = image[65:-25,:,:]
        original_img = image.copy()
        # angle independent augs

  
        augs['random_brightness'] = random.uniform(0, 1) > 0.5 
        augs['random_shadow'] = random.uniform(0, 1) > 0.5
        augs['random_blur'] = random.uniform(0, 1) > 0.5

        angle = self.camera_csv['angle'].iloc[idx]

        image, angle = apply_augs(image, angle, augs)
        angle_t = torch.tensor(angle)#.clamp(-1, 1)

  
        if self.transform:
            image_transformed = self.transform(cv2.resize(image, tuple(self.img_size)))

        if self.optical_flow:
            if idx != 0:
                path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx - 1])
                prev = cv2.imread(path)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25,:,:]
                prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            else:
                prev = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            cur = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

            # Convert to tensors and add batch and channel dimensions
            prev = torch.tensor(prev, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            cur = torch.tensor(cur, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            output = self.model(prev, cur, iters=4)
            flow_final = output['flow'][-1]  # shape [B, 2, H, W]
            flow_2d = flow_final[0]         # remove batch, now [2, H, W]
            # Create a magnitude channel
            flow_mag = torch.sqrt(flow_2d[0]**2 + flow_2d[1]**2).unsqueeze(0)  # [1, H, W]
            flow_3d = torch.cat([flow_2d, flow_mag], dim=0)                    # [3, H, W]
            flow_3d = F.interpolate(flow_3d.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            flow_3d = flow_3d.squeeze(0)  # final shape: [3, 224, 224]

            # Convert to numpy
            optical_rgb = flow_3d.cpu().numpy().detach().transpose(1, 2, 0) # [224, 224, 3]
            optical_rgb = cv2.normalize(optical_rgb, None, 0, 255, cv2.NORM_MINMAX)
            optical_rgb = optical_rgb.astype(np.uint8)
            
            # Use Hue, Saturation, Value colour model
            # flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # hsv = np.zeros(original_img.shape, dtype=np.uint8)
            # hsv[..., 1] = 255
            
            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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
            DESCRIPTION.
            in case of list:
                if len(idx) == batch_size -> do not choose augmentations since it will be applied to the whole batch
                if len(idx) == sequence_length -> apply augmentations
            in case of int:
                apply augmentations
        augs: a dict of augmentations
        Returns
        -------
        image(s), angle(s), (optical_flow: optional)
        """
        if (isinstance(idx, int) and self.seq_len == 0) or (isinstance(idx, list) and len(idx) == self.seq_len):
            flip_horizontally = random.uniform(0, 1) > 0.5
            translate = None
            if random.uniform(0, 1) > 0.65: # translate
                translation_x = np.random.randint(-10, 10) 
                translation_y = np.random.randint(-10, 10)
                translate = (translation_x, translation_y, 0.35/100.0)
            rotate = None
            if random.uniform(0, 1) > 0.5: # rotate
                random_rot = random.uniform(-1, 1)#np.random.randint(5, 15+1)
                #rotation_angle = random.choice([-random_rot, random_rot])
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
                for stack_idx in [0, 1, 2, 3]: # we don't stack timestamp and frame_id since those are string data
                    data[stack_idx] = torch.stack(data[stack_idx])
            else:
                for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
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

