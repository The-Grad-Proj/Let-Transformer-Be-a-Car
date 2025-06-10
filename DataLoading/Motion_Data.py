import os
import cv2
from PIL import Image
import numpy as np
import torch 
from torch.utils.data import Dataset
from torchvision import transforms as T


class Motion_Data(Dataset):
    def __init__(self, root, data_folders, seq_len=5, img_size=(224, 224)):
        self.root = root
        self.seq_len = seq_len
        self.img_size = img_size
        
        # Lists to store sequences
        self.front_img_sequences = []
        self.speed_sequences = []
        self.angle_sequences = []  # Using theta as steering angle
        
        # Process each subroot separately
        for sub_root in data_folders:
            data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()
            
            # Get data for this subroot
            front_imgs = data['front_img']
            speeds = data['speed']
            thetas = data['input_theta']  # Using theta as steering angle
            
            # Create sequences of 5 consecutive frames
            num_sequences = len(front_imgs) // seq_len
            print(f"Found {len(front_imgs)} frames, creating {num_sequences} sequences")
            
            for i in range(num_sequences):
                start_idx = i * seq_len
                end_idx = start_idx + seq_len
                
                # Extract sequence
                img_seq = front_imgs[start_idx:end_idx]
                speed_seq = speeds[start_idx:end_idx]
                theta_seq = thetas[start_idx:end_idx]
                
                # Only add complete sequences
                if len(img_seq) == seq_len:
                    self.front_img_sequences.append(img_seq)
                    self.speed_sequences.append(speed_seq)
                    self.angle_sequences.append(theta_seq)
                else:
                    print(f"Skipping sequence {i} due to insufficient frames")
        
        # Image transform (same as in data.py)
        self._im_transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.front_img_sequences)
    
    def calculate_optical_flow(self, prev_img, curr_img):
        """Calculate optical flow between two images"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert flow to HSV representation
        hsv = np.zeros((*curr_gray.shape, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to RGB
        optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return optical_rgb
    
    def __getitem__(self, index):
        """Returns a sequence of images with optical flow"""
        # Get sequences for this index
        img_paths = self.front_img_sequences[index]
        speeds = self.speed_sequences[index]
        angles = self.angle_sequences[index]
        
        # Lists to store processed data
        images = []
        optical_flows = []
        
        prev_img = None
        
        for i in range(self.seq_len):
            # Load and preprocess image
            img_path = self.root + img_paths[i][0]
            img = np.array(Image.open(img_path))
            
            # Resize image
            img_resized = cv2.resize(img, self.img_size)
            
            # Calculate optical flow
            if i == 0:
                # For first frame, use zero optical flow
                optical_flow = np.zeros_like(img_resized)
            else:
                optical_flow = self.calculate_optical_flow(prev_img, img)
                optical_flow = cv2.resize(optical_flow, self.img_size)
            
            # Apply transforms
            img_tensor = self._im_transform(img_resized)
            optical_tensor = self._im_transform(optical_flow)
            
            images.append(img_tensor)
            optical_flows.append(optical_tensor)
            
            prev_img = img.copy()
        
        # Stack tensors
        images = torch.stack(images)  # Shape: [seq_len, 3, H, W]
        optical_flows = torch.stack(optical_flows)  # Shape: [seq_len, 3, H, W]
        
        # Fix for nan angles
        angles_arr = []
        for a in angles:
            # Check if a is a list/array or a float
            if hasattr(a, '__getitem__') and not isinstance(a, (str, bytes)):
                val = a[0] if not np.isnan(a[0]) else 0.0
            else:
                val = a if not np.isnan(a) else 0.0
            angles_arr.append(val)
        
        speeds_arr = []
        for s in speeds:
            # Check if s is a list/array or a float
            if hasattr(s, '__getitem__') and not isinstance(s, (str, bytes)):
                speeds_arr.append(s[0])
            else:
                speeds_arr.append(s)
        
        # Convert to numpy arrays
        angles = np.array(angles_arr)
        # print(f"angles length: {len(angles)}")
        speeds = np.array(speeds_arr)
        # print(f"speeds length: {len(speeds)}")
        
        # Convert to tensors
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        speeds_tensor = torch.tensor(speeds, dtype=torch.float32)
        
        # Return in format similar to UdacityDataset
        sample = {
            'image': images,
            'angle': angles_tensor,
            'optical': optical_flows,
            'speed': speeds_tensor
        }
        
        return sample 