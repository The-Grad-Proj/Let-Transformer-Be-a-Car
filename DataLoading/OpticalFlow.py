import sys
sys.path.append('core')
from core.raft import RAFT
from core.utils.utils import load_ckpt
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
import os



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



def calculate_opticalFlow(prev,cur):

    model= load_pretrained_model()
    prev = torch.tensor(prev, dtype=torch.float32).permute(2, 0, 1)
    cur = torch.tensor(cur, dtype=torch.float32).permute(2, 0, 1)
    prev = prev[None].to(device)
    cur = cur[None].to(device)

    output= model(prev,cur,iters=4)
    flow_final = output['flow'][-1]
    flow_2d=flow_final[0]
    # Create a magnitude channel
    flow_mag = torch.sqrt(flow_2d[0]**2 + flow_2d[1]**2).unsqueeze(0)  # [1, H, W]
    flow_3d = torch.cat([flow_2d, flow_mag], dim=0)                    # [3, H, W]
    flow_3d = F.interpolate(flow_3d.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    flow_3d = flow_3d.squeeze(0)  # final shape: [3, 224, 224]
    # Convert to numpy
    optical_rgb = flow_3d.detach().cpu().numpy().transpose(1, 2, 0) # [224, 224, 3]
    optical_rgb = cv2.normalize(optical_rgb, None, 0, 255, cv2.NORM_MINMAX)
    optical_rgb = optical_rgb.astype(np.uint8)

    return optical_rgb



def main():

    image_folder= "DataLoading\images"
    output_folder= "DataLoading\opticalflow_output"
    csv_file="DataLoading\interpolated.csv"
    
    camera_csv=pd.read_csv(csv_file)
    
    # Iterate over csv file columns
    if 'frame_id' in camera_csv.columns:
        
        previous = None  # Initialize previous as None before the loop
        current = None  

        # Iterate in the rows of the selected column 
        for index, row in camera_csv.iterrows():
            if row['frame_id']=='center_camera':
                previous= current
                current= index
                current_file_name = camera_csv.loc[current, 'filename']
                previous_file_name = camera_csv.loc[previous, 'filename'] if previous is not None else None

                current_path= os.path.join(image_folder, current_file_name)
                previous_path= os.path.join(image_folder, previous_file_name)

                cur= cv2.imread(current_path)
                prev= cv2.imread(previous_path)

                optical_rgb= calculate_opticalFlow(prev, cur)

                # Save optical flow image
                output_file_name = f"{previous_file_name}_optical.png"
                output_path = os.path.join(output_folder, output_file_name)
                cv2.imwrite(output_path, optical_rgb)

                print(f"Saved optical flow image: {output_file_name}")



if __name__ == "__main__":
    main()



