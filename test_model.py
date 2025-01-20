import os
import random
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from model.MotionTransformer import MotionTransformer
from easydict import EasyDict as edict
from DataLoading.UdacityDataset import UdacityDataset
from DataLoading import ConsecutiveBatchSampler as CB
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
data_dir = '/home/ibraa04/grad_project/udacity/output/'
checkpoint_path = '/home/ibraa04/grad_project/Let-Transformer-Be-a-Car/saved_models/MotionTransformer/epoch_145.tar'
seq_len = 5

parameters = edict(
    batch_size = 1,
    seq_len = 5,
    num_workers = 1,
    model_name = 'MotionTransformer',
    normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    image_size=(224, 224),
    all_frames=True,
    optical_flow=True,
    checkpoint='/home/ibraa04/grad_project/models/dinov2/epoch_120.tar'
)

# Load the model
ckpt = torch.load(checkpoint_path)
network = MotionTransformer(seq_len)
network.load_state_dict(ckpt['model_state_dict'])
network.to(device)
network.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create an instance of UdacityDataset
dataset = UdacityDataset(csv_file=f"{data_dir}/interpolated.csv",
                         root_dir=data_dir,
                         transform=transform,
                         img_size=parameters.image_size,
                         select_camera='center_camera',
                         optical_flow=parameters.optical_flow,
                         seq_len=parameters.seq_len,
                         select_range=(0, 10))

# Sample 20 sequences from the dataset
# indices = random.sample(range(len(dataset)), 20)
# subset = Subset(dataset, indices)

def collate_fn(batch):
    return batch[0]

# Create a DataLoader using the sampled subset
validation_cbs = CB.ConsecutiveBatchSampler(data_source=dataset, batch_size=parameters.batch_size, use_all_frames=parameters.all_frames, shuffle=True, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(dataset, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=collate_fn)

predictions = []
network.eval()
with torch.no_grad():    
    for validation_sample in validation_loader:
        param_values = [v for v in validation_sample.values()]
        image, angle, optical, speed = param_values
        image = image.to(device)
        optical = optical.to(device)
        speed = speed.float().reshape(-1, 1).to(device)
        angle_hat, speed_hat = network(image, optical)
        speed_hat = speed_hat.reshape(-1, 1)
        angle_hat = angle_hat.reshape(-1, 1)
        angle = angle.float().reshape(-1, 1).to(device)
        angle_hat = angle_hat.mean(dim=0)
        angle = angle.mean(dim=0)
        predictions.append((angle_hat, angle, speed_hat, speed))
        print("Actual Angle: ", angle, "\tPredicted Angle: ", angle_hat)
        # print("Actual Speed: ", speed, "Predicted Speed: ", speed_hat)

# how the predictions
# print("Predictions: ", predictions)