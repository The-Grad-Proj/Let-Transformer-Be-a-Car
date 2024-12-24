#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:24:40 2021

@author: chingis
"""

from easydict import EasyDict as edict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from DataLoading import UdacityDataset as UD
from DataLoading import ConsecutiveBatchSampler as CB
from model.MotionTransformer import MotionTransformer
from model.SimpleTransformer import SimpleTransformer
from model.LSTM import SequenceModel
import wandb
import os
import re
# noinspection PyAttributeOutsideInit

def load_model(model_path, default_parameters, device="cpu"):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        parameters = edict(checkpoint['parameters'])
        print(f"Loading Model with paramaters:{parameters}")
        if parameters.model_name == 'LSTM':
            model_object = SequenceModel
        elif parameters.model_name == 'MotionTransformer' :
            model_object = MotionTransformer
        elif parameters.model_name == 'SimpleTransformer' :
            model_object = SimpleTransformer
        else:
            raise KeyError("Unknown Architecture")
        network = model_object(parameters.seq_len)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(network.parameters(), lr=parameters.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        last_epoch = checkpoint['epoch']
    else:
        parameters = default_parameters
        if parameters.model_name == 'LSTM':
            model_object = SequenceModel
        elif parameters.model_name == 'MotionTransformer' :
            model_object = MotionTransformer
        elif parameters.model_name == 'SimpleTransformer' :
            model_object = SimpleTransformer
        else:
            raise KeyError("Unknown Architecture")
        network = model_object(parameters.seq_len)
        optimizer = optim.Adam(network.parameters(), lr=parameters.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        last_epoch = 0

    return network, optimizer, parameters, last_epoch

# Load parameters if available else use default
default_parameters = edict(
    learning_rate = 0.0001,
    batch_size = 26,
    seq_len = 5,
    num_workers = 8,
    model_name = 'MotionTransformer',
    normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    image_size=(224, 224),
    epochs=161,
    all_frames=False,
    optical_flow=True
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Directory containing the saved models
data_dir = "saved_models" # Change this to the directory containing the saved models
model_dir = os.path.join(data_dir, "MotionTransformer")

# Get the file with the highest epoch number
def get_latest_model_path(directory, base_filename="last_epoch"):
    highest_epoch = -1
    latest_file = None
    
    # Iterate through files in the directory
    for file_name in os.listdir(directory):
        # Match the pattern: last_epoch_number.tar
        match = re.match(f"{base_filename}_(\\d+)\\.tar", file_name)
        if match:
            epoch = int(match.group(1))
            if epoch > highest_epoch:
                highest_epoch = epoch
                latest_file = file_name
    
    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        raise FileNotFoundError(f"No files matching '{base_filename}_<number>.tar' found in '{directory}'.")

# Get the path of the latest model
try:
    model_path = get_latest_model_path(model_dir)
    print(f"Latest model path: {model_path}")
except FileNotFoundError as e:
    model_path = ""

# Load the model
device = torch.device("cuda")
network, optimizer, parameters, last_epoch = load_model(model_path, default_parameters, device)
network.to(device)

wandb.init(config=parameters, project='self-driving-car-commai')
wandb.watch(network)

DATASET_PATH = r'E:\\GradProject\\Output'

training_set = UD.UdacityDataset(csv_file=fr'{DATASET_PATH}\\train_split.csv',
                             root_dir=DATASET_PATH,
                             transform=transforms.Compose([
                                 #transforms.Resize((224,224)),#(120,320)
                                 transforms.ToTensor(),
                                 transforms.Normalize(*parameters.normalization)
                                 ]),
                             img_size=parameters.image_size,
                             seq_len=parameters.seq_len,
                             optical_flow=parameters.optical_flow,
                             select_camera='center_camera',
                             select_range=(0,218740))
validation_set = UD.UdacityDataset(csv_file=fr'{DATASET_PATH}\\val_split.csv',
                             root_dir=DATASET_PATH,
                             transform=transforms.Compose([
                                # transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(*parameters.normalization)
                                 ]),
                             img_size=parameters.image_size,
                             seq_len=parameters.seq_len,
                             optical_flow=parameters.optical_flow,
                             select_camera='center_camera',
                             select_range=(0,24305))


training_cbs = CB.ConsecutiveBatchSampler(data_source=training_set, batch_size=parameters.batch_size,use_all_frames=parameters.all_frames, shuffle=True, drop_last=False, seq_len=parameters.seq_len)
training_loader = DataLoader(training_set, sampler=training_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))

validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, use_all_frames=False, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
criterion = torch.nn.MSELoss()
criterion.to(device)
speed_criterion =  torch.nn.SmoothL1Loss()
speed_criterion.to(device)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = parameters.learning_rate
    if epoch in [30, 90, 150]:
        lr = parameters.learning_rate * 0.1
        parameters.learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

experiment_epochs = 40
last_epoch_saved = None


# Training Loop
for epoch in range(last_epoch, min(parameters.epochs, last_epoch + experiment_epochs)):
    train_angle_losses = AverageMeter()
    train_speed_losses = AverageMeter()
    
    network.train()
    adjust_learning_rate(optimizer, epoch)

    for training_sample in tqdm(training_loader):
        param_values = [v for v in training_sample.values()]
        if parameters.optical_flow:
            image, angle, optical, speed = param_values
            optical = optical.to(device)
            speed = speed.float().reshape(-1, 1).to(device)
        else:
            image, angle = param_values

        loss = 0
        image = image.to(device)
        if parameters.optical_flow:
            angle_hat, speed_hat = network(image, optical)
            speed_hat = speed_hat.reshape(-1, 1)
            training_loss_speed = speed_criterion(speed_hat, speed)
            loss += training_loss_speed
            train_speed_losses.update(training_loss_speed.item())
        else:
            angle_hat = network(image)

        angle_hat = angle_hat.reshape(-1, 1)
        angle = angle.float().reshape(-1, 1).to(device)
        training_loss_angle = torch.sqrt(criterion(angle_hat, angle) + 1e-6)
        loss += training_loss_angle

        train_angle_losses.update(training_loss_angle.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Training Loss: Angle = {train_angle_losses.avg}, Speed = {train_speed_losses.avg}")
    last_epoch_saved = epoch

    # Save model after every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join('saved_models', parameters.model_name, f'last_epoch_{epoch + 1}.tar')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'parameters': dict(parameters)  # Save parameters for reference
        }, save_path)

    # Validation (if applicable)
    network.eval()
    val_speed_losses = AverageMeter()
    val_angle_losses = AverageMeter()
    with torch.no_grad():
        for validation_sample in tqdm(validation_loader):
            param_values = [v for v in validation_sample.values()]
            if parameters.optical_flow:
                image, angle, optical, speed = param_values
                optical = optical.to(device)
                speed = speed.float().reshape(-1, 1).to(device)
            else:
                image, angle = param_values

            loss = 0
            image = image.to(device)
            if parameters.optical_flow:
                angle_hat, speed_hat = network(image, optical)
                speed_hat = speed_hat.reshape(-1, 1)
                validation_loss_speed = speed_criterion(speed_hat, speed)
                loss += validation_loss_speed
                val_speed_losses.update(validation_loss_speed.item())
            else:
                angle_hat = network(image)
            angle_hat = angle_hat.reshape(-1, 1)
            angle = angle.float().reshape(-1, 1).to(device)

            validation_loss_angle = torch.sqrt(criterion(angle_hat, angle) + 1e-6)
            loss += validation_loss_angle

            val_angle_losses.update(validation_loss_angle.item())

    print(f"Epoch {epoch} Validation Loss: Angle = {val_angle_losses.avg}, Speed = {val_speed_losses.avg}")

    # Log results to WandB
    report = {
        'training_angle_loss': train_angle_losses.avg,
        'epoch': epoch,
    }
    if parameters.optical_flow:
        report['training_speed_loss'] = train_speed_losses.avg
        report['validation_angle_loss'] = val_angle_losses.avg
        report['validation_speed_loss'] = val_speed_losses.avg

    wandb.log(report)

# Save model after the last epoch
if last_epoch_saved is not None:
    save_path = os.path.join('saved_models', parameters.model_name, f'last_epoch_{epoch + 1}.tar')
    torch.save({
        'epoch': last_epoch_saved + 1,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'parameters': dict(parameters)  # Save parameters for reference
    }, save_path)