import argparse
import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from DataLoading.Motion_Data import Motion_Data
from model.MotionTransformer import MotionTransformer
from model.SimpleTransformer import SimpleTransformer
from model.LSTM import SequenceModel


class TransformerLightningModule(LightningModule):
    def __init__(self, model_name, learning_rate, optical_flow, seq_len=5):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.optical_flow = optical_flow
        self.seq_len = seq_len
        
        # Initialize the model based on model_name
        if model_name == 'motion':
            self.model = MotionTransformer(seq_len)
        elif model_name == 'simple':
            self.model = SimpleTransformer(seq_len)
        elif model_name == 'LSTM':
            self.model = SequenceModel(seq_len)
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose from: motion, simple, LSTM")
        
        # Loss functions
        self.angle_criterion = torch.nn.MSELoss()
        self.speed_criterion = torch.nn.SmoothL1Loss()
    
    def forward(self, images, optical_flow=None):
        if self.optical_flow and optical_flow is not None:
            return self.model(images, optical_flow)
        else:
            return self.model(images)
    
    def training_step(self, batch, batch_idx):
        if self.optical_flow:
            images, angles, optical, speeds = batch['image'], batch['angle'], batch['optical'], batch['speed']
            angle_pred, speed_pred = self(images, optical)
            
            # Reshape predictions and targets
            angle_pred = angle_pred.reshape(-1, 1)
            speed_pred = speed_pred.reshape(-1, 1)
            angles = angles.float().reshape(-1, 1)
            speeds = speeds.float().reshape(-1, 1)
            
            # Calculate losses
            angle_loss = torch.sqrt(self.angle_criterion(angle_pred, angles) + 1e-6)
            speed_loss = self.speed_criterion(speed_pred, speeds)
            total_loss = angle_loss + speed_loss
            
            # Log metrics
            self.log('train_angle_loss', angle_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_speed_loss', speed_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            
        else:
            images, angles = batch['image'], batch['angle']
            angle_pred = self(images)
            
            # Reshape predictions and targets
            angle_pred = angle_pred.reshape(-1, 1)
            angles = angles.float().reshape(-1, 1)
            
            # Calculate loss
            total_loss = torch.sqrt(self.angle_criterion(angle_pred, angles) + 1e-6)
            
            # Log metrics
            self.log('train_angle_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if self.optical_flow:
            images, angles, optical, speeds = batch['image'], batch['angle'], batch['optical'], batch['speed']
            angle_pred, speed_pred = self(images, optical)
            
            # Reshape predictions and targets
            angle_pred = angle_pred.reshape(-1, 1)
            speed_pred = speed_pred.reshape(-1, 1)
            angles = angles.float().reshape(-1, 1)
            speeds = speeds.float().reshape(-1, 1)
            
            # Calculate losses
            angle_loss = torch.sqrt(self.angle_criterion(angle_pred, angles) + 1e-6)
            speed_loss = self.speed_criterion(speed_pred, speeds)
            total_loss = angle_loss + speed_loss
            
            # Log metrics
            self.log('val_angle_loss', angle_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_speed_loss', speed_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
            
        else:
            images, angles = batch['image'], batch['angle']
            angle_pred = self(images)
            
            # Reshape predictions and targets
            angle_pred = angle_pred.reshape(-1, 1)
            angles = angles.float().reshape(-1, 1)
            
            # Calculate loss
            total_loss = torch.sqrt(self.angle_criterion(angle_pred, angles) + 1e-6)
            
            # Log metrics
            self.log('val_angle_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 90, 150], gamma=0.1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss" if self.optical_flow else "val_angle_loss",
            },
        }


class TCPDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define town splits
        self.train_towns = ['town01', 'town03', 'town04', 'town05', 'town06']
        self.val_towns = ['town02', 'town07', 'town10']
    
    def setup(self, stage=None):
        # Prepare training data folders
        train_data = []
        for town in self.train_towns:
            train_data.append(os.path.join(self.data_dir, town))
            train_data.append(os.path.join(self.data_dir, town + '_addition'))
        
        # Prepare validation data folders
        val_data = []
        for town in self.val_towns:
            val_data.append(os.path.join(self.data_dir, town + '_val'))
        
        # Create datasets
        self.train_dataset = Motion_Data(root=self.data_dir, data_folders=train_data)
        self.val_dataset = Motion_Data(root=self.data_dir, data_folders=val_data)
        
        print(f"Created Training Dataset with {len(self.train_dataset)} Datapoints")
        print(f"Created Validation Dataset with {len(self.val_dataset)} Datapoints")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else 2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else 2
        )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Transformer model with PyTorch Lightning')
    parser.add_argument('--batch_size', type=int, default=13, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--model', type=str, choices=['motion', 'simple', 'LSTM'], default='motion',
                        help='Model to train: motion (MotionTransformer), simple (SimpleTransformer), or LSTM')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--optical_flow', action='store_true', default=True, help='Use optical flow')
    parser.add_argument('--data_dir', type=str, default='/home/user/tcp_carla_data', help='Data directory')
    parser.add_argument('--ckpt_dir', type=str, default='./lightning_checkpoints', help='Checkpoint directory')
    parser.add_argument('--max_epochs', type=int, default=161, help='Maximum number of epochs')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb_id', type=str, default=None, help='Wandb run ID to resume')
    parser.add_argument('--wandb_project', type=str, default='Let-Transformer-Be-a-Car', help='Wandb project name')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Initialize data module
    data_module = TCPDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = TransformerLightningModule(
        model_name=args.model,
        learning_rate=args.learning_rate,
        optical_flow=args.optical_flow
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f'{args.model}-{{epoch:02d}}-{{val_total_loss:.2f}}' if args.optical_flow else f'{args.model}-{{epoch:02d}}-{{val_angle_loss:.2f}}',
        save_top_k=3,
        monitor='val_total_loss' if args.optical_flow else 'val_angle_loss',
        mode='min',
        save_last=True
        # period=10
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=args.ckpt_dir,
        name=f'{args.model}_logs'
    )
    loggers.append(tb_logger)
    
    # Wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{args.model}_lr{args.learning_rate}_bs{args.batch_size}",
        id=args.wandb_id,  # Set to None to create a new run, or provide an ID to resume
        resume="allow",  # Resume if ID exists or create new run
        log_model="all"  # Log all checkpoints to wandb
    )
    loggers.append(wandb_logger)
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=loggers,
        log_every_n_steps=5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        num_sanity_val_steps=0
     )
    
    # Print training info
    print(f"Training {args.model} model")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Optical flow: {args.optical_flow}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.ckpt_dir}")
    print(f"Max epochs: {args.max_epochs}")
    
    if args.resume_from_checkpoint:
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        trainer.fit(model, data_module)
        print("Training started from scratch")
    
    # Save the wandb run ID for future resuming
    if wandb_logger.experiment.id:
        with open(os.path.join(args.ckpt_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb_logger.experiment.id)
        print(f"Wandb run ID: {wandb_logger.experiment.id}")
    
    print("Training completed!")


if __name__ == "__main__":
    main() 