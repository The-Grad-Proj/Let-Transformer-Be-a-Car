import torch
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import os

# Check for CUDA availability in a Windows-friendly way
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, seq_len):
        super(SimpleTransformer, self).__init__()
        self.seq_len = seq_len
        
        # Load DINOv2 model with error handling
        try:
            self.position_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False)
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Attempting to load from cache if available...")
            # Implement fallback loading mechanism if needed
            raise
        
        self.d_model = 512
        self.position_embedder = nn.Linear(384, self.d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dropout=0.1,
            batch_first=False
        )
        
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2,
            norm=None
        )
        
        self.reduce_combined = nn.Linear(self.d_model, 64)
        self.steering_predictor = nn.Linear(64, 1)
        self.speed_predictor = nn.Linear(64, 1)
        
        # Move model to available device
        self.to(device)
    
    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(device)
    
    def forward(self, frames):
        # Ensure inputs are on the correct device
        frames = frames.to(device)
        
        # Reshape input for DINOv2
        frames = frames.reshape(-1, 3, 224, 224)
        
        # Get features from DINOv2 with gradient disabled
        with torch.no_grad():
            frame_features = self.position_encoder(frames)
        
        # Process through the transformer
        frames = F.relu(self.position_embedder(frame_features))
        frames = frames.reshape(-1, self.seq_len, self.d_model).permute(1, 0, 2)
        
        # Generate attention mask
        attn_mask = self.generate_square_subsequent_mask(frames.shape[0])
        
        # Process through transformer encoder
        fused_embedding = F.relu(self.transformer_encoder(frames, mask=attn_mask))
        
        # Reshape and reduce dimensionality
        fused_embedding = fused_embedding.permute(1, 0, 2)
        fused_embedding = fused_embedding.reshape(-1, self.d_model)
        reduced = F.relu(self.reduce_combined(fused_embedding))
        
        # Generate steering angle prediction
        angle = self.steering_predictor(reduced)
        
        return angle

def load_model(seq_len, checkpoint_path=None):
    """
    Initialize and optionally load a checkpoint for the SimpleTransformer model.
    
    Args:
        seq_len (int): Sequence length for the transformer
        checkpoint_path (str, optional): Path to model checkpoint
        
    Returns:
        SimpleTransformer: Initialized model
    """
    model = SimpleTransformer(seq_len=seq_len)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    model.eval()
    return model

# Example usage
if __name__ == "__main__":
    # Example configuration
    SEQ_LEN = 10
    BATCH_SIZE = 1
    
    # Initialize model
    model = load_model(SEQ_LEN)
    
    # Example input (you should replace this with your actual data)
    example_input = torch.randn(BATCH_SIZE, SEQ_LEN, 3, 224, 224)
    
    # Get prediction
    with torch.no_grad():
        angle_pred = model(example_input)
        print(f"Predicted angle shape: {angle_pred.shape}")