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

class TransformerMotionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerMotionEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src_tuple, mask=None, src_key_padding_mask=None):
        frames, optical = src_tuple
        output = frames + optical  # Simple fusion of features

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class MotionTransformer(nn.Module):
    def __init__(self, seq_len):
        super(MotionTransformer, self).__init__()
        self.seq_len = seq_len
        
        # Load DINOv2 model with error handling
        try:
            self.position_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False)
            self.motion_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False)
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Attempting to load from cache if available...")
            # Implement fallback loading mechanism if needed
            raise

        self.d_model = 512
        self.position_embedder = nn.Linear(384, self.d_model)  
        self.motion_embedder = nn.Linear(384, self.d_model)
        
        self.encoder_layer = TransformerMotionEncoderLayer(d_model=self.d_model, nhead=4, dropout=0.1)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = TransformerMotionEncoder(self.encoder_layer, num_layers=2)
        
        self.reduce_combined = nn.Linear(self.d_model, 64)
        self.steering_predictor = nn.Linear(64, 1)
        self.speed_predictor = nn.Linear(64, 1)
        
        # Move model to available device
        self.to(device)
    
    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(device)
    
    def forward(self, frames, optical):
        # Ensure inputs are on the correct device
        frames = frames.to(device)
        optical = optical.to(device)
        
        frames = frames.reshape(-1, 3, 224, 224)
        optical = optical.reshape(-1, 3, 224, 224)
        
        with torch.no_grad():
            frames_features = self.position_encoder(frames)
            optical_features = self.motion_encoder(optical)
        
        frames = F.relu(self.position_embedder(frames_features))
        frames = frames.reshape(-1, self.seq_len, self.d_model).permute(1, 0, 2)

        optical = F.relu(self.motion_embedder(optical_features))
        optical = optical.reshape(-1, self.seq_len, self.d_model).permute(1, 0, 2)
        
        attn_mask = self.generate_square_subsequent_mask(frames.shape[0])
        fused_embedding = F.relu(self.transformer_encoder((frames, optical), mask=attn_mask))

        fused_embedding = fused_embedding.permute(1, 0, 2)
        fused_embedding = fused_embedding.reshape(-1, self.d_model)
        reduced = F.relu(self.reduce_combined(fused_embedding))

        speed = self.speed_predictor(reduced)
        angle = self.steering_predictor(reduced)

        return angle, speed

# Example usage
def load_model(seq_len, checkpoint_path=None):
    model = MotionTransformer(seq_len=seq_len)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model