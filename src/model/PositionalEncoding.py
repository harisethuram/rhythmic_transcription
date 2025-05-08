import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Note that we only go up to d_model in steps of 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = self.pe.unsqueeze(0)
        
    def forward(self, x, batch_first=True):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) if batch_first=True
               or (seq_len, batch_size, d_model) if batch_first=False.
        """
        if batch_first:
            seq_len = x.shape[1]
            x = x + self.pe[:, :seq_len, :]
        else:
            seq_len = x.shape[0]
            x = x + self.pe.transpose(0, 1)[:seq_len, ...]
        
        return x
    
    def to(self, device):
        self.pe = self.pe.to(device)
        return super().to(device)
        
        
        