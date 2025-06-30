import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_seq_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]