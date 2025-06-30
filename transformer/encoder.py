import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionWiseFeedForward
from positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super(Encoder, self).__init__()

        self.self_attention = MultiHeadAttention(d, heads)
        self.feed_forward = PositionWiseFeedForward(d, d_ff)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x+ self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x+ self.dropout(ff_output))

        return x
