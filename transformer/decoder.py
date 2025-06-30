import torch 
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionWiseFeedForward
from positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super(Decoder, self).__init__()

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        self.self_attention = MultiHeadAttention(d, heads)
        self.cross_attention = MultiHeadAttention(d, heads)
        self.feed_forward = PositionWiseFeedForward(d, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoded_output, source_mask, target_mask):
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.self_attention(x, encoded_output, encoded_output, source_mask)
        x = self.norm2(x+ self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x+self.dropout(ff_output))

        return x