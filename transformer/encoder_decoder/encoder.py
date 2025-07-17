import torch.nn as nn
from ..attention import MultiHeadAttention
from ..feedforward import PositionWiseFeedForward

class Encoder(nn.Module):
    '''
    Transformer Encoder block:
    - Self-attention (unmasked)
    - Position-wise Feed Forward
    - LayerNorm + residual connections
    '''
    def __init__(self, d, heads, d_ff, dropout):
        '''
        d: model dimension
        heads: number of attention heads
        d_ff: feedforward hidden size
        dropout: dropout rate
        '''
        super(Encoder, self).__init__()

        self.self_attention = MultiHeadAttention(d, heads)    # self-attention layer
        self.feed_forward = PositionWiseFeedForward(d, d_ff)  # feedforward layer
        self.norm1 = nn.LayerNorm(d) # norm after attention
        self.norm2 = nn.LayerNorm(d) # norm after feedforward
        self.dropout = nn.Dropout(dropout) # dropout for regularization
    
    def forward(self, x, mask):
        '''
        x: input to encoder (batch, seq_len, d)
        mask: padding mask for attention
        '''
        attention_output = self.self_attention(x, x, x, mask) # self-attention
        x = self.norm1(x + self.dropout(attention_output)) # add & norm
        ff_output = self.feed_forward(x) # feedforward
        x = self.norm2(x + self.dropout(ff_output)) # add & norm

        return x
