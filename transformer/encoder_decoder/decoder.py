import torch.nn as nn
from ..attention import MultiHeadAttention
from ..feedforward import PositionWiseFeedForward

class Decoder(nn.Module):
    '''
    Transformer Decoder block:
    - Self-attention (masked)
    - Cross-attention (attend to encoder output)
    - Position-wise Feed Forward
    '''

    def __init__(self, d, heads, d_ff, dropout):
        '''
        d: model dimension
        heads: number of attention heads
        d_ff: feedforward hidden size
        dropout: dropout rate
        '''
        super(Decoder, self).__init__()

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        self.self_attention = MultiHeadAttention(d, heads)    # masked self-attention
        self.cross_attention = MultiHeadAttention(d, heads)   # attention over encoder output
        self.feed_forward = PositionWiseFeedForward(d, d_ff)  # feedforward layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoded_output, source_mask, target_mask):
        '''
        x: decoder input
        encoded_output: output from encoder
        source_mask: mask for encoder input (padding)
        target_mask: mask for decoder input (causal + padding)
        '''
        # masked self-attention
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))

        # cross-attention (queries from decoder, keys/values from encoder)
        attention_output = self.cross_attention(x, encoded_output, encoded_output, source_mask)
        x = self.norm2(x + self.dropout(attention_output))

        # feedforward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
