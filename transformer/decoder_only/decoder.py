import torch.nn as nn
from ..attention import MultiHeadAttention
from ..feedforward import PositionWiseFeedForward

# this is different from the encoder-decoder transformer architecture since theres no need for any sort of cross attention here.
class Decoder(nn.Module):
    '''
    Transformer Decoder block:
    - Self-attention (masked)
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

        self.self_attention = MultiHeadAttention(d, heads)    # masked self-attention
        self.feed_forward = PositionWiseFeedForward(d, d_ff)  # feedforward layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target_mask):
        '''
        x: decoder input
        target_mask: mask for decoder input (causal + padding)
        '''
        # masked self-attention (only once self attention block as theres no enc output to combine this with)
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))

        # feedforward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
