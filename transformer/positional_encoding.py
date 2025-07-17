import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    '''
    adds positional information to token embeddings using sine and cosine functions
    '''
    def __init__(self, d, max_seq_len):
        '''
        d: embedding dimension
        max_seq_len: maximum length of input sequences
        '''
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d) # (max_seq_len, d)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)

        # compute the frequency terms for even dimensions
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))

        # apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # saved with the model but not trained
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        '''
        x: input tensor of shape (batch, seq_len, d)
        returns: input with positional encoding added
        '''
        return x + self.pe[:, :x.size(1)]
