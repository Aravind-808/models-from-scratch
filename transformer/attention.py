import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    '''
    d: Total dimension of the model (e.g., 512).
    heads: Number of attention heads (e.g., 8).
    d_k = d // heads: Dimensionality per head
    '''
    def __init__(self, d, heads):
        super(MultiHeadAttention, self).__init__()
        assert d%heads == 0

        self.d = d
        self.heads = heads
        self.d_k = d//heads

        self.W_q = nn.Linear(d, d) # query
        self.W_k = nn.Linear(d, d) # key
        self.W_v = nn.Linear(d, d) # value
        self.W_o = nn.Linear(d, d) # output projection 
    
    '''
    Takes input parameters query, key and value and a mask. The mask is used in decoder where masked attention is needed. 
    '''
    def scaled_dotproduct_attention(self, Q, K, V, mask = None):
        # attention(q,k,v) = softmax((q*k_transpose)/root(dimensionality)) * v
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.d_k)
        if mask is not None: # apply the mask (optional)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probabilities = torch.softmax(attention_scores, dim = -1)

        output = torch.matmul(attention_probabilities, V)

        return output

    # split the heads induvidually
    def split_heads(self, x):
        batch_size, seq_length, d = x.size()
        return x.view(batch_size, seq_length, self.heads, self.d_k).transpose(1, 2)
    
    # combine them back
    def combine_heads(self, x):
        batch_size, num, seq_length,  d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d)

    # main mechanism that is done during inference
    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.scaled_dotproduct_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output
    