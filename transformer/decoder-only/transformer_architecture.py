import torch 
import torch.nn as nn   
from ..positional_encoding import PositionalEncoding
from .decoder import Decoder

class DecoderOnlyTransformer(nn.Module):
    '''
    full transformer model: encoder-decoder architecture
    '''
    def __init__(self, target_vocab_size, d, heads, num_layers, d_ff, max_seq_len, dropout):
        '''
        target_vocab_size: size of output vocabulary
        d: model dimension
        heads: number of attention heads
        num_layers: number of encoder and decoder layers
        d_ff: hidden size in feedforward layers
        max_seq_len: maximum sequence length
        dropout: dropout rate
        '''
        super(DecoderOnlyTransformer, self).__init__()

        self.decoder_embeddings = nn.Embedding(target_vocab_size, d)  # embedding for target tokens
        self.positional_encoding = PositionalEncoding(d, max_seq_len)  # positional encoding shared by encoder and decoder

        self.decoder_layers = nn.ModuleList([Decoder(d, heads, d_ff, dropout) for _ in range(num_layers)])  # list of decoder layers

        self.fc = nn.Linear(d, target_vocab_size)  # final projection to vocabulary
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt):
        '''
        creates padding and causal masks for encoder and decoder
        '''
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # shape: (batch, 1, tgt_len, 1)

        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()  # causal mask

        tgt_mask = tgt_mask & nopeak_mask  # combine padding and causal masks
        return tgt_mask

    def forward(self, tgt):
        '''
        src: input token ids (batch, src_len)
        tgt: target token ids (batch, tgt_len)
        returns: logits over target vocabulary
        '''
        tgt_mask = self.generate_mask(tgt)

        # apply embedding and positional encoding to inputs
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddings(tgt)))

        # pass through encoder layers

        # pass through decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        # project decoder output to target vocabulary
        output = self.fc(dec_output)
        return output
