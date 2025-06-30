import torch 
import torch.nn as nn   
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d, heads, num_layers, d_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()

        self.encoder_embeddings = nn.Embedding(source_vocab_size, d)
        self.decoder_embeddings = nn.Embedding(target_vocab_size, d)
        self.positional_encoding = PositionalEncoding(d, max_seq_len)

        self.encoder_layers = nn.ModuleList([Encoder(d, heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(d, heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddings(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddings(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
