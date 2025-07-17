import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
import pickle
from tqdm import tqdm
from .transformer_architecture import Transformer

spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text): return [tok.text.lower() for tok in spacy_en(text)]
def tokenize_fr(text): return [tok.text.lower() for tok in spacy_fr(text)]

with open("transformer/vocabs/src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("transformer/vocabs/tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

print("Loaded vocabularies")

PAD_IDX = src_vocab["<pad>"]

def encode(text, vocab, tokenizer):
    return [vocab["<sos>"]] + [vocab.get(tok, vocab["<unk>"]) for tok in tokenizer(text)] + [vocab["<eos>"]]

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src = torch.tensor(encode(item["translation"]["en"], src_vocab, tokenize_en), dtype=torch.long)
        tgt = torch.tensor(encode(item["translation"]["fr"], tgt_vocab, tokenize_fr), dtype=torch.long)
        src_batch.append(src)
        tgt_batch.append(tgt)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch

dataset = load_dataset("opus_books", "en-fr", split="train[:40000]")

print(len(dataset))

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print("Loading dataset and collation success")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    source_vocab_size=len(src_vocab),
    target_vocab_size=len(tgt_vocab),
    d=512, heads=4, num_layers=2, d_ff=2048,
    max_seq_len=512, dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

EPOCHS = 1
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for src_batch, tgt_batch in loop:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        output = model(src_batch, tgt_input)
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        loop.set_postfix(loss=total_loss / (loop.n + 1e-9))

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "transformer/models/transformer_final_2.pth")
