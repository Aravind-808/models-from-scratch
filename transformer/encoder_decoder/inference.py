import torch
import torch.nn.functional as F
import spacy
import pickle
from .transformer_architecture import Transformer
import warnings
warnings.filterwarnings('ignore')

spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en(text)]

with open("transformer/vocabs/src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)
with open("transformer/vocabs/tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

inv_tgt_vocab = {i: w for w, i in tgt_vocab.items()}
PAD_IDX = src_vocab["<pad>"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    source_vocab_size=len(src_vocab),
    target_vocab_size=len(tgt_vocab),
    d=512, heads=8, num_layers=4, d_ff=2048,
    max_seq_len=512, dropout=0.1
).to(device)

model.load_state_dict(torch.load(r"C:\Users\rajes\models-from-scratch\transformer\models\encoder_decoder.pth", map_location=device))
model.eval()

def encode(text, vocab, tokenizer):
    return [vocab["<sos>"]] + [vocab.get(tok, vocab["<unk>"]) for tok in tokenizer(text)] + [vocab["<eos>"]]

def decode(model, src_tensor, max_len=50):
    src_tensor = src_tensor.unsqueeze(0).to(device)
    tgt_indices = [tgt_vocab["<sos>"]]
    translated = []
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        #print(output)
        '''
        # Probabilistic sampling instead of greedy search using softmax (top-k sampling).
        # Temperature and topk values are hardcoded for testing.

        temperature = 0.5  
        logits = output[0, -1]
        topk_logs, topk_idx = torch.topk(logits, 20)
        probabilities = torch.softmax(topk_logs, dim=-1)
        sample_idx = torch.multinomial(probabilities, num_samples=1)
        next_token = topk_idx.gather(0, sample_idx).item()
        '''
        # This is the greedy search method that just uses argmax. No sampling done here.

        next_token = output[0, -1].argmax(dim=-1).item()  
        
        if next_token == tgt_vocab["<eos>"]:
            break
        #print(tgt_indices)
        tgt_indices.append(next_token)
        key = [k for k, value in tgt_vocab.items() if value == next_token]
        #print(key)
        translated.append(key)

    return [inv_tgt_vocab[i] for i in tgt_indices[1:]]

def translate(text):
    tokens = encode(text, src_vocab, tokenize_en)
    src_tensor = torch.tensor(tokens, dtype=torch.long)
    translated_tokens = decode(model, src_tensor)
    return " ".join(translated_tokens)

if __name__ == "__main__":
    while True:
        text = input("\nenglish: ")
        if text.lower() in ["quit", "exit"]:
            break
        print("french: ", translate(text))

