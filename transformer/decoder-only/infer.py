from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from .transformer_architecture import DecoderOnlyTransformer 
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(sample):
    return tokenizer(sample["text"], padding="max_length", truncation=True, max_length=256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DecoderOnlyTransformer(
    target_vocab_size=tokenizer.vocab_size,
    d=512,
    heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_len=256,
    dropout=0.1
).to(device)

model.load_state_dict(torch.load(r"C:\Users\rajes\models-from-scratch\transformer\models\decoder_only.pth", map_location=device))
model.eval()

# def generate_lyrics(prompt, max_new_tokens = 50):
#     model.eval()

#     tokens = tokenizer(prompt, return_tensors = "pt")
#     input_ids = tokens["input_ids"].to(device)

#     for token in range(max_new_tokens):
#         if input_ids.size(1) >= 256:
#             break

#         with torch.no_grad():
#             output = model(input_ids)
#             logits = output[:, -1, :]
#             # greedy: next_token = torch.argmax(logits, dim = -1).unsqueeze(1)
#             # sampling:
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#         input_ids = torch.cat((input_ids, next_token), dim=1)
    
#     generated_text = tokenizer.decode(input_ids[0], skip_special_tokens = True)

#     return generated_text

def generate_lyrics(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        input_trimmed = generated[:, -256:]

        with torch.no_grad():
            outputs = model(input_trimmed)  # (1, seq_len, vocab)
            logits = outputs[:, -1, :] / temperature  # (1, vocab)

            topk_logits, topk_indices = torch.topk(logits, top_k)
            probs = torch.softmax(topk_logits, dim=-1)
            sample_idx = torch.multinomial(probs, num_samples=1)         # shape: [1, 1]
            next_token = topk_indices.gather(1, sample_idx) 

        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    datasets = load_dataset("huggingartists/kendrick-lamar")

    train_percentage = 0.9
    validation_percentage = 0.07
    test_percentage = 0.03

    full_text = datasets['train']['text']
    train_split, val_split, test_split = np.split(
        full_text,
        [int(len(full_text)*train_percentage), int(len(full_text)*(train_percentage + validation_percentage))]
    )

    datasets = DatasetDict({
        'train': Dataset.from_dict({'text': list(train_split)}),
        'validation': Dataset.from_dict({'text': list(val_split)}),
        'test': Dataset.from_dict({'text': list(test_split)})
    })

    tokenized_val = datasets["validation"].map(tokenize, batched=True)

    text = input("Enter prompt: ")
    prompt_tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=80)
    prompt = tokenizer.decode(prompt_tokens["input_ids"][0], skip_special_tokens=True)
    generated = generate_lyrics(model, tokenizer, prompt)
    print(generated)

    # samples = 3
    # for i in range(samples):
    #     input_text = datasets["test"][i]["text"]
    #     prompt_tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=50)
    #     prompt = tokenizer.decode(prompt_tokens["input_ids"][0], skip_special_tokens=True)
    #     generated = generate_lyrics(prompt)
    #     print(f"Input text:\n{input_text[:50]}")
    #     print(f"Expected generation:\n{input_text[:100]}")
    #     print(f"Output:\n{generated[:100]}\n\n")

    # print(repr(datasets["test"][0]["text"]))