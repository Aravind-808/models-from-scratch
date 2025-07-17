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

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(sample):
    return tokenizer(sample["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = datasets.map(tokenize, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=16)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=16)

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

optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

EPOCHS = 40
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch in loop:
        tgt_batch = batch["input_ids"].to(device)

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        output = model(tgt_input) 
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        loop.set_postfix(loss=total_loss / (loop.n + 1e-9))

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), r"C:\Users\rajes\models-from-scratch\transformer\models\decoder_only.pth")