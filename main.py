import torch
import json
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformer_decoder import Decoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from training import train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

path_lines = ["/content/Harry Potter 1 Sorcerer's_Stone.txt",
              "/content/Harry Potter 2 Chamber_of_Secrets.txt",
              "/content/Harry Potter 3 Prisoner of Azkaban.txt",
              "/content/Harry Potter 4 and the Goblet of Fire.txt",
              "/content/Book 5 - The Order of the Phoenix.txt",
              "/content/Book 6 - The Half Blood Prince.txt",
              "/content/Book 7 - The Deathly Hallows.txt"]

#every setence is in the list sentence
sentences = []
for path in path_lines:
  with open(path, mode="r") as file:
    content = file.read()
    sentences = content.split("\n\n")

#context length
max = 512
text = 0

#transform sentence to avoid noisy character
for _ in range(2):
  for idx, i in enumerate(sentences):
    if i =="" or i==" ":
      del sentences[idx]
    if i.startswith("Page | ") or len(i) < 3:
      del sentences[idx]
    if len(i) >= max:
      temp = sentences.pop(idx).split(".")
      sentences.extend(temp)
      text = idx

#convert setence into json file name harry.json
with open("harry.json", "w") as f:
  for x in sentences:
    j = {"sentence": x}
    s = json.dumps(j)
    f.write(f"{s}\n")
    

#convert json file into dataset
raw_datasets = load_dataset("json", data_files='harry.json', split="train")

#tokenize
def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

#tokeniz padding and truncation
tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#remove setnece column
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])

#set into a dataloader for use
train_loader = DataLoader(
    tokenized_datasets["train"],
    shuffle = True,
    batch_size = 32,
    collate_fn=data_collator
)

#maximum context length
tokenizer.max_model_input_sizes[checkpoint]

#transformer model
model = Decoder(
    vocab_size = tokenizer.vocab_size,
    max_len = tokenizer.max_model_input_sizes[checkpoint],
    d_k = 32,
    d_model=128,
    n_heads=8,
    n_layers=4,
    dropout_prob=0.1
)
model.to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) #avoid padding tokens for the loss
optimizer = torch.optim.Adam(model.parameters())

train_losses = train(
    model, criterion, optimizer, train_loader, epochs=100
)

# generate something
prompt = input(" ")

tokenized_prompt = tokenizer(prompt, return_tensors='pt')

# prepare inputs + get rid of SEP token at the end
input_ids = tokenized_prompt['input_ids'][:, :-1].to(device)
mask = tokenized_prompt['attention_mask'][:, :-1].to(device)

for _ in range(100):
  outputs = model(input_ids, mask)
  prediction_id = torch.argmax(outputs[:, -1, :], axis=-1)

  input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))
  mask = torch.ones_like(input_ids)

  if prediction_id == tokenizer.sep_token_id:
    break

tokenizer.decode(input_ids[0][1:-1])