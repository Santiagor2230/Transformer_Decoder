import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# A function to encapsulate the training loop
def train(model, criterion, optimizer, train_loader, epochs):
  train_losses = np.zeros(epochs)

  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = []
    for batch in train_loader:
      # move data to GPU
      batch = {k: v.to(device) for k, v in batch.items()}

      # zero the parameter gradients
      optimizer.zero_grad()

      # shift targets backwards
      targets = batch['input_ids'].clone().detach()
      targets = torch.roll(targets, shifts=-1, dims=1)
      targets[:, -1] = tokenizer.pad_token_id

      # Forward pass
      outputs = model(batch['input_ids'], batch['attention_mask'])
      # outputs are N x T x V
      # but PyTorch expects N x V x T
      # print("outputs:", outputs)
      # print("targets:", targets)
      loss = criterion(outputs.transpose(2, 1), targets)
      # N, T, V = outputs.shape
      # loss = criterion(outputs.view(N * T, V), targets.view(N * T))

      # Backward and optimize
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())

    # Get train loss and test loss
    train_loss = np.mean(train_loss)

    # Save losses
    train_losses[it] = train_loss

    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Duration: {dt}')

  return train_losses