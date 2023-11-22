import torch.nn as nn
from transformer_block import TransformerBlock
from positional_encoder import PositionalEncoding

class Decoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               dropout_prob):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        TransformerBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob
        ) for _ in range(n_layers)]

    self.transformer_blocks = nn.Sequential(*transformer_blocks) #encapsulate in sequential
    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, vocab_size) #outputs vocab size


  def forward(self, x, pad_mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, pad_mask)

    x = self.ln(x)
    x = self.fc(x) #many-to-many

    return x