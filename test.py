import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x): # x.shape = (batch_size, seq_len)
        # Make the input scale more reasonable
        # avoid self.embed(x) too small
        return self.embed(x) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        posIdx = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(posIdx * div_term)
        pe[:, 1::2] = torch.cos(posIdx * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x): # x.shape = (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]