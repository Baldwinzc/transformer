import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q, k, v, mask = None, dropout_layer = None): # (batch_size, num_heads, seq_len, d_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k) # (batch_size, num_heads, seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weight = torch.softmax(scores, dim=-1)

    if dropout_layer is not None:
        weight = dropout_layer(weight)

    return torch.matmul(weight, v) # (batch_size, num_heads, seq_len, d_k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask = None): # (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        # (batch_size, num_heads, seq_len, d_k)
        q = self.linear[0](q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear[1](k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear[2](v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
     
        x = scaled_dot_product_attention(q, k, v, mask, self.dropout).transpose(1, 2) # (batch_size, seq_len, num_heads, d_k)
        x = x.contiguous().view(batch_size, -1, self.num_heads * self.d_k) # (batch_size, seq_len, d_model)

        return self.output_linear(x)