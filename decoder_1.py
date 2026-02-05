import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        return self.linear2(x)

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

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        # 修正: 使用 nn.ModuleList 而不是 nn.Module
        self.layerNorm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        x_norm = self.layerNorm[0](x)

        # 修正: self attention 应该使用 tgt_mask (掩盖未来信息)
        self_atten_output = self.self_attention(
            x_norm,
            x_norm,
            x_norm,
            tgt_mask
        )

        x  = x + self.dropout(self_atten_output)

        # 修正: cross attention 应该使用 src_mask (掩盖 encoder 的 padding)
        cross_atten_output = self.cross_attention(
            self.layerNorm[1](x),
            memory,
            memory,
            src_mask
        )

        x = x + self.dropout(cross_atten_output)

        # 修正: 加上 Residual Connection
        x = x + self.dropout(self.ffn(self.layerNorm[2](x)))

        return x 
