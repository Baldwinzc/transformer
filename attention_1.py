import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q, k, v, mask=None, dropout_layer=None):
    # (batch_size, num_heads, seq_len, seq_len)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:  # 修正: 判断 mask 是否存在
        scores = scores.masked_fill(mask == 0, float('-inf')) # 修正: 使用 scores.masked_fill
    
    weight = torch.softmax(scores, dim=-1)

    if dropout_layer is not None: # 修正: 判断 dropout 是否存在
        weight = dropout_layer(weight)
    
    # (batch_size, num_heads, seq_len, d_k)
    return torch.matmul(weight, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__() # 修正: 加上括号
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads # 修正: 使用整数除法 //
        self.num_heads = num_heads

        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 修正: 必须使用 -1 来自动推断 seq_len，或者先获取 seq_len = q.size(1)
        # 这里使用 -1 是最安全的做法
        
        # Q: (batch, heads, seq, dk)
        q = self.linear[0](q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # K: (batch, heads, seq, dk)
        # 修正: 拼写错误 self,num_heads -> self.num_heads
        k = self.linear[1](k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # V: (batch, heads, seq, dk)
        v = self.linear[2](v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        x = scaled_dot_product_attention(q, k, v, mask, self.dropout).transpose(1, 2)
        
        # Concat
        x = x.contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.output_linear(x)
