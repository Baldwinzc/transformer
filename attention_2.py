import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q, k, v, mask=None, dropout_layer=None):
    # (batch_size, num_heads, seq_len, seq_len)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:  # 修正: 使用 is not None 判断
        scores = scores.masked_fill(mask == 0, float('-inf')) # 修正: 拼写 mak -> mask
    
    weight = torch.softmax(scores, dim=-1) # 修正: 必须指定 dim=-1

    if dropout_layer is not None: # 修正: 增加 None 判断
        weight = dropout_layer(weight)
    
    # (batch_size, num_heads, seq_len, d_k)
    return torch.matmul(weight, v) # 修正: 变量名 x -> v

class MultiHeadAttention(nn.Module): # 修正: 拼写 Mutil -> Multi
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__() # 修正: super.__init__() -> super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" # 修正: d_mode -> d_model
        self.d_model = d_model
        self.d_k = d_model // num_heads # 修正: 使用整数除法 //
        self.num_heads = num_heads
        
        # 修正: liner -> linear
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None): # (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # (batch_size, num_heads, seq_len, d_k)
        # 修正: self.liner -> self.linear
        q = self.linear[0](q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear[1](k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear[2](v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # (batch_size, num_heads, seq_len, d_k)
        x = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        
        # 修正: 缺少 transpose(1, 2) 把头合并回来
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 修正: self.output_liner -> self.output_linear
        return self.output_linear(x)
