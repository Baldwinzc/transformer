import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__() # 修正: 必须加上括号 ()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x): # x.shape = (batch_size, seq_len)
        # 修正: 使用 self.d_model 而不是外部变量 d_model
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__() # 修正: 必须加上括号 ()
        pe = torch.zeros(max_len, d_model) # 修正: 不需要 self.pe，因为后面要 register_buffer
        posIdx = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1) #(max_len,1)
        div_temp = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) ##(d_model/2,)

        pe[:, 0::2] = torch.sin(posIdx * div_temp)
        pe[:, 1::2] = torch.cos(posIdx * div_temp)
        
        # 修正: 注册的是 pe 变量，不是 self.pe
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): # (batch_size, seq_len, d_model)
        # 修正: 切片语法错误，应该是 :x.size(1)
        return x + self.pe[:, :x.size(1), :]

