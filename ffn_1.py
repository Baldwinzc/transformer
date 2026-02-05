import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__() # 必须先初始化父类
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    batch_size = 6 
    seq_len = 2
    d_model = 10
    x = torch.rand(batch_size, seq_len, d_model)
    print(f"x.shape:{x.shape}")
    print(f"x.value:{x}")
    ffn = FeedForward(d_model)
    x = ffn(x)
    print(f"x.shape:{x.shape}")
    print(f"x.value:{x}")
