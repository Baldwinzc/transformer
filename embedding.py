import torch
import torch.nn as nn
import math

# from transformer.explain_pos_2i import d_model
# from transformer.explain_script import vocab_size

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # 修正拼写错误: embeb -> embed
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x): # x.shape = (batch_size, seq_len)
        # 修正: 使用 self.embed 和 self.d_model
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__() # 必须调用父类初始化
        # 修正拼写错误: zeors -> zeros
        pe = torch.zeros(max_len, d_model)
        posIdx = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1) # shape:(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape:(d_model/2,)

        pe[:, 0::2] = torch.sin(posIdx * div_term)
        pe[:, 1::2] = torch.cos(posIdx * div_term)
        # 注册 buffer
        self.register_buffer('pe', pe.unsqueeze(0)) # pe.shape = (1,max_len,d_model)

    def forward(self, x): # x.shape = (batch_size, seq_len, d_model)
        # 修正: 使用 self.pe，并且 pe 应该是类属性
        return x + self.pe[:, :x.size(1), :]

if __name__ == "__main__":
    vocab_size = 10000
    d_model = 6
    max_len=5000
    emb = Embeddings(vocab_size,d_model).embed
    print(emb.weight.shape)
    # x = torch.random()
    x = torch.randint(0, vocab_size, (3, 2))
    # print(x)
    output = emb(x)
    # print(output)
    print(output.shape)

    pe = PositionalEncoding(d_model,max_len)
    output_2 = pe(output)
    print(output_2.shape)
    # time()
