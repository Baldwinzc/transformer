import torch
import torch.nn as nn
import math

print("=== 问题二演示: Embedding 形状变换 ===")
vocab_size = 10  # 词表大小
d_model = 6      # 每个词的向量维度
batch_size = 2   # 2句话
seq_len = 3      # 每句话3个词

# 模拟输入 x: (batch_size, seq_len)
x = torch.tensor([
    [1, 5, 9],  # 第一句话的单词ID
    [2, 0, 3]   # 第二句话的单词ID
])
print(f"输入 x 形状: {x.shape}") # [2, 3]

embedding_layer = nn.Embedding(vocab_size, d_model)
output = embedding_layer(x)
print(f"经过 Embedding 后形状: {output.shape}") # [2, 3, 6]
print("解释: 每个单词ID (标量) 被替换成了一个长度为 6 的向量。")
print("-" * 30)

print("=== 问题三演示: 位置编码细节 (d_model=4, max_len=5) ===")
d_model = 4
max_len = 5

# Line 21 分解
# torch.arange(0, d_model, 2) -> [0, 2]
indices = torch.arange(0, d_model, 2).float()
print(f"1. 偶数索引 (indices): {indices}") 

# 常数部分
scale = -math.log(10000.0) / d_model
print(f"2. 缩放常数: {scale:.4f}")

# 指数部分
div_term = torch.exp(indices * scale)
print(f"3. div_term (Line 21 结果): {div_term}")
print(f"   含义: 这是频率项，形状是 (d_model/2,) = ({d_model//2},)")

# Line 20 分解
posIdx = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
print(f"4. posIdx (Line 20) 形状: {posIdx.shape}") # (5, 1)

# 计算乘积 (广播机制)
# (5, 1) * (2,) -> (5, 2)
angle = posIdx * div_term 
print(f"5. posIdx * div_term 形状: {angle.shape}")
print(f"   前两行数值:\n{angle[:2]}")

# Line 23 & 24 填充
pe = torch.zeros(max_len, d_model)
# 偶数列填 sin
pe[:, 0::2] = torch.sin(angle)
# 奇数列填 cos
pe[:, 1::2] = torch.cos(angle)

print(f"6. 最终 PE 矩阵形状: {pe.shape}")
print(f"   前两个位置的编码:\n{pe[:2]}")
