import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== 问题三演示: transpose(-2, -1) ===")
# 创建一个形状为 (Batch=1, Heads=1, SeqLen=2, Dim=3) 的张量
K = torch.tensor([[[[1, 2, 3], 
                    [4, 5, 6]]]])
print(f"原始 K 形状: {K.shape}")
print(f"原始 K 最后两维:\n{K[0,0]}")

# 转置最后两个维度
K_T = K.transpose(-2, -1)
print(f"转置后 K_T 形状: {K_T.shape}")
print(f"转置后 K_T 最后两维:\n{K_T[0,0]}")
print("解释: (2, 3) 变成了 (3, 2)，行变成了列。")
print("-" * 30)

print("=== 问题四演示: masked_fill ===")
scores = torch.tensor([[10.0, 20.0, 30.0], 
                       [40.0, 50.0, 60.0]])
# 0 表示要遮挡，1 表示保留
mask = torch.tensor([[1, 1, 0], 
                     [1, 0, 0]]) 

print("原始分数:\n", scores)
print("Mask (0是被遮挡):\n", mask)

# 将 mask==0 的位置填为 -inf
masked_scores = scores.masked_fill(mask == 0, float('-inf'))
print("masked_fill 后:\n", masked_scores)
print("-" * 30)

print("=== 问题六演示: Dropout ===")
dropout = nn.Dropout(p=0.5) # 50% 概率丢弃
weight = torch.ones(5, 5)
output = dropout(weight)
print("原始全1矩阵经过 Dropout(0.5):\n", output)
print("解释: 约一半元素变成 0，剩下的元素变大(乘以 1/(1-p)=2)以保持期望不变。")
