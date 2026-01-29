import torch
import torch.nn as nn

print("=== 问题三演示: 广播机制 vs 矩阵乘法 ===")
# 形状 (5, 1)
posIdx = torch.tensor([[0.], [1.], [2.], [3.], [4.]]) 
# 形状 (2,) -> 等价于 (1, 2)
div_term = torch.tensor([10., 100.]) 

print(f"posIdx 形状: {posIdx.shape}")
print(f"div_term 形状: {div_term.shape}")

# 逐元素相乘 (广播)
result = posIdx * div_term
print(f"posIdx * div_term 结果形状: {result.shape}")
print("结果值:\n", result)
print("解释: 这不是矩阵乘法(matmul)。这是广播机制：")
print("1. div_term 被复制了5行，变成 (5, 2)")
print("2. posIdx 被复制了2列，变成 (5, 2)")
print("3. 然后对应位置相乘")
print("-" * 30)

print("=== 问题四演示: 切片语法 [:, 0::2] ===")
# 创建一个 5行 4列 的矩阵
pe = torch.arange(20).reshape(5, 4)
print("原始矩阵 pe (5x4):\n", pe)

# [:, 0::2] 解析
# 第一部分 `:` 表示取所有行
# 第二部分 `0::2` 是 python 切片语法 start:stop:step
# 从索引0开始，到末尾，步长为2 -> 取索引 0, 2
col_even = pe[:, 0::2]
print("\npe[:, 0::2] (取所有行，取偶数列 0, 2):\n", col_even)

# [:, 1::2] 解析
# 从索引1开始，到末尾，步长为2 -> 取索引 1, 3
col_odd = pe[:, 1::2]
print("\npe[:, 1::2] (取所有行，取奇数列 1, 3):\n", col_odd)
