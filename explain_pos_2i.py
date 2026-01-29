import torch
import math

# 设定简单的参数以便观察
max_len = 4   # 假设句子最长 4 个词: ["I", "love", "neural", "networks"]
d_model = 6   # 每个词用 6 维向量表示

print(f"=== 参数设置: 句子长度={max_len}, 向量维度={d_model} ===\n")

# 1. posIdx (位置索引)
# 对应公式中的 pos
posIdx = torch.arange(0, max_len).float().unsqueeze(-1)
print("1. posIdx (位置索引):")
print(posIdx)
print("   含义: 代表单词在句子中的顺序。")
print("   pos=0 -> 'I'")
print("   pos=1 -> 'love'")
print("   pos=2 -> 'neural'")
print("   pos=3 -> 'networks'\n")

# 2. 2i (维度索引)
# 对应公式中的 2i
# 代码: torch.arange(0, d_model, 2)
two_i = torch.arange(0, d_model, 2).float()
print("2. 2i (维度索引 - 偶数部分):")
print(two_i)
print("   含义: 代表词向量中的第几个分量。")
print(f"   我们的向量有 {d_model} 维，索引分别是 0, 1, 2, 3, 4, 5")
print("   这里的 2i 取的是偶数索引: 0, 2, 4")
print("   (奇数索引 2i+1 会使用对应的 cos 函数)\n")

# 3. 结合意义：频率计算
# 公式: 1 / 10000^(2i/d_model)
div_term = torch.exp(two_i * (-math.log(10000.0) / d_model))
print("3. 不同 2i 对应的频率 (div_term):")
for idx, val in zip(two_i, div_term):
    print(f"   维度 2i={int(idx.item())}: 频率值 = {val.item():.4f}")

print("\n=== 总结 ===")
print("posIdx 控制行: 不同的单词 (哪一行)")
print("2i     控制列: 不同的频率 (哪一列)")
print("低维 (2i=0) 频率高，变化快 (像秒针)")
print("高维 (2i=4) 频率低，变化慢 (像时针)")
