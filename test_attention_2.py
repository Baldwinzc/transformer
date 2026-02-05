import torch
from attention_2 import MultiHeadAttention

def test_attention_2():
    print("=== 开始测试 MultiHeadAttention (attention_2.py) ===")
    
    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4
    
    try:
        mha = MultiHeadAttention(d_model, num_heads)
        print("1. 模型初始化成功! ✅")
    except Exception as e:
        print(f"1. 模型初始化失败! ❌\n   错误信息: {e}")
        return

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    try:
        output = mha(q, k, v)
        print(f"   输出形状: {output.shape} (预期: [{batch_size}, {seq_len}, {d_model}])")
        print("2. 前向传播测试通过! ✅")
    except Exception as e:
        print(f"2. 前向传播失败! ❌\n   错误信息: {e}")

if __name__ == "__main__":
    test_attention_2()
