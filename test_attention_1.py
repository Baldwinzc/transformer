import torch
from attention_1 import MultiHeadAttention

def test_attention():
    print("=== 开始测试 MultiHeadAttention (attention_1.py) ===")
    
    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4
    
    # 1. 实例化模型
    try:
        mha = MultiHeadAttention(d_model, num_heads)
        print("1. 模型初始化成功! ✅")
    except Exception as e:
        print(f"1. 模型初始化失败! ❌\n   错误信息: {e}")
        return

    # 2. 准备输入数据
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 模拟 mask (假设后面 2 个词被遮挡)
    # mask 形状通常是 (batch_size, 1, 1, seq_len) 或者 (batch_size, seq_len, seq_len)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, -2:] = 0 # 遮挡最后两个
    
    print(f"   输入形状: (batch={batch_size}, seq={seq_len}, d_model={d_model})")

    # 3. 前向传播
    try:
        output = mha(q, k, v, mask=mask)
        print(f"   输出形状: {output.shape} (预期: [{batch_size}, {seq_len}, {d_model}])")
        
        if output.shape == (batch_size, seq_len, d_model):
            print("2. 前向传播测试通过! ✅")
        else:
            print("2. 前向传播输出形状错误! ❌")
            
    except Exception as e:
        print(f"2. 前向传播失败! ❌\n   错误信息: {e}")

if __name__ == "__main__":
    test_attention()
