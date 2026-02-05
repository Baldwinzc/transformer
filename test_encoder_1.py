import torch
from encoder_1 import Encoder

def test_encoder():
    print("=== 开始测试 Encoder (encoder_1.py) ===")
    
    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4
    d_ff = 32
    
    # 1. 实例化模型
    try:
        encoder = Encoder(d_model, num_heads, d_ff)
        print("1. 模型初始化成功! ✅")
    except Exception as e:
        print(f"1. 模型初始化失败! ❌\n   错误信息: {e}")
        return

    # 2. 准备输入数据
    x = torch.randn(batch_size, seq_len, d_model)
    # mask 形状: (batch_size, 1, 1, seq_len)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    
    print(f"   输入形状: (batch={batch_size}, seq={seq_len}, d_model={d_model})")

    # 3. 前向传播
    try:
        output = encoder(x, mask)
        print(f"   输出形状: {output.shape} (预期: [{batch_size}, {seq_len}, {d_model}])")
        print("2. 前向传播测试通过! ✅")
    except Exception as e:
        print(f"2. 前向传播失败! ❌\n   错误信息: {e}")

if __name__ == "__main__":
    test_encoder()
