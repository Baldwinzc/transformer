import torch
from decoder_1 import Decoder

def test_decoder():
    print("=== 开始测试 Decoder (decoder_1.py) ===")
    
    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4
    d_ff = 32
    
    try:
        decoder = Decoder(d_model, num_heads, d_ff)
        print("1. 模型初始化成功! ✅")
    except Exception as e:
        print(f"1. 模型初始化失败! ❌\n   错误信息: {e}")
        return

    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    memory = torch.randn(batch_size, seq_len, d_model) # Encoder 的输出
    
    # src_mask (Encoder padding mask)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    
    # tgt_mask (Decoder causal mask)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len)
    
    print(f"   输入形状: (batch={batch_size}, seq={seq_len}, d_model={d_model})")

    try:
        output = decoder(x, memory, src_mask, tgt_mask)
        print(f"   输出形状: {output.shape} (预期: [{batch_size}, {seq_len}, {d_model}])")
        print("2. 前向传播测试通过! ✅")
    except Exception as e:
        print(f"2. 前向传播失败! ❌\n   错误信息: {e}")

if __name__ == "__main__":
    test_decoder()
