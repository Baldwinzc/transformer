import torch
from embedding import Embeddings, PositionalEncoding

def test_code():
    print("=== 开始测试代码 ===")
    vocab_size = 100
    d_model = 16
    max_len = 50
    batch_size = 2
    seq_len = 10

    print("1. 测试 Embeddings 类...")
    try:
        emb = Embeddings(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = emb(x)
        print(f"   Embeddings 输出形状: {output.shape} (预期: [{batch_size}, {seq_len}, {d_model}])")
        print("   Embeddings 测试通过! ✅")
    except Exception as e:
        print(f"   Embeddings 测试失败! ❌\n   错误信息: {e}")

    print("\n2. 测试 PositionalEncoding 类...")
    try:
        pe = PositionalEncoding(d_model, max_len)
        # 模拟 Embeddings 的输出作为输入
        x_emb = torch.randn(batch_size, seq_len, d_model)
        output_pe = pe(x_emb)
        print(f"   PositionalEncoding 输出形状: {output_pe.shape} (预期: [{batch_size}, {seq_len}, {d_model}])")
        print("   PositionalEncoding 测试通过! ✅")
    except Exception as e:
        print(f"   PositionalEncoding 测试失败! ❌\n   错误信息: {e}")

if __name__ == "__main__":
    test_code()
