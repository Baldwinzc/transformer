import torch
from attention_2 import MultiHeadAttention

def test_attention_2():
    print("=============test====================")

    batch_size = 2
    seq_len = 10
    d_model = 16
    num_heads = 4

    try:
        mha =MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败!  错误信息:{e}")
        return 

    q = torch.randn(batch_size,seq_len,d_model)
    k = torch.randn(batch_size,seq_len,d_model)
    v = torch.randn(batch_size,seq_len,d_model)

    try:
        output = mha(q, k, v)
        print(f"输出形状:{output.shape}")
        print("前向传播成功")
    except Exception as e:
        print("前向传播失败")

if __name__ == "__main__":
    test_attention_2()