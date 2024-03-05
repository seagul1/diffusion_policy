import math
import torch
import torch.nn as nn

# 使用了正余弦编码来嵌入编码去噪过程中的时间步T
# 接受一个shape是(batch, 1)的tensor作为输入（即批处理过程中，所代表的此时的加噪时刻T），
# 然后返回一个shape是(batch_size, dim)的tensor. dim是每一个T嵌入的维度
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 广播相乘 [x.shape[0], 1] * [1, half_dim] = [x.shape[0], half_dim]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

