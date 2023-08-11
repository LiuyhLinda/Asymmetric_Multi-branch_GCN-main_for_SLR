"""
来自DecoupleGCN-DropGraph网络（完全一样）
spatial(skeleton) attention-guided DropGraph
"""
import torch
import torch.nn.functional as F
from torch import nn
import warnings


class DropBlock_Ske(nn.Module):
    def __init__(self, num_point, block_size=7):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point

    def forward(self, input, keep_prob, A):  # input:(n,c,t,v)
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()  # 分别为样本数，channel，时间帧数，节点数量

        input_abs = torch.mean(torch.mean(
            torch.abs(input), dim=2), dim=1).detach()  # 剩余维度为n，v，只保留空间维度
        # 采用绝对值，用于评估单元的重要性，从而进行attention。
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        if self.num_point == 25:  # Kinect V2（Kinect v2最多可支持25个骨骼节点）
            gamma = (1. - self.keep_prob) / (1 + 1.92)  # 猜测分母为经验数值
        elif self.num_point == 20:  # Kinect V1（Kinect v1最多可以支持20个骨骼点）
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            warnings.warn('undefined skeleton graph')
        M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed, A)  # matmul()返回两个矩阵乘积,A为邻接矩阵。
        M[M > 0.001] = 1.0  # 相当于大于0的元素赋值为1；其余元素为0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_point)  # (n,1,1,v) 只保留空间特征。
        return input * mask * mask.numel() / mask.sum()
