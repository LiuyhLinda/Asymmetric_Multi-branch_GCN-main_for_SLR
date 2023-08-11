"""
来自DecoupleGCN-DropGraph网络（完全一样）
temporal attention-guided DropGraph
采用DropBlock
dropout:随机丢弃元素，对于全连接层有效，但对于卷积层无效。因为尽管某个单元被dropout掉，但相邻元素依然可以保留该位置的语义信息。
针对卷积网络，采用DropBlock，将feature map中相邻区域中的单元一起drop掉。
DropBlock主要包括两个参数：block_size是要删除的块的大小，γ(gamma)控制要删除多少个激活单元.
block_size：对所有特征图设置一个恒定的块大小
希望保持每个激活单元的概率为keep_prob（在0.75到0.95之间）,作者采用γ的计算公式为：γ=(1-keep_prob)/block_size
"""

import torch
import torch.nn.functional as F
from torch import nn

class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()  # 剩余维度为n，t
        # 激活单元的绝对值表示单元的重要性，从而确定attention区域。
        # abs函数：获取绝对值；detach函数：不计算梯度，使得requires_grad=false
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)
        # numel():返回数组中元素个数；view函数：改变形状
        gamma = (1. - self.keep_prob) / self.block_size  # gamma控制着样本概率
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)  # (n,c,v,t)→(n,c*v,t)
        # permute():维度换位;contiguous()为了使内存连续，否则view报错;
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
        # clamp():将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。这里的作用是将非零元素化为1。
        # bernoulli():通过伯努利分布，生成二进制随机数（0或1）。输出值为1表示激活单元。
        # bernoulli()：第i个元素的输出值要根据输入的第i个概率值，来决定是否生成1值。输入值的范围[0,1].
        # repeat():对张量进行复制，在axis=1维度上复制。
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        # //：取整除，返回商的整数部分；
        # 进行1维最大池化，对通道、节点进行最大池化，只保留时间特征
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        # mask：挡住到根节点邻域中最大阶数为K阶的节点。这部分节点不进行激活，即丢弃了。丢弃的=（1-保留的）
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)  # 变回(n,c,t,v)

