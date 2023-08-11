"""
AM-GCN主干网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d
from graph.graph_part import parts as pa_parts  # --add
from model.PA_TA_attention import ST_Joint_Att5,Tpart_Att,ACCAblock, SCCAblock

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    """
    这里修改成膨胀卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, autopad=True,
                 num_point=27, block_size=41):
        super(unit_tcn, self).__init__()
        if autopad:
            pad = int((kernel_size - 1) * dilation // 2)
        else:
            pad = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))  # 卷积+归一化
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    """ 修改 """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, autopad=True):
        super(unit_tcn_skip, self).__init__()
        if autopad:
            pad = int((kernel_size - 1) * dilation // 2)
        else:
            pad = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)



class MultiScale_TemporalConv4(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 dilations=[1, 2, 3],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations) + 3) == 0, '# out channels should be multiples of # branches' # 需要整除6！

        self.num_branches = len(dilations) + 3
        branch_channels = out_channels // self.num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=(1, 5),
                    padding=(0, 2)),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=5,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(7, 1), padding=(3, 0), stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)
        self.dropS = DropBlock_Ske(num_point=27)
        self.dropT = DropBlockT_1d(block_size=41)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, keep_prob, A):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out = self.dropT(self.dropS(self.bn(out), keep_prob, A), keep_prob)

        return out


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
            3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True)
                                      .repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)

    def norm(self, A):
        b, c, h, w = A.size()  # batch_size,channel,h,w
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()

        x = x + self.Linear_bias  # x0·w+b
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, dilation=1,
                 autopad=True, residual=True,
                 attention_ST=False, st_joint=False,
                 tpart=False,
                 attention_C=False, acca=False,
                                    scca=False):

        super(TCN_GCN_unit, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)


        self.tcn1 = MultiScale_TemporalConv4(out_channels, out_channels, stride=stride)

        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
            3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'),
                              requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

        if attention_ST:
            print('Attention Enabled!')
            self.sigmoid = nn.Sigmoid()

            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

        # =================加tpart时间增强============
        if tpart:
            self.Tpart_Att = Tpart_Att(out_channels, 3)

        # =================加STjoint时空注意力============
        if st_joint:
            self.STjoint = ST_Joint_Att5(out_channels, 2, 3, True, pa_parts)

        if attention_C:
            # channel attention
            rr = 2
            self.sigmoid = nn.Sigmoid()
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

        # =================加ACCA通道注意力============
        if acca:
            self.acca_block = ACCAblock(out_channels, 2)
        elif scca:
            self.scca_block = SCCAblock(out_channels, 6)


    def forward(self, x, keep_prob):
        y = self.gcn1(x)

        if self.attention_ST:
            se = y.mean(-2)
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y

            se = y.mean(-1)  # N C T V → N C T
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y

        # =================加STjoint时空注意力============
        if self.st_joint:
            y = self.STjoint(y)

        # =================加tpart时间增强============
        if self.tpart:
            y = self.Tpart_Att(y)

        if self.attention_C:
            # channel attention
            se = y.mean(-1).mean(-1)  # N C T V → N C T → N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        # =================ACCA模块=================
        if self.acca:
            y=self.acca_block(y)
        if self.scca:
            y=self.scca_block(y)

        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class TCN_GCN_unit2(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, dilation=1,
                 autopad=True, residual=True,
                 attention_ST=False, st_joint=False,
                 tpart=False,
                 attention_C=False, acca=False,
                                    scca=False):
        super(TCN_GCN_unit2, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)

        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride, num_point=num_point, dilation=dilation, autopad=autopad)

        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
            3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'),
                              requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(in_channels, out_channels,
                            kernel_size=1, stride=stride, dilation=dilation, autopad=autopad)

        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)


        if attention_ST:
            print('Attention Enabled!')
            self.sigmoid = nn.Sigmoid()
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
        # =================加STjoint时空注意力============
        if st_joint:
            self.STjoint = ST_Joint_Att5(out_channels, 2, 3, True, pa_parts)

        if tpart:
            self.Tpart_Att = Tpart_Att(out_channels, 3)

        if attention_C:
            rr = 2
            self.sigmoid = nn.Sigmoid()
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

        # =================ACCA模块=================
        if acca:
            self.acca_block = ACCAblock(out_channels, 2)
        elif scca:
            self.scca_block = SCCAblock(out_channels, 6)


    def forward(self, x, keep_prob):
        y = self.gcn1(x)

        if self.attention_ST:
            se = y.mean(-2)
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y

            se = y.mean(-1)  # N C T V → N C T
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y

        # =================加STjoint时空注意力============
        if self.st_joint:
            y = self.STjoint(y)
        # =================加tpart时间增强============
        if self.tpart:
            y = self.Tpart_Att(y)

        if self.attention_C:
            # channel attention
            se = y.mean(-1).mean(-1)  # N C T V → N C T → N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        # =================ACCA模块=================
        if self.acca:
            y = self.acca_block(y)

        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(),
                 in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)


        self.l1 = TCN_GCN_unit(in_channels, 96, A, groups, num_point,
                               block_size, residual=False)
        self.l2 = TCN_GCN_unit(96, 96, A, groups, num_point, block_size)
        self.l3 = TCN_GCN_unit(96, 96, A, groups, num_point, block_size)
        self.l5 = TCN_GCN_unit(
            96, 192, A, groups, num_point, block_size, stride=2)
        self.l6 = TCN_GCN_unit(192, 192, A, groups, num_point, block_size)
        self.l4 = TCN_GCN_unit2(96, 96, A, groups, num_point, block_size)
        self.l7 = TCN_GCN_unit2(192, 192, A, groups, num_point, block_size)
        self.l8 = TCN_GCN_unit2(192, 384, A, groups, num_point, block_size)  # 改

        self.fc = nn.Linear(384, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # -------------mstcn10
        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)

        c_new = x.size(1)

        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
