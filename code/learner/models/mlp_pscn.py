import sys
import torch
import torch.nn as nn  # for builtin modules including Linear, Conv2d, MultiheadAttention, LayerNorm, etc
import torch.nn.functional as F
from torch.nn import ModuleDict  # for layer naming when nn.Sequential is not viable
import numpy as np  # for some basic dimention computation, might be redundent

from math import ceil, floor
from collections import OrderedDict

# typing
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple
from ctypes import Union


# 权重初始化
def init_weights(layer, init_type="uniform"):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if init_type == "uniform":
            init_type = np.random.choice(
                a=["kaiming", "xavier", "orthogonal"], p=[0.3, 0.3, 0.4]
            )

        if init_type == "kaiming":
            nn.init.kaiming_uniform_(layer.weight)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(layer.weight)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


# 全连接层
class MLP(nn.Module):
    def __init__(
            self,
            dim_list,
            activation=nn.PReLU(),
            last_act=False,
            use_norm=False,
            linear=nn.Linear,
            *args,
            **kwargs,
    ):
        super(MLP, self).__init__()
        assert dim_list, "Dim list can't be empty!"
        layers = []
        for i in range(len(dim_list) - 1):
            layer = init_weights(linear(dim_list[i], dim_list[i + 1], *args, **kwargs))
            layers.append(layer)
            if i < len(dim_list) - 2:
                if use_norm:
                    layers.append(nn.LayerNorm(dim_list[i + 1]))
                layers.append(activation)
        if last_act:
            if use_norm:
                layers.append(nn.LayerNorm(dim_list[-1]))
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 一种兼顾宽度和深度的全连接层，提取信息效率更高
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim, depth, linear=nn.Linear):
        super(PSCN, self).__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert (
                output_dim >= min_dim
        ), f"output_dim must be >= {min_dim} for depth {depth}"
        assert (
                output_dim % min_dim == 0
        ), f"output_dim must be divisible by {min_dim} for depth {depth}"

        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim

        for _ in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True, linear=linear))
            in_dim = out_dim // 2
            out_dim //= 2

    def forward(self, x):
        out_parts = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                split_size = self.output_dim // (2 ** (i + 1))
                part, x = torch.split(x, [split_size, split_size], dim=-1)
                out_parts.append(part)
            else:
                out_parts.append(x)

        out = torch.cat(out_parts, dim=-1)
        return out


if __name__ == '__main__':
    input = torch.zeros((32, 512))
    net = PSCN(512, 512, 3)
    out = net(input)
    print(net)
