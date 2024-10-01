from typing import Union, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileOneBlock"]


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)
class MobileOneBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = int,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_se: bool = True,
        use_act: bool = True,
        use_scale_branch: bool = False,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.pool = nn.AvgPool2d(3, 2, 1)  # 平均池化作为非卷积分支
        self.cov=nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        if use_se:
            self.se = eca_layer(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        # Conv branches
        if num_conv_branches > 0:
            rbr_conv = [self._conv_bn(kernel_size=kernel_size, padding=padding)
                        for _ in range(self.num_conv_branches)]
            self.rbr_conv = nn.ModuleList(rbr_conv)
        else:
            self.rbr_conv = None

        self.rbr_scale = self._conv_bn(kernel_size=1, padding=0) if use_scale_branch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # 使用平均池化分支

        scale_out = self.rbr_scale(x) if self.rbr_scale is not None else 0

         # 直接相加两个分支的输出

        if self.rbr_conv is not None:
            for conv in self.rbr_conv:
                out =  self.pool(conv(x))
        return self.activation(self.se(out+self.bn(self.cov(x))))

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.out_channels)
        )

