#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Conv2D(nn.Module):
    """ conv -> batchnorm -> relu の標準的な畳み込み層クラス． """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class BasicConv(nn.Module):
    """ ECOの2D Netモジュール内の最初の畳み込みネットワーク """

    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1 = nn.Sequential(
            Conv2D(3, 64, kernel_size=7, stride=2, padding=3),      # size 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size 1/2
        )
        self.conv2 = Conv2D(64, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Sequential(
            Conv2D(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size 1/2
        )

    def forward(self, x):
        c1 = self.conv1(x)    # [3, 224, 224] -> [64, 56, 56]
        c2 = self.conv2(c1)   # [64, 56, 56] -> [64, 56, 56]
        out = self.conv3(c2)  # [64, 56, 56] -> [192, 28, 28]
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 224, 224)  # 入力用のテストtensor

    # Basic Convモジュールのテスト
    basic_conv = BasicConv()
    basic_out = basic_conv(input_tensor)
    print('Basic Conv output:', basic_out.shape)
