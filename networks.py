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

class Inception_A(nn.Module):
    """ ECOの2D Netモジュール内のInceptionモジュールの1つ目 """
    
    def __init__(self):
        super(Inception_A, self).__init__()

        self.inception1 = Conv2D(192, 64, kernel_size=1, stride=1)
        self.inception2 = nn.Sequential(
            Conv2D(192, 64, kernel_size=1, stride=1),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.inception3 = nn.Sequential(
            Conv2D(192, 64, kernel_size=1, stride=1),
            Conv2D(64, 96, kernel_size=3, stride=1, padding=1),
            Conv2D(96, 96, kernel_size=3, stride=1, padding=1),
        )
        self.inception4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2D(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out1 = self.inception1(x)  # [192, 28, 28] -> [64, 28, 28]
        out2 = self.inception2(x)  # [192, 28, 28] -> [64, 28, 28]
        out3 = self.inception3(x)  # [192, 28, 28] -> [96, 28, 28]
        out4 = self.inception4(x)  # [192, 28, 28] -> [32, 28, 28]
        # channels方向に結合，shape: [64+64+96+32, 28, 28]
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 224, 224)  # 入力用のテストtensor

    # Basic Convモジュールのテスト
    basic_conv = BasicConv()
    basic_out = basic_conv(input_tensor)
    print('Basic Conv output:', basic_out.shape)

    # InceptionAモジュールのテスト
    inception_a = Inception_A()
    inception_a_out = inception_a(basic_out)
    print('Inception A output:', inception_a_out.shape)
