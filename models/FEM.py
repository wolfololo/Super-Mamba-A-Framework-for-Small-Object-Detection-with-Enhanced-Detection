import torch
import torch.nn as nn


class FEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.SiLU()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv1(x)))
        x2 = self.act2(self.bn2(self.conv2(x1)))
        x3 = self.act3(self.bn3(self.conv3(x2)))
        out = x1 + x2 + x3
        return self.upsample(out)