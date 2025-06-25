import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C2f, Conv, Bottleneck


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class RFCA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(RFCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            h_sigmoid()
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        z = torch.cat([avg_out, max_out], dim=1)
        z = self.conv(z)
        att = self.sigmoid(z)

        return x * y.expand_as(x) * att.expand_as(x)


class RFCACONV(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(RFCACONV, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.attention = RFCA(c2)

    def forward(self, x):
        return self.attention(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.attention(self.act(self.conv(x)))


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def replace_module(model, old_module_type, new_module_class):
    for name, module in model.named_children():
        if isinstance(module, old_module_type):
            in_channels = module.conv.in_channels
            out_channels = module.conv.out_channels
            kernel_size = module.conv.kernel_size
            stride = module.conv.stride
            padding = module.conv.padding
            groups = module.conv.groups
            act = module.act
            setattr(model, name, new_module_class(in_channels, out_channels, kernel_size, stride, padding, groups, act))
        else:
            replace_module(module, old_module_type, new_module_class)
    return model


