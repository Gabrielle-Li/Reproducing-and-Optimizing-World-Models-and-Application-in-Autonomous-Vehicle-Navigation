import sys
sys.path.append("../..")
from model import iresnet
import torch.nn.functional as F
import torch
import torch.nn as nn


class MapModule(nn.Module):
    def __init__(self, args):
        super(MapModule, self).__init__()
        self.map_features = iresnet.iresnet50(num_classes=args.map_embedding_size)  # 网络框架，提取特征
        self.ca = ChannelSENet(in_channels=args.map_embedding_size)
        self.sa = SpatialSENet(in_channels=args.map_embedding_size)
        self.reconstruction = Reconstruction(in_channels=args.map_embedding_size, out_height=1500, out_width=844)

    def forward(self, x):
        x = self.map_features(x)
        attention = self.ca(x)
        attention = self.sa(attention)
        fake_map = self.reconstruction(attention)
        return x, fake_map


class VisionModule(nn.Module):
    def __init__(self, args):
        super(VisionModule, self).__init__()
        self.view_features = iresnet.iresnet18(num_classes=args.view_embedding_size)  # 网络框架，提取特征
        self.ca = ChannelSENet(in_channels=args.map_embedding_size)
        self.sa = SpatialSENet(in_channels=args.map_embedding_size)
        self.reconstruction = Reconstruction(in_channels=512, out_height=100, out_width=100)

    def forward(self, x):
        x = self.view_features(x)
        attention = self.ca(x)
        attention = self.sa(attention)
        fake_view = self.reconstruction(attention)
        return x, fake_view


class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelSENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelSENet, self).__init__()
        self.se_block = SEBlock(in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        return self.se_block(x)


class SpatialSENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialSENet, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        return channel_out * spatial_out


# 定义重建模块
class Reconstruction(nn.Module):
    def __init__(self, in_channels, out_height=100, out_width=100):
        super(Reconstruction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)
        self.out_height = out_height
        self.out_width = out_width

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        # 改变输出形状
        x = F.interpolate(x, size=(self.out_height, self.out_width), mode='bilinear', align_corners=False)
        return x


def ssim(image1, image2):
    # 将张量转换为浮点数类型
    image1 = image1.float()
    image2 = image2.float()

    # 计算 SSIM
    ssim_value = F.mse_loss(image1, image2)

    return ssim_value.item()
