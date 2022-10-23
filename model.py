import torch
import torch.nn as nn
from module import ConvNorm, LinearNorm

class ResidualUnit(nn.Module):
    def __init__(self, in_channel,  out_channel, kernel= 3, down_sample= False):
        super(ResidualUnit, self).__init__()
        self.down_sample = down_sample
        self.conv= nn.Sequential(
            ConvNorm(in_channel, out_channel, kernel, stride=2 if down_sample else 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            ConvNorm(out_channel, out_channel, kernel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        if self.down_sample:
            self.x_conv = nn.Sequential(
                ConvNorm(in_channel, out_channel, kernel, 2),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x):
        f = self.conv(x)
        if self.down_sample:
            x = self.x_conv(x)
        return f + x


class Resnet(nn.Module):
    def __init__(self, n_class= 10):
        super(Resnet, self).__init__()
        self.prenet = nn.Sequential(
            ConvNorm(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride= 2),
        )
        resnet=[]
        resnet += self.make_residual_block(64, 64, 3)
        resnet += self.make_residual_block(64, 128, 4, down_sample= True)
        resnet += self.make_residual_block(128, 256, 6, down_sample= True)
        resnet += self.make_residual_block(256, 512, 3, down_sample= True)
        self.resnet = nn.Sequential(*resnet)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = LinearNorm(512, 10)

    def make_residual_block(self, in_channel, out_channel, layer, down_sample=False):
        layers = []
        layers.append(ResidualUnit(in_channel, out_channel, down_sample= down_sample))
        for i in range(1, layer):
            layers.append(ResidualUnit(out_channel, out_channel, down_sample= False))
        return layers

    def forward(self, x):
        B = x.shape[0]
        x = self.prenet(x)
        x = self.resnet(x)
        x = self.pool(x)
        out = self.linear(x.view(B, -1))
        return out
