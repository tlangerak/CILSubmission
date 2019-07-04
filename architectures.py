'''
Three architectures
UNET
WNET
FCN32 with a pretrained VGG (not used).
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture_utils import *
from constants import *
import math
from torchvision.models.vgg import VGG
from torchvision import models

def select_model(name, args):
    intermediate_supervision = False
    if name == "U-Net":
        model = UNet(3, 2)
    elif name == "R2U-Net":
        model = R2U_Net()
    elif name == "AttU-Net":
        model = AttU_Net()
    elif name == "R2AttU-Net":
        model = R2AttU_Net()
    elif name == "U-Net2":
        model = UNet2(3, 2)
    elif name == "W2-Net":
        model = WNet(3, 2)
    elif name == "W16-Net":
        model = WNet(3, 16)
    elif name == "PW-Net":
        model = PWNet(3, args.premodel)
    elif name == "Leaky-UNet":
        model = LeakyUNet(3, 2)
    elif name == "Leaky-R2UNet":
        model = LeakyR2Net()
    elif name == "W64-Net":
        model = WNet(3, 64)
    elif name == "PCNet":
        model = PCNet(3, args.premodel)
    elif name == "DeeplabV3+":
        model = DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True, _print=True)
    elif name == "Leaky-R2UNet-NoSigmoid":
        model = LeakyR2NetNoSigmoid()
    elif name == "W-Net":
        model = WNet(3, 1, intermediate=False, leaky=args.leaky, batch_norm=args.batch_norm, drop_rate=args.drop_rate,
                     dilate_first=args.dilate_first, dilate_second=args.dilate_second)
    elif name == "W-Net-Intermediate":
        model = WNet(3, 1, intermediate=True, leaky=args.leaky, batch_norm=args.batch_norm, drop_rate=args.drop_rate,
                     dilate_first=args.dilate_first, dilate_second=args.dilate_second)
        intermediate_supervision = True
    elif name == "R2-W-Net-Intermediate":
        model = R2_WNet(3, t=2, leaky=args.leaky, batch_norm=args.batch_norm, drop_rate=args.drop_rate,
                        dilate_first=args.dilate_first, dilate_second=args.dilate_second)
        intermediate_supervision = True

    elif name == "FCN32":
        model = FCN32s()
        intermediate_supervision = False
    else:
        model = None
        print("Not a valid model")
        exit(1)
    return intermediate_supervision, model

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# our work.
class WNet(nn.Module):
    def __init__(self, n_channels, n_midlayer=1, intermediate=False, leaky=False, batch_norm=True, drop_rate=0.0,
                 dilate_first=1, dilate_second=1):
        super(WNet, self).__init__()
        self.intermediate = intermediate
        self.inc = inconv(n_channels, 64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                          dilation=dilate_first)
        self.down1 = down(64, 128, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.down2 = down(128, 256, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.down3 = down(256, 512, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.down4 = down(512, 512, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.up1 = up(1024, 256, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.up2 = up(512, 128, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.up3 = up(256, 64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.up4 = up(128, 64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_first)
        self.outc = outconv(64, n_midlayer)
        self.inc2 = inconv(n_midlayer, 64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate,
                           dilation=dilate_second)
        self.down12 = down(128, 128, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.down22 = down(128, 256, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.down32 = down(256, 512, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.down42 = down(512, 512, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.up12 = up(1024, 256, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.up22 = up(512, 128, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.up32 = up(256, 64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.up42 = up(128, 64, leaky=leaky, batch_norm=batch_norm, drop_rate=drop_rate, dilation=dilate_second)
        self.outc2 = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        y5 = self.outc(y4)
        z1 = self.inc2(y5)
        z2 = self.down12(torch.cat([z1, y4], dim=1))
        z3 = self.down22(z2)
        z4 = self.down32(z3)
        z5 = self.down42(z4)
        w1 = self.up12(z5, z4)
        w2 = self.up22(w1, z3)
        w3 = self.up32(w2, z2)
        w4 = self.up42(w3, z1)
        w5 = self.outc2(w4)
        return torch.cat([y5, w5], dim=1) if self.intermediate else w5

## not used at the moment. 
class FCN32s(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_class = 1
        self.pretrained_net = VGGNet(requires_grad=True)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        return score