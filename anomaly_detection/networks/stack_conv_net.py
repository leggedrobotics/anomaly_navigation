import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly_detection.base.base_net import BaseNet


class StackConvNet(BaseNet):

    def __init__(self, in_channels=3, use_bn=False, use_dropout=True):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        if use_dropout:
            self.drop = nn.Dropout2d(p=0.05)

        self.conv1 = nn.Conv2d(in_channels, 32, 5, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False)
        self.fconv1 = nn.Conv2d(128, self.rep_dim, 1, bias=False)
        if use_bn:
            self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
            self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
            self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn2d1(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2d2(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.pool(x)

        x = self.conv3(x)
        if self.use_bn:
            x = self.bn2d3(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = self.fconv1(x)
        return x


class StackConvNet_Autoencoder(BaseNet):

    def __init__(self, in_channels=3, use_bn=False, use_dropout=True):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        if use_dropout:
            self.drop = nn.Dropout2d(p=0.05)

        # Encoder (must match the network above)
        self.conv1 = nn.Conv2d(in_channels, 32, 5, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.fconv1 = nn.Conv2d(128, self.rep_dim, 1, bias=False)
        if use_bn:
            self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
            self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
            self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
            self.bn2d = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(self.rep_dim, 128, 1, bias=False)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.deconv4 = nn.ConvTranspose2d(32, in_channels, 5, bias=False)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        if use_bn:
            self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
            self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
            self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn2d1(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2d2(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.pool(x)

        x = self.conv3(x)
        if self.use_bn:
            x = self.bn2d3(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = self.fconv1(x)
        if self.use_bn:
            x = self.bn2d(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = self.deconv1(x)
        if self.use_bn:
            x = self.bn2d4(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = self.deconv2(x)
        if self.use_bn:
            x = self.bn2d5(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.deconv3(x)
        if self.use_bn:
            x = self.bn2d6(x)
        x = F.leaky_relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.deconv4(x)
        x = torch.tanh(x)

        return x
