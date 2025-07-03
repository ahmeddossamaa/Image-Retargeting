import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ARSU(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, height=7):
        super(ARSU, self).__init__()
        self.height = height
        self.rebnconvin = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.rebnconv1 = nn.Sequential(
            nn.Conv2d(out_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(height - 2):
            self.encoder_layers.append(nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True)
            ))

        # Attention gates
        self.attention_gates = nn.ModuleList()
        for i in range(height - 1):
            self.attention_gates.append(AttentionGate(mid_ch))

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(height - 2):
            self.decoder_layers.append(nn.Sequential(
                nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True)
            ))

        self.rebnconv_out = nn.Sequential(
            nn.Conv2d(mid_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        # Encoder with attention
        features = [hxin]
        hx = self.rebnconv1(hxin)
        hx = self.attention_gates[0](hx)
        features.append(hx)

        for i, layer in enumerate(self.encoder_layers):
            hx = F.max_pool2d(hx, 2, stride=2, ceil_mode=True)
            hx = layer(hx)
            hx = self.attention_gates[i + 1](hx)
            features.append(hx)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            hx = F.interpolate(hx, size=features[-(i + 2)].shape[2:], mode='bilinear')
            hx = layer(torch.cat([hx, features[-(i + 2)]], 1))

        hx = F.interpolate(hx, size=features[0].shape[2:], mode='bilinear')
        hx = self.rebnconv_out(torch.cat([hx, features[0]], 1))

        return hx + hxin


class AttU2Net(nn.Module):
    def __init__(self):
        super(AttU2Net, self).__init__()

        # Encoder
        self.stage1 = ARSU(3, 32, 64, 7)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = ARSU(64, 32, 128, 6)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = ARSU(128, 64, 256, 5)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = ARSU(256, 128, 512, 4)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = ARSU(512, 256, 512, 4)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = ARSU(512, 256, 512, 4)

        # Decoder
        self.stage5d = ARSU(1024, 256, 512, 4)
        self.stage4d = ARSU(1024, 128, 256, 4)
        self.stage3d = ARSU(512, 64, 128, 5)
        self.stage2d = ARSU(256, 32, 64, 6)
        self.stage1d = ARSU(128, 16, 64, 7)

        # Side outputs
        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)

        # Output fusion
        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        hx = x

        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # Decoder
        hx5d = self.stage5d(torch.cat([F.interpolate(hx6, hx5.size()[2:], mode='bilinear'), hx5], 1))
        hx4d = self.stage4d(torch.cat([F.interpolate(hx5d, hx4.size()[2:], mode='bilinear'), hx4], 1))
        hx3d = self.stage3d(torch.cat([F.interpolate(hx4d, hx3.size()[2:], mode='bilinear'), hx3], 1))
        hx2d = self.stage2d(torch.cat([F.interpolate(hx3d, hx2.size()[2:], mode='bilinear'), hx2], 1))
        hx1d = self.stage1d(torch.cat([F.interpolate(hx2d, hx1.size()[2:], mode='bilinear'), hx1], 1))

        # Side outputs
        d1 = self.side1(hx1d)
        d2 = F.interpolate(self.side2(hx2d), d1.size()[2:], mode='bilinear')
        d3 = F.interpolate(self.side3(hx3d), d1.size()[2:], mode='bilinear')
        d4 = F.interpolate(self.side4(hx4d), d1.size()[2:], mode='bilinear')
        d5 = F.interpolate(self.side5(hx5d), d1.size()[2:], mode='bilinear')
        d6 = F.interpolate(self.side6(hx6), d1.size()[2:], mode='bilinear')

        # Fusion
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))

        return torch.sigmoid(d0)