import torch
import torch.nn as nn

# ===== DoubleConv（加入 Dropout）=====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        """
        in_ch: 输入通道数
        out_ch: 输出通道数
        dropout: Dropout 概率
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # 🔥 加入 dropout

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)   # 🔥 加入 dropout
        )

    def forward(self, x):
        return self.conv(x)

# ===== UNet =====
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout=0.2):
        super().__init__()

        self.down1 = DoubleConv(n_channels, 64, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256, dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(256, 512, dropout)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bridge
        br = self.bridge(p3)

        # Decoder
        u3 = self.up3(br)
        u3 = torch.cat([u3, d3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, d2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, d1], dim=1)
        d1 = self.dec1(u1)

        return self.out(d1)
