import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.conv = DoubleConv(cout * 2, cout)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]      # 16, 32, 64, 128, 256
        self.enc1, self.enc2 = DoubleConv(3, c[0]), DoubleConv(c[0], c[1])
        self.enc3, self.enc4 = DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4])
        self.up4, self.up3 = Up(c[4], c[3]), Up(c[3], c[2])
        self.up2, self.up1 = Up(c[2], c[1]), Up(c[1], c[0])
        self.head = nn.Conv2d(c[0], 1, kernel_size=1)

    def forward(self, x):                       # x: B×3×120×160
        h = x.shape[-2]
        x = F.pad(x, (0, 0, 0, (-h) % 16), mode="replicate")       # pad bottom → height divisible by 16 (128)
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))
        b = self.bottleneck(F.max_pool2d(s4, 2))
        y = self.up1(self.up2(self.up3(self.up4(b, s4), s3), s2), s1)
        return self.head(y)[..., :h, :]          # logits B×1×120×160 (padding cropped)