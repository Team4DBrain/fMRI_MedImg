import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # 1. Encoder (Downwards)
        # We increase filters from 32/64 to 64/128/256
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        # 2. Decoder (Upwards)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Bridge from bottom layer
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch), # Added BatchNorm to prevent blurriness
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU is better for detail than ReLU
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.pool(e1)
        
        e2 = self.enc2(e2)
        e3 = self.pool(e2)
        
        e3 = self.enc3(e3)
        
        # Decoder path with Skip Connections
        d2 = self.up(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)