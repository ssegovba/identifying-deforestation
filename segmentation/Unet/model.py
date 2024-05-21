### Final model, adding batch normalization and more dropout layers ####
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        self.enc5 = self.contracting_block(512, 1024)
        self.enc6 = self.contracting_block(1024, 2048)  # Added an additional layer

        self.upconv5 = self.expansive_block(2048 + 1024, 1024)
        self.upconv4 = self.expansive_block(1024 + 512, 512)
        self.upconv3 = self.expansive_block(512 + 256, 256)
        self.upconv2 = self.expansive_block(256 + 128, 128)
        self.upconv1 = self.expansive_block(128 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)  # Addded Dropout layer with 50% probability

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        enc6 = self.enc6(self.pool(enc5))  # Added forward pass for the additional layer

        upconv5 = self.upconv5(self.up(enc6, enc5))
        upconv4 = self.upconv4(self.up(upconv5, enc4))
        upconv3 = self.upconv3(self.up(upconv4, enc3))
        upconv2 = self.upconv2(self.up(upconv3, enc2))
        upconv1 = self.dropout(self.upconv1(self.up(upconv2, enc1)))  # Added dropout before final output

        return torch.sigmoid(self.final(upconv1))

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def expansive_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def up(self, x, bridge):
        up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.cat([up, bridge], 1)

    def pool(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)


