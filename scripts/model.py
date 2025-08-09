import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.se(x)
        return self.relu(x + residual)

class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, use_se=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        filters = [64, 128, 256, 512, 1024]

        self.enc1 = ResidualConvBlock(in_channels, filters[0], use_se)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(filters[0], filters[1], use_se)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(filters[1], filters[2], use_se)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualConvBlock(filters[2], filters[3], use_se)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ResidualConvBlock(filters[3], filters[4], use_se)

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(filters[3], filters[3], filters[3] // 2) if use_attention else nn.Identity()
        self.dec4 = ResidualConvBlock(filters[4], filters[3], use_se)

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(filters[2], filters[2], filters[2] // 2) if use_attention else nn.Identity()
        self.dec3 = ResidualConvBlock(filters[3], filters[2], use_se)

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(filters[1], filters[1], filters[1] // 2) if use_attention else nn.Identity()
        self.dec2 = ResidualConvBlock(filters[2], filters[1], use_se)

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(filters[0], filters[0], filters[0] // 2) if use_attention else nn.Identity()
        self.dec1 = ResidualConvBlock(filters[1], filters[0], use_se)

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        # For deep supervision
        self.ds4 = nn.Conv2d(filters[3], out_channels, kernel_size=1)
        self.ds3 = nn.Conv2d(filters[2], out_channels, kernel_size=1)
        self.ds2 = nn.Conv2d(filters[1], out_channels, kernel_size=1)

        # For Grad-CAM targeting
        #self.target_layer = self.enc4  # <-- Layer to visualize with Grad-CAM
        self.target_layer = self.dec4

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final_conv(d1)

        # Deep Supervision
        ds_out4 = F.interpolate(self.ds4(d4), size=out.shape[2:], mode="bilinear", align_corners=False)
        ds_out3 = F.interpolate(self.ds3(d3), size=out.shape[2:], mode="bilinear", align_corners=False)
        ds_out2 = F.interpolate(self.ds2(d2), size=out.shape[2:], mode="bilinear", align_corners=False)

        return out, ds_out2, ds_out3, ds_out4
