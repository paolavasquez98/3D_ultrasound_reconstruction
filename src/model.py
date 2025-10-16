import torch
import torch.nn as nn
import torch.nn.init
import complextorch.nn as ctnn

# ------------- CID Net implementation in 3D + AMU activation -------------
class AMU(nn.Module):
    def __init__(self, pieces=4):
        super().__init__()
        self.pieces = pieces

    def forward(self, x):  # x is complex: (B, C, D, H, W)
        x_a = x.abs()
        feature_maps = x_a.shape[1] // self.pieces
        out_shape = (x_a.shape[0], feature_maps, self.pieces, *x_a.shape[2:])
        x_a = x_a.view(out_shape)
        idx = x_a.argmax(dim=2, keepdim=True)
        del x_a

        x = x.view(out_shape)
        x = x.gather(dim=2, index=idx).squeeze(dim=2)
        return x


class CIDNet3D(nn.Module):
    def __init__(self, in_channels = 9, pieces = 4):
        super(CIDNet3D, self).__init__()
        self.pieces = pieces
        self.amu = AMU(pieces=pieces)
        self.layer1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, dtype=torch.cfloat) #192
        self.layer2 = nn.Conv3d(in_channels=32//pieces, out_channels=64, kernel_size=3, padding=1, dtype=torch.cfloat) #192
        self.layer3 = nn.Conv3d(in_channels=64//pieces, out_channels=32, kernel_size=3, padding=1, dtype=torch.cfloat) #192

        # The muklti path with different kernel sizes to get different spatial features
        self.layer4_1 = nn.Conv3d(in_channels=32//pieces, out_channels=4, kernel_size=(7, 7, 7), padding=(3, 3, 3), dtype=torch.cfloat) #192
        self.layer4_2 = nn.Conv3d(in_channels=32//pieces, out_channels=4, kernel_size=(5, 5, 5), padding=(2, 2, 2), dtype=torch.cfloat)
        self.layer4_3 = nn.Conv3d(in_channels=32//pieces, out_channels=4, kernel_size=(3, 3, 3), padding=(1, 1, 1), dtype=torch.cfloat)
        # self.layer4_4 = nn.Conv3d(in_channels=64//pieces, out_channels=4, kernel_size=3, padding=1, dtype=torch.cfloat)
        
        # Final refinement layer
        self.layer5 = nn.Conv3d(in_channels=3, out_channels=4, kernel_size=1, padding=0, dtype=torch.cfloat)

    def forward(self, x):
        x = self.layer1(x)
        x = self.amu(x)

        x = self.layer2(x)
        x = self.amu(x)

        x = self.layer3(x)
        x = self.amu(x)

        x_1 = self.layer4_1(x)
        x_1 = self.amu(x_1)
        x_2 = self.layer4_2(x)
        x_2 = self.amu(x_2)
        x_3 = self.layer4_3(x)
        x_3 = self.amu(x_3)

        x = torch.cat([x_1, x_2, x_3], dim=1)
        del x_1, x_2, x_3
        x = self.layer5(x)
        x = self.amu(x)

        return x

# ---------------------- 3D UNet with complex operations ----------------------
    
class ComplexBatchNorm3D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, **kwargs):
        super().__init__()
        self.real_bn = nn.BatchNorm3d(num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm3d(num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats)

    def forward(self, x):
        real = self.real_bn(x.real)
        imag = self.imag_bn(x.imag)
        return torch.complex(real, imag)
    
class ComplexGroupNorm3D(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()
        self.real_gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        self.imag_gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
    
    def forward(self, x):
        real = self.real_gn(x.real)
        imag = self.imag_gn(x.imag)
        return torch.complex(real, imag)

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.complex(self.relu(x.real), self.relu(x.imag))

class ComplexLReLU(nn.Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.Lrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        return torch.complex(self.Lrelu(x.real), self.Lrelu(x.imag))

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, dtype=torch.cfloat),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU()
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, padding=0, stride=2, dtype=torch.cfloat),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU()
        )
    def forward(self, x):
        return self.block(x)
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(Bottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU()
        )
    def forward(self, x):
        return self.block(x)

class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(UNetResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, dtype=torch.cfloat),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            ComplexGroupNorm3D(num_groups, out_channels)
        )

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, dtype=torch.cfloat)
            if in_channels != out_channels else nn.Identity()
        )

        self.relu = ComplexReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))
    
class ComplexCBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ComplexCBAM, self).__init__()
        
        # Channel attention
        self.avg_pool = ctnn.modules.pooling.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, dtype=torch.cfloat),
            ComplexReLU(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, dtype=torch.cfloat)
        )

        # Spatial attention
        self.conv_spatial = nn.Conv3d(1, 1, kernel_size=7, padding=3, dtype=torch.cfloat)

    def forward(self, x):
        # ----- Channel Attention -----
        avg_out = self.avg_pool(x)            # shape: [B, C, 1, 1, 1]
        channel_att = torch.sigmoid(self.mlp(avg_out))
        x = x * channel_att

        # ----- Spatial Attention -----
        avg = torch.mean(x, dim=1, keepdim=True)         # [B, 1, D, H, W]
        spatial_att = torch.sigmoid(self.conv_spatial(avg))
        x = x * spatial_att

        return x
    
class BottleneckWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        self.bottleneck = Bottleneck(in_channels, out_channels, num_groups)
        self.attn = ComplexCBAM(out_channels)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.attn(x)
        return x
    
class CxUnet_RB(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()

        self.encoder1 = UNetResidualBlock(in_channels, 16, num_groups=2)
        self.encoder2 = UNetResidualBlock(16, 32, num_groups=4)
        self.encoder3 = UNetResidualBlock(32, 64, num_groups=4)
        self.encoder4 = UNetResidualBlock(64, 128, num_groups=8)

        self.bottleneck = BottleneckWithAttention(128, 256, num_groups=8)

        self.decoder1 = DecoderBlock(256 + 128, 128, num_groups=8)
        self.decoder2 = DecoderBlock(128 + 64, 64, num_groups=4)
        self.decoder3 = DecoderBlock(64 + 32, 32, num_groups=4)
        self.decoder4 = DecoderBlock(32 + 16, 16, num_groups=2)

        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1, dtype=torch.cfloat)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d1 = self.decoder1(torch.cat([b, e4], dim=1))
        d2 = self.decoder2(torch.cat([d1, e3], dim=1))
        d3 = self.decoder3(torch.cat([d2, e2], dim=1))
        d4 = self.decoder4(torch.cat([d3, e1], dim=1))

        out = self.final_conv(d4)

        return out