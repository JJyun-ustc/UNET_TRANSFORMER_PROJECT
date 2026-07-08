import torch
from torch import nn

from .blocks import ConvBlock, DownsampleBlock, TransformerBottleneck, UpsampleBlock


class UNetTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        transformer_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8)

        self.bottleneck_conv = ConvBlock(base_channels * 8, base_channels * 8)
        self.transformer = TransformerBottleneck(
            channels=base_channels * 8,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=transformer_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = UpsampleBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x = self.down3(s3)

        x = self.bottleneck_conv(x)
        x = x + self.transformer(x)

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        return self.head(x)
