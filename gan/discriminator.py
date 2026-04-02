import torch
import torch.nn as nn


class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator for GliGAN.

    Based on Ferreira et al. (2022) with modifications from the paper:
    - Stack of 3D Conv layers with increasing filters
    - InstanceNorm + LeakyReLU
    - Final conv + Sigmoid for real/fake probability map

    Input: concat(scan, label) → (B, 5, 96, 96, 96)
    Output: (B, 1, P, P, P) probability map
    """

    def __init__(self, in_channels=5, base_filters=64):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.layers = nn.Sequential(
            # No norm on first layer
            nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(base_filters, base_filters * 2, stride=2),
            conv_block(base_filters * 2, base_filters * 4, stride=2),
            conv_block(base_filters * 4, base_filters * 8, stride=1),
            # Final layer: 1-channel output + sigmoid
            nn.Conv3d(base_filters * 8, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
