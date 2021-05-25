import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torchreg.nn as tnn
import torchreg.settings as settings
import numpy as np
from src.models.unet import UNet


class SegNet(nn.Module):
    """ 
    U-net
    """

    def __init__(self, features, in_channels=1, out_channels=1, conv_layers_per_stage=2, bnorm=False, dropout=True):
        """
        Parameters:
            enc_feat: List of encoder features. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder features. e.g. [32, 32, 32, 16]
            in_channels: input channels, eg 1 for a single greyscale image. Default 1.
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
            skip_connections: bool, Set for U-net like skip cnnections
        """
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            enc_feat=features,
            dec_feat=features[::-1],
            conv_layers_per_stage=conv_layers_per_stage,
            bnorm=bnorm,
            dropout=dropout,
        )
        layers = [
            tnn.Conv(self.unet.output_channels[-1], out_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            tnn.Conv(out_channels * 2, out_channels, 3, 1, 1),
        ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        feats = self.unet(x)
        return self.classifier(feats[-1])
