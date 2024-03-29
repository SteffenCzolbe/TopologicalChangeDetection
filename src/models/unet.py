import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torchreg.nn as tnn
import torchreg.settings as settings
import numpy as np


class UNet(nn.Module):
    """ 
    U-net
    """

    def __init__(self, enc_feat, dec_feat, in_channels=1, conv_layers_per_stage=2, bnorm=False, dropout=True, skip_connections=True):
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

        self.upsample = tnn.Upsample(
            scale_factor=2, mode="linear", align_corners=False)
        self.skip_connections = skip_connections

        # configure encoder (down-sampling path)
        prev_feat = in_channels
        self.encoder = nn.ModuleList()
        for feat in enc_feat:
            self.encoder.append(
                Stage(prev_feat, feat, stride=2,
                      conv_layers=conv_layers_per_stage, dropout=dropout, bnorm=bnorm)
            )
            prev_feat = feat

        if self.skip_connections:
            # pre-calculate decoder sizes and channels
            enc_stages = len(enc_feat)
            dec_stages = len(dec_feat)
            enc_history = list(reversed([in_channels] + enc_feat))
            decoder_out_channels = [
                enc_history[i + 1] + dec_feat[i] for i in range(dec_stages)
            ]
            decoder_in_channels = [enc_history[0]] + decoder_out_channels[:-1]

        else:
            # pre-calculate decoder sizes and channels
            decoder_out_channels = dec_feat
            decoder_in_channels = enc_feat[-1:] + decoder_out_channels[:-1]

        # pre-calculate return sizes and channels
        self.output_length = len(dec_feat) + 1
        self.output_channels = [enc_feat[-1]] + decoder_out_channels
        self.enc_feat = enc_feat
        self.dec_feat = dec_feat

        # configure decoder (up-sampling path)
        self.decoder = nn.ModuleList()
        for i, feat in enumerate(dec_feat):
            self.decoder.append(
                Stage(
                    decoder_in_channels[i], feat, stride=1, conv_layers=conv_layers_per_stage, dropout=dropout, bnorm=bnorm
                )
            )

    def forward(self, x):
        """
        Feed x throught the U-Net

        Parameters:
            x: the input

        Return:
            list of decoder activations, from coarse to fine. Last index is the full resolution output.
        """
        # pass through encoder, save activations
        x_enc = [x]
        for layer in self.encoder:
            x_enc.append(layer(x_enc[-1]))

        # pass through decoder
        x = x_enc.pop()
        x_dec = [x]
        for layer in self.decoder:
            x = layer(x)
            x = self.upsample(x)
            if self.skip_connections:
                x = torch.cat([x, x_enc.pop()], dim=1)
            x_dec.append(x)

        return x_dec


class Stage(nn.Module):
    """
    Specific U-net stage
    """

    def __init__(self, in_channels, out_channels, conv_layers=2, stride=1, bnorm=True, dropout=True):
        super().__init__()

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise ValueError("stride must be 1 or 2")

        # build stage
        layers = []
        layers.append(tnn.Conv(in_channels, out_channels, ksize, stride, 1))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(1, conv_layers):
            layers.append(tnn.Conv(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.LeakyReLU(0.2))
        if bnorm:
            layers.append(tnn.BatchNorm(out_channels))
        if dropout:
            layers.append(tnn.Dropout())

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)
