import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg
import torchreg.nn as tnn
from torch.distributions.normal import Normal

from .unet import UNet


class Encoder(nn.Module):
    """
    UNet  with some smoothin layers at the end
    """

    def __init__(
        self, in_channels, enc_feat, dec_feat, conv_layers_per_stage, bnorm=False, dropout=True
    ):
        """ 
        Parameters:
            in_channels: channels of the input
            enc_feat: List of encoder filters. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder filters. e.g. [32, 32, 32, 16]
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
        """
        super().__init__()

        # configure backbone
        self.unet = UNet(
            enc_feat, dec_feat, in_channels=in_channels*2, conv_layers_per_stage=conv_layers_per_stage, dropout=dropout, bnorm=bnorm,
        )
        unet_out_channels = self.unet.output_channels[-1]

        # some smoothing layers
        self.smooth = nn.Sequential(tnn.Conv(unet_out_channels, unet_out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(0.2),
                                    tnn.Conv(
                                        unet_out_channels, unet_out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(0.2),
                                    )

        # configure output layer
        self.mu = FlowPredictor(
            in_channels=unet_out_channels, init_mean=0, init_std=1e-5)
        self.log_var = FlowPredictor(
            in_channels=unet_out_channels, init_mean=-20, init_std=1e-2)

    def forward(self, x0, x1):
        # feed through network
        x = torch.cat([x0, x1], dim=1)
        h = self.unet(x)[-1]
        h = self.smooth(h)
        mu = self.mu(h)
        log_var = self.log_var(h)

        return mu, log_var


class FlowPredictor(nn.Module):
    """
    A layer intended for flow prediction. Initialied with small weights for faster training.
    """

    def __init__(self, in_channels, init_mean=0, init_std=1e-5):
        super().__init__()
        """
        instantiates the flow prediction layer.
        
        Parameters:
            in_channels: input channels
        """
        self.ndims = torchreg.settings.get_ndims()
        # configure cnn
        self.layer = tnn.Conv(in_channels, 3, kernel_size=3, padding=1)

        # init final cnn layer with small weights and bias
        self.layer.weight = nn.Parameter(
            Normal(0, init_std).sample(self.layer.weight.shape)
        )
        self.layer.bias = nn.Parameter(
            init_mean * torch.ones(self.layer.bias.shape))

    def forward(self, x):
        # predict the flow
        flow = self.layer(x)
        if self.ndims == 2:
            flow[:, 2] = 0  # no depth-whise flow in 2d case
        return flow
