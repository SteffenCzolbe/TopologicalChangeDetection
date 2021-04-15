import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchreg
from .models.encoder import Encoder
from .models.decoder import FixedDecoder
import pprint


class RegistrationModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # set net
        self.ndims = torchreg.settings.get_ndims()
        self.encoder = Encoder(
            in_channels=self.hparams.data_dims[0],
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            conv_layers_per_stage=self.hparams.conv_layers_per_stage,
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )
        self.decoder = FixedDecoder()

        # various components for loss caluclation and evaluation
        self.mse = torch.nn.MSELoss()
        self.diffusion_reg = torchreg.metrics.GradNorm()
        self.dice_overlap = torchreg.metrics.DiceOverlap(
            classes=list(range(self.hparams.data_classes))
        )
        self.transformer = torchreg.nn.SpatialTransformer()
        self.integrate = torchreg.nn.FlowIntegration(nsteps=4)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            factor=0.1,
            patience=self.hparams.lr_decline_patience,
            verbose=True,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val/loss",
                "reduce_on_plateau": True,
            },
        }

    def sample(self, mu, log_var):
        #import ipdb
        # ipdb.set_trace()
        std = torch.exp(log_var / 2)
        # prior distribution p(z)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std))
        # conditional posterior distribution q(z | x)
        q = torch.distributions.Normal(mu, std)
        # sample z ~ q(z | x)
        transform = q.rsample()
        if self.ndims == 2:
            transform[:, 2] = 0  # no depth-whise flow in 2d case
        return p, q, transform

    def forward(self, I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            I0 (torch.Tensor): moving image
            I1 (torch.Tensor): fixed image

        Returns:
            [torch.Tensor]: Pixel-wise upper bound on -log p(I1, I0)
        """
        mu, log_var = self.encoder(I0, I1)
        prior, posterior, transform = self.sample(mu, log_var)
        # TODO: return elbo instead
        return transform, self.decoder(transform, I0)

    def step(self, I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            I0 (torch.Tensor): moving image
            I1 (torch.Tensor): fixed image

        Returns:
            [torch.Tensor]: Pixel-wise upper bound on -log p(I1, I0)
        """
        mu, log_var = self.encoder(I0, I1)
        p, q, transform = self.sample(mu, log_var)
        I01 = self.decoder(transform, I0)

        recon_loss = 1/(2 * self.hparams.var) * \
            F.mse_loss(I01, I1, reduction='mean')

        log_qz = q.log_prob(transform)
        log_pz = p.log_prob(transform)  # TODO: not sure about this part
        kl_loss = torch.mean(log_qz - log_pz)

        loss = kl_loss + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "loss": loss,
            "mean_latent_log_var": log_var.mean(),
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        I0 = batch['I0']['data']
        I1 = batch['I1']['data']
        loss, logs = self.step(I0, I1)
        self.log_dict({f"train/{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        I0 = batch['I0']['data']
        I1 = batch['I1']['data']
        loss, logs = self.step(I0, I1)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        I0 = batch['I0']['data']
        I1 = batch['I1']['data']
        loss, logs = self.step(I0, I1)
        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Registration Model")

        parser.add_argument(
            "--var", type=float, default=1, help="Variance on the reconstruction distribution. Balances the VAE loss terms. A lower variance strengthens the reconstruction loss."
        )
        parser.add_argument(
            "--channels",
            nargs="+",
            type=int,
            default=[64, 128, 256, 512],
            help="U-Net encoder channels. Decoder uses the reverse. Defaukt: [64, 128, 256, 512]",
        )
        parser.add_argument(
            "--bnorm", action="store_true", help="use batchnormalization."
        )
        parser.add_argument(
            "--dropout", action="store_true", help="use dropout")
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)"
        )
        parser.add_argument(
            "--lr_decline_patience", type=int, default=10, help="LR halving after x epochs of no improvement"
        )
        parser.add_argument(
            "--conv_layers_per_stage", type=int, default=2, help="Convolutional layer sper network stage. Default: 2"
        )

        return parent_parser
