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
from .models.elbo import ELBO
import pprint
from typing import Dict, Optional


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
        self.decoder = FixedDecoder(integrate=self.hparams.integrate)
        self.elbo = ELBO(data_dims=self.hparams.data_dims,
                         use_analytical_solution=self.hparams.use_analytical_solution_for_alpha_beta,
                         init_prior_log_alpha=self.hparams.prior_log_alpha,
                         trainable_alpha=self.hparams.trainable_alpha,
                         init_prior_log_beta=self.hparams.prior_log_beta,
                         trainable_beta=self.hparams.trainable_beta,
                         init_recon_log_var=self.hparams.recon_log_var,
                         trainable_recon_var=self.hparams.trainable_recon_var,)

        # various components for loss caluclation and evaluation
        self.dice_overlap = torchreg.metrics.DiceOverlap(
            classes=list(range(self.hparams.data_classes))
        )
        self.jacobian_determinant = torchreg.metrics.JacobianDeterminant(
            reduction='none')

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

    def forward(self, I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            I0 (torch.Tensor): moving image
            I1 (torch.Tensor): fixed image

        Returns:
            [torch.Tensor]: Pixel-wise upper bound on -log p(I1, I0)
        """
        mu, log_var = self.encoder(I0, I1)
        transform = mu  # take mean during forward (no sample)
        transform, transform_inv = self.decoder.get_transform(
            mu, inverse=True)  # get transform and inverse
        I01, _ = self.decoder.apply_transform(transform, I0)  # morph image

        bound_1, _, _ = self.elbo.loss(
            mu, log_var, I01, I1, reduction='none')  # calculate bound
        return bound_1, transform, I01

    def sample_transformation(self, mu, log_var):
        std = torch.exp(log_var / 2)
        # conditional posterior distribution q(z | x)
        q = torch.distributions.Normal(mu, std)
        # sample z ~ q(z | x)
        transform = q.rsample()
        if self.ndims == 2:
            transform[:, 2] = 0  # no depth-whise flow in 2d case
        return transform

    def step(self, I0: torch.Tensor, I1: torch.Tensor, S0: Optional[torch.Tensor] = None, S1: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """[summary]

        Args:
            I0 (torch.Tensor): [description]
            I1 (torch.Tensor): [description]
            S0 ([type], optional): [description]. Defaults to None:Optional[torch.Tensor].
            S1 ([type], optional): [description]. Defaults to None:Optional[torch.Tensor].

        Returns:
            Dict[str, float]: log values
        """
        mu, log_var = self.encoder(I0, I1)
        transform = self.sample_transformation(mu, log_var)
        I01, S01 = self.decoder(transform, I0, seg=S0)

        loss, recon_loss, kl_loss = self.elbo.loss(
            mu, log_var, I01, I1, reduction='mean')

        with torch.no_grad():
            jacdet = self.jacobian_determinant(mu)

        logs = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mean_latent_log_var": log_var.mean(),
            "prior_log_alpha": self.elbo.prior_log_alpha.mean(),
            "prior_log_beta": self.elbo.prior_log_beta.mean(),
            "recon_log_var": self.elbo.recon_log_var.mean(),
            "transformation_smoothness": -jacdet.var(),
            "transformation_folding": (jacdet <= 0).float().mean(),
        }

        if S0 is not None:
            # evaluate supervised measures
            with torch.no_grad():
                logs["seg_dice_overlap"] = self.dice_overlap(S01, S1)
                logs["seg_accuracy"] = torch.mean((S01 == S1).float())

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
        S0 = batch['S0']['data']
        S1 = batch['S1']['data']
        loss, logs = self.step(I0, I1, S0=S0, S1=S1)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        I0 = batch['I0']['data']
        I1 = batch['I1']['data']
        S0 = batch['S0']['data']
        S1 = batch['S1']['data']
        loss, logs = self.step(I0, I1, S0=S0, S1=S1)
        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Registration Model")
        parser.add_argument(
            "--channels",
            nargs="+",
            type=int,
            default=[64, 128, 256, 512],
            help="U-Net encoder channels. Decoder uses the reverse. Defaukt: [64, 128, 256, 512]",
        )
        parser.add_argument(
            "--integrate", action="store_true", help="set to integrate flow field."
        )
        parser.add_argument(
            "--use_analytical_solution_for_alpha_beta", action="store_true", help="Set to use analytical solution for prior parameters alpha, beta"
        )
        parser.add_argument(
            "--prior_log_alpha", type=float, default=-4, help="Prior parameter log alpha"
        )
        parser.add_argument(
            "--trainable_alpha", action="store_true", help="Set to make trainable"
        )
        parser.add_argument(
            "--prior_log_beta", type=float, default=7.8, help="Parameter initialization"
        )
        parser.add_argument(
            "--trainable_beta", action="store_true", help="Set to make trainable"
        )
        parser.add_argument(
            "--recon_log_var", type=float, default=-5, help="Parameter initialization"
        )
        parser.add_argument(
            "--trainable_recon_var", action="store_true", help="Set to make trainable"
        )
        parser.add_argument(
            "--bnorm", action="store_true", help="set to use batchnormalization."
        )
        parser.add_argument(
            "--dropout", action="store_true", help="set to use dropout")
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate. Default: 0.0001"
        )
        parser.add_argument(
            "--lr_decline_patience", type=int, default=10, help="LR halving after x epochs of no improvement. Default: 10"
        )
        parser.add_argument(
            "--conv_layers_per_stage", type=int, default=1, help="Convolutional layer sper network stage. Default: 2"
        )

        return parent_parser
