import torch
import torch.nn as nn
import torchreg.nn as tnn
import numpy as np
from typing import *
import pytorch_lightning as pl


class VAEAnaomalyDetection(pl.LightningModule):
    """Anomaly detection from paper:
    Variational Autoencoder based Anomaly Detection using Reconstruction Probability
    """

    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams

        # fixes an earlier bug, required to load old models
        if self.hparams.channels[0] > self.hparams.channels[-1]:
            self.hparams.channels.reverse()

        modules = []
        in_channels = self.hparams.data_dims[0]
        hidden_dims = self.hparams.channels
        stages = len(hidden_dims)
        self.last_hidden_dim = (hidden_dims[-1],
                                self.hparams.data_dims[1] // 2**stages,
                                self.hparams.data_dims[2] // 2**stages,
                                max(self.hparams.data_dims[3] // 2**stages, 1))
        latent_dim = self.hparams.latent_dim
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    tnn.Conv(in_channels, out_channels=h_dim,
                             kernel_size=3, stride=2, padding=1),
                    tnn.BatchNorm(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.mu_layer = tnn.Conv(in_channels, out_channels=latent_dim,
                                 kernel_size=3, stride=2, padding=1)
        self.var_layer = tnn.Conv(in_channels, out_channels=latent_dim,
                                  kernel_size=3, stride=2, padding=1)
        # Build Decoder
        modules = []

        hidden_dims = hidden_dims[::-1]
        decoder_input = tnn.ConvTranspose(latent_dim,
                                          hidden_dims[0],
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)
        modules.append(decoder_input)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    tnn.ConvTranspose(hidden_dims[i],
                                      hidden_dims[i + 1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                    tnn.BatchNorm(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tnn.ConvTranspose(hidden_dims[-1],
                              hidden_dims[-1],
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              output_padding=1),
            tnn.BatchNorm(hidden_dims[-1]),
            nn.LeakyReLU(),
            tnn.Conv(hidden_dims[-1], out_channels=self.hparams.data_dims[0],
                     kernel_size=3, padding=1),
            nn.Tanh())

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

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu_layer(result)
        log_var = self.var_layer(result)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self, I, recon, mu, log_var) -> dict:
        recon_loss = nn.functional.mse_loss(recon, I)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var -
                                              mu ** 2 - log_var.exp(), dim=(1, 2, 3, 4)), dim=0)

        loss = 1 / (self.hparams.sigma ** 2) * recon_loss + kl_loss

        return loss, recon_loss, kl_loss

    def forward(self, I) -> dict:
        mu, log_var = self.encode(I)
        recon = self.decode(mu)
        recon_loss = torch.mean((I - recon)**2, dim=1, keepdim=True)
        return -recon_loss

    def bound(self, I0: torch.Tensor, I1: torch.Tensor, bidir=False):
        """Fake-implementation of the probability bound

        Args:
            I0 (torch.Tensor): [description]
            I1 (torch.Tensor): [description]
            bidir (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        bound_1 = self.forward(I1)

        if not bidir:
            return bound_1, None
        else:
            return None, bound_1, None, None

    def step(self, I: torch.Tensor) -> Dict[str, float]:
        mu, log_var = self.encode(I)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)

        loss, recon_loss, kl_loss = self.loss_function(I, recon, mu, log_var)

        logs = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "latent_mu_norm": (torch.sum(mu**2, dim=1)**0.5).mean(),
            "latent_log_var": log_var.mean(),
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        I = batch['I']['data']
        loss, logs = self.step(I)
        self.log_dict({f"train/{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        I = batch['I']['data']
        loss, logs = self.step(I)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        I = batch['I']['data']
        loss, logs = self.step(I)
        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VAE")
        parser.add_argument(
            "--channels",
            nargs="+",
            type=int,
            default=[64, 128, 256, 512],
            help="Encoder and decoder chnanels",
        )
        parser.add_argument(
            "--sigma", type=float, default=1, help="Recon loss sigma"
        )
        parser.add_argument(
            "--latent_dim", type=int, default=1024, help="Latent dim"
        )
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
