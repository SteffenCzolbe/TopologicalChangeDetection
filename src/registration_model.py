import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
import torchreg
from .models.voxelmorph import Voxelmorph
import pprint


class RegistrationModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.mcdropout = False

        # set dimensionalty for torchreg layers
        torchreg.settings.set_ndims(2)

        # set net
        self.net = Voxelmorph(
            in_channels=1,
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            conv_layers_per_stage=self.hparams.conv_layers_per_stage,
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )

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

    def forward(self, I0, I1, omit_integration=False):
        # activate dropout layers for probabilistic model
        if self.mcdropout:
            dropout_layers = self.get_dropout_layers(self)
            for layer in dropout_layers:
                layer.train()  # activate dropout even when not in training mode

        # run model
        flow = self.net(I0, I1)

        if omit_integration:
            # can be set to avoid doing unneded work, e.g. to speed up training
            return flow
        else:
            # integrate
            transform = self.integrate(flow)
            transform_inv = self.integrate(- flow)
            return transform, transform_inv

    def enable_mcdropout(self, p):
        self.mcdropout = True
        dropout_layers = self.get_dropout_layers(self)
        for layer in dropout_layers:
            layer.train()  # activate dropout
            layer.p = p  # set dropout probability

    def disable_mcdropout(self):
        self.mcdropout = False
        dropout_layers = self.get_dropout_layers(self)
        for layer in dropout_layers:
            layer.p = 0.5  # reset to default

    def get_dropout_layers(self, model):
        """
        Collects all the dropout layers of the model
        """
        ret = []
        for obj in model.children():
            if hasattr(obj, 'children'):
                ret += self.get_dropout_layers(obj)
            if isinstance(obj, torch.nn.Dropout3d) or isinstance(obj, torch.nn.Dropout2d):
                ret.append(obj)
        return ret

    def segmentation_to_onehot(self, S):
        return (
            torch.nn.functional.one_hot(
                S[:, 0], num_classes=self.dataset_config("classes")
            )
            .unsqueeze(1)
            .transpose(1, -1)
            .squeeze(-1)
            .float()
        )

    def _step(self, batch, batch_idx, subset="train"):
        # unpack batch
        I0 = batch['I0']['data']
        I1 = batch['I1']['data']
        seg_available = batch.get('S0')
        if seg_available:
            S0 = batch['S0']['data']
            S1 = batch['S1']['data']

        # predict flowfield
        flow = self.forward(I0, I1, omit_integration=True)
        transform = self.integrate(flow)

        # morph image and segmentation
        Im = self.transformer(I0, transform)
        if seg_available:
            Sm = self.transformer(S0.float(), transform,
                                  mode="nearest").round().long()
            S0_onehot = self.segmentation_to_onehot(S0)
            Sm_onehot = self.transformer(S0_onehot, transform)
            S1_onehot = self.segmentation_to_onehot(S1)

        # calculate loss
        similarity_loss = self.mse(Im, I1)
        diffusion_regularization = self.diffusion_reg(flow)
        loss = similarity_loss + self.hparams.lam * diffusion_regularization
        self.log(f"{subset}/loss", loss)
        self.log(f"{subset}/regularization",
                 diffusion_regularization)
        self.log(f"{subset}/similarity_loss", similarity_loss)

        # calculate other (supervised) evaluation mesures
        if seg_available:
            with torch.no_grad():
                dice_overlap = self.dice_overlap(S_m, S_1)
                accuracy = torch.mean((S_m == S_1).float())
                self.log(f"{subset}/dice_overlap", dice_overlap)
                self.log(f"{subset}/accuracy", accuracy)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, subset="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, subset="val")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, subset="test")

    @staticmethod
    def model_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False
            )
        else:
            parser = argparse.ArgumentParser()

        parser.add_argument(
            "--lam", type=float, default=0.5, help="Diffusion regularizer strength"
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

        return parser
