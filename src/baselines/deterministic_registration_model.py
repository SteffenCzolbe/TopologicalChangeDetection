import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg.nn as tnn
import pytorch_lightning as pl
import torchreg
from src.models.encoder import Encoder
import src.util as util
from src.semantic_loss import SemanticLossModel
from typing import Dict, Optional


class DeterministicRegistrationModel(pl.LightningModule):
    """
    Deterministic registration model following "Semantic similarity metrics for learned image registration", Czolbe 2021
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
            fixed_model_var=None,
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )

        # various components for loss caluclation and evaluation
        self.regularizer = torchreg.metrics.GradNorm(
            penalty="l2", reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.dice_overlap = torchreg.metrics.DiceOverlap(
            classes=list(range(self.hparams.data_classes))
        )
        self.jacobian_determinant = torchreg.metrics.JacobianDeterminant(
            reduction='none')
        self.transformer = tnn.SpatialTransformer()
        if self.hparams.semantic_loss:
            self.load_semantic_loss_model(
                model_path=self.hparams.semantic_loss)

    def load_semantic_loss_model(self, model_path):
        # load semantic loss model
        model_checkpoint = util.get_checkoint_path_from_logdir(model_path)
        self.semantic_loss_model = SemanticLossModel.load_from_checkpoint(
            model_checkpoint)
        util.freeze_model(self.semantic_loss_model)
        # adjust image channels for augmented data
        channel_cnt = sum(self.semantic_loss_model.net.enc_feat)
        self.hparams.data_dims = (channel_cnt, *self.hparams.data_dims[1:])

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

    def bound(self, I0: torch.Tensor, I1: torch.Tensor, bidir=False):
        I0_to_I1 = self.forward(I0, I1)

        if not bidir:
            return I0_to_I1["bound"], I0_to_I1
        else:
            I1_to_I0 = self.forward(I1, I0)

            bound_0 = I1_to_I0["bound"] + \
                self.transformer(I0_to_I1["bound"], I1_to_I0["transform"])
            bound_1 = I0_to_I1["bound"] + \
                self.transformer(I1_to_I0["bound"], I0_to_I1["transform"])
            return bound_0, bound_1, I0_to_I1, I1_to_I0,

    def forward(self, I0: torch.Tensor, I1: torch.Tensor) -> dict:
        """calculates the upper bound on -log p(I1 | I0)

        In addition, a dict with additional information is returned.

        Args:
            I0 (torch.Tensor): [description]
            I1 (torch.Tensor): [description]
            bidir (bool, optional): [description]. Defaults to False.

        Returns:
            Dictionary with various information
        """
        raise NotImplementedError("overwrite this method")

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
        # register the images
        transform, _ = self.encoder(I0, I1)

        # augment the images
        if self.hparams.semantic_loss:
            I0 = self.semantic_loss_model.augment_image(I0)
            I1 = self.semantic_loss_model.augment_image(I1)

        # apply the transformation
        I01 = self.transformer(I0, transform)

        # calculate the loss
        reg_loss = self.regularizer(transform)
        sim_loss = self.mse(I01, I1)
        loss = self.hparams.regularizer_strengh * reg_loss + sim_loss
        loss = loss.mean()

        with torch.no_grad():
            jacdet = self.jacobian_determinant(transform)

        logs = {
            "loss": loss,
            "transformation_smoothness": -jacdet.var(),
            "transformation_folding": (jacdet <= 0).float().mean(),
        }

        if S0 is not None:
            # evaluate supervised measures
            with torch.no_grad():
                S01 = self.transformer(S0, transform, mode="nearest")
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
            "--regularizer_strengh", type=float, default=0.1, help="Regularizer strength"
        )
        parser.add_argument(
            "--semantic_loss", type=str, help="Path to semantic model. Set to use semantic reconstruction loss."
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
