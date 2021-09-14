import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchreg
from src.models.unet import UNet
from typing import Dict, Optional


class SemanticLossModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # set net
        self.ndims = torchreg.settings.get_ndims()
        self.net = UNet(
            in_channels=self.hparams.data_dims[0],
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-
                                           1][:-1] + [self.hparams.data_classes],
            conv_layers_per_stage=self.hparams.conv_layers_per_stage,
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )
        # various components for loss caluclation and evaluation
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.dice_overlap = torchreg.metrics.DiceOverlap(
            classes=list(range(self.hparams.data_classes))
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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

    def class_probabilities(self, x):
        # run model
        y_pred_onehot, y_pred_raw = self.net(x)
        return y_pred_onehot

    def forward(self, x):
        # run model
        feat = self.net(x)
        # take last feature activation
        y_pred_raw = feat[-1]
        # softmax
        y_pred_onehot = F.softmax(y_pred_raw, dim=1)
        # class prediction
        y_pred = torch.argmax(y_pred_onehot, dim=1, keepdim=True)
        return y_pred, y_pred_onehot, y_pred_raw

    def extract_features(self, x):
        """
        Extracts deep features from the input.
        """
        self.eval()
        feat = []
        for stage in self.net.encoder:
            x = stage(x)
            feat.append(x)
        return feat

    def augment_image(self, x):
        """
        Augments an image by extracting deep features.
        """
        H, W, D = x.shape[2:]
        # get features
        feats = self.extract_features(x)

        # expand deeper layers to same dimensionality as input
        if D == 1:
            feats = [F.interpolate(feat.squeeze(-1), size=(H, W), mode='bicubic', align_corners=False).unsqueeze(-1)
                     for feat in feats]
        else:
            feats = [F.interpolate(feat, size=(H, W, D), mode='trilinear', align_corners=False)
                     for feat in feats]  # TODO: cubic interpolation for 2d data

        # stack along channel dimension
        feats = torch.cat(feats, dim=1)

        # normalize by channel
        channel_norms = (feats**2).sum(dim=[2, 3, 4], keepdim=True) ** 0.5
        feats = feats / channel_norms

        return feats

    def step(self, x: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        # predict
        y_pred, y_pred_onehot, y_pred_raw = self.forward(x)

        loss = self.cross_entropy_loss(y_pred_raw, y_true.squeeze(1))
        dice_overlap = self.dice_overlap(y_true, y_pred)
        accuracy = torch.mean((y_true == y_pred).float())

        logs = {
            "loss": loss,
            "dice_overlap": dice_overlap,
            "accuracy": accuracy,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        x = batch['I']['data']
        y = batch['S']['data']
        loss, logs = self.step(x, y)
        self.log_dict({f"train/{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['I']['data']
        y = batch['S']['data']
        loss, logs = self.step(x, y)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['I']['data']
        y = batch['S']['data']
        loss, logs = self.step(x, y)
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
            "--weight_decay", type=float, default=0., help="Weight decay factor. Default 0."
        )
        parser.add_argument(
            "--conv_layers_per_stage", type=int, default=1, help="Convolutional layer sper network stage. Default: 2"
        )

        return parent_parser
