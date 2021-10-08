import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchio as tio
import os
import glob
import random
import pandas as pd
from .brats_datamodule import BraTSDataset
import random


class BraTSDataModule4SegBaseline(pl.LightningDataModule):
    def __init__(self, pairs=True, atlasreg=True, volumetric=True, load_train_seg=False, load_val_seg=True, data_dir: str = "../BraTS/", batch_size: int = 32):
        """The BraTS datamodule.

        Args:
            pairs (bool, optional): Set True to return image pairs. Defaults to True.
            atlasreg (bool, optional): Set True for registration to atlas. Defaults to True.
            volumetric (bool, optional): Set True for 3D images. Defaults to True.
            loadseg (bool, optional): Set True to load segmentation maps. Defaults to True.
            data_dir (str, optional): Data directory. Defaults to "../brain_mris/".
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = (1, 160, 192, 224) if volumetric else (1, 224, 160, 1)
        self.class_cnt = 3
        self.class_names = [
            'Normal', 'Necrotic/Cystic Core', 'Edema', 'Enhancing Core']
        self.num_workers = 32
        self.pairs = pairs
        self.load_train_seg = load_train_seg
        self.load_val_seg = load_val_seg
        self.atlasreg = atlasreg
        self.volumetric = volumetric

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "preprocessed_data")):
            raise Exception('BraTS data not found.')

    def train_dataloader(self, shuffle=False):
        augmentation = tio.RandomAffine(
            scales=(0.9, 1.2),
            degrees=10,
        )
        dataset = BraTSDataset(
            self.data_dir, "train", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.load_val_seg, volumetric=self.volumetric, augmentation=augmentation,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        dataset = BraTSDataset(
            self.data_dir, "val", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.load_val_seg, volumetric=self.volumetric,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        dataset = BraTSDataset(
            self.data_dir, "val", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.load_val_seg, volumetric=self.volumetric,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
