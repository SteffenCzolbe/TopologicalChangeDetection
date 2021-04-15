import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchio as tio
import os
import glob
import random
import pandas as pd
import random


class BrainMRIDataModule(pl.LightningDataModule):
    def __init__(self, pairs=True, atlasreg=True, volumetric=True, loadseg=True, data_dir: str = "../brain_mris/", batch_size: int = 32):
        """The Brain-MRI datamodule, combining the train, validation and test set.

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
        self.class_cnt = 24
        self.class_names = ["Background", "Cerebral G. Matter", "Cerebral W. Matter", "Lateral Ventricle", "Inf Lat Ventricle", "Cerebellum W. Matter", "Cerebellum Cortex", "Thalamus", "Caudate", "Putamen", "Pallidum",
                            "3rd Ventricle", "4th Ventricle", "Brain Stem", "Hippocampus", "Amygdala", "CSF", "Accumbens area", "Ventral DC", "Vessel", "Choroid plexus", "5th Ventricle", "Cingulate Cortex", "WM hypointensities"]
        self.num_workers = 32
        self.pairs = pairs
        self.loadseg = loadseg
        self.atlasreg = atlasreg
        self.volumetric = volumetric

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "data")):
            raise Exception('BrainMRI data not found.')

    def train_dataloader(self):
        dataset = BrainMRIDataset(
            self.data_dir, "train", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.loadseg, volumetric=self.volumetric,
            limitsize=4000, deterministic=False, augmentation=None,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = BrainMRIDataset(
            self.data_dir, "val", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.loadseg, volumetric=self.volumetric,
            limitsize=250, deterministic=True, augmentation=None,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = BrainMRIDataset(
            self.data_dir, "test", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.loadseg, volumetric=self.volumetric,
            limitsize=250, deterministic=True, augmentation=None,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def take_slice_from_tensor(tensor):
    # custom preprocessing function
    return tensor[:, :, [95], :].permute(0, 3, 1, 2).flip(1)


class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, datasplit, pairs, atlasreg, loadseg, volumetric=True, limitsize=None, deterministic=True, augmentation=None):
        """Brain MRI dataset

        Args:
            data_dir ([type]): Path to the data directory
            datasplit ([type]): split of the data. e.g. 'train', 'val', 'test'
            pairs ([type]): set to true to get pairs of images.
            atlasreg ([type]): set true to register to atlas
            loadseg ([type]): set true to load segmentation masks
            volumetric (bool, optional): Set for 3d images. Defaults to True.
            limitsize ([type]): artifical limit on size of dataset, only applicable if atlasreg=False
            deterministic (bool, optional): Set to obtain deterministic behaviour. Only applicable if atlasreg=False. Defaults to True.
            augmentation ([type], optional): Augmentation Transforms. Defaults to None.
        """
        self.data_dir = data_dir
        self.pairs = pairs
        self.atlasreg = atlasreg
        self.loadseg = loadseg
        self.limitsize = limitsize
        self.deterministic = deterministic
        self.augmentation = augmentation

        # preprocessing, performed before augmentation
        transforms = []
        to_2d_transform = tio.Lambda(take_slice_from_tensor,
                                     types_to_apply=[tio.INTENSITY, tio.LABEL])
        intensity_scale_transform = tio.Lambda(
            lambda t: t.float() / 128, types_to_apply=[tio.INTENSITY])

        if volumetric:
            self.preprocess = tio.Compose([
                intensity_scale_transform,
            ])
        else:
            self.preprocess = tio.Compose([
                to_2d_transform,
                intensity_scale_transform,
            ])

        # load image paths from csv
        df = pd.read_csv(os.path.join(
            self.data_dir, "metadata.csv"), dtype=str)
        df.set_index("subject_id", inplace=True)
        subjects = list(df[df["SPLIT"] == datasplit].index)

        # gather list of files
        self.image_nii_files = list(
            map(lambda s: os.path.join(self.data_dir, "data",
                                       s, "brain_aligned.nii.gz"), subjects)
        )
        self.image_nii_label_files = list(
            map(
                lambda s: os.path.join(
                    self.data_dir, "data", s, "seg_coalesced_aligned.nii.gz"),
                subjects,
            )
        )

        # load atlas
        if self.atlasreg:
            self.atlas = tio.ScalarImage(os.path.join(
                self.data_dir, "atlas", "brain_aligned.nii.gz"))
            self.atlas_seg = tio.LabelMap(os.path.join(
                self.data_dir, "atlas", "seg_coalesced_aligned.nii.gz"))

    def __len__(self):
        if self.atlasreg or not self.pairs:
            return len(self.image_nii_files)
        else:
            return self.limitsize if self.limitsize else len(self.image_nii_files)

    def get_subject_indices_from_index(self, index):
        if self.atlasreg or not self.pairs:
            # register one image to atlas
            return index, None
        if not self.deterministic:
            # pick two images at random
            N = len(self.image_nii_files)
            idx0 = random.randint(0, N-1)
            idx1 = random.randint(0, N-1)
            return idx0, idx1
        else:
            # pick two pseudo-random images
            N = len(self.image_nii_files)
            index *= (N ** 2) // self.limitsize
            idx0 = index // N
            idx1 = index % N
            return idx0, idx1

    def __getitem__(self, index):
        idx0, idx1 = self.get_subject_indices_from_index(index)
        # load images
        I0 = tio.ScalarImage(self.image_nii_files[idx0])
        S0 = tio.LabelMap(self.image_nii_label_files[idx0])

        if self.pairs:
            if self.atlasreg:
                I1 = self.atlas
                S1 = self.atlas_seg
            else:
                I1 = tio.ScalarImage(self.image_nii_files[idx1])
                S1 = tio.LabelMap(self.image_nii_label_files[idx1])

            # build subject
            if self.loadseg:
                subject = tio.Subject(I0=I0, S0=S0, I1=I1, S1=S1)
            else:
                subject = tio.Subject(I0=I0, I1=I1)
        else:
            if self.loadseg:
                subject = tio.Subject(I0=I0, S0=S0)
            else:
                subject = tio.Subject(I0=I0)

        # apply preprocessing and augmentation
        subject = self.preprocess(subject)
        if self.augmentation:
            subject = self.augmentation(subject)

        return subject


if __name__ == '__main__':
    batchsize = 4
    dm = BrainMRIDataModule(pairs=True, atlasreg=True, volumetric=False,
                            loadseg=True, batch_size=batchsize)

    # load image
    dataloader = dm.train_dataloader()
    batch = next(iter(dataloader))
    image = batch['I0']
    print(batch['S0']['data'].unique())

    print('shape: ', image["data"][0].shape)

    for i in range(batchsize):
        # save output
        output = tio.ScalarImage(tensor=image["data"][i].detach(),
                                 affine=image["affine"][i],
                                 check_nans=True)

        if output.is_2d():
            output['data'] = output['data'] * 128
            output.as_pil().save(f"test{i}.png")
            print('saved 2d')
        else:
            output.save(f"test{i}.nii.gz")
            print('saved 3d')
