import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchio as tio
import os
import glob
import random
import pandas as pd
import random


class BraTSDataModule(pl.LightningDataModule):
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
        self.class_cnt = 4
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

    def test_dataloader(self):
        dataset = BraTSDataset(
            self.data_dir, "train", pairs=self.pairs, atlasreg=self.atlasreg,
            loadseg=self.load_val_seg, volumetric=self.volumetric, augmentation=None,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def enumerate_labels(tensor):
    # relabel annotations:
    # Background: 0
    # necrotic/cycstic core: 1
    # edema: 2
    # enhancing core: 4 => 3
    tensor[tensor == 4] = 3
    return tensor


def take_slice_from_tensor(tensor):
    # custom preprocessing function
    return tensor[:, :, [95], :].permute(0, 3, 1, 2).flip(1)


class BraTSDataset(Dataset):
    def __init__(self, data_dir, datasplit, pairs, atlasreg, loadseg, volumetric=True, limitsize=None, deterministic=True, augmentation=None):
        """BraTSDataset dataset

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
        enumerate_labels_transform = tio.Lambda(
            enumerate_labels, types_to_apply=[tio.LABEL])
        to_long_transform = tio.Lambda(
            lambda t: t.long(), types_to_apply=[tio.LABEL])

        if volumetric:
            self.preprocess = tio.Compose([
                intensity_scale_transform,
                enumerate_labels_transform,
                to_long_transform,
            ])
        else:
            self.preprocess = tio.Compose([
                to_2d_transform,
                intensity_scale_transform,
                enumerate_labels_transform,
                to_long_transform,
            ])

        # load image paths from csv
        df = pd.read_csv(os.path.join(
            self.data_dir, "metadata.csv"), dtype=str)
        subjects = df.loc[(df['SPLIT'] == "train") & (
            df['AUTO_PROCESSING'] == "OK")]['subject_id'].values

        # gather list of files
        self.image_nii_files = list(
            map(lambda s: os.path.join(self.data_dir, 'preprocessed_data',
                                       s, "t1_aligned_normalized.nii.gz"), subjects)
        )
        self.image_nii_label_files = list(
            map(lambda s: os.path.join(self.data_dir,
                                       'preprocessed_data', s, "seg_aligned.nii.gz"), subjects)
        )

        # load atlas
        if self.atlasreg:
            self.atlas = tio.ScalarImage(os.path.join(
                self.data_dir, "atlas.nii.gz"))
            self.atlas_seg = tio.LabelMap(
                tensor=torch.zeros(1, 160, 192, 224, dtype=torch.long))

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
            index *= (N ** 2) // ((self.limitsize or N)+17)
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
                subject = tio.Subject(I=I0, S=S0)
            else:
                subject = tio.Subject(I=I0)

        # apply preprocessing and augmentation
        subject = self.preprocess(subject)
        if self.augmentation:
            subject = self.augmentation(subject)

        return subject


if __name__ == '__main__':
    batchsize = 4
    dm = BraTSDataModule(pairs=False, volumetric=False, batch_size=batchsize)
    print('size: ', dm.dims)
    print('length: ', len(dm.test_dataloader()))

    # load image
    dataloader = dm.test_dataloader()
    batch = next(iter(dataloader))
    image = batch['S']
    print(image["data"].unique())

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
