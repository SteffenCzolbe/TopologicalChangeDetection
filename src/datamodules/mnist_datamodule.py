import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchio as tio
import os
import glob
import random


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, label: int = 7, data_dir: str = "./data/MNIST/", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.label = label
        self.batch_size = batch_size
        self.dims = (1, 28, 28, 1)
        self.classes = 2
        self.num_workers = 4

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "train")):
            print("Data not found. Downloading...")
            from data.MNIST.download import main as download
            download()

    def train_dataloader(self):
        # augmentation = tio.transforms.Compose(
        #     [  # tio.transforms.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10), translation=0.),
        #         # tio.transforms.RandomFlip(axes=(0, 1), flip_probability=0.5),
        #         # tio.transforms.RandomElasticDeformation(max_displacement=5) # not supported for 2d
        #     ]
        # )
        dataset = MnistLabelDataset(
            self.data_dir, "train", label=self.label, augmentation=None, size=100000, deterministic=False,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = MnistLabelDataset(
            self.data_dir, "val", label=self.label, size=10000, deterministic=True,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = MnistLabelDataset(
            self.data_dir, "test", label=self.label, size=10000, deterministic=True,
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def calligraphy_split_dataloader(self):
        dataset = MnistSevenCalligraphySplitDataset(
            self.data_dir, "test",
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


class MnistDataset(Dataset):
    def __init__(self, data_dir, datasplit, augmentation=None):
        self.augmentation = augmentation

    def __len__(self):
        raise NotImplementedError()

    def get_file_paths(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        I0_path, I1_path = self.get_file_paths(index)

        # load images
        I0 = tio.ScalarImage(I0_path)
        I0['data'] = I0['data'].float().transpose(1, 2) / 255
        I1 = tio.ScalarImage(I1_path)
        I1['data'] = I1['data'].float().transpose(1, 2) / 255
        subject = tio.Subject(I0=I0, I1=I1)

        # apply data augmentation
        if self.augmentation:
            subject = self.augmentation(subject)

        return subject


class MnistLabelDataset(MnistDataset):
    def __init__(self, data_dir, datasplit, label, size, deterministic=True, augmentation=None):
        super().__init__(data_dir, datasplit, augmentation)
        self.size = size
        self.deterministic = deterministic

        # read images
        self.image_paths = glob.glob(os.path.join(
            data_dir, datasplit, str(label), '*.png'))

    def __len__(self):
        return self.size

    def get_file_paths(self, index):
        # get image pair
        N = len(self.image_paths)
        if self.deterministic:
            index *= (N ** 2) // self.size
            idx0 = index // N
            idx1 = index % N
        else:
            idx0 = random.randint(0, N-1)
            idx1 = random.randint(0, N-1)

        return self.image_paths[idx0], self.image_paths[idx1]


class MnistSevenCalligraphySplitDataset(MnistDataset):
    def __init__(self, data_dir, datasplit, augmentation=None):
        super().__init__(data_dir, datasplit, augmentation)
        # read images
        self.dashed = glob.glob(os.path.join(
            data_dir, datasplit, '7_dash', '*.png'))
        self.palmer = glob.glob(os.path.join(
            data_dir, datasplit, '7_palmer', '*.png'))

    def __len__(self):
        return len(self.dashed) * len(self.palmer)

    def get_file_paths(self, index):
        # get image pair
        N = len(self.palmer)
        idx0 = index // N
        idx1 = index % N
        return self.dashed[idx0], self.palmer[idx1]


if __name__ == '__main__':
    dm = MnistDataModule(label=1)
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    image = batch['I0']
    for i in range(16):
        # save output
        output = tio.ScalarImage(tensor=image["data"][i].detach(),
                                 affine=image["affine"][i],
                                 check_nans=True)
        if output.is_2d():
            output['data'] = output['data'] * 255
            output.as_pil().save(f"test_{i:02}.png")
        else:
            output.save("test.nii")  # only nii write format supported
