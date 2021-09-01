import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchio as tio
import os
from .tif_image_stack_dataset import TifImageStackDataset


class PhCU373DataModule(pl.LightningDataModule):
    def __init__(self, pairs=True, data_dir: str = "./data/PhC-U373/", batch_size: int = 32):
        """The Brain-MRI datamodule, combining the train, validation and test set.

        Args:
            pairs (bool, optional): Set True to return image pairs. Defaults to True.
            data_dir (str, optional): Data directory. Defaults to "../data/platelet_em_reduced/".
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = (1, 512, 688, 1)
        self.class_cnt = 3
        # labels for visualization
        self.class_names = ["Background", "Cell"]
        # colors for visualization
        self.class_colors = [(0, 0, 0), (27, 247, 156)]
        self.num_workers = 8
        self.pairs = pairs

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "images")):
            raise Exception('PhC-U373 data not found.')

    def train_dataloader(self, shuffle=False):
        intensity_file = os.path.join(self.data_dir, "images", "01.tif")
        segmentation_file = os.path.join(
            self.data_dir, "labels-class", "01.tif")
        augmentations = tio.Compose([
            tio.transforms.RandomFlip(axes=(0, 1)),
            tio.transforms.RandomAffine(),
        ])
        dataset = TifImageStackDataset(intensity_file, segmentation_file,
                                       pairs=self.pairs, slice_pair_max_z_diff=2, augmentations=augmentations)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        intensity_file = os.path.join(self.data_dir, "images", "02.tif")
        segmentation_file = os.path.join(
            self.data_dir, "labels-class", "02.tif")
        dataset = TifImageStackDataset(intensity_file, segmentation_file,
                                       pairs=self.pairs, min_slice=0, max_slice=50, slice_pair_max_z_diff=1)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        intensity_file = os.path.join(self.data_dir, "images", "02.tif")
        segmentation_file = os.path.join(
            self.data_dir, "labels-class", "02.tif")
        dataset = TifImageStackDataset(intensity_file, segmentation_file,
                                       pairs=self.pairs, min_slice=60, max_slice=115, slice_pair_max_z_diff=1)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)


if __name__ == '__main__':
    batchsize = 4
    dm = PhCU373DataModule(pairs=True, batch_size=batchsize)

    # load a batch (manually, normally pytorch lightning does this for us)
    dataloader = dm.test_dataloader()
    print('dataset length:', len(dataloader.dataset))

    batch = next(iter(dataloader))
    image = batch['I0']
    seg = batch['S0']
    print('batch shape: ', image["data"].shape)
    print('segmentation classes: ', seg['data'].unique())

    for i in range(batchsize):
        # save output
        output = tio.ScalarImage(tensor=image["data"][i].detach(),
                                 affine=image["affine"][i],
                                 check_nans=True)

        if output.is_2d():
            path = f"test{i}.png"
            output['data'] = output['data'] * 256
            output.as_pil().save(path)
            print(f'saved image in {path}')
        else:
            path = f"test{i}.nii.gz"
            output.save(path)
            print(f'saved image in {path}')
