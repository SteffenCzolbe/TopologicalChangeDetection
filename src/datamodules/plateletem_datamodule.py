import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchio as tio
import os
from .tif_image_stack_dataset import TifImageStackDataset


class PlateletemDataModule(pl.LightningDataModule):
    def __init__(self, pairs=True, data_dir: str = "./data/platelet_em_reduced/", batch_size: int = 32, **kwargs):
        """The Brain-MRI datamodule, combining the train, validation and test set.

        Args:
            pairs (bool, optional): Set True to return image pairs. Defaults to True.
            data_dir (str, optional): Data directory. Defaults to "../data/platelet_em_reduced/".
            batch_size (int, optional): Batch size. Defaults to 32.
            kwargs: other args are voided, required for compatibility with Brain-Datamodule interface
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = (1, 800, 800, 1)
        self.class_cnt = 3
        # labels for visualization
        self.class_names = ["Background", "Cytoplasm", "Organelle"]
        # colors for visualization
        self.class_colors = [(0, 40, 97), (0, 40, 255), (255, 229, 0)]
        self.num_workers = 8
        self.pairs = pairs

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "images")):
            raise Exception('Platelet-EM data not found.')

    def train_dataloader(self, shuffle=False):
        intensity_file = os.path.join(self.data_dir, "images", "50-images.tif")
        segmentation_file = os.path.join(
            self.data_dir, "labels-class", "50-class.tif")
        augmentations = tio.Compose([
            tio.transforms.RandomFlip(axes=(0, 1)),
            tio.transforms.RandomAffine(),
        ])
        dataset = TifImageStackDataset(intensity_file, segmentation_file,
                                       pairs=self.pairs, slice_pair_max_z_diff=2, augmentations=augmentations)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        intensity_file = os.path.join(self.data_dir, "images", "24-images.tif")
        segmentation_file = os.path.join(
            self.data_dir, "labels-class", "24-class.tif")
        dataset = TifImageStackDataset(intensity_file, segmentation_file,
                                       pairs=self.pairs, min_slice=0, max_slice=11, slice_pair_max_z_diff=1)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        intensity_file = os.path.join(self.data_dir, "images", "24-images.tif")
        segmentation_file = os.path.join(
            self.data_dir, "labels-class", "24-class.tif")
        dataset = TifImageStackDataset(intensity_file, segmentation_file,
                                       pairs=self.pairs, min_slice=12, max_slice=23, slice_pair_max_z_diff=1)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)


if __name__ == '__main__':
    batchsize = 4
    dm = PlateletemDataModule(pairs=True, batch_size=batchsize)

    # load a batch (manually, normally pytorch lightning does this for us)
    dataloader = dm.train_dataloader()
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
