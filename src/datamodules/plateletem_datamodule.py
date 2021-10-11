import random
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
import torchio as tio
import os
from .tif_image_stack_dataset import TifImageStackDataset
from .tif_topology_change_dataset import TifTopologyChangeDataset
import glob


class PlateletemDataModule(pl.LightningDataModule):
    def __init__(self, pairs=True, data_dir: str = "./data/platelet_em/", batch_size: int = 32, **kwargs):
        """The platelet-em datamodule, combining the train, validation and test set.

        Args:
            pairs (bool, optional): Set True to return image pairs. Defaults to True.
            data_dir (str, optional): Data directory. Defaults to "../data/platelet_em_reduced/".
            batch_size (int, optional): Batch size. Defaults to 32.
            kwargs: other args are voided, required for compatibility with Brain-Datamodule interface
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = (1, 256, 256, 1)
        self.class_cnt = 3
        # labels for visualization
        self.class_names = ["Background", "Cytoplasm", "Organelle"]
        # colors for visualization
        self.class_colors = [(0, 40, 97), (0, 40, 255), (255, 229, 0)]
        self.num_workers = 8
        self.pairs = pairs

    def prepare_data(self):
        # check if data available
        if not os.path.isdir(os.path.join(self.data_dir, "train")):
            raise Exception('Platelet-EM data not found.')

    def train_dataloader(self, shuffle=False):
        return self.get_dataloader("train", shuffle)

    def val_dataloader(self, shuffle=False):
        return self.get_dataloader("val", shuffle)

    def test_dataloader(self, shuffle=False, bootstrap=False):
        return self.get_dataloader("test", shuffle, bootstrap)

    def get_dataloader(self, split: str, shuffle: bool, bootstrap: bool = False):
        if split == "train":
            augmentations = tio.Compose([
                tio.transforms.RandomFlip(axes=(0, 1)),
                tio.transforms.RandomAffine(),
            ])
        else:
            augmentations = None

        # read file paths
        intensity_files = sorted(glob.glob(os.path.join(
            self.data_dir, split, "image", "*.tif")))
        label_files = sorted(glob.glob(os.path.join(
            self.data_dir, split, "label", "*.tif")))
        topology_appear_files = sorted(glob.glob(os.path.join(
            self.data_dir, split, "topology_appear", "*.tif")))
        topology_disappear_files = sorted(glob.glob(os.path.join(
            self.data_dir, split, "topology_disappear", "*.tif")))
        topology_combined_files = sorted(glob.glob(os.path.join(
            self.data_dir, split, "topology_combined", "*.tif")))

        datasets = []

        # load datasets
        if (not self.pairs) or split == "train":
            # train set or single image:
            for intensity_file, label_file in zip(intensity_files, label_files):
                dataset = TifImageStackDataset(intensity_file, label_file,
                                               pairs=self.pairs, slice_pair_max_z_diff=2 if split == "train" else 1, augmentations=augmentations)
                datasets.append(dataset)
        else:
            # pairs and validation/test: load annotated topological differences
            for intensity_file, label_file, topology_appear_file, topology_disappear_file, topology_combined_file in zip(intensity_files, label_files, topology_appear_files, topology_disappear_files, topology_combined_files):
                dataset = TifTopologyChangeDataset(intensity_file, label_file, topology_appear_file,
                                                   topology_disappear_file, topology_combined_file, augmentations=augmentations)
                datasets.append(dataset)

        if bootstrap:
            # bootstrap dataset by sampling with replacement
            dataset = random.choices(dataset, k=len(dataset))

        # concatinate individual TIF stack datasets
        dataset = ConcatDataset(datasets)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)


if __name__ == '__main__':
    def save_as_png(tio_img, batch_idx, fname):
        output = tio.ScalarImage(tensor=tio_img["data"][batch_idx].detach() * 128,
                                 affine=tio_img["affine"][batch_idx],
                                 check_nans=True)
        output.as_pil().save(fname)
        print(f'saved image in {fname}')

    batchsize = 4
    dm = PlateletemDataModule(pairs=True, batch_size=batchsize)

    # load a batch (manually, normally pytorch lightning does this for us)
    dataloader = dm.test_dataloader(shuffle=True)
    print('dataset length:', len(dataloader.dataset))

    batch = next(iter(dataloader))
    image = batch['S0']
    seg = batch['S0']
    print('batch shape: ', image["data"].shape)
    print('segmentation classes: ', seg['data'].unique())

    for i in range(batchsize):
        save_as_png(batch['I0'], i, f'{i}_I0.png')
        save_as_png(batch['I1'], i, f'{i}_I1.png')
        save_as_png(batch['S0'], i, f'{i}_S0.png')
        save_as_png(batch['S1'], i, f'{i}_S1.png')
        save_as_png(batch['T0'], i, f'{i}_T0.png')
        save_as_png(batch['T1'], i, f'{i}_T1.png')
        save_as_png(batch['Tcombined'], i, f'{i}_Tcombined.png')
