from torch.utils.data import DataLoader, ConcatDataset
import torchio as tio
import os
from .tif_topology_change_dataset import TifTopologyChangeDataset
import glob
from .plateletem_datamodule import PlateletemDataModule
import random


class PlateletemDataModule4SegBaseline(PlateletemDataModule):
    def __init__(self, data_dir: str = "./data/platelet_em/", batch_size: int = 32, **kwargs):
        """The platelet-em datamodule for the segmentation baseline

        Args:
            data_dir (str, optional): Data directory. Defaults to "../data/platelet_em_reduced/".
            batch_size (int, optional): Batch size. Defaults to 32.
            kwargs: other args are voided, required for compatibility with Brain-Datamodule interface
        """
        super().__init__(pairs=True, data_dir=data_dir, batch_size=batch_size)
        self.class_cnt = 2
        self.dims = (2, 256, 256, 1)

    def collect_files_for_datasplit(self, split):
        # we re-do the datasplit 75% train 25% val+test for training the segmentation baseline model

        # get all file names
        intensity_files = []
        label_files = []
        topology_appear_files = []
        topology_disappear_files = []
        topology_change_combined_files = []
        for original_split in ["val", "test"]:
            intensity_files += sorted(glob.glob(os.path.join(
                self.data_dir, original_split, "image", "*.tif")))
            label_files += sorted(glob.glob(os.path.join(
                self.data_dir, original_split, "label", "*.tif")))
            topology_appear_files += sorted(glob.glob(os.path.join(
                self.data_dir, original_split, "topology_appear", "*.tif")))
            topology_disappear_files += sorted(glob.glob(os.path.join(
                self.data_dir, original_split, "topology_disappear", "*.tif")))
            topology_change_combined_files += sorted(glob.glob(os.path.join(
                self.data_dir, original_split, "topology_combined", "*.tif")))

        # re-do split
        n = len(intensity_files)
        s = int(n*0.75)
        if split == "train":
            return intensity_files[:s], label_files[:s], topology_appear_files[:s], topology_disappear_files[:s], topology_change_combined_files[:s]
        else:
            return intensity_files[s:], label_files[s:], topology_appear_files[s:], topology_disappear_files[s:], topology_change_combined_files[s:]

    def get_dataloader(self, split: str, shuffle: bool, bootstrap: bool = False):
        if split == "train":
            augmentations = tio.Compose([
                tio.transforms.RandomFlip(axes=(0, 1)),
                tio.transforms.RandomAffine(),
            ])
        else:
            augmentations = None

        # collect file paths across validation and test split
        intensity_files, label_files, topology_appear_files, topology_disappear_files, topology_combined_files = self.collect_files_for_datasplit(
            split)

        datasets = []
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
