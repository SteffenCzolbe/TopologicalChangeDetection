import torch
from torch.utils.data import Dataset
import torchio as tio


class TifImageStackDataset(Dataset):
    def __init__(self, tif_intensity_file: str, tif_seg_file: str, pairs: bool, min_slice: int = None, max_slice: int = None, slice_pair_max_z_diff=2, augmentations: tio.Transform = None):
        """Load a dataset from a stacked .tif file

        Args:
            tif_intensity_file (str): path to the intensity file
            tif_seg_file (str): path to the segmentation class file
            pairs (bool): should pairs be returned?
            min_slice (int, optional): Slice from. Set to none for all images. Defaults to None.
            max_slice (int, optional): Slice to. Set to none for all images. Inclusive range. Defaults to None.
            slice_pair_max_z_diff (int): max z-diff beween slice pairs
            augmentations (tio.Transform, optional): Data-augmentation transforms
        """
        # load data (kept in mem, as data is small)
        self.intensity_stack = tio.ScalarImage(tif_intensity_file)
        self.segmentation_stack = tio.LabelMap(tif_seg_file)
        self.min_slice = min_slice or 0
        self.max_slice = max_slice or self.intensity_stack['data'].shape[-1] - 1
        self.pairs = pairs
        self.max_z_diff = slice_pair_max_z_diff
        self.augmentations = augmentations

        # set-up preprocessing
        self.slice_idxs = self.get_slice_idxs()
        self.dynamic_range = self.get_dynamic_range_of_dtype(
            self.intensity_stack['data'].dtype)
        self.image_intensity_scale_transform = tio.Lambda(
            lambda t: t.float() / 2**self.dynamic_range, types_to_apply=[tio.INTENSITY])
        self.segmentation_to_long_transform = tio.Lambda(
            lambda t: t.long(), types_to_apply=[tio.LABEL])
        self.preprocess = tio.Compose([
            self.image_intensity_scale_transform,
            self.segmentation_to_long_transform,
        ])

    @staticmethod
    def get_dynamic_range_of_dtype(dtype):
        if dtype == torch.uint8:
            return 8
        elif dtype == torch.int32:
            return 16  # We use 16 bit data in our datasets.
        else:
            raise Exception("dtype {dtype} not implemented")

    def __len__(self):
        return len(self.slice_idxs)

    def get_slice_idxs(self):
        first_idxs = list(range(self.min_slice, self.max_slice+1))

        if not self.pairs:
            # single samples
            return first_idxs

        # sample pairs
        combinations = []
        for first_idx in first_idxs:
            # all combinations within allowed max_z_diff
            second_idxs = list(range(first_idx - self.max_z_diff, first_idx)) + \
                list(range(first_idx + 1, first_idx + self.max_z_diff + 1))
            # filter out non-existsant slices
            second_idxs = filter(
                lambda x: x >= self.min_slice and x <= self.max_slice, second_idxs)
            combinations += [(first_idx, second_idx)
                             for second_idx in second_idxs]

        return combinations

    def load_image_slice(self, slice_idx):
        i = self.intensity_stack['data'][..., slice_idx]
        # transpose and add empty 3rd spatial dimension
        i = i.transpose(1, 2).unsqueeze(-1)
        # build scalar image of slice
        I = tio.ScalarImage(
            tensor=i, affine=self.intensity_stack['affine'], path=self.intensity_stack['path'])

        s = self.segmentation_stack['data'][..., slice_idx]
        # add empty 3rd spatial dimension
        s = s.transpose(1, 2).unsqueeze(-1)
        # build label map of slice
        S = tio.LabelMap(
            tensor=s, affine=self.segmentation_stack['affine'], path=self.segmentation_stack['path'])

        # preprocess
        I = self.preprocess(I)
        S = self.preprocess(S)
        return I, S

    def __getitem__(self, index):
        if self.pairs:
            slice_idx0, slice_idx1 = self.slice_idxs[index]
            I0, S0 = self.load_image_slice(slice_idx0)
            I1, S1 = self.load_image_slice(slice_idx1)

            # build subject
            subject = tio.Subject(I0=I0, S0=S0, I1=I1, S1=S1,
                                  subject_id0=slice_idx0, subject_id1=slice_idx1)
        else:
            slice_index = self.slice_idxs[index]
            I, S = self.load_image_slice(slice_index)
            subject = tio.Subject(I=I, S=S, subject_id=slice_index)

        # apply data augmentation
        if self.augmentations:
            subject = self.augmentations(subject)
            # map labels back to long
            subject = self.segmentation_to_long_transform(subject)

        return subject
