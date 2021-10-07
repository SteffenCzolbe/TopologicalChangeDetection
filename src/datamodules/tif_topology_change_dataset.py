import torch
from torch.utils.data import Dataset
import torchio as tio


class TifTopologyChangeDataset(Dataset):
    def __init__(self, tif_intensity_file: str, tif_seg_file: str, tif_topology_change_appearing_file: str, tif_topology_change_disappearing_file: str, tif_topology_change_combined_file: str, augmentations: tio.Transform = None):
        """Dataset of image pairs with annotated topological difference

        Args:
            tif_intensity_file (str): [description]
            tif_seg_file (str): [description]
            tif_topology_change_appearing_file (str): [description]
            tif_topology_change_disappearing_file (str): [description]
            augmentations (tio.Transform, optional): [description]. Defaults to None.
        """
        # load data (kept in mem, as data is small)
        self.intensity_stack = tio.ScalarImage(tif_intensity_file)
        self.segmentation_stack = tio.LabelMap(tif_seg_file)
        self.topology_change_appearing_stack = tio.LabelMap(
            tif_topology_change_appearing_file)
        self.topology_change_disappearing_stack = tio.LabelMap(
            tif_topology_change_disappearing_file)
        self.tif_topology_change_combined_stack = tio.LabelMap(
            tif_topology_change_combined_file)
        self.augmentations = augmentations
        self.slice_count = self.intensity_stack['data'].shape[-1]

        # set-up preprocessing
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
        return self.slice_count - 1

    def load_slice_from_imagestack(self, image, slice_idx):
        i = image['data'][..., slice_idx]
        # transpose and add empty 3rd spatial dimension
        i = i.transpose(1, 2).unsqueeze(-1)
        # build scalar image of slice
        if type(image) == tio.ScalarImage:
            # intensity image
            I = tio.ScalarImage(
                tensor=i, affine=image['affine'], path=image['path'])
        elif type(image) == tio.LabelMap:
            # annotation
            I = tio.LabelMap(
                tensor=i, affine=image['affine'], path=image['path'])
        else:
            raise Exception(f"cannot process type {type(image)}")

        # apply preprocessing
        I = self.preprocess(I)
        return I

    def __getitem__(self, index):
        I0 = self.load_slice_from_imagestack(self.intensity_stack, index)
        S0 = self.load_slice_from_imagestack(self.segmentation_stack, index)
        I1 = self.load_slice_from_imagestack(self.intensity_stack, index+1)
        S1 = self.load_slice_from_imagestack(self.segmentation_stack, index+1)
        T0 = self.load_slice_from_imagestack(
            self.topology_change_disappearing_stack, index)
        T1 = self.load_slice_from_imagestack(
            self.topology_change_appearing_stack, index+1)
        Tcombined = self.load_slice_from_imagestack(
            self.tif_topology_change_combined_stack, index+1)

        subject = tio.Subject(I0=I0, S0=S0, I1=I1, S1=S1, T0=T0, T1=T1, Tcombined=Tcombined,
                              subject_id0=index, subject_id1=index+1)

        # apply data augmentation
        if self.augmentations:
            subject = self.augmentations(subject)
            # map labels back to long
            subject = self.segmentation_to_long_transform(subject)

        return subject
