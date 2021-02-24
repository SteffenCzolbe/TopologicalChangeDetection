"""
functions providing dimension-agnostic contructors to popular torch.nn building blocks
"""
import torch.nn as nn
import torchreg.settings as settings


def Conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True):
    ndims = settings.get_ndims()
    if ndims == 2:
        # adjust for faked 3rd dimension
        kernel_size = (kernel_size, kernel_size, 1)
        padding = (padding, padding, 0)
        stride = (stride, stride, 1)
    return nn.Conv3d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=bias)


def Upsample(size=None, scale_factor=None, mode="nearest", align_corners=False):
    ndims = settings.get_ndims()
    mode = interpol_mode(mode)

    if ndims == 2:
        scale_factor = (scale_factor, scale_factor, 1)
    return nn.Upsample(
        size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )


def Upsample2x(mode="nearest"):
    return Upsample(scale_factor=2, mode=mode)


def BatchNorm(*args, **kwargs):
    return nn.BatchNorm3d(*args, **kwargs)


def Dropout(*args, **kwargs):
    """
    performs channel-whise dropout. As described in the paper Efficient Object Localization Using Convolutional 
    Networks , if adjacent pixels within feature maps are strongly correlated (as is normally the case in early 
    convolution layers) then i.i.d. dropout will not regularize the activations and will otherwise just result 
    in an effective learning rate decrease.
    """
    return nn.Dropout3d(*args, **kwargs)


def interpol_mode(mode):
    """
    returns an interpolation mode for the current dimensioanlity.
    """
    ndims = settings.get_ndims()
    if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
        mode = "trilinear"
    return mode
