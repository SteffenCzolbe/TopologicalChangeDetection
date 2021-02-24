# Torchreg

A toolkit for deep-learning based image registration with pytorch.

Similar to torchvision, but for dimensionality-agnostic (works in 2d AND 3d out of the box), and nonvinient transformations for both images and annotations (segmentation masks and landmarks!).

## Handling of 2d data

All images are in the format of BxCxHxWxD. If the image is 2d, D=1.

## Guide

We first import torchreg and set up the dimensionality. Many operations infere their working based on the dimensionality of the data. Currently supported are 2d and 3d voxel-based images

```
import torchreg
torchreg.settings.set_dims(3)
```
