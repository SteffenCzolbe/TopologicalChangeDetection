import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import torchreg.settings as settings
from .dim_agnostic import interpol_mode

"""
basic spatial transformer layers
"""


class Identity(nn.Module):
    def __init__(self):
        """
        Creates a identity transform
        """
        super().__init__()

    def forward(self, flow):
        # create identity grid
        size = flow.shape[2:]
        vectors = [
            torch.arange(0, s, dtype=flow.dtype, device=flow.device) for s in size
        ]
        grids = torch.meshgrid(vectors)
        identity = torch.stack(grids)  # z, y, x
        identity = identity.expand(
            flow.shape[0], -1, -1, -1, -1
        )  # add batch
        return identity


class FlowComposition(nn.Module):
    """
    A flow composer, composing two flows /transformations / displacement fields.
    """

    def __init__(self):
        """
        instantiates the FlowComposition
        """
        super().__init__()
        self.transformer = SpatialTransformer()

    def forward(self, *args):
        """
        compose any number of flows

        Parameters:
            *args: flows, in order from left to right
        """
        if len(args) == 0:
            raise Exception("Can not compose 0 flows")
        elif len(args) == 1:
            return args[0]
        else:
            composition = self.compose(args[0], args[1])
            return self.forward(composition, *args[2:])

    def compose(self, flow0, flow1):
        """
        compose the flows

        Parameters:
            flow0: the first flow
            flow1: the next flow
        """
        return flow0 + self.transformer(flow1, flow0)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the spatial transformer. 
        A spatial transformer transforms a src image with a flow of displacement vectors.

        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.identity = Identity()
        self.grid_sampler = GridSampler(mode=mode)
        self.ndims = settings.get_ndims()

    def forward(self, src, flow, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            flow: Tensor  (B x C x D x H x W) of displacement vextors. 
                Channels run along the axis of the tensor. Channel 0 indexes 1st spatial axis, Channel 1 indexes 2nd spatial axis, etc...
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        dtype = src.dtype
        if not dtype.is_floating_point:
            # convert to float, we will convert back later
            src = src.float()

        if self.ndims == 2:
            # fix depth-value to 0 in 2d case
            flow[:, 2] = 0

        if self.ndims == 2:
            # there is a critical error with gradient computation when doing pseudo 2d sampling on 3d images.
            # we circumvant the issue by expanding (in a memory saving way) the pseudo 3d images to proper 3d.s
            flow = flow.expand(-1, -1, -1, -1, 3)
            src = src.expand(-1, -1, -1, -1, 3)

        # map from displacement vectors to absolute coordinates
        coordinates = self.identity(flow) + flow
        sampled = self.grid_sampler(
            src, coordinates, mode=mode, padding_mode=padding_mode)

        if self.ndims == 2:
            # remove previously added expansion
            sampled = sampled[..., [1]]

        # convert back to original dtype
        sampled = sampled.to(dtype)

        return sampled


class AffineSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer for affine input
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the spatial transformer. 
        A spatial transformer transforms a src image with a flow of displacement vectors.

        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        raise Exception(
            'The Affine spatial transformer is untested, and the code outdated')
        self.identity = Identity()
        self.grid_sampler = GridSampler(mode=mode)
        self.ndims = settings.get_ndims()

    def forward(self, src, affine, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            affine: Tensor  (B x 4 x 4) the affine transformation matrix
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        coordinates = self.identity(src)

        # add homogenous coordinate
        coordinates = torch.cat(
            (
                coordinates,
                torch.ones(
                    coordinates.shape[0],
                    1,
                    *coordinates.shape[2:],
                    device=coordinates.device,
                    dtype=coordinates.dtype
                ),
            ),
            dim=1,
        )

        # center the coordinate grid, so that rotation happens around the center of the domain
        size = coordinates.shape[2:]
        for i in range(self.ndims):
            coordinates[:, i] -= size[i] / 2.0

        # permute for batched matric multiplication
        coordinates = (
            coordinates.permute(0, 2, 3, 4, 1)
            if self.ndims == 3
            else coordinates.permute(0, 2, 3, 1)
        )
        # we need to do this for each member of the batch separately
        for i in range(len(coordinates)):
            coordinates[i] = torch.matmul(coordinates[i], affine[i])
        coordinates = (
            coordinates.permute(0, -1, 1, 2, 3)
            if self.ndims == 3
            else coordinates.permute(0, -1, 1, 2)
        )
        # de-homogenize
        coordinates = coordinates[:, : self.ndims]

        # un-center the coordinate grid
        for i in range(self.ndims):
            coordinates[:, i] += size[i] / 2

        return self.grid_sampler(src, coordinates, mode=mode, padding_mode=padding_mode)


class GridSampler(nn.Module):
    """
    A simple Grid sample operation
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the grid sampler.
        The grid sampler samples a grid of values at coordinates.

        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.mode = mode
        self.ndims = settings.get_ndims()

    def forward(self, values, coordinates, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            flow: Tensor  (B x C x D x H x W) of displacement vectors. Channel 0 indicates the flow in the depth dimension.
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        mode = mode if mode else self.mode

        # make mode dimentionality-agnostic
        # mode = interpol_mode(mode)

        # clone the coordinate field as we will modift it.
        coordinates = coordinates.clone()
        # normalize coordinates to be within [-1..1]
        size = values.shape[2:]
        for i in range(len(size)):
            coordinates[:, i, ...] = 2 * \
                (coordinates[:, i, ...] / (size[i] - 1) - 0.5)

        # put coordinate channels in last position and
        # reverse channels (in-build pytorch function indexes axis D x H x W and pixel coordinates z,y,x)
        coordinates = coordinates.permute(0, 2, 3, 4, 1)
        coordinates = coordinates[..., [2, 1, 0]]

        # sample
        sampled = nnf.grid_sample(
            values,
            coordinates,
            mode=mode,
            padding_mode=padding_mode,
            # align = True is nessesary to behave similar to indexing the transformation.
            align_corners=True,
        )

        return sampled


if __name__ == 'main':
    settings.set_ndims(2)
    img = torch.rand(2, 1, 6, 7, 1)
    transform = torch.zeros(2, 3, 6, 7, 1)
    sampler = SpatialTransformer()
    img_sampled = sampler(img, transform)
    print("images identical:", torch.allclose(img, img_sampled))
