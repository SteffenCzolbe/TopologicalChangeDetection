import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchreg.settings as settings
from typing import Tuple, Union, List


class CovFilterLayer(nn.Module):
    def __init__(self, ksize: int, channels=1):
        super().__init__()
        self.ndims = settings.get_ndims()
        if ksize % 2 == 0:
            ksize += 1
        self.ksize = ksize

        if self.ndims == 2:
            weights = self._2d_weights(self.ksize)
            self.conv_layer = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(
                ksize, ksize, 1), padding=(ksize//2, ksize//2, 0), bias=False, groups=channels)
            self.conv_layer.weight = torch.nn.Parameter(
                weights.view(1, 1, ksize, ksize, 1), requires_grad=False)
        elif self.ndims == 3:
            weights = self._3d_weights(self.ksize)
            self.conv_layer = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(
                ksize, ksize, ksize), padding=(ksize//2, ksize//2, ksize//2), bias=False, groups=channels)
            self.conv_layer.weight = torch.nn.Parameter(
                weights.view(1, 1, ksize, ksize, ksize), requires_grad=False)

    def _2d_weights(self, ksize):
        """
        return weight-matrix of size ksize x ksize
        """
        raise NotImplementedError()

    def _3d_weights(self, ksize):
        """
        return weight-matrix of size ksize x ksize x ksize
        """
        raise NotImplementedError()

    def forward(self, x):
        return self.conv_layer(x)


def gauss(x, mu, sigma):
    # gaussian
    return 1 / (sigma * (2 * np.pi)**0.5) * np.exp(- (x - mu)**2 / (2 * sigma**2))


def d_gauss(x, mu, sigma):
    # gaussian derivitive
    return - (x - mu) / (sigma**3 * (2 * np.pi)**0.5) * np.exp(- (x - mu)**2 / (2 * sigma**2))


def make_coordinate_list(ksize):
    l = ksize // 2
    return torch.tensor(list(range(-l, l+1)), dtype=torch.float)


class GaussianSmoothing(CovFilterLayer):
    def __init__(self, sigma):
        """Image smoothing with a gaussian filter

        Args:
            sigma (float): standard deviation
        """
        self.sigma = sigma
        super().__init__(ksize=int(sigma*6 + 1))

    def _2d_weights(self, ksize):
        """
        return weight-matrix of size ksize x ksize
        """
        mu = 0
        sigma = self.sigma
        c = make_coordinate_list(ksize)
        weights = torch.ones(ksize, ksize)
        for i in range(ksize):
            for j in range(ksize):
                x, y = c[i], c[j]
                weights[i, j] = gauss(x, mu, sigma) * gauss(y, mu, sigma)

        return weights

    def _3d_weights(self, ksize):
        """
        return weight-matrix of size ksize x ksize
        """
        mu = 0
        sigma = self.sigma
        c = make_coordinate_list(ksize)
        weights = torch.ones(ksize, ksize, ksize)
        for i in range(ksize):
            for j in range(ksize):
                for k in range(ksize):
                    x, y, z = c[i], c[j], c[k]
                    weights[i, j, k] = gauss(
                        x, mu, sigma) * gauss(y, mu, sigma) * gauss(z, mu, sigma)

        return weights


class GaussianDerivative(CovFilterLayer):
    def __init__(self, sigma, dim):
        """A Guassian Derivitive Filter

        Args:
            sigma (float): standard deviation
            dim (int): dimension of the derivitive. Uses pytorch dim numbering. valid values: 2, 3, 4
        """
        self.sigma = sigma
        self.dim = dim
        super().__init__(ksize=int(sigma*8 + 1))

    def _2d_weights(self, ksize):
        """
        return weight-matrix of size ksize x ksize
        """
        assert(self.dim in [2, 3])
        mu = 0
        sigma = self.sigma
        c = make_coordinate_list(ksize)
        weights = torch.ones(ksize, ksize)
        for i in range(ksize):
            for j in range(ksize):
                x, y = c[i], c[j]
                if self.dim == 2:
                    weights[i, j] = d_gauss(x, mu, sigma) * gauss(y, mu, sigma)
                elif self.dim == 3:
                    weights[i, j] = gauss(x, mu, sigma) * d_gauss(y, mu, sigma)

        return weights

    def _3d_weights(self, ksize):
        """
        return weight-matrix of size ksize x ksize
        """
        assert(self.dim in [2, 3, 4])
        mu = 0
        sigma = self.sigma
        c = make_coordinate_list(ksize)
        weights = torch.ones(ksize, ksize, ksize)
        for i in range(ksize):
            for j in range(ksize):
                for k in range(ksize):
                    x, y, z = c[i], c[j], c[k]
                    if self.dim == 2:
                        weights[i, j, k] = d_gauss(
                            x, mu, sigma) * gauss(y, mu, sigma) * gauss(z, mu, sigma)
                    elif self.dim == 3:
                        weights[i, j, k] = gauss(
                            x, mu, sigma) * d_gauss(y, mu, sigma) * gauss(z, mu, sigma)
                    elif self.dim == 4:
                        weights[i, j, k] = gauss(
                            x, mu, sigma) * gauss(y, mu, sigma) * d_gauss(z, mu, sigma)

        return weights


class GradientMagnitude(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.ndims = settings.get_ndims()
        self.dx = GaussianDerivative(sigma, dim=2)
        self.dy = GaussianDerivative(sigma, dim=3)
        if self.ndims == 3:
            self.dz = GaussianDerivative(sigma, dim=4)

    def forward(self, x):
        dx = self.dx(x)
        dy = self.dy(x)
        dz = self.dz(x) if self.ndims == 3 else 0

        return dx**2 + dy**2 + dz**2


def _se_to_mask(se: torch.Tensor) -> torch.Tensor:
    # structuring element to mask for convolution
    se_h, se_w, se_d = se.size()
    se_flat = se.view(-1)
    num_feats = se_h * se_w * se_d
    out = torch.zeros(num_feats, 1, se_h, se_w, se_d,
                      dtype=se.dtype, device=se.device)
    for i in range(num_feats):
        y = (i // se_d) // se_w
        x = (i // se_d) % se_w
        z = i % se_d
        out[i, 0, y, x, z] = (se_flat[i] >= 0).float()
    return out


def dilation(tensor: torch.Tensor, dilation_size: int) -> torch.Tensor:
    """channel wise-dilation. Each channel of the image is treated independently.

    Args:
        tensor (torch.Tensor): Image or Mask
        dilation_size (int): size of the morphological operation

    Returns:
        torch.Tensor: The eroded input
    """
    return -erosion(-tensor, dilation_size)


# erosion
def erosion(tensor: torch.Tensor, erosion_size: int) -> torch.Tensor:
    """channel wise erosion. Each channel of the image is treated independently.

    Args:
        tensor (torch.Tensor): Image or Mask
        erosion_size (int): size of the morphological operation

    Returns:
        torch.Tensor: The eroded input
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) != 5:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            tensor.dim()))

    kernel_size = 2*erosion_size + 1

    ndims = settings.get_ndims()
    kernel = torch.ones(kernel_size, kernel_size,
                        1 if ndims == 2 else kernel_size, device=tensor.device)

    # prepare kernel
    se_e: torch.Tensor = kernel - 1.
    kernel_e: torch.Tensor = _se_to_mask(se_e)

    # pad
    pad_e: List[int] = [0 if ndims == 2 else kernel_size // 2,
                        0 if ndims == 2 else kernel_size // 2,
                        kernel_size // 2, kernel_size // 2,
                        kernel_size // 2, kernel_size // 2, ]

    output: torch.Tensor = tensor.view(
        tensor.shape[0] * tensor.shape[1], 1, tensor.shape[2], tensor.shape[3], tensor.shape[4]).float()
    output = F.pad(output, pad_e, mode='constant', value=1.)
    output = F.conv3d(output, kernel_e) - se_e.view(1, -1, 1, 1, 1)
    output = torch.min(output, dim=1)[0]

    return output.view_as(tensor).type_as(tensor)


if __name__ == "__main__":
    settings.set_ndims(2)
    img = torch.zeros(1, 1, 5, 5, 1, dtype=torch.long)
    img[:, :, 1, 1, 0] = 3
    img[:, :, 2, 2, 0] = 1

    print(dilation(img, 1).squeeze())
