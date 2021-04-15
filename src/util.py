import glob
import os
import re
import yaml
import torch
import torchreg


def to_device(obj, device):
    """Maps a torch tensor, or a collection containing torch tesnors recursively onto the gpu

    Args:
        obj ([type]): [description]
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif hasattr(obj, "__iter__"):
        return [to_device(o, device) for o in obj]
    else:
        raise Exception(f"Do not know how to map object {obj} to {device}")


def get_supported_datamodules():
    from src.datamodules.mnist_datamodule import MnistDataModule
    from src.datamodules.brainmri_datamodule import BrainMRIDataModule

    supported_datamodels = {"mnist": (MnistDataModule, {}),
                            "brain": (BrainMRIDataModule, {'volumetric': True, 'atlasreg': False}),
                            "brain2d": (BrainMRIDataModule, {'volumetric': False, 'atlasreg': False}),
                            "brainatlas": (BrainMRIDataModule, {'volumetric': True, 'atlasreg': True}),
                            "brain2datlas": (BrainMRIDataModule, {'volumetric': False, 'atlasreg': True})}

    return supported_datamodels


def load_datamodule_from_name(dataset_name, batch_size=32):
    """Loads a Datamodule

    Args:
        dataset_name (str): Name of dataset
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        Datamodule
    """
    supported_datamodels = get_supported_datamodules()
    if dataset_name not in supported_datamodels:
        raise Exception(
            f"Dataset {dataset_name} unknown. Supported datasets: {supported_datamodels.keys()}"
        )

    datamodule_cls, args = supported_datamodels[dataset_name]
    datamodule = datamodule_cls(batch_size=batch_size, **args)

    if datamodule.dims[-1] == 1:
        torchreg.settings.set_ndims(2)
    else:
        torchreg.settings.set_ndims(3)

    return datamodule


def load_datamodule_for_model(model, batch_size=None):
    """Loads the datamodule for the model. kwargs set will overwrite model defaults.

    Args:
        model: The model
        batchsize (bool, optional): Set to overwrite batch size.
    """
    batch_size = batch_size if batch_size is not None else model.hparams.batch_size
    datamodule_name = model.hparams.dataset
    return load_datamodule(datamodule_name, batch_size=batch_size)


def entropy(p):
    """
    Calculates the entropy (uncertainty) of p

    Args:
        p (Tensor BxCxHxW): probability per class

    Returns:
        Tensor Bx1xHxW
    """
    mask = p > 0.00001
    h = torch.zeros_like(p)
    h[mask] = torch.log2(1 / p[mask])
    H = torch.sum(p * h, dim=1, keepdim=True)
    return H


def binary_entropy(p):
    """
    Calculates the entropy (uncertainty) of p

    Args:
        p (Tensor Bx1xHxW): probability per class

    Returns:
        Tensor Bx1xHxW
    """
    p = torch.cat([p, 1 - p], dim=1)
    return entropy(p)


def onehot(tensor, num_classes):
    """converts an int-tensor into one-hot encoding

    Args:
        Long tensor: Bx1xHxW


    Returns:
        Float tensor Bx1xHxW
    """
    return torch.nn.functional.one_hot(tensor[:, 0], num_classes=num_classes).permute(0, -1, 1, 2)


def wasserstein(p, q):
    """Discrete wasserstein / earthmovers distance along the channel dimension

    Args:
        p (torch.Tensor): Probability distribution p (channel-wise), tensor of shape BxCx...
        q (torch.Tensor): Probability distribution q (channel-wise), tensor of shape BxCx...

    Returns:
        torch.Tensor: tensor of shape Bx1x...
    """
    return (p-q).abs().sum(dim=1, keepdim=True)


def kl_divergence(p, q, eps=0.):
    """
    Pixel-whise KL-divergence along the channel dimension

    For two vectors p, q
    KL(p, q) = sum_i p_i * log(p_i / q_i)

    Note! KL-divergence is not defined for 
    p > 0, q == 0, this function will return inf in this case.

    Args:
        p (torch.Tensor): Probability distribution p (channel-wise), tensor of shape BxCx...
        q (torch.Tensor): Probability distribution q (channel-wise), tensor of shape BxCx...
        eps (float, optional): Smoothing epsilon. Can be set to avoid p,q ==0 or ==1. Defaults to 0.

    Returns:
        torch.Tensor: tensor of shape Bx1x...
    """
    if eps > 0:
        eps = torch.tensor(eps, dtype=p.dtype, device=p.device)
        # bound with epsilon
        p = torch.max(p, eps)
        p = torch.min(p, 1-eps)
        q = torch.max(q, eps)
        q = torch.min(q, 1-eps)

    # p == 0 implies kl == 0. We use a mask to avoid computing 0 * -inf
    mask = p > 0.00001
    h = torch.zeros_like(p)
    h[mask] = torch.log(p[mask] / q[mask])
    H = torch.sum(p * h, dim=1, keepdim=True)
    if torch.isinf(H).any():
        print("Warning: KL-divergence contains inf values, probably due to invalid input!")
    return H
