"""Determine the impact of KL loss vs Reconstruction loss
    """
import argparse
import os
import shutil
import torch
import torchreg
import pytorch_lightning as pl
from src.registration_model import RegistrationModel
import src.util as util
from tqdm import tqdm
from src.datamodules.brainmri_datamodule import BrainMRIDataset
import torchreg.viz as viz
import src.eval.config as config
import numpy as np
"""
      We caluclate the term

      p_tumor(J) = E_{I}[log p(J|I)] - E_{I, K}[ log p(I|K) transform_{I -> J}]
                 = 1/N sum_I (log p(J|I) - (1/N sum_K log p(I|K)) transform_{I -> J} )
                                           |--------------------|
                                              pre-computed in mean_pIK()
                   |--- computed in pJ() --------------------------------------------|

"""


def value_range(values, low_cutoff=0., high_cutoff=1.):
    values = values.flatten().cpu().numpy()
    N = len(values)
    values = np.sort(values)
    lo = values[int(low_cutoff*N)]
    hi = values[int(high_cutoff*N)]
    return hi-lo


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    brains_dm = util.load_datamodule_from_name(
        "brain2d", batch_size=args.samples, load_val_seg=False, pairs=False)
    test_data = iter(brains_dm.test_dataloader(shuffle=True))
    Is = next(test_data)
    Ks = next(test_data)
    subject_ids = Is['subject_id']
    Is = Is['I']['data'].to(device)
    Ks = Ks['I']['data'].to(device)

    # load model
    weights = config.MODELS[args.model_name]["path"]
    model_cls = config.MODELS[args.model_name]["model_cls"]
    model = util.load_model_from_logdir(weights, model_cls=model_cls)
    model.eval()
    model.to(device)

    d = model.forward(Ks, Is)
    recon_loss = d["recon_loss"]
    kl_loss = d["kl_loss"]

    recon_loss_value_range = value_range(
        recon_loss, low_cutoff=0.03, high_cutoff=0.97)
    kl_loss_value_range = value_range(
        kl_loss, low_cutoff=0.03, high_cutoff=0.97)

    print(f"Model {args.model_name}:")
    print(f"recon_loss value range: {recon_loss_value_range:.1f}")
    print(f"kl_loss value range: {kl_loss_value_range:.1f}")
    pct = kl_loss_value_range / \
        (kl_loss_value_range + recon_loss_value_range) * 100
    print(f"kl loss contributes {pct:.2f}% to total loss")
    print("")


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--model_name",
        type=str,
        help="name of model to test",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="sample count of images to use",
    )

    hparams = parser.parse_args()
    pl.seed_everything(42)
    with torch.no_grad():
        main(hparams)
