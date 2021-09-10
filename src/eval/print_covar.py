"""prints the models reconstruction-loss covariance matrix
    """
import argparse
import os
import torch
import torchreg
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util


def load_model(hparams):
    model = util.load_model_from_logdir(hparams.weights)
    model.eval()
    return model


def plot_covar(file, model):

    covar = model.elbo.covar
    print(covar)


def main(hparams):
    model = load_model(hparams)
    plot_covar(hparams.file, model)


if __name__ == "__main__":
    torchreg.settings.set_ndims(2)
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights",
        type=str,
        default="./weights.ckpt",
        help="model checkpoint to initialize with",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/sample",
        help="outputfile, without extension",
    )

    hparams = parser.parse_args()
    main(hparams)
