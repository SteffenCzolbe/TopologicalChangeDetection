import argparse
import os
import torch
import torchreg
import pytorch_lightning as pl
from src.datamodules.brainmri_datamodule import BrainMRIDataModule
from src.datamodules.brats_datamodule import BraTSDataModule
import torchreg.viz as viz
import src.util as util
import src.eval.config as config
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def load_L_sym(model, I, J):
    with torch.no_grad():
        _, bound_1, _, _ = model.bound(
            I.unsqueeze(0), J.unsqueeze(0), bidir=True)

    return bound_1.squeeze(0)


def crop(img, x_low=0, x_high=-1, y_low=0, y_high=-1):
    return img[:, x_low:x_high, y_low:y_high]


def plot(args, model_names, models, I, J):
    cols = 2 + len(model_names)
    rows = len(I)
    """
    # TODO: figure spacing
    # we set up margins with gridspec
    m = 0.01
    axs = np.empty((rows, cols), dtype=object)
    col0 = GridSpec(rows, 1)
    col0.update(left=0, right=1./6 - m, wspace=0.05, hspace=0.05)
    col1 = GridSpec(rows, 1)
    col1.update(left=1./6 + m, right=2./6, wspace=0.05, hspace=0.05)
    col2 = GridSpec(rows, 1)
    col2.update(left=2./6, right=3./6 - m, wspace=0.05, hspace=0.05)
    col3 = GridSpec(rows, 1)
    col3.update(left=3./6 + m, right=4./6, wspace=0.05, hspace=0.05)
    col4 = GridSpec(rows, 1)
    col4.update(left=4./6 + 2*m, right=5./6 + m, wspace=0.05, hspace=0.05)
    col5 = GridSpec(rows, 1)
    col5.update(left=5./6 + m, right=1, wspace=0.05, hspace=0.05)

    for row in range(rows):
        axs[row, 0] = plt.subplot(col0[row, 0])
        axs[row, 1] = plt.subplot(col1[row, 0])
        axs[row, 2] = plt.subplot(col2[row, 0])
        axs[row, 3] = plt.subplot(col3[row, 0])
        axs[row, 4] = plt.subplot(col4[row, 0])
        axs[row, 5] = plt.subplot(col5[row, 0])

    # set-up fig
    fig = viz.Fig(rows, cols, title=None, figsize=(5.3, 1+rows*1.1), axs=axs)
        """
    fig = viz.Fig(rows, cols, title=None, figsize=(1+cols*1.1, 1+rows*1.1))

    for row in range(rows):
        # plot I
        fig.plot_img(row, 0, crop(I[row]), vmin=0, vmax=1)

        # plot J
        fig.plot_img(row, 1, crop(J[row]), vmin=0, vmax=1)

        # plot L_sym(J|I)
        for col, (model_name, model) in enumerate(zip(model_names, models), start=2):
            l_sym = load_L_sym(model, I[row], J[row])
            vmin, vmax = config.MODELS[model_name]["probability_range"]["platelet-em"]
            fig.plot_img(row, col, crop(J[row]), vmin=0, vmax=1)
            fig.plot_overlay(row, col, crop(l_sym), vmin=vmin,
                             vmax=vmax, cbar=False, alpha=0.45)

    fig.set_col_labels(["$\mathbf{I}$", "$\mathbf{J}$"] + model_names)
    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def load_samples(dataset, device):
    torch.manual_seed(0)
    dm = util.load_datamodule_from_name(
        dataset, batch_size=5, pairs=True)
    dl = dm.test_dataloader(shuffle=True)
    batch = next(iter(dl))
    I0 = batch['I0']['data'].to(device)
    I1 = batch['I1']['data'].to(device)
    return I0, I1


def load_models(model_names, dataset, device):
    models = []

    for model_name in model_names:
        model = util.load_model_from_logdir(
            config.MODELS[model_name]["path"][dataset])
        model.to(device)
        models.append(model)

    return models


def main(args):
    torchreg.settings.set_ndims(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    I, J = load_samples(args.dataset, device)
    model_names = ["mse", "semantic_loss"]
    models = load_models(model_names, args.dataset, device)

    plot(args, model_names, models, I, J)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/pub_fig",
        help="outputfile, without extension",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="platelet-em",
        help="dataset",
    )

    hparams = parser.parse_args()
    main(hparams)
