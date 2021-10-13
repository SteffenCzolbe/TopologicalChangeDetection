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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        _, bound_1, _, _ = model.bound(
            I.unsqueeze(0).to(device), J.unsqueeze(0).to(device), bidir=True)

    return bound_1.squeeze(0).cpu()


def crop(img, x_low=78, x_high=-60, y_low=138, y_high=-1):
    return img[:, x_low:x_high, y_low:y_high]


def plot(args, model_name, model, batch):
    sample_id = 31
    cols = 3
    rows = 1

    # we set up margins with gridspec
    m = 0.05
    axs = np.empty((rows, cols), dtype=object)
    col0 = GridSpec(rows, 1)
    col0.update(left=m, right=1./3, wspace=0.001, hspace=0.001)
    col1 = GridSpec(rows, 1)
    col1.update(left=1./3, right=2./3 - m, wspace=0.001, hspace=0.001)
    col2 = GridSpec(rows, 1)
    col2.update(left=2./3 + m, right=1., wspace=0.001, hspace=0.001)

    for row in range(rows):
        axs[row, 0] = plt.subplot(col0[row, 0])
        axs[row, 1] = plt.subplot(col1[row, 0])
        axs[row, 2] = plt.subplot(col2[row, 0])

    fig = viz.Fig(rows+1, cols, title=None,
                  figsize=(cols*2, rows*2), axs=axs)

    # plot I
    fig.plot_img(0, 0, crop(batch["I0"]["data"][sample_id]), vmin=0, vmax=1)
    if args.overlay_seg:
        fig.plot_overlay_class_mask(0, 0, crop(batch["S0"]["data"][sample_id]), colors=[
                                   (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=0.2)
        fig.plot_contour(0, 0, crop(
            batch["S0"]["data"][sample_id]), contour_class=1, width=2, rgba=(0, 40, 255, 255))
        fig.plot_contour(0, 0, crop(
            batch["S0"]["data"][sample_id]), contour_class=2, width=2, rgba=(255, 229, 0, 255))

    # if args.overlay_contour:
    #    fig.plot_contour(0, 0, crop(batch["T0"]["data"][sample_id]),
    #                     contour_class=1, width=2, rgba=(255, 0, 0, 255))

    # plot J
    fig.plot_img(0, 1, crop(batch["I1"]["data"][sample_id]), vmin=0, vmax=1)
    if args.overlay_seg:
        fig.plot_overlay_class_mask(row, 1, crop(batch["S1"]["data"][sample_id]), colors=[
                                   (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=0.2)
        fig.plot_contour(0, 1, crop(
            batch["S1"]["data"][sample_id]), contour_class=1, width=2, rgba=(0, 40, 255, 255))
        fig.plot_contour(0, 1, crop(
            batch["S1"]["data"][sample_id]), contour_class=2, width=2, rgba=(255, 229, 0, 255))

    if args.overlay_contour:
        fig.plot_contour(0, 1, crop(batch["T1"]["data"][sample_id]),
                         contour_class=1, width=2, rgba=(255, 0, 0, 255))

    # plot L_sym(J|I)
    l_sym = load_L_sym(
        model, batch["I0"]["data"][sample_id], batch["I1"]["data"][sample_id])
    vmin, vmax = config.MODELS[model_name]["probability_range"]["platelet-em"]
    fig.plot_img(0, 2, crop(
        batch["I1"]["data"][sample_id]), vmin=0, vmax=1)
    fig.plot_overlay(0, 2, crop(l_sym), vmin=vmin,
                     vmax=vmax, cbar=False, alpha=0.45)

    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def load_samples(dataset):
    torch.manual_seed(0)
    dm = util.load_datamodule_from_name(
        dataset, batch_size=32, pairs=True)
    dl = dm.test_dataloader(shuffle=True)
    batch = next(iter(dl))
    return batch


def load_model(model_name, dataset, device):
    model = util.load_model_from_logdir(
        config.MODELS[model_name]["path"][dataset], model_cls=config.MODELS[model_name]["model_cls"])
    model.to(device)
    return model


def main(args):
    torchreg.settings.set_ndims(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = load_samples("platelet-em")
    model_name = "semantic_loss"
    model = load_model(model_name, "platelet-em", device)

    plot(args, model_name, model, batch)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/intro_fig",
        help="outputfile, without extension",
    )
    parser.add_argument(
        "--overlay_contour",
        action="store_true",
        help="set to overlay segmentations",
    )
    parser.add_argument(
        "--overlay_seg",
        action="store_true",
        help="set to overlay segmentations",
    )

    hparams = parser.parse_args()
    main(hparams)
