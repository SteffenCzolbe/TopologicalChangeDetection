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


def plot(args, batch):
    cols = 4
    rows = 5
    # samples avoiding showing the same patch multiple times
    sample_ids = [21, 18, 3, 2, 6]

    # we set up margins with gridspec
    m = 0.01
    axs = np.empty((rows, cols), dtype=object)
    col0 = GridSpec(rows, 1)
    col0.update(left=m, right=1./4, wspace=0.05, hspace=0.05)
    col1 = GridSpec(rows, 1)
    col1.update(left=1./4, right=2./4 - m, wspace=0.05, hspace=0.05)
    col2 = GridSpec(rows, 1)
    col2.update(left=2./4 + m, right=3./4, wspace=0.05, hspace=0.05)
    col3 = GridSpec(rows, 1)
    col3.update(left=3./4, right=1. - m, wspace=0.05, hspace=0.05)

    for row in range(rows):
        axs[row, 0] = plt.subplot(col0[row, 0])
        axs[row, 1] = plt.subplot(col1[row, 0])
        axs[row, 2] = plt.subplot(col2[row, 0])
        axs[row, 3] = plt.subplot(col3[row, 0])
    fig = viz.Fig(rows, cols, title=None, figsize=(6.5, 10), axs=axs)

    for row, idx in enumerate(sample_ids):
        # plot I
        fig.plot_img(row, 0, batch["I0"]["data"][idx], vmin=0, vmax=1)

        # plot I seg
        fig.plot_overlay_class_mask(row, 1, batch["S0"]["data"][idx], colors=[
                                    (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=1)
        fig.plot_contour(row, 1, batch["T0"]["data"][idx],
                         contour_class=1, width=2, rgba=(255, 0, 0, 255))
        fig.plot_contour(row, 1, batch["T1"]["data"][idx],
                         contour_class=1, width=2, rgba=(0, 255, 0, 255))

        # plot J
        fig.plot_img(row, 2, batch["I1"]["data"][idx], vmin=0, vmax=1)

        # plot J seg
        fig.plot_overlay_class_mask(row, 3, batch["S1"]["data"][idx], colors=[
                                    (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=1)
        fig.plot_contour(row, 3, batch["T1"]["data"][idx],
                         contour_class=1, width=2, rgba=(255, 0, 0, 255))
        fig.plot_contour(row, 3, batch["T0"]["data"][idx],
                         contour_class=1, width=2, rgba=(0, 255, 0, 255))

    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def load_samples(dataset):
    torch.manual_seed(0)
    dm = util.load_datamodule_from_name(
        dataset, batch_size=32, pairs=True)
    dl = dm.test_dataloader(shuffle=True)
    batch = next(iter(dl))
    return batch


def main(args):
    batch = load_samples("platelet-em")

    plot(args, batch)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/platelet_dataset_samples",
        help="outputfile, without extension",
    )

    hparams = parser.parse_args()
    main(hparams)
