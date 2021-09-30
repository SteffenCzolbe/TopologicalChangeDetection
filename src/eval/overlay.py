import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util


def load_model(hparams):
    # load model
    model = util.load_model_from_logdir(hparams.weights)
    model.eval()
    return model


def get_batch(ds1, ds2, device):
    if ds1 == ds2:
        dm = util.load_datamodule_from_name(
            ds1, batch_size=32, pairs=True)
        dl = dm.test_dataloader()
        batch = next(iter(dl))
        I0 = batch['I0']['data'].to(device)
        I1 = batch['I1']['data'].to(device)

    else:
        dm1 = util.load_datamodule_from_name(
            ds1, batch_size=32, pairs=False)
        dm2 = util.load_datamodule_from_name(
            ds2, batch_size=32, pairs=False)
        dl1 = dm1.test_dataloader()
        batch1 = next(iter(dl1))
        I0 = batch1['I']['data'].to(device)

        dl2 = dm2.test_dataloader(shuffle=True)
        batch2 = next(iter(dl2))
        I1 = batch2['I']['data'].to(device)

    return I0, I1


def predict(model, I0, I1):
    bound_0, bound_1, info, _ = model.bound(I0, I1, bidir=True)

    return info["morphed"], bound_0, bound_1


def plot(args, I0, I1, bound_0, bound_1):
    rows = 8
    # set-up fig
    fig = viz.Fig(rows, 4, None, figsize=(5, 8))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # define intensity range
    vmin = args.range[0]
    vmax = args.range[1]

    for row in range(rows):
        fig.plot_img(row, 0, I0[row], vmin=0, vmax=1,
                     title="I" if row == 0 else None)
        fig.plot_img(row, 1, I0[row], vmin=0, vmax=1,
                     title="I, annot." if row == 0 else None)
        fig.plot_overlay(
            row, 1, bound_0[row], vmin=vmin, vmax=vmax, cbar=True, alpha=0.45)
        fig.plot_img(row, 2, I1[row], vmin=0, vmax=1,
                     title="J, annot." if row == 0 else None)
        fig.plot_overlay(
            row, 2, bound_1[row], vmin=vmin, vmax=vmax, cbar=True, alpha=0.45)
        fig.plot_img(row, 3, I1[row], vmin=0, vmax=1,
                     title="J" if row == 0 else None)

    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    I0, I1 = get_batch(hparams.ds1, hparams.ds2, device)
    model = load_model(hparams)
    model.to(device)
    I01, bound_0, bound_1 = predict(
        model, I0, I1)
    plot(args, I0, I1, bound_0, bound_1)


if __name__ == "__main__":
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
        "--ds1",
        type=str,
        default="brain2d",
        help="Dataset 1",
    )
    parser.add_argument(
        "--ds2",
        type=str,
        default="brats2d",
        help="Dataset 2",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs="+",
        default=[1, 20],
        help="Dataset 2",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/overlay",
        help="outputfile, without extension",
    )

    hparams = parser.parse_args()
    main(hparams)
