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
    bound, info = model.bound(I0, I1, bidir=False)
    bound_bidir_0, bound_bidir_1, info_0to1, info_1to0 = model.bound(
        I0, I1, bidir=True)

    return info_0to1["morphed"], info_1to0["morphed"], bound, bound_bidir_1, bound_bidir_0


def plot(file, I0, I10, I01, I1, bound, bound_bidir_1, bound_bidir_0):
    rows = 8
    # set-up fig
    fig = viz.Fig(rows, 7, None, figsize=(6, 8))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for row in range(rows):
        fig.plot_img(row, 0, I0[row], vmin=0, vmax=1,
                     title="I" if row == 0 else None)
        fig.plot_img(row, 1, I10[row], vmin=0, vmax=1,
                     title="J to I" if row == 0 else None)
        fig.plot_img(row, 2, I01[row], vmin=0, vmax=1,
                     title="I to J" if row == 0 else None)
        fig.plot_img(row, 3, I1[row], vmin=0, vmax=1,
                     title="J" if row == 0 else None)
        fig.plot_img(row, 4, bound[row], cmap='jet',
                     title="p(J | I)" if row == 0 else None)
        fig.plot_img(row, 5, bound_bidir_1[row], cmap='jet',
                     title="bidir J" if row == 0 else None)
        fig.plot_img(row, 6, bound_bidir_0[row], cmap='jet',
                     title="bidir I" if row == 0 else None)

    fig.save(file + ".pdf", close=False)
    fig.save(file + ".png")


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    I0, I1 = get_batch(hparams.ds1, hparams.ds2, device)
    model = load_model(hparams)
    model.to(device)
    I01, I10, bound, bound_bidir_1, bound_bidir_0 = predict(
        model, I0, I1)
    plot(hparams.file, I0, I10, I01, I1, bound, bound_bidir_1, bound_bidir_0)


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
        "--file",
        type=str,
        default="./plots/sample",
        help="outputfile, without extension",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test-set metrics")

    hparams = parser.parse_args()
    with torch.no_grad():
        main(hparams)
