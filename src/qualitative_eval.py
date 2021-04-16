import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util


def load_module_and_dataset(hparams):
    # load data
    dm1 = util.load_datamodule_from_name(hparams.ds1, batch_size=32)
    dm2 = util.load_datamodule_from_name(hparams.ds2, batch_size=32)
    # load model
    model = util.load_model_from_logdir(hparams.weights)
    model.eval()
    return model, dm1, dm2


def get_batch(dm1, dm2, device):
    dl1 = dm1.test_dataloader()
    batch1 = next(iter(dl1))
    I0 = batch1['I0']['data'].to(device)

    dl2 = dm2.test_dataloader()
    batch2 = next(iter(dl2))
    I1 = batch2['I1']['data'].to(device)

    return I0, I1


def predict(model, I0, I1):
    bound, transform, I01 = model.forward(I0, I1)
    return bound, I01, transform


def plot(file, I0, I01, I1, bound):
    rows = 8
    # set-up fig
    fig = viz.Fig(rows, 4, None, figsize=(5, 8))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for row in range(rows):
        fig.plot_img(row, 0, I0[row], vmin=0, vmax=1)
        fig.plot_img(row, 1, I01[row], vmin=0, vmax=1)
        fig.plot_img(row, 2, I1[row], vmin=0, vmax=1)
        fig.plot_img(row, 3, bound[row], cmap='jet')

    fig.save(file + ".pdf", close=False)
    fig.save(file + ".png")


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, dm1, dm2 = load_module_and_dataset(hparams)
    model.to(device)
    I0, I1 = get_batch(dm1, dm2, device)
    bound, I01, transform = predict(model, I0, I1)
    plot(hparams.file, I0, I01, I1, bound)


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
        default="brats2d",
        help="Dataset 1",
    )
    parser.add_argument(
        "--ds2",
        type=str,
        default="brain2d",
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
    main(hparams)
