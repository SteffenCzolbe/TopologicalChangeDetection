import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util


def load_module_and_dataset(hparams):
    # load data
    dm1 = util.load_datamodule_from_name(
        hparams.ds1, batch_size=32, pairs=False)
    dm2 = util.load_datamodule_from_name(
        hparams.ds2, batch_size=32, pairs=False)
    # load model
    model = util.load_model_from_logdir(hparams.weights)
    model.eval()
    return model, dm1, dm2


def get_batch(dm1, dm2, device):
    dl1 = dm1.test_dataloader()
    batch1 = next(iter(dl1))
    I0 = batch1['I']['data'].to(device)

    dl2 = dm2.test_dataloader()
    batch2 = next(iter(dl2))
    I1 = batch2['I']['data'].to(device)

    return I0, I1


def predict(model, I0, I1):
    bound, transform, I01 = model.forward(I0, I1)
    u, log_var = model.encoder(I0, I1)
    recon_loss = model.elbo.recon_loss(I01, I1, reduction="none")
    kl_div = model.elbo.kl_loss(u, log_var, reduction="none")

    return bound, I01, transform, recon_loss, kl_div


def plot(file, I0, I01, I1, bound, recon_loss, kl_div):
    rows = 8
    # set-up fig
    fig = viz.Fig(rows, 6, None, figsize=(5, 8))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for row in range(rows):
        fig.plot_img(row, 0, I0[row], vmin=0, vmax=1,
                     title="I0" if row == 0 else None)
        fig.plot_img(row, 1, I01[row], vmin=0, vmax=1,
                     title="I01" if row == 0 else None)
        fig.plot_img(row, 2, I1[row], vmin=0, vmax=1,
                     title="I1" if row == 0 else None)
        fig.plot_img(row, 3, bound[row], cmap='jet',
                     title="p(I1 | I0)" if row == 0 else None)
        fig.plot_img(row, 4, recon_loss[row], cmap='jet',
                     title="$log p(J|I, \Phi)$" if row == 0 else None)
        fig.plot_img(row, 5, kl_div[row], cmap='jet',
                     title="$D_{KL}$" if row == 0 else None)

    fig.save(file + ".pdf", close=False)
    fig.save(file + ".png")


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, dm1, dm2 = load_module_and_dataset(hparams)
    model.to(device)
    I0, I1 = get_batch(dm1, dm2, device)
    bound, I01, transform, recon_loss, kl_div = predict(
        model, I0, I1)
    plot(hparams.file, I0, I01, I1, bound,
         recon_loss, kl_div)


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
    main(hparams)
