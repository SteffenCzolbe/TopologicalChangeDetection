import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.semantic_loss import SemanticLossModel
import src.util as util


def load_module_and_dataset(hparams):
    # load data
    dm = util.load_datamodule_from_name(
        hparams.ds, batch_size=32, pairs=False)
    # load model
    checkpoint = util.get_checkoint_path_from_logdir(
        hparams.weights)
    model = SemanticLossModel.load_from_checkpoint(
        checkpoint_path=checkpoint, strict=True
    )
    model.eval()
    return model, dm


def get_batch(dm, device):
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    I = batch['I']['data'].to(device)

    return I


def get_feature_stack(model, I):
    feats = model.augment_image(I)

    return feats


def plot(file, I, feats):
    rows = 8
    # set-up fig
    fig = viz.Fig(rows, 6, None, figsize=(5, 8))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for row in range(rows):
        fig.plot_img(row, 0, I[row], vmin=0, vmax=1,
                     title="I0" if row == 0 else None)
        fig.plot_img(row, 1, feats[row, 0],
                     title="0" if row == 0 else None)
        fig.plot_img(row, 2, feats[row, 16],
                     title="16" if row == 0 else None)
        fig.plot_img(row, 3, feats[row, 439],
                     title="439" if row == 0 else None)
        fig.plot_img(row, 4, feats[row, 440],
                     title="440" if row == 0 else None)

    fig.save(file + ".pdf", close=False)
    fig.save(file + ".png")


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, dm = load_module_and_dataset(hparams)
    model.to(device)
    I = get_batch(dm, device)
    feats = get_feature_stack(
        model, I)
    plot(hparams.file, I, feats)


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
        "--ds",
        type=str,
        default="brain2d",
        help="Dataset 1",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/sample",
        help="outputfile, without extension",
    )
    hparams = parser.parse_args()
    main(hparams)
