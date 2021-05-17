import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util
import src.eval.config as config


def get_images(device):
    idxs = [3, 4, 22]
    # load data
    dm1 = util.load_datamodule_from_name("brain2d", batch_size=32, pairs=False)
    dm2 = util.load_datamodule_from_name("brats2d", batch_size=32, pairs=False)

    # extract image
    dl1 = dm1.test_dataloader()
    batch1 = next(iter(dl1))
    I0 = batch1['I']['data'][idxs].to(device)

    dl2 = dm2.test_dataloader()
    batch2 = next(iter(dl2))
    I1 = batch2['I']['data'][idxs].to(device)

    return I0, I1


def load_models(model_names, device):
    models = []
    for model_name in model_names:
        model_logdir = config.MODELS[model_name]["path"]
        model_cls = config.MODELS[model_name]["model_cls"]
        model = util.load_model_from_logdir(model_logdir, model_cls=model_cls)
        model.to(device)
        models.append(model)
    return models


def predict(model, I0, I1):
    with torch.no_grad():
        _, bound_1, _, _ = model.bound(I0, I1, bidir=True)

    return bound_1


def crop(img, x_low=25, x_high=224, y_low=0, y_high=160):
    return img[:, :, x_low:x_high, y_low:y_high]


def plot(args, model_names, models, I0, I1):
    cols = 2 + len(model_names)
    rows = len(I0)
    # set-up fig
    fig = viz.Fig(rows, cols, title=None, figsize=(4.5*1.5, 1.5+rows*1.65))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # plot images
    for row in range(rows):
        fig.plot_img(row, 0, crop(I0)[row], vmin=0, vmax=1)
        fig.plot_img(row, 1, crop(I1)[row], vmin=0, vmax=1)

    # plot model predictions
    for i, (model_name, model) in enumerate(zip(model_names, models)):

        bound_1 = predict(model, I0, I1)
        vmin, vmax = config.MODELS[model_name]["probability_range"]
        model_display_name = config.MODELS[model_name]["display_name"]

        for j in range(rows):
            fig.plot_img(j, i + 2, crop(I1)[j], vmin=0, vmax=1,
                         title=model_display_name if j == 0 else None)
            fig.plot_overlay(
                j, i + 2, crop(bound_1)[j], vmin=vmin, vmax=vmax, cbar=False, alpha=0.45)

    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    I0, I1 = get_images(device)
    model_names = config.FULL_MODELS
    models = load_models(model_names, device)
    plot(args, model_names, models, I0, I1)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/qualitative_comparison",
        help="outputfile, without extension",
    )

    hparams = parser.parse_args()
    main(hparams)
