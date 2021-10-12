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


def crop(img, x_low=0, x_high=-1, y_low=0, y_high=-1):
    return img[:, x_low:x_high, y_low:y_high]


def plot(args, model_names, models, batch):
    sample_ids = [0, 6, 3, 18]
    cols = 2 + len(model_names)
    rows = len(sample_ids)
    fig = viz.Fig(rows, cols, title=None, figsize=(cols*2, rows*2))

    for row, idx in enumerate(sample_ids):
        # plot I
        fig.plot_img(row, 0, crop(batch["I0"]["data"][idx]), vmin=0, vmax=1)
        if args.overlay_seg:
            # fig.plot_overlay_class_mask(row, 0, crop(batch["S0"]["data"][idx]), colors=[
            #                            (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=0.1)
            fig.plot_contour(row, 0, crop(
                batch["S0"]["data"][idx]), contour_class=1, width=2, rgba=(0, 40, 255, 255))
            fig.plot_contour(row, 0, crop(
                batch["S0"]["data"][idx]), contour_class=2, width=2, rgba=(255, 229, 0, 255))

        if not args.combine_annotations:
            fig.plot_contour(row, 0, crop(batch["T0"]["data"][idx]),
                             contour_class=1, width=2, rgba=(255, 0, 0, 255))

        # plot J
        fig.plot_img(row, 1, crop(batch["I1"]["data"][idx]), vmin=0, vmax=1)
        if args.overlay_seg:
            # fig.plot_overlay_class_mask(row, 1, crop(batch["S1"]["data"][idx]), colors=[
            #                            (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=0.1)
            fig.plot_contour(row, 1, crop(
                batch["S1"]["data"][idx]), contour_class=1, width=2, rgba=(0, 40, 255, 255))
            fig.plot_contour(row, 1, crop(
                batch["S1"]["data"][idx]), contour_class=2, width=2, rgba=(255, 229, 0, 255))

        if not args.combine_annotations:
            fig.plot_contour(row, 1, crop(batch["T1"]["data"][idx]),
                             contour_class=1, width=2, rgba=(255, 0, 0, 255))

        # plot L_sym(J|I)
        for col, (model_name, model) in enumerate(zip(model_names, models), start=2):
            l_sym = load_L_sym(
                model, batch["I0"]["data"][idx], batch["I1"]["data"][idx])
            vmin, vmax = config.MODELS[model_name]["probability_range"]["platelet-em"]
            fig.plot_img(row, col, crop(
                batch["I1"]["data"][idx]), vmin=0, vmax=1)
            fig.plot_overlay(row, col, crop(l_sym), vmin=vmin,
                             vmax=vmax, cbar=False, alpha=0.45)
            if args.overlay_seg:
                # fig.plot_overlay_class_mask(row, col, crop(batch["S1"]["data"][idx]), colors=[
                #                            (0, 40, 97), (0, 40, 255), (255, 229, 0)], alpha=0.1)
                fig.plot_contour(row, col, crop(
                    batch["S1"]["data"][idx]), contour_class=1, width=2, rgba=(0, 40, 255, 255))
                fig.plot_contour(row, col, crop(
                    batch["S1"]["data"][idx]), contour_class=2, width=2, rgba=(255, 229, 0, 255))

            if args.combine_annotations:
                fig.plot_contour(row, col, crop(batch["Tcombined"]["data"][idx]),
                                 contour_class=1, width=2, rgba=(255, 0, 0, 255))

    if len(model_names) > 1:
        fig.set_row_labels(["$\mathbf{I}$", "$\mathbf{J}$"] + model_names)
    fig.fig.tight_layout()
    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def load_samples(dataset):
    torch.manual_seed(0)
    dm = util.load_datamodule_from_name(
        dataset, batch_size=32, pairs=True)
    dl = dm.test_dataloader(shuffle=True)
    batch = next(iter(dl))
    return batch


def load_models(model_names, dataset, device):
    models = []

    for model_name in model_names:
        model = util.load_model_from_logdir(
            config.MODELS[model_name]["path"][dataset], model_cls=config.MODELS[model_name]["model_cls"])
        model.to(device)
        models.append(model)

    return models


def filter_for_existing_models(model_names, dataset):
    existing_models = []
    for model_name in model_names:
        if util.checkpoint_exists(config.MODELS[model_name]["path"][dataset]):
            existing_models.append(model_name)
    return existing_models


def main(args):
    torchreg.settings.set_ndims(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = load_samples(args.dataset)

    if args.all_models:
        model_names = filter_for_existing_models(
            config.ALL_MODELS, args.dataset)
    else:
        model_names = ["semantic_loss"]
    models = load_models(model_names, args.dataset, device)

    plot(args, model_names, models, batch)


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
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="set to compare all models",
    )
    parser.add_argument(
        "--combine_annotations",
        action="store_true",
        help="set to compare all models",
    )
    parser.add_argument(
        "--overlay_seg",
        action="store_true",
        help="set to overlay segmentations",
    )

    hparams = parser.parse_args()
    main(hparams)
