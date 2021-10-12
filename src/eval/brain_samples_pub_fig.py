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


def load_images(brain_subject_id, brats_subject_id):
    I, _ = util.load_subject_from_dataset("brain2d", "test", brain_subject_id)
    J, tumor = util.load_subject_from_dataset(
        "brats2d", "test", brats_subject_id)
    return I['data'], J['data'], tumor['data']


def load_q_tumor(model_name, subject_id):
    dir = os.path.join(config.MODELS[model_name]['path']["brain2d"], "p_tumor")
    q_tumor = torch.load(os.path.join(dir, subject_id))
    return q_tumor


def load_L_sym(model, I, J):
    with torch.no_grad():
        _, bound_1, _, _ = model.bound(
            I.unsqueeze(0), J.unsqueeze(0), bidir=True)

    return bound_1.squeeze(0)


def crop(img, x_low=25, x_high=224, y_low=0, y_high=160):
    return img[:, x_low:x_high, y_low:y_high]


def plot(args, model_names, brain_subject_ids, brats_subject_ids, mse_model, sem_model):
    cols = 6
    rows = len(brats_subject_ids)
    # we set up margins with gridspec
    m = 0.01
    axs = np.empty((rows, cols), dtype=object)
    col0 = GridSpec(rows, 1)
    col0.update(left=0, right=1./6 - m, wspace=0.05, hspace=0.05)
    col1 = GridSpec(rows, 1)
    col1.update(left=1./6 + m, right=2./6, wspace=0.05, hspace=0.05)
    col2 = GridSpec(rows, 1)
    col2.update(left=2./6+2*m, right=3./6 + m, wspace=0.05, hspace=0.05)
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

    tumor_color = (255, 0, 0, 255)

    # plot images
    for row, (brain_subject_id, brats_subject_id) in enumerate(zip(brain_subject_ids, brats_subject_ids)):
        I, J, tumor_seg = load_images(brain_subject_id, brats_subject_id)

        # plot I
        fig.plot_img(row, 0, crop(I), vmin=0, vmax=1)

        # plot tumor brain J
        fig.plot_img(row, 1, crop(J), vmin=0, vmax=1)

        # plot sem model topological differences
        l_sym = load_L_sym(sem_model, I, J)
        vmin, vmax = config.MODELS[model_names[1]
                                   ]["probability_range"]["brain2d"]
        fig.plot_img(row, 2, crop(J), vmin=0, vmax=1)
        fig.plot_overlay(row, 2, crop(l_sym), vmin=vmin,
                         vmax=vmax, cbar=False, alpha=0.45)

        # plot mse model topological differences
        l_sym = load_L_sym(mse_model, I, J)
        vmin, vmax = config.MODELS[model_names[0]
                                   ]["probability_range"]["brain2d"]
        fig.plot_img(row, 3, crop(J), vmin=0, vmax=1)
        fig.plot_overlay(row, 3, crop(l_sym), vmin=vmin,
                         vmax=vmax, cbar=False, alpha=0.45)

        # plot sem model tumor prediction
        q_tumor = load_q_tumor(model_names[1], brats_subject_id)
        vmin, vmax = config.MODELS[model_names[1]
                                   ]["p_tumor_probability_range"]
        fig.plot_img(row, 4, crop(J), vmin=0, vmax=1)
        fig.plot_overlay(row, 4, crop(q_tumor), vmin=vmin,
                         vmax=vmax, cbar=False, alpha=0.45)
        fig.plot_contour(row, 4, crop(tumor_seg),
                         contour_class=1, width=2, rgba=tumor_color)

        # plot mse model tumor prediction
        q_tumor = load_q_tumor(model_names[0], brats_subject_id)
        vmin, vmax = config.MODELS[model_names[0]
                                   ]["p_tumor_probability_range"]
        fig.plot_img(row, 5, crop(J), vmin=0, vmax=1)
        fig.plot_overlay(row, 5, crop(q_tumor), vmin=vmin,
                         vmax=vmax, cbar=False, alpha=0.45)
        fig.plot_contour(row, 5, crop(tumor_seg),
                         contour_class=1, width=2, rgba=tumor_color)

    fig.set_col_labels([None, None, "SEM", "MSE", "SEM", "MSE"])
    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def main(args):
    torchreg.settings.set_ndims(2)
    # we use some brats subjects selected at random in an earlier version of the code,
    # since we had already written the results section for these
    if args.dataset == "brain2d":
        brats_subject_ids = ["BraTS20_Training_309", "BraTS20_Training_087",
                             "BraTS20_Training_169", "BraTS20_Training_323", "BraTS20_Training_149"]
        brain_subject_ids = BrainMRIDataModule(
        ).test_dataloader().dataset.subjects  # [:args.sample_cnt]
    else:
        raise Exception(f"not implemented for dataset {args.dataset}")
    mse_model = util.load_model_from_logdir(
        config.MODELS[config.FULL_MODELS[0]]["path"][args.dataset])
    sem_model = util.load_model_from_logdir(
        config.MODELS[config.FULL_MODELS[1]]["path"][args.dataset])

    plot(args, config.FULL_MODELS, brain_subject_ids,
         brats_subject_ids, mse_model, sem_model)


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
        default="brain2d",
        help="dataset",
    )

    hparams = parser.parse_args()
    main(hparams)
