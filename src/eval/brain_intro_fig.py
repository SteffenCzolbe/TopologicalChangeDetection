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


def plot(args, model_names, brain_subject_ids, brats_subject_ids, sem_model):
    cols = 4
    rows = len(brats_subject_ids)
    # we set up margins with gridspec
    m = 0.05
    axs = np.empty((rows, cols), dtype=object)
    col0 = GridSpec(rows, 1)
    col0.update(left=0, right=1./4 - m, wspace=0.05, hspace=0.05)
    col1 = GridSpec(rows, 1)
    col1.update(left=1./4, right=2./4 - m, wspace=0.05, hspace=0.05)
    col2 = GridSpec(rows, 1)
    col2.update(left=2./4+m, right=3./4, wspace=0.05, hspace=0.05)
    col3 = GridSpec(rows, 1)
    col3.update(left=3./4 + m, right=1., wspace=0.05, hspace=0.05)

    for row in range(rows):
        axs[row, 0] = plt.subplot(col0[row, 0])
        axs[row, 1] = plt.subplot(col1[row, 0])
        axs[row, 2] = plt.subplot(col2[row, 0])
        axs[row, 3] = plt.subplot(col3[row, 0])

    # set-up fig
    fig = viz.Fig(rows+1, cols, title=None, figsize=(8, 3), axs=axs)

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

        # plot sem model tumor prediction
        q_tumor = load_q_tumor(model_names[1], brats_subject_id)
        vmin, vmax = config.MODELS[model_names[1]
                                   ]["p_tumor_probability_range"]
        fig.plot_img(row, 3, crop(J), vmin=0, vmax=1)
        fig.plot_overlay(row, 3, crop(q_tumor), vmin=vmin,
                         vmax=vmax, cbar=False, alpha=0.45)
        if args.overlay_tumor:
            fig.plot_contour(row, 3, crop(tumor_seg),
                             contour_class=1, width=2, rgba=tumor_color)

    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def main(args):
    torchreg.settings.set_ndims(2)
    # we use some brats subjects selected at random in an earlier version of the code,
    # since we had already written the results section for these
    brats_subject_ids = ["BraTS20_Training_169"]
    brain_subject_ids = BrainMRIDataModule(
    ).test_dataloader().dataset.subjects  # [:args.sample_cnt]
    sem_model = util.load_model_from_logdir(
        config.MODELS[config.FULL_MODELS[1]]["path"]["brain2d"])

    plot(args, config.FULL_MODELS, brain_subject_ids,
         brats_subject_ids, sem_model)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/brain_intro_fig",
        help="outputfile, without extension",
    )
    parser.add_argument(
        "--overlay_tumor",
        action="store_true",
        help="set to overlay the tumor",
    )

    hparams = parser.parse_args()
    main(hparams)
