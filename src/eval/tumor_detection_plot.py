import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util
import src.eval.config as config


def load_q_tumor(model_name, subject_ids):
    # load I
    dir = os.path.join(config.MODELS[model_name]['path'], "p_tumor")
    q_tumors = [torch.load(os.path.join(dir, subject_id))
                for subject_id in subject_ids]
    q_tumors = torch.stack(q_tumors)
    return q_tumors


def load_images(subject_ids):
    idxs = [3, 4, 22]
    Is, Ss = [], []

    for s in subject_ids:
        I, S = util.load_subject_from_dataset("brats2d", "test", s)
        Is.append(I['data'])
        Ss.append(S['data'])

    Is = torch.stack(Is)
    Ss = torch.stack(Ss)
    return Is, Ss


def crop(img, x_low=25, x_high=224, y_low=0, y_high=160):
    return img[:, :, x_low:x_high, y_low:y_high]


def plot(args, model_names, J, S, subject_ids):
    cols = 1 + len(model_names)
    rows = len(J)
    # set-up fig
    fig = viz.Fig(rows, cols, title=None, figsize=(4.5*1.5, 1.5+rows*1.65))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # plot images
    for row in range(rows):
        fig.plot_img(row, 0, crop(J)[row], vmin=0, vmax=1)

    # plot model predictions
    for i, model_name in enumerate(model_names):
        Q_tumor = load_q_tumor(model_name, subject_ids)

        vmin, vmax = config.MODELS[model_name]["p_tumor_probability_range"]
        model_display_name = config.MODELS[model_name]["display_name"]

        for j in range(rows):
            fig.plot_img(j, i + 1, crop(J)[j], vmin=0, vmax=1,
                         title=model_display_name if j == 0 else None)
            fig.plot_overlay(
                j, i + 1, crop(Q_tumor)[j], vmin=vmin, vmax=vmax, cbar=False, alpha=0.45)
            fig.plot_contour(  # edema
                j,
                i+1,
                crop(S)[j],
                contour_class=2,
                width=1,
                rgba=(236, 200, 12, 255),
            )
            fig.plot_contour(  # tumor
                j,
                i+1,
                crop(S)[j],
                contour_class=1,
                width=1,
                rgba=(36, 255, 12, 255),
            )

    fig.set_row_labels(subject_ids)
    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")


def main(args):
    subject_ids = os.listdir('trained_models/semantic_loss/p_tumor/')[:12]
    J, S = load_images(subject_ids)
    model_names = config.FULL_MODELS
    plot(args, model_names, J, S, subject_ids)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/tumor",
        help="outputfile, without extension",
    )

    hparams = parser.parse_args()
    main(hparams)
