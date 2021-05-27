import argparse
import torch
import torchreg
import pytorch_lightning as pl
import src.util as util
import torchreg.viz as viz
import src.eval.config as config
from src.datamodules.brainmri_datamodule import BrainMRIDataModule
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def crop(img, x_low=25, x_high=224, y_low=0, y_high=160):
    return img[:, x_low:x_high, y_low:y_high]


if __name__ == "__main__":
    # load samples
    brain0_subject = BrainMRIDataModule().test_dataloader().dataset.subjects[2]
    brain1_subject = BrainMRIDataModule().test_dataloader().dataset.subjects[0]
    brain0 = util.load_subject_from_dataset(
        "brain2d", "test", brain0_subject)[0]["data"].unsqueeze(0)
    brain1 = util.load_subject_from_dataset(
        "brain2d", "test", brain1_subject)[0]["data"].unsqueeze(0)

    # load model
    weights = config.MODELS["semantic_loss"]["path"]
    model_cls = config.MODELS["semantic_loss"]["model_cls"]
    model = util.load_model_from_logdir(weights, model_cls=model_cls)
    model.eval()

    # transform
    _, info = model.bound(brain0, brain1, bidir=False)
    #morphed = info["morphed"]
    flow = info["transform"]
    integrate = torchreg.nn.FlowIntegration(nsteps=7)
    transform = integrate(flow)
    inv_transform = integrate(-flow)
    morphed = model.transformer(brain0, transform)

    crop_area = (25, 224, 0, 160)
    highlight_area = (140, 195, 37, 81)

    fig = viz.Fig(1, 3, title=None, figsize=(4, 1.52))
    plt.tight_layout(pad=0)
    fig.plot_img(0, 0, crop(brain0[0]), vmin=0, vmax=1)
    fig.plot_img(0, 1, crop(brain1[0]), vmin=0, vmax=1)
    fig.plot_img(0, 2, crop(morphed[0], *highlight_area), vmin=0, vmax=1)
    fig.plot_transform_grid(0, 2, crop(
        inv_transform[0], *highlight_area), interval=1, linewidth=0.1, color="#0031FFFF", overlay=True)

    # add highlight-frame
    w = highlight_area[1] - highlight_area[0] - 1
    h = highlight_area[3] - highlight_area[2] - 1
    p = (highlight_area[2] - crop_area[2],
         highlight_area[0] - crop_area[0])

    # frame in left col
    rect = patches.Rectangle(p, h, w, linewidth=1,
                             edgecolor="#fa7f0f", facecolor='none')
    fig.axs[0, 0].add_patch(rect)

    # frame in conter col
    rect = patches.Rectangle(p, h, w, linewidth=1,
                             edgecolor="#1f77b4", facecolor='none')
    fig.axs[0, 1].add_patch(rect)

    # frame in right col
    rect = patches.Rectangle((0, 0), h, w, linewidth=1,
                             edgecolor="#FF0000", facecolor='none', zorder=1000)
    fig.axs[0, 2].add_patch(rect)

    # add arrow
    arrow = patches.FancyArrow(
        x=34, y=30, dx=-8, dy=3, width=2.5, head_length=7, linewidth=1, zorder=1000)
    fig.axs[0, 2].add_patch(arrow)

    fig.save("plots/intro.pdf", close=False)
    fig.save("plots/intro.png")
