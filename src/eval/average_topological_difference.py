import argparse
import torch
import torchreg
import pytorch_lightning as pl
import src.util as util
from tqdm import tqdm
import torchreg.viz as viz
import src.eval.config as config
import matplotlib.pyplot as plt
import numpy as np


def crop(img, x_low=25, x_high=224, y_low=0, y_high=160):
    return img[:, x_low:x_high, y_low:y_high]


def get_class_masks(atlas_seg):
    def merge_mask_channels(mask):
        return mask.sum(dim=1, keepdim=True).clamp(0, 1)

    def add_mask(a, b):
        return (a+b).clamp(0, 1)

    def substract_mask(a, b):
        return (a-b).clamp(0, 1)
    atlas_seg = util.onehot(atlas_seg, num_classes=24).long()

    # ventricles = dilated ventricles
    ventricle_dilation = 1
    ventricle_mask = torchreg.nn.dilation(
        atlas_seg[:, [3, 4, 11]], ventricle_dilation)
    ventricle_mask = merge_mask_channels(ventricle_mask)

    # cortical surface = dilated GM
    gm_dilation = 4
    cortical_surface_mask = torchreg.nn.dilation(
        atlas_seg[:, [2]], gm_dilation)
    cortical_surface_mask = substract_mask(
        cortical_surface_mask, ventricle_mask)

    # subcortical structures: non-background, not part of the previous masks
    not_subcortex = merge_mask_channels(atlas_seg[:, [0, 2, 5, 6, 12, 13, 18]])
    not_subcortex = add_mask(not_subcortex, cortical_surface_mask)
    not_subcortex = add_mask(not_subcortex, ventricle_mask)
    subcortical_structures_mask = substract_mask(
        torch.ones_like(not_subcortex), not_subcortex)

    background_mask = substract_mask(
        torch.ones_like(not_subcortex), cortical_surface_mask)
    background_mask = substract_mask(background_mask, ventricle_mask)
    background_mask = substract_mask(
        background_mask, subcortical_structures_mask)

    return background_mask, cortical_surface_mask, ventricle_mask, subcortical_structures_mask


def mean_difference_in_atlas_domain(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load atlas
    atlas_dm = util.load_datamodule_from_name(
        "brain2d", batch_size=args.samples, load_val_seg=True, pairs=True, atlasreg=True)
    atlas_data = next(iter(atlas_dm.test_dataloader()))
    atlas = atlas_data['I1']['data'].to(device)
    atlas_seg = atlas_data['S1']['data'][[0]].to(device)

    # load data
    brains_dm = util.load_datamodule_from_name(
        "brain2d", batch_size=args.samples, load_val_seg=False, pairs=False)
    test_data = iter(brains_dm.test_dataloader(shuffle=True))
    Is = next(test_data)
    Js = next(test_data)
    Is = Is['I']['data'].to(device)
    Js = Js['I']['data'].to(device)

    # load model
    weights = config.MODELS[args.model_name]["path"]
    model_cls = config.MODELS[args.model_name]["model_cls"]
    model = util.load_model_from_logdir(weights, model_cls=model_cls)
    model.eval()
    model.to(device)

    # calculate mean bounds
    bounds = torch.zeros_like(Is)

    for i in tqdm(range(args.samples), desc="calculating p(J|I) composed Phi"):
        _, b, _, _ = model.bound(Is, Js, bidir=True)
        _, J_to_atlas = model.bound(Js, atlas, bidir=False)
        b = model.transformer(b, J_to_atlas["transform"])
        bounds += b
        Js = torch.cat([Js[1:], Js[:1]], dim=0)  # round robin

    bounds = torch.mean(bounds, dim=0, keepdim=True) / args.samples

    background_mask, cortical_surface_mask, ventricle_mask, subcortical_structures_mask = get_class_masks(
        atlas_seg)

    # visualize bound and images
    fig = viz.Fig(1, 2, title=None, figsize=(3.5, 2))
    plt.tight_layout(pad=0)
    vmin, vmax = config.MODELS[args.model_name]["probability_range"]
    # plot heatmap
    fig.plot_img(0, 0, crop(atlas[0]), vmin=0, vmax=1)
    fig.plot_overlay(0, 0, crop(bounds[0]), vmin=vmin,
                     vmax=vmax, cbar=False, alpha=0.45)

    # plot class-mask
    class_mask_onehot = torch.cat(
        [background_mask, cortical_surface_mask, ventricle_mask, subcortical_structures_mask], dim=1)
    class_mask = util.from_onehot(class_mask_onehot)
    cortical_surface_color = (31, 119, 180)
    ventricle_color = (250, 127, 15)
    subcortical_color = (44, 160, 44)
    class_colors = [None, cortical_surface_color,
                    ventricle_color, subcortical_color]
    fig.plot_img(0, 1, crop(atlas[0]), vmin=0, vmax=1)
    fig.plot_overlay_class_mask(0, 1, crop(
        class_mask[0]), class_colors, alpha=0.6)
    fig.plot_contour(0, 1, crop(class_mask[0]), contour_class=1,
                     rgba=cortical_surface_color, width=1)
    fig.plot_contour(0, 1, crop(class_mask[0]),
                     contour_class=2, rgba=ventricle_color, width=1)

    fig.save(args.file + ".pdf", close=False)
    fig.save(args.file + ".png")

    # boxplot
    data = [bounds[cortical_surface_mask == 1].cpu(),
            bounds[ventricle_mask == 1].cpu(), bounds[subcortical_structures_mask == 1].cpu()]
    labels = ["Cortical Surface", "Ventricles", "Subcortical Structures"]
    plt.rcParams["boxplot.medianprops.color"] = "k"
    plt.rcParams["boxplot.medianprops.linewidth"] = 3.0
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.95, bottom=0.25)
    bplot = ax.boxplot(data, labels=labels,
                       showfliers=False, patch_artist=True)
    # color boxes
    for patch, color in zip(bplot["boxes"], class_colors[1:]):
        patch.set_facecolor(np.array(color) / 255.)

    # rotate labels
    ax.set_xticklabels(labels, rotation=15,
                       ha="center")

    ax.set_ylabel(r"$\bar{L}_{sym}$")
    plt.savefig(args.file + "_boxplot.pdf")
    plt.savefig(args.file + "_boxplot.png")


def main(args):
    with torch.no_grad():
        mean_difference_in_atlas_domain(args)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--model_name",
        type=str,
        help="name of model to test",
    )
    parser.add_argument(
        "--file",
        default="./plots/mean_Lsym",
        type=str,
        help="Plot file name, without extension",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="count of samples I, K to draw",
    )

    hparams = parser.parse_args()
    pl.seed_everything(42)
    with torch.no_grad():
        main(hparams)
