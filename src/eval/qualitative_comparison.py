import argparse
import os
import torch
import pytorch_lightning as pl
import torchreg.viz as viz
from src.registration_model import RegistrationModel
import src.util as util
import src.eval.config as config

def get_images(device):
    # load data
    dm1 = util.load_datamodule_from_name("brain2d", batch_size=32, pairs=False)
    dm2 = util.load_datamodule_from_name("brats2d", batch_size=32, pairs=False)
    
    # extract image
    dl1 = dm1.test_dataloader()
    batch1 = next(iter(dl1))
    I0 = batch1['I']['data'][[3]].to(device)

    dl2 = dm2.test_dataloader()
    batch2 = next(iter(dl2))
    I1 = batch2['I']['data'][[3]].to(device)

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


def plot(args, model_names, models, I0, I1):
    rows = len(model_names)
    # set-up fig
    fig = viz.Fig(rows, 3, title=None, figsize=(5, 1.5+rows*2))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)


    for row, (model_name, model) in enumerate(zip(model_names, models)):
        
        bound_1 = predict(model, I0, I1)
        vmin, vmax = config.MODELS[model_name]["probability_range"]
        
        fig.plot_img(row, 0, I0[0], vmin=0, vmax=1,
                     title="$I$" if row == 0 else None)
        fig.plot_img(row, 1, I1[0], vmin=0, vmax=1,
                     title="$J$" if row == 0 else None)
        fig.plot_img(row, 2, I1[0], vmin=0, vmax=1,
                     title="$p(J | I)$" if row == 0 else None)
        fig.plot_overlay(
            row, 2, bound_1[0], vmin=vmin, vmax=vmax, cbar=False, alpha=0.45)
        fig.axs[row, 0].text(x=-30, y=112, s=config.MODELS[model_name]["display_name"], fontsize=12, rotation=90, verticalalignment='center', horizontalalignment='center')
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
