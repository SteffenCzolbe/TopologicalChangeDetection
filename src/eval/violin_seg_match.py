import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import torch
import src.util as util


def load_module_and_dataset(args):
    # load data
    dm = util.load_datamodule_from_name(
        "brain2d", batch_size=32, pairs=True)
    # load model
    model = util.load_model_from_logdir(args.weights)
    model.eval()
    return model, dm


def get_batch(dm, device):
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    I0 = batch['I0']['data'].to(device)
    S0 = batch['S0']['data'].to(device)
    I1 = batch['I1']['data'].to(device)
    S1 = batch['S1']['data'].to(device)
    return I0, S0, I1, S1


def predict(model, I0, I1, S0, S1):
    _, bound_1, info_0_to_1, _ = model.bound(I0, I1, bidir=True)
    transform = info_0_to_1["transform"]
    S01 = model.transformer(S0, transform, mode="nearest")
    return bound_1, S01


def group_by_category(S01, S1, bound1):
    # get foreground mask
    foreground = (S01 > 0) & (S1 > 0)
    # get seg mismatch mask
    seg_agreement = S01 == S1
    # mask by background
    seg_agreement = seg_agreement & foreground
    seg_disagreement = torch.logical_not(seg_agreement) & foreground

    seg_agreement_bounds = bound1[seg_agreement].flatten().cpu().tolist()
    seg_disagreement_bounds = bound1[seg_disagreement].flatten().cpu().tolist()

    bounds = seg_agreement_bounds + seg_disagreement_bounds
    categories = ["Segmentation Match"] * len(seg_agreement_bounds) + [
        "Segmentation Mismatch"] * len(seg_disagreement_bounds)
    return categories, bounds


def prepare_data(args):
    # load model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, dm = load_module_and_dataset(args)
    model.to(device)
    I0, S0, I1, S1 = get_batch(dm, device)

    # get bounds
    bound1, S01 = predict(model, I0, I1, S0, S1)

    # group bounds to classes
    categories, bounds = group_by_category(S01, S1, bound1)

    data = {'Category': categories,
            '$-log p(J|I)$': bounds}
    return pd.DataFrame(data)
    # return sns.load_dataset("tips")


def plot(data, fname):
    # set-up figure
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # plot
    ax = sns.violinplot(data=data, x="Category",
                        y="$-log p(J|I)$", inner="quartile")
    plt.xticks(rotation=20, ha='right')
    # save
    fig = ax.get_figure()
    fig.savefig(fname + '.png')
    fig.savefig(fname + '.pdf')


def main(args):
    with torch.no_grad():
        data = prepare_data(args)
        plot(data, args.file)


if __name__ == '__main__':
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
        "--file",
        type=str,
        default="./plots/violin_seg_match",
        help="outputfile, without extension",
    )
    args = parser.parse_args()
    main(args)
