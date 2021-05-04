import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import torch
import src.util as util


def load_module_and_dataset(args):
    # load data
    brains_dm = util.load_datamodule_from_name(
        "brain2d", batch_size=32, pairs=False)
    brats_dm = util.load_datamodule_from_name(
        "brats2d", batch_size=32, pairs=False)
    # load model
    model = util.load_model_from_logdir(args.weights)
    model.eval()
    return model, brains_dm, brats_dm


def get_batch(brains_dm, brats_dm, device):
    dl1 = brains_dm.test_dataloader()
    batch1 = next(iter(dl1))
    I0 = batch1['I']['data'].to(device)
    S0 = batch1['S']['data'].to(device)

    dl2 = brats_dm.test_dataloader()
    batch2 = next(iter(dl2))
    I1 = batch2['I']['data'].to(device)
    S1 = batch2['S']['data'].to(device)

    return I0, S0, I1, S1


def predict(model, I0, I1):
    bound_0, bound_1, _, _ = model.bound(I0, I1, bidir=True)
    return bound_0, bound_1


def get_bounds_for_seg_class_list(bound, S, seg_class_list):
    filtered_bounds = []
    for s in seg_class_list:
        filtered_bounds += bound[S == s].flatten().cpu().tolist()
    return filtered_bounds


def get_brain_classes(bound_brains, S, brain_seg_class_names):
    b = get_bounds_for_seg_class_list(bound_brains, S, [
                                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    categories = ["Any"]*len(b)
    bounds = b
    b = get_bounds_for_seg_class_list(bound_brains, S, [1])
    categories += [brain_seg_class_names[1]]*len(b)
    bounds += b
    b = get_bounds_for_seg_class_list(bound_brains, S, [2])
    categories += [brain_seg_class_names[2]]*len(b)
    bounds += b
    b = get_bounds_for_seg_class_list(bound_brains, S, [3, 4, 11, 12])
    categories += ["Ventricle"]*len(b)
    bounds += b
    return categories, bounds


def get_brats_classes(bound_brats, S, brats_seg_class_names):
    b = get_bounds_for_seg_class_list(bound_brats, S, [1])
    categories = ["Tumor " + brats_seg_class_names[1]]*len(b)
    bounds = b
    b = get_bounds_for_seg_class_list(bound_brats, S, [2])
    categories += ["Tumor " + brats_seg_class_names[2]]*len(b)
    bounds += b
    b = get_bounds_for_seg_class_list(bound_brats, S, [3])
    categories += ["Tumor " + brats_seg_class_names[3]]*len(b)
    bounds += b
    return categories, bounds


def prepare_data(args):
    # load model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, brains_dm, brats_dm = load_module_and_dataset(args)
    model.to(device)
    I0, S0, I1, S1 = get_batch(brains_dm, brats_dm, device)

    # get bounds
    bound_brains, bound_brats = predict(model, I0, I1)

    # group bounds to classes
    categories0, bounds0 = get_brain_classes(
        bound_brains, S0, brains_dm.class_names)
    categories1, bounds1 = get_brats_classes(
        bound_brats, S1, brats_dm.class_names)

    data = {'Category': categories0 + categories1,
            '$-log p(J|I)$': bounds0 + bounds1}
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
        default="./plots/violin",
        help="outputfile, without extension",
    )
    args = parser.parse_args()
    main(args)
