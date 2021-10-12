from .roc_brainmri import get_bounds_for_model_brain_dataset
from .roc_plateletem import get_bounds_for_model_plateletem_dataset
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
import src.util as util
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import src.eval.config as config


def get_bounds_for_model(model_name, dataset, include_edema, bootstrap=False):
    """Get model-specific bounds grouped by ground-truth values for computation of the ROC curves.

    Args:
        model_name (str): name of model
        dataset (str): name of dataset
        include_edema (bool): srt to include edema into brain evaluation
        bootstrap (bool, optional): Set to true to induce bootstrap-randomness. Set to false for deterministic behaviour. Defaults to False.

    Returns:
        Tuple[List]: bounds_topological_change, bounds_no_topological_change
            Model-predicted bounds grouped by ground truth
    """
    if dataset == "brain2d":
        bounds_topological_change, bounds_no_topological_change = get_bounds_for_model_brain_dataset(
            model_name, dataset, include_edema, bootstrap=bootstrap)
    elif dataset == "platelet-em":
        bounds_topological_change, bounds_no_topological_change = get_bounds_for_model_plateletem_dataset(
            model_name, bootstrap=bootstrap)

    return bounds_topological_change, bounds_no_topological_change


def get_roc_curve(model_name, dataset, include_edema, bootstrap_sample_cnt):
    random.seed(42)
    fprs, tprs, aucs = [], [], []
    for i in range(bootstrap_sample_cnt):
        tumor_bounds, non_tumor_bounds = get_bounds_for_model(
            model_name, dataset, include_edema, bootstrap=(i > 0))

        # get true positive rate, false negative rate
        class_labels = [0] * len(non_tumor_bounds) + [1] * \
            len(tumor_bounds)  # 0 = Non-tumor, 1=Tumor
        bounds = non_tumor_bounds + tumor_bounds
        if i == 0:
            fpr, tpr, thresholds = roc_curve(class_labels, bounds)
        # calculate area under the courve (AUC)
        auc = roc_auc_score(class_labels, bounds)
        aucs.append(auc)

    auc = np.mean(aucs)
    auc_std = np.std(aucs)
    print(f'AUC of {model_name}: {auc:.3f} +- {auc_std:.3f}')
    return fpr, tpr, auc


def plot_setup():
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.set_aspect('equal', 'box')
    plt.plot([0, 1], [0, 1], color='lightgrey',
             linestyle='--')
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.25)


def plot_finish(args):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=5)
    plt.title("Brains" if "brain" in args.dataset else "Cells")
    plt.axis([0, 1, 0, 1])
    if args.legend:
        plt.legend(loc='lower right')
    # save
    fig = plt.gcf()
    fig.savefig(args.file + '.png')
    fig.savefig(args.file + '.pdf')


def plot_model(model_name, dataset, include_edema, bootstrap_sample_cnt):
    if not util.checkpoint_exists(config.MODELS[model_name]["path"][dataset]):
        return
    fpr, tpr, auc = get_roc_curve(
        model_name, dataset, include_edema, bootstrap_sample_cnt)

    label = config.MODELS[model_name]["display_name"] + f", {auc:.2f} AUC"
    plt.plot(fpr, tpr, label=label, color=config.MODELS[model_name]["color"])


def main(args):
    # torchreg.settings.set_ndims(2)
    model_names = config.ALL_MODELS
    plot_setup()
    for model_name in model_names:
        plot_model(model_name, args.dataset, args.include_edema,
                   args.bootstrap_sample_cnt)
    plot_finish(args)


if __name__ == '__main__':
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/roc",
        help="outputfile, without extension",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
    )
    parser.add_argument(
        "--bootstrap_sample_cnt",
        type=int,
        default=1,
        help="Bootstrapping iterations. Default = 1 (no bootstrapping)",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="set to plot with legend",
    )
    parser.add_argument(
        "--include_edema",
        action="store_true",
        help="Set to include edema of the brain datasets with the tumor",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
