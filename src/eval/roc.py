import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import torch
import src.util as util
import torchreg
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def load_model(weights):
    # load model
    model = util.load_model_from_logdir(weights)
    model.eval()
    return model


def load_datasets():
    # load data
    brains_dm = util.load_datamodule_from_name(
        "brain2d", batch_size=32, pairs=False)
    brats_dm = util.load_datamodule_from_name(
        "brats2d", batch_size=32, pairs=False)
    return brains_dm, brats_dm


def get_batch(dm1, dm2, device):
    dl1 = dm1.test_dataloader()
    batch1 = next(iter(dl1))
    I0 = batch1['I']['data'].to(device)
    S0 = batch1['S']['data'].to(device)

    dl2 = dm2.test_dataloader()
    batch2 = next(iter(dl2))
    I1 = batch2['I']['data'].to(device)
    S1 = batch2['S']['data'].to(device)

    return I0, S0, I1, S1


def predict(model, I0, I1, S0, S1):
    bound_1, info = model.bound(I0, I1, bidir=False)
    transform = info["transform"]
    S01 = model.transformer(S0, transform, mode="nearest")
    return bound_1, S01


def group_bounds_by_category(S1, S01, bound1):
    # get foreground mask from annotated brain dataset
    foreground_idx = (S01 > 0)
    # get tumor mask
    tumor_idx = (S1 == 1) | (S1 == 3)  # Tumor core (necrotic + enhanching)
    # get non-tumor mask
    non_tumor_idx = torch.logical_not(tumor_idx) & foreground_idx

    tumor_bounds = bound1[tumor_idx].flatten().cpu().tolist()
    non_tumor_bounds = bound1[non_tumor_idx].flatten().cpu().tolist()

    # build list of classes and bounds
    class_labels = [0] * len(non_tumor_bounds) + [1] * \
        len(tumor_bounds)  # 0 = Non-tumor, 1=Tumor
    bounds = non_tumor_bounds + tumor_bounds

    return class_labels, bounds


def test_model(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # load data
    brains_dm, brats_dm = load_datasets()
    I0, S0, I1, S1 = get_batch(brains_dm, brats_dm, device=device)

    # predict
    bound1, S01 = predict(model, I0, I1, S0, S1)
    class_labels, bounds = group_bounds_by_category(S1, S01, bound1)

    return class_labels, bounds


def plot_setup(title):
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.title(title)


def plot_finish(fname):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # save
    fig = plt.gcf()
    fig.savefig(fname + '.png')
    fig.savefig(fname + '.pdf')


def plot_roc_curve(true_labels, predicted_probs, model_name):
    """plots the ROC curve and calculates the AUC

    Args:
        true_labels ([type]): list of true labels, encoded as [0, 1]
        predicted_probs ([type]): scores/probailities of class 1
    """

    # get true positive rate, false negative rate
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)

    # calculate area under the courve (AUC)
    auc = roc_auc_score(true_labels, predicted_probs)
    print(f'AUC of {model_name}: {auc}')

    plt.plot(fpr, tpr, label=model_name)


def main(args):
    torchreg.settings.set_ndims(2)
    plot_setup(title="ROC Cruves of annotated Tumors")
    models = [("MSE", "./lightning_logs/mse_analytical_prior_trainable_recon"),
              ("Semantic Loss", "./lightning_logs/semantic_loss_analytical_prior_trainable_recon")]
    for model_name, weights in models:
        model = load_model(weights)
        class_labels, bounds = test_model(model)
        plot_roc_curve(class_labels, bounds, model_name)
    plot_finish(args.file)


if __name__ == '__main__':
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="./plots/roc",
        help="outputfile, without extension",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
