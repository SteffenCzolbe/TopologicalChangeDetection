import matplotlib.pyplot as plt
import argparse
import torch
import src.util as util
import os
import random
import imageio
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import src.eval.config as config


def load_bg_mask():
    # load a mask of potential background areas
    mask = imageio.imread("./src/eval/mask.png")
    mask = torch.as_tensor(mask).view(1,224,160,1)
    potential_bg = mask == 255 
    return potential_bg

def group_bounds_by_tumor_or_notumor(I1, S1, bound1, potential_bg):
    # get foreground mask from annotated brain dataset
    background_idx = potential_bg & (I1 <= 0.01)
    # get tumor mask
    tumor_idx = (S1 == 1) | (S1 == 3)  # Tumor core (necrotic + enhanching)
    # get non-tumor mask
    non_tumor_idx = torch.logical_not(tumor_idx) & torch.logical_not(background_idx)

    tumor_bounds = bound1[tumor_idx].flatten().cpu().tolist()
    non_tumor_bounds = bound1[non_tumor_idx].flatten().cpu().tolist()
    
    # sample 500 pixels for plotting (proportionally)
    K = len(tumor_bounds) + len(non_tumor_bounds)
    tumor_bounds = random.sample(tumor_bounds, k=int(len(tumor_bounds) / K * 500))
    non_tumor_bounds = random.sample(non_tumor_bounds, k=int(len(non_tumor_bounds) / K * 500))

    return tumor_bounds, non_tumor_bounds

def get_bounds_for_model(model_name):
    potential_bg = load_bg_mask()
    subject_ids = os.listdir(os.path.join(config.MODELS[model_name]["path"], "p_tumor"))
    
    tumor_bounds, non_tumor_bounds = [], []
    
    for subject_id in subject_ids:
        p_tumor = torch.load(os.path.join(config.MODELS[model_name]["path"], "p_tumor", subject_id))
        I, S = util.load_subject_from_dataset("brats2d", "test", subject_id)
        I = I["data"]
        S = S["data"]
        tb, ntb = group_bounds_by_tumor_or_notumor(I, S, p_tumor, potential_bg)
        tumor_bounds += tb
        non_tumor_bounds += ntb
        
    return tumor_bounds, non_tumor_bounds
    



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


def plot_model(model_name):
    tumor_bounds, non_tumor_bounds = get_bounds_for_model(model_name)

    # get true positive rate, false negative rate
    class_labels = [0] * len(non_tumor_bounds) + [1] * \
        len(tumor_bounds)  # 0 = Non-tumor, 1=Tumor
    bounds = non_tumor_bounds + tumor_bounds
    fpr, tpr, thresholds = roc_curve(class_labels, bounds)

    # calculate area under the courve (AUC)
    auc = roc_auc_score(class_labels, bounds)
    print(f'AUC of {model_name}: {auc}')

    label = config.MODELS[model_name]["display_name"] + f", {auc:.2f} AUC"
    plt.plot(fpr, tpr, label=label)


def main(args):
    # torchreg.settings.set_ndims(2)
    model_names = config.FULL_MODELS
    plot_setup(title=None)
    for model_name in model_names:
        plot_model(model_name)
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
