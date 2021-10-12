import torch
import src.util as util
import os
import random
import imageio
import src.eval.config as config


def load_2d_brain_background_mask():
    # load a mask of potential background areas
    mask = imageio.imread("./src/eval/brain_background_mask.png")
    mask = torch.as_tensor(mask).view(1, 224, 160, 1)
    potential_bg = mask == 255
    return potential_bg


def group_bounds_by_tumor_or_notumor(I1, S1, bound1, potential_bg, include_edema):
    # get foreground mask from annotated brain dataset
    background_idx = potential_bg & (I1 <= 0.01)
    # get tumor mask
    if include_edema:
        tumor_idx = (S1 == 1) | (S1 == 2)
    else:
        tumor_idx = (S1 == 1)
    # get non-tumor mask
    non_tumor_idx = torch.logical_not(
        tumor_idx) & torch.logical_not(background_idx)

    tumor_bounds = bound1[tumor_idx].flatten().cpu().tolist()
    non_tumor_bounds = bound1[non_tumor_idx].flatten().cpu().tolist()

    # sample 1000 pixels for plotting (proportionally)
    K = len(tumor_bounds) + len(non_tumor_bounds)
    tumor_bounds = random.sample(
        tumor_bounds, k=int(len(tumor_bounds) / K * 1000))
    non_tumor_bounds = random.sample(
        non_tumor_bounds, k=int(len(non_tumor_bounds) / K * 1000))

    return tumor_bounds, non_tumor_bounds


def get_bounds_for_model_brain_dataset(model_name, dataset, include_edema, bootstrap=False):
    """Returns topological-change probability values, grouped by ground truth into classes of topological_change, no_topological_change

    Args:
        model_name ([type]): [description]
        dataset ([type]): [description]
        include_edema ([type]): [description]
        bootstrap (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    potential_bg = load_2d_brain_background_mask()
    precomputed_p_tumor_dir = os.path.join(
        config.MODELS[model_name]["path"][dataset], "p_tumor")
    if ("brain" in dataset) and include_edema:
        # check if precomputed values exist specifically for edema
        precomputed_p_tumor_or_edema_dir = os.path.join(
            config.MODELS[model_name]["path"][dataset], "p_tumor_or_edema")
        if os.path.isdir(precomputed_p_tumor_or_edema_dir):
            precomputed_p_tumor_dir = precomputed_p_tumor_or_edema_dir
    subject_ids = os.listdir(precomputed_p_tumor_dir)

    if bootstrap:
        # sample with replacement
        subject_ids = random.choices(subject_ids, k=len(subject_ids))

    tumor_bounds, non_tumor_bounds = [], []

    for subject_id in subject_ids:
        p_tumor = torch.load(os.path.join(precomputed_p_tumor_dir, subject_id))
        I, S = util.load_subject_from_dataset("brats2d", "test", subject_id)
        I = I["data"]
        S = S["data"]
        tb, ntb = group_bounds_by_tumor_or_notumor(
            I, S, p_tumor, potential_bg, include_edema)
        tumor_bounds += tb
        non_tumor_bounds += ntb

    return tumor_bounds, non_tumor_bounds
