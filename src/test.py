import torch
import pytorch_lightning as pl
import torchreg
from argparse import ArgumentParser
from tqdm import tqdm
import src.util as util
from src.registration_model import RegistrationModel
import glob
import os
import torchreg.viz as viz
import numpy as np


def load_model(path):
    # find oldest checkpoint
    checkpoint_files = glob.glob(os.path.join(path, 'checkpoints', '*'))
    checkpoint_file = sorted(
        checkpoint_files, key=os.path.getmtime, reverse=True)[0]
    print(f'Loading RegistrationModel from {checkpoint_file}')
    reg_model = RegistrationModel.load_from_checkpoint(
        checkpoint_path=checkpoint_file).eval()
    datamodule = util.load_datamodule_for_model(reg_model)

    return reg_model, datamodule


class RepresentationModel(torch.nn.Module):
    """
    A dummy-model to extract an abstract feature representation
    """

    def forward(self, I):
        # expand to probability distribution over 1 classes: white and black
        return torch.cat([I, 1-I], dim=1)


class SegmentationModel(torch.nn.Module):
    """
    A dummy-model to extract a segmentation mask
    """

    def forward(self, I):
        # one-hot encoded label map
        return torch.cat([I, 1-I], dim=1)  # .round()


def draw_model_samples(reg_model, I0, I1, sample_cnt=32):
    # take samples (1st sample is maximum likelyhood estimate, following are samples)

    reg_model.disable_mcdropout()
    transform_map, transform_map_inv = reg_model.forward(I0, I1)

    reg_model.enable_mcdropout(p=0.5)
    transform_samples, transform_samples_inv = [], []
    for _ in range(sample_cnt):
        t, t_inv = reg_model.forward(I0, I1)
        transform_samples.append(t)
        transform_samples_inv.append(t_inv)
    return transform_map, transform_map_inv, transform_samples, transform_samples_inv


def calculate_covar(sample_list):
    # N inputs of Bx3xHxWxD
    N = len(sample_list)
    # stack inputs, now NxBx3xHxWxD
    samples = torch.stack(sample_list)
    # reshape to BxHxWxDx3xN
    samples = samples.permute(1, 3, 4, 5, 2, 0)
    # calculate M = X - mu_x
    m = samples - samples.mean(dim=-1, keepdim=True)
    # calculate M^T
    mT = m.permute(0, 1, 2, 3, 5, 4)
    # calculate covar = 1/(n-1) MM^T
    covar = torch.matmul(m, mT) / (N-1)
    # re-add channel dimension
    return covar.unsqueeze(1)


def transformation_uncertainty(transforms):
    # we are assuming a gaussian distribution here.
    # A more accurate method would be using a kernel density estimator. then H = 1/N sum_i p(s_i),
    # with p(s_i) estimated by the kernel density estimator (e.g. sum of normal distributions, or k-NN estimator)

    covar = calculate_covar(transforms)
    # if we are in 2d, the 3rd row of covar will be 0, leading to determinant = 0
    if torchreg.settings.get_ndims() == 2:
        covar = covar[..., :-1, :-1]
    return 0.5 * torch.log(torch.det(2 * np.pi * np.e * covar))


def label_uncertainty(transforms, label_map):
    """
    Calculates the label uncertainty U_l. The flows point from a moving image to the static atlas. The segmentation classes are of the atlas
    """
    # morph and mean segmentation
    # memory-saving implementation
    transformer = torchreg.nn.SpatialTransformer()
    n = len(transforms)
    mean_label = transformer(label_map, transforms[0], mode='bilinear')
    for i in range(1, n):
        # transform segmentation mask
        morphed_segmentation = transformer(
            label_map, transforms[i], mode='bilinear')
        mean_label += morphed_segmentation
    mean_label /= n

    # entropy
    return util.entropy(mean_label)


def anatomical_uncertainty(repr0, repr1, transform, transform_inv):
    transformer = torchreg.nn.SpatialTransformer()
    repr0_aligned = transformer(repr0, transform, mode='bilinear')
    u_a = util.kl_divergence(repr0_aligned, repr1, eps=5e-3)
    u_a = transformer(u_a, transform_inv, mode='bilinear')
    return u_a


def anatomical_uncertainty_prob(repr0, repr1, transforms, transform_invs):
    mean_u_a = anatomical_uncertainty(
        repr0, repr1, transforms[0], transform_invs[0])
    n = len(transforms)
    for i in range(1, n):
        mean_u_a += anatomical_uncertainty(repr0,
                                           repr1,
                                           transforms[i],
                                           transform_invs[i])
    return mean_u_a / n


def calculate_metrics(reg_model, I0, I1):

    transform_map, transform_map_inv, transform_samples, transform_samples_inv = draw_model_samples(
        reg_model, I0, I1)

    # morphed image
    transformer = torchreg.nn.SpatialTransformer()
    Im = transformer(I0, transform_map)

    # transformation uncertainty
    U_t = transformation_uncertainty(transform_samples_inv)

    # label uncertainty
    seg_model = SegmentationModel()
    label_map = seg_model(I1)
    U_l = label_uncertainty(transform_samples_inv, label_map)

    # anatomical uncertainty (1 sample)
    repr_model = RepresentationModel()
    repr0 = repr_model(I0)
    repr1 = repr_model(I1)
    U_a = anatomical_uncertainty(
        repr0, repr1, transform_map, transform_map_inv)

    # anatomical uncertainty (probabilistic)
    U_a_prob = anatomical_uncertainty_prob(
        repr0, repr1, transform_samples, transform_samples_inv)

    return Im, transform_map, transform_map_inv, U_t, U_l, U_a, U_a_prob


def tune_dropout_p(model, datamodule):
    """
    tune dropout probability, by testing various values
    """
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
    # test different dropout levels
    for p in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # configure MC-dropout
        model.enable_mcdropout(p)
        print(f'testing for p={p}')
        trainer.test(model, datamodule=datamodule)


def cli_main(args):
    pl.seed_everything(1234)

    # load model and data
    reg_model, datamodule = load_model(path=args.model_path)
    reg_model = reg_model.to('cuda')
    dataloader = datamodule.calligraphy_split_dataloader()

    if args.tune_dropout:
        tune_dropout_p(reg_model, datamodule)
        exit()

    # process image
    batch = next(iter(dataloader))
    I0 = batch['I0']['data'].to('cuda')
    I1 = batch['I1']['data'].to('cuda')

    Im, transform, transform_inv, U_t, U_l, U_a, U_a_prob = calculate_metrics(
        reg_model, I0, I1)

    # plot
    columns = ['Moving', 'Fixed', 'Morphed', 'Transform',
               '$U_T$', '$U_L$', '$U_A$', '$U_A$ prob.']
    cols = len(columns)
    rows = min(8, len(I0))
    fig = viz.Fig(rows, cols, None, figsize=(
        cols, rows), transparent_background=False, column_titles=columns)
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for r in range(rows):
        # moving
        fig.plot_img(r, 0, I0[r], vmin=0, vmax=1)
        # fixed
        fig.plot_img(r, 1, I1[r], vmin=0, vmax=1)
        # morphed
        fig.plot_img(r, 2, Im[r], vmin=0, vmax=1)
        # morphed + grid
        fig.plot_img(r, 3, Im[r], vmin=0, vmax=1)
        fig.plot_transform_grid(
            r,
            3,
            transform_inv[r],
            interval=2,
            linewidth=0.4,
            color="#00FF00FF",
            overlay=True,
        )

        # transformation uncetainty
        fig.plot_img(r, 4, I0[r], vmin=0, vmax=1)
        fig.plot_overlay(r, 4, U_t[r], vmin=0)

        # label uncertainty
        fig.plot_img(r, 5, I0[r], vmin=0, vmax=1)
        fig.plot_overlay(r, 5, U_l[r], vmin=0)

        # Anatomical uncertainty
        fig.plot_img(r, 6, I0[r], vmin=0, vmax=1)
        fig.plot_overlay(r, 6, U_a[r], vmin=0)

        # Anatomical uncertainty
        fig.plot_img(r, 7, I0[r], vmin=0, vmax=1)
        fig.plot_overlay(r, 7, U_a_prob[r], vmin=0)

    os.makedirs("./plots", exist_ok=True)
    fig.save(args.file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, default='./trained_models/mnist/', help=f'Path to the trained model.')
    parser.add_argument(
        '--file', type=str, default='./plots/samples.png', help=f'File to save the results in.')
    parser.add_argument(
        '--tune_dropout', action='store_true', help=f'set to tune dropout p')
    args = parser.parse_args()
    cli_main(args)
