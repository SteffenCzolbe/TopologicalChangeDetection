import argparse
import os
import shutil
import torch
import torchreg
import pytorch_lightning as pl
from src.registration_model import RegistrationModel
import src.util as util
from tqdm import tqdm
from src.datamodules.brainmri_datamodule import BrainMRIDataset
import torchreg.viz as viz
import src.eval.config as config


def p_tumor(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load brats
    brats_dm = util.load_datamodule_from_name(
        "brats2d", batch_size=1, load_val_seg=True, pairs=False)
    brats_dl = brats_dm.val_dataloader()

    # load model
    weights = config.MODELS[args.model_name]["path"]["brain2d"]
    model_cls = config.MODELS[args.model_name]["model_cls"]
    model = util.load_model_from_logdir(weights, model_cls=model_cls)
    model.eval()
    model.to(device)

    # set-up fig
    fig = viz.Fig(8, 4, None, figsize=(5, 8))

    for i, batch in enumerate(tqdm(brats_dl, desc="calculating p_tumor(J)")):
        J = batch["I"]["data"].to(device)

        # bound and transform
        _, y_pred_onehot, _ = model.forward(J)

        ptumor_or_edema = y_pred_onehot[0, [1]] + \
            y_pred_onehot[0, [2]]  # edema or tumor
        ptumor = y_pred_onehot[0, [1]]  # tumor

        # save result
        torch.save(ptumor.cpu(), os.path.join(
            args.p_tumor_dir, batch["subject_id"][0]))
        torch.save(ptumor_or_edema.cpu(), os.path.join(
            args.p_tumor_or_edema_dir, batch["subject_id"][0]))

        # viz
        if i < 8:
            fig.plot_img(i, 0, J[0], vmin=0, vmax=1, title="J")
            fig.plot_img(i, 2, ptumor, title="p_tumor")
            fig.plot_img(i, 3, ptumor_or_edema, title="p_tumor or edema")
            print(f"p_tumor min: {ptumor.min()}, max: {ptumor.max()}")
        if i == 8:
            fig.save(os.path.join(
                config.MODELS[args.model_name]["path"]["brain2d"], "p_tumor.png"))


def main(args):
    args.mean_pIK_dir = os.path.join(
        config.MODELS[args.model_name]["path"]["brain2d"], "mean_pIK")
    if os.path.isdir(args.mean_pIK_dir) and not args.non_cached:
        print('using cached mean_pIK')
    else:
        print('calculating mean_pIK')

        # remove dir
        if os.path.isdir(args.mean_pIK_dir):
            shutil.rmtree(args.mean_pIK_dir)

        # make new dir
        os.mkdir(args.mean_pIK_dir)

    args.p_tumor_dir = os.path.join(
        config.MODELS[args.model_name]["path"]["brain2d"], "p_tumor")
    args.p_tumor_or_edema_dir = os.path.join(
        config.MODELS[args.model_name]["path"]["brain2d"], "p_tumor_or_edema")
    if os.path.isdir(args.p_tumor_dir) and not args.non_cached:
        print('using cached p_tumor')
    else:
        print('calculating p_tumor')

        # remove dir
        if os.path.isdir(args.p_tumor_dir):
            shutil.rmtree(args.p_tumor_dir)
            shutil.rmtree(args.p_tumor_or_edema_dir)

        # make new dir
        os.mkdir(args.p_tumor_dir)
        os.mkdir(args.p_tumor_or_edema_dir)

        # compute means
        p_tumor(args)


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
        "--non_cached", action="store_true", help="Set to overwrite cached results")
    parser.add_argument(
        "--datasplit",
        type=str,
        default="test",
        help="Datasplit for evaluation",
    )

    hparams = parser.parse_args()
    pl.seed_everything(42)
    with torch.no_grad():
        main(hparams)
