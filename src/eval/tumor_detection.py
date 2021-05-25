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
"""
      We caluclate the term

      p_tumor(J) = E_{I}[log p(J|I)] - E_{I, K}[ log p(I|K) transform_{I -> J}]
                 = 1/N sum_I (log p(J|I) - (1/N sum_K log p(I|K)) transform_{I -> J} )
                                           |--------------------|
                                              pre-computed in mean_pIK()
                   |--- computed in pJ() --------------------------------------------|

"""


def mean_pIK(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    brains_dm = util.load_datamodule_from_name(
        "brain2d", batch_size=args.samples, load_val_seg=False, pairs=False)
    test_data = iter(brains_dm.test_dataloader(shuffle=True))
    Is = next(test_data)
    Ks = next(test_data)
    subject_ids = Is['subject_id']
    Is = Is['I']['data'].to(device)
    Ks = Ks['I']['data'].to(device)

    # load model
    weights = config.MODELS[args.model_name]["path"]
    model_cls = config.MODELS[args.model_name]["model_cls"]
    model = util.load_model_from_logdir(weights, model_cls=model_cls)
    model.eval()
    model.to(device)

    # calculate mean bounds
    bounds = torch.zeros_like(Is)

    for i in tqdm(range(args.samples), desc="pre-calculating p(I|K)"):
        _, b, _, _ = model.bound(Ks, Is, bidir=True)
        bounds += b
        Ks = torch.cat([Ks[1:], Ks[:1]], dim=0)  # round robin

    bounds /= args.samples

    # save bounds and images
    for bound, subject_id in zip(bounds, subject_ids):
        torch.save(bound.cpu(), os.path.join(
            args.mean_pIK_dir, subject_id))


def p_tumor(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer = torchreg.nn.SpatialTransformer()

    # load sample means of I
    healthy_subject_ids = os.listdir(args.mean_pIK_dir)
    # load I
    mean_pIKs = [torch.load(os.path.join(args.mean_pIK_dir, subject_id))
                 for subject_id in healthy_subject_ids]
    mean_pIKs = torch.stack(mean_pIKs).to(device)

    Is = []
    for s in healthy_subject_ids:
        I, _ = util.load_subject_from_dataset("brain2d", "test", s)
        Is.append(I["data"])
    Is = torch.stack(Is).to(device)

    # load brats
    brats_dm = util.load_datamodule_from_name(
        "brats2d", batch_size=1, load_val_seg=True, pairs=False)
    brats_dl = brats_dm.test_dataloader()

    # load model
    weights = config.MODELS[args.model_name]["path"]
    model_cls = config.MODELS[args.model_name]["model_cls"]
    model = util.load_model_from_logdir(weights, model_cls=model_cls)
    model.eval()
    model.to(device)

    # set-up fig
    fig = viz.Fig(8, 4, None, figsize=(5, 8))

    for i, batch in enumerate(tqdm(brats_dl, desc="calculating p_tumor(J)")):
        J = batch["I"]["data"].to(device)

        # expand J to size I
        J = J.expand(Is.shape)

        # bound and transform
        _, pJI, I_to_J, _ = model.bound(Is, J, bidir=True)
        transform_I_to_J = I_to_J["transform"]

        # morph mean_pIK to J
        mean_pIKs_to_J = transformer(mean_pIKs, transform_I_to_J)

        # remove non-tumor differences
        ptumor = pJI - mean_pIKs_to_J

        # mean across samples
        ptumor = ptumor.mean(dim=0)

        # save result
        torch.save(ptumor.cpu(), os.path.join(
            args.p_tumor_dir, batch["subject_id"][0]))

        # viz
        if i < 8:
            fig.plot_img(i, 0, J[0], vmin=0, vmax=1, title="J")
            fig.plot_img(i, 1, pJI.mean(dim=0), title="p(J|I)")
            fig.plot_img(i, 2, mean_pIKs_to_J.mean(dim=0), title="p(I|K)")
            fig.plot_img(i, 3, ptumor, title="p_tumor")
            print(f"p(J | I) min: {pJI.min()}, max: {pJI.max()}")
            print(f"p_tumpr min: {ptumor.min()}, max: {ptumor.max()}")
        if i == 8:
            fig.save(os.path.join(
                config.MODELS[args.model_name]["path"], "p_tumor.png"))


def main(args):
    args.mean_pIK_dir = os.path.join(
        config.MODELS[args.model_name]["path"], "mean_pIK")
    if os.path.isdir(args.mean_pIK_dir) and not args.non_cached:
        print('using cached mean_pIK')
    else:
        print('calculating mean_pIK')

        # remove dir
        if os.path.isdir(args.mean_pIK_dir):
            shutil.rmtree(args.mean_pIK_dir)

        # make new dir
        os.mkdir(args.mean_pIK_dir)

        # compute means
        mean_pIK(args)

    args.p_tumor_dir = os.path.join(
        config.MODELS[args.model_name]["path"], "p_tumor")
    if os.path.isdir(args.p_tumor_dir) and not args.non_cached:
        print('using cached p_tumor')
    else:
        print('calculating p_tumor')

        # remove dir
        if os.path.isdir(args.p_tumor_dir):
            shutil.rmtree(args.p_tumor_dir)

        # make new dir
        os.mkdir(args.p_tumor_dir)

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
        "--samples",
        type=int,
        default=2,
        help="count of samples I, K to draw",
    )

    hparams = parser.parse_args()
    pl.seed_everything(42)
    with torch.no_grad():
        main(hparams)
