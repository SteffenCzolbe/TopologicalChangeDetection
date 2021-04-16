"""
segments a TIFF image stack
"""
import argparse
import os
import pytorch_lightning as pl
from .registration_model import RegistrationModel
from


def main(hparams):
    # load model
    model = RegistrationModel.load_from_checkpoint(
        checkpoint_path=hparams.weights)
    model.eval()

    print(
        f"Evaluating model for dataset {model.hparams.dataset}, loss {model.hparams.loss}, lambda {model.hparams.lam}"
    )

    # create grid animation
    dataloader = model.test_dataloader()
    batch = next(iter(dataloader))

    I0 = batch['I0']
    I1 = batch['I1']
    transform, transform_inv = model.forward(
        I0['data'], I1['data'])
    Im = model.transformer(I0['data'], transform)

    # save result
    os.makedirs(hparams.out, exist_ok=True)
    for i in range(len(Im)):
        sample_dir = os.path.join(hparams.out, f"{i:02}")
        os.makedirs(sample_dir, exist_ok=True)
        output = tio.ScalarImage(tensor=Im[i].detach().cpu(),
                                 affine=I0["affine"][i],
                                 check_nans=True)
        if I0.is_2d():
            output['data'] = output['data'] * 255
            output.as_pil().save(os.path.join(sample_dir, "morphed.png"))
            I0.as_pil().save(os.path.join(sample_dir, "moving.png"))
            I1.as_pil().save(os.path.join(sample_dir, "fixed.png"))
        else:
            output.save(os.path.join(sample_dir, "morphed.nii.gz"))
            I0.save(os.path.join(sample_dir, "moving.nii.gz"))
            I1.save(os.path.join(sample_dir, "fixed.nii.gz"))


if __name__ == "__main__":
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
        "--out",
        type=str,
        default="./out/",
        help="path to save the result in",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test-set metrics")

    hparams = parser.parse_args()
    main(hparams)
