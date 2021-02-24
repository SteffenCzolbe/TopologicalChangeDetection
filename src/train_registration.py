from argparse import ArgumentParser
import pytorch_lightning as pl
from .registration_model import RegistrationModel
import src.util as util


def main(hparams):
    pl.seed_everything(42)

    # load data
    dataset = util.load_damodule(
        hparams.dataset, batch_size=hparams.batch_size)
    hparams.data_dims = dataset.dims
    hparams.data_classes = dataset.classes

    # load model
    if hparams.load_from_checkpoint:
        model = RegistrationModel.load_from_checkpoint(
            hparams.load_from_checkpoint)
        hparams.resume_from_checkpoint = hparams.load_from_checkpoint
    else:
        model = RegistrationModel(hparams)

    # add some hints for better experiments tracking
    hparams.task = "registration"

    # save model with best validation loss
    checkpointing_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", mode="min"
    )
    # early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=hparams.early_stop_patience, verbose=True, mode="min"
    )

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        checkpoint_callback=checkpointing_callback,
        callbacks=[early_stop_callback],
    )

    # fit
    trainer.fit(model, dataset)

    # test
    if not hparams.notest:
        trainer.test()


if __name__ == "__main__":
    # add model args
    parser = RegistrationModel.model_args()
    # add program level args
    parser.add_argument(
        "--load_from_checkpoint", help="optional model checkpoint to initialize with"
    )
    parser.add_argument(
        "--notest", action="store_true", help="Set to not run test after training."
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset."
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=100,
        help="Early stopping oatience, in Epcohs. Default: 10",
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=32, help="batchsize")
    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()
    main(hparams)
