from argparse import ArgumentParser
import pytorch_lightning as pl
from src.baselines.segmentation_model import SegmentationModel
import src.util as util


def build_arg_parser():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SegmentationModel.add_model_specific_args(parser)
    parser = add_program_level_args(parser)
    return parser


def add_program_level_args(parent_parser):
    parser = parent_parser.add_argument_group("Program Level Arguments")
    parser.add_argument(
        "--load_from_checkpoint", help="optional model checkpoint to initialize with"
    )
    parser.add_argument(
        "--notest", action="store_true", help="Set to not run test after training."
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", help=f"Dataset. Options: {util.get_supported_datamodules().keys()}"
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
    return parent_parser


def load_datamodule_from_params(hparams):
    datamodule = util.load_datamodule_from_name(
        hparams.dataset, batch_size=hparams.batch_size, pairs=False, load_train_seg=True)
    # pass data properties to model
    hparams.data_dims = datamodule.dims
    hparams.data_classes = datamodule.class_cnt
    return datamodule, hparams


def load_model_from_hparams(hparams):
    if hparams.load_from_checkpoint:
        model = SegmentationModel.load_from_checkpoint(
            checkpoint_path=hparams.load_from_checkpoint)
        hparams.resume_from_checkpoint = hparams.load_from_checkpoint
    else:
        model = SegmentationModel(hparams)

    return model, hparams


def config_trainer_from_hparams(hparams):
    # save model with best validation loss
    checkpointing_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/dice_overlap", mode="max"
    )
    # early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/dice_overlap", min_delta=0.00, patience=hparams.early_stop_patience, verbose=True, mode="max"
    )

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        checkpoint_callback=checkpointing_callback,
        callbacks=[early_stop_callback],
    )
    return trainer


def main(hparams):
    # set-up
    pl.seed_everything(42)
    datamodule, hparams = load_datamodule_from_params(hparams)
    model, hparams = load_model_from_hparams(hparams)
    trainer = config_trainer_from_hparams(hparams)

    # train
    trainer.fit(model, datamodule)

    # test
    if not hparams.notest:
        trainer.test()


if __name__ == "__main__":
    # add model args
    parser = build_arg_parser()

    hparams = parser.parse_args()
    main(hparams)
