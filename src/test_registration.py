from argparse import ArgumentParser
import pytorch_lightning as pl
from src.registration_model import RegistrationModel
import src.util as util
import yaml


def build_arg_parser():
    parser = ArgumentParser()
    parser = add_program_level_args(parser)
    return parser


def add_program_level_args(parent_parser):
    parser = parent_parser.add_argument_group("Program Level Arguments")
    parser.add_argument(
        "--load_from_checkpoint", help="optional model checkpoint to initialize with"
    )
    return parent_parser


def load_datamodule_from_params(hparams):
    datamodule = util.load_datamodule_from_name(
        hparams.dataset, batch_size=hparams.batch_size, pairs=True)
    # pass data properties to model
    hparams.data_dims = datamodule.dims
    hparams.data_classes = datamodule.class_cnt
    return datamodule, hparams


def load_model(logdir):
    checkpoint = util.get_checkoint_path_from_logdir(logdir)
    model = RegistrationModel.load_from_checkpoint(
        checkpoint_path=checkpoint)
    model.eval()
    hparams = model.hparams

    return model, hparams


def config_trainer_from_hparams(hparams):
    # trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams
    )
    return trainer


def main(args):
    # set-up
    pl.seed_everything(42)
    hparams = util.load_hparams_from_logdir(args.load_from_checkpoint)
    datamodule, hparams = load_datamodule_from_params(hparams)
    model, hparams = load_model(args.load_from_checkpoint)
    trainer = config_trainer_from_hparams(hparams)

    # test
    trainer.test(model, datamodule=datamodule)
    print(f"^ results from model {args.load_from_checkpoint} above")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
