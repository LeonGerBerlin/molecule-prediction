"""Console script for molproperty_prediction."""
import argparse
import sys

from molproperty_prediction.train import train
from molproperty_prediction.setup_wandb import sweep_wb
from molproperty_prediction.inference import inference


def setup_calculation(args):
    if args.mode == "train":
        train(args.config_path, args.file_path, args.checkpoint_path)
    elif args.mode == "wb_sweep":
        sweep_wb(args.file_path, args.config_path, args.wb_config_path)
    elif args.mode == "inference":
        inference(args.checkpoint_path, args.file_path, args.prediction_file_path)
    else:
        raise ValueError(f"Mode {args.mode} is not yet implemented.")


def main():
    """Console script for molproperty_prediction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("_", nargs="*")
    parser.add_argument(
        "--mode",
        "-m",
        required=True,
        type=str,
        default="inference",
        choices=["train", "inference", "wb_sweep"],
        help="Choose between 'train', 'inference' and 'wb_sweep'."
        "In case of 'train' the code assumes a training label in the dataset file.",
    )

    parser.add_argument(
        "--file-path",
        "-f",
        required=True,
        type=str,
        default="data/biogen_solubility.csv",
        help="Path to a .csv file containing SMILES strings. "
        "In case of training mode it assumes labels as well.",
    )

    parser.add_argument(
        "--wb-config-path",
        required=False,
        type=str,
        default=None,
        help="Path to W&B config for hyperparameter sweep. Make sure to choose wb_sweep mode.",
    )

    parser.add_argument(
        "--config-path",
        "-c",
        required=False,  # When using inference mode we use the config from the checkpoint.
        default="configs/config_mpnn.yml",
        help="Path to config file, defining model parameters and learning hyperparameters.",
    )

    parser.add_argument(
        "--prediction-file-path",
        required=False,
        help="Path to prediction file for the given .csv file containing the SMILES string.",
    )

    parser.add_argument(
        "--checkpoint-path",
        required=False,
        default="checkpoints/",
        help="Path to checkpoint folder for training. In case you want to use inference please specify the file path.",
    )

    args = parser.parse_args()

    setup_calculation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
