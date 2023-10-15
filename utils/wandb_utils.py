import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from pathlib import Path

from eval import eval_nn
from utils.loggers import Logger
from train import train_nn
from utils.data_utils import load_csv_data, load_config, prepare_dataloader


def sweep_agent():
    loss_type = "pearson"

    # Load data
    dataset_solubility = load_csv_data()
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset_solubility)), test_size=0.2
    )

    smiles, labels = (
        dataset_solubility.SMILES.to_numpy(),
        dataset_solubility.logS.to_numpy(),
    )
    smiles_train, labels_train = smiles[train_idx], labels[train_idx]
    smiles_test, labels_test = smiles[test_idx], labels[test_idx]

    root_path = Path(__file__).absolute().parent.parent
    config, raw_dict = load_config(root_path / "/configs/config_mpnn.yaml")
    config.logging.experiment_name = None
    logger = Logger(config, raw_dict)

    wandb_config = wandb.config

    # wandb sweep
    lr = wandb_config["lr"]
    patience = wandb_config["patience"]
    order_mag_reduce = wandb_config["order_mag_reduce"]
    nb_layers = wandb_config["nb_layers"]
    num_heads = wandb_config["num_heads"]
    hidden_channels = wandb_config["hidden_channels"]

    logger.update(wandb_config)

    # adapt config
    config.optimization.lr = lr
    config.optimization.patience_lr_scheduler = patience
    config.optimization.order_mag_reduce = order_mag_reduce

    config.model.nb_layer = nb_layers
    config.model.num_heads = num_heads
    config.model.hidden_channels = hidden_channels

    loader_train = prepare_dataloader(smiles_train, labels_train, config)
    model = train_nn(loader_train, config, logger)

    loader_test = prepare_dataloader(smiles_test, labels_test, config)
    loss = eval_nn(model, loader_test, loss_type=loss_type)

    logger.log({"loss": loss})


def sweep_wb():
    sweep_config = {"method": "grid"}
    metric = {"name": "loss", "goal": "minimize"}

    sweep_config["metric"] = metric
    parameters_dict = {
        "lr": {"values": [0.001, 0.0001]},
        "patience": {"values": [2, 5, 10]},
        "order_mag_reduce": {"values": [10, 100]},
        "nb_layers": {"values": [2, 4]},
        "num_heads": {"values": [2, 4, 8]},
        "hidden_channels": {"values": [128, 256]},
    }
    sweep_config["parameters"] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="mol-pred-sweep")
    wandb.agent(sweep_id, sweep_agent, count=50)
