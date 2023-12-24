import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from molproperty_prediction.utils.train_utils import eval_nn
from molproperty_prediction.utils.loggers import Logger
from molproperty_prediction.train import train_nn
from molproperty_prediction.utils.data_utils import (
    load_csv_data,
    load_config,
    prepare_dataloader,
)


def setup_wbconfig(wb_config_path):
    if not wb_config_path:
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
    else:
        _, sweep_config = load_config(wb_config_path)
    return sweep_config


def sweep_wb(dataset_path, config_path, wb_config_path=None):
    sweep_config = setup_wbconfig(wb_config_path)

    def sweep_agent():
        loss_type = "pearson"

        # Load data
        data = load_csv_data(dataset_path)
        train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2)

        smiles, labels = (
            data.smiles.to_numpy(),
            data.label.to_numpy(),
        )
        smiles_train, labels_train = smiles[train_idx], labels[train_idx]
        smiles_test, labels_test = smiles[test_idx], labels[test_idx]

        config, raw_dict = load_config(config_path)
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

    sweep_id = wandb.sweep(sweep_config, project="mol-pred-sweep")
    wandb.agent(sweep_id, sweep_agent, count=50)
