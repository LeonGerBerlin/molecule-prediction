import numpy as np
import torch
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from molproperty_prediction.utils.data_utils import (
    load_config,
    normalize_label,
    prepare_dataloader,
    prepare_gbm_feature,
    load_csv_data,
)
from molproperty_prediction.utils.loggers import Logger
from molproperty_prediction.utils.train_utils import (
    get_model,
    get_lr_scheduler,
    eval_nn,
    eval_gbm,
    save_checkpoint,
)


def train(config_path, dataset_path, checkpoint_path=None):
    config, raw_dict = load_config(config_path)
    logger = Logger(config, raw_dict)

    data = load_csv_data(dataset_path)
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2)

    smiles, labels = (
        data.smiles.to_numpy(),
        data.label.to_numpy(),
    )
    smiles_train, labels_train = smiles[train_idx], labels[train_idx]
    smiles_test, labels_test = smiles[test_idx], labels[test_idx]

    if config.model.type in ["gcn", "gat", "transformer"]:
        if config.features.normalize_labels:
            labels_train, normalizing_constant = normalize_label(labels_train)
            labels_test, _ = normalize_label(labels_test, normalizing_constant)

        loader_train = prepare_dataloader(smiles_train, labels_train, config)
        model = train_nn(loader_train, config, logger, checkpoint_path)

        loader_test = prepare_dataloader(smiles_test, labels_test, config)
        loss = eval_nn(model, loader_test, loss_type=config.loss.type)

    elif config.model.type in ["gbm"]:
        descriptor_train = prepare_gbm_feature(smiles_train, config)
        model = train_lgbm(descriptor_train, labels_train, config)

        descriptor_test = prepare_gbm_feature(smiles_test, config)
        loss = eval_gbm(model, descriptor_test, labels_test, loss_type=config.loss.type)
    else:
        raise ValueError(f"Model type {config.model_type} is not yet defined!")

    logger.log(dict(loss_eval=loss))


def train_nn(loader, config, logger=None, checkpoint_path=None):
    model = get_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.optimization.lr, weight_decay=5e-4
    )
    scheduler = get_lr_scheduler(config.optimization, optimizer)
    loss_fct = torch.nn.MSELoss()

    nb_epochs = config.optimization.n_epochs
    with tqdm(range(nb_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            for batch in loader:
                model.train()
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = loss_fct(out, batch.y.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            loss_eval = eval_nn(model, loader)
            if scheduler is not None:
                scheduler.step(loss_eval)

            if checkpoint_path:
                save_checkpoint(
                    epoch, loss_eval, model.state_dict(), checkpoint_path, config
                )

            tepoch.set_postfix(loss=loss_eval, lr=optimizer.param_groups[0]["lr"])
            logger.log(dict(loss_opt=loss_eval, lr=optimizer.param_groups[0]["lr"]))
    return model


def train_lgbm(descriptor, label, config):
    if config.model.type == "lgbm":
        model = LGBMRegressor()
        model.fit(np.stack(descriptor), label)
    else:
        raise ValueError("GBM model not defined!")
    return model
