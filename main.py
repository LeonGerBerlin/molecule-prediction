from sklearn.model_selection import train_test_split
import numpy as np
from train import train_nn, train_lgbm
from eval import eval_nn, eval_gbm
from utils.data_utils import (
    load_csv_data,
    fix_seed,
    prepare_gbm_feature,
    prepare_dataloader,
    normalize_label,
    load_config,
)

from utils.loggers import Logger


def optimize_and_evaluate(
    smiles_train,
    labels_train,
    smiles_test,
    labels_test,
    model_type,
    loss_type,
    config_path,
):
    config, raw_dict = load_config(config_path)
    logger = Logger(config, raw_dict)

    if model_type == "mpnn":
        if config.features.normalize_labels:
            labels_train, normalizing_constant = normalize_label(labels_train)
            labels_test, _ = normalize_label(labels_test, normalizing_constant)

        loader_train = prepare_dataloader(smiles_train, labels_train, config)
        model = train_nn(loader_train, config, logger)

        loader_test = prepare_dataloader(smiles_test, labels_test, config)
        loss = eval_nn(model, loader_test, loss_type=loss_type)

    elif model_type == "gbm":
        descriptor_train = prepare_gbm_feature(smiles_train, config)
        model = train_lgbm(descriptor_train, labels_train, config)

        descriptor_test = prepare_gbm_feature(smiles_test, config)
        loss = eval_gbm(model, descriptor_test, labels_test, loss_type=loss_type)
    else:
        raise ValueError(f"Model type {model_type} is not yet defined!")

    logger.log(dict(loss_eval=loss))
    print(f"Loss {model_type}: {loss}")


def main():
    fix_seed()

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

    methods = ["mpnn"]  # mpnn gbm
    for m in methods:
        optimize_and_evaluate(
            smiles_train,
            labels_train,
            smiles_test,
            labels_test,
            model_type=m,
            loss_type=loss_type,
            config_path=f"configs/config_{m}.yaml",
        )


if __name__ == "__main__":
    main()
    # sweep_wb()
