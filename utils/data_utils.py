import pandas as pd
import numpy as np
import torch
import yaml
from ml_collections import config_dict
from molfeat.trans import MoleculeTransformer
from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model.input_features import featurizer_simple, featurizer_advanced


def fix_seed(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_csv_data(dataset="solubility"):
    root_path = Path(__file__).absolute().parent.parent
    if dataset == "solubility":
        return pd.read_csv(root_path / "data/biogen_solubility.csv")
    else:
        raise ValueError(f"Dataset {dataset} not yet defined!")


def load_config(config_path):
    with open(config_path, "r") as stream:
        cfg_dict = yaml.safe_load(stream)
    return config_dict.ConfigDict(cfg_dict), cfg_dict


def normalize_label(labels, normalizing_constant=None):
    if normalizing_constant is None:
        normalizing_constant = np.max(labels)
    return labels / normalizing_constant, normalizing_constant


def prepare_dataloader(smiles, property, config):
    data_processed = []
    for smile, label in zip(smiles, property):
        if config.features.advanced_feat:
            edge_index, edge_attr, x = featurizer_advanced(smile)
        else:
            edge_index, edge_attr, x = featurizer_simple(smile)

        data_processed.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.float32),  # TODO Add normalize values
            )
        )

    loader = DataLoader(
        data_processed, batch_size=config.optimization.batch_size, shuffle=True
    )
    return loader


def prepare_gbm_feature(smiles, config):
    transformer = MoleculeTransformer(featurizer=config.features.type, dype=float)
    descriptor = transformer(smiles)
    return descriptor
