import pandas as pd
import torch

from molproperty_prediction.utils.data_utils import (
    load_csv_data,
    prepare_dataloader,
)
from molproperty_prediction.utils.train_utils import get_model


def inference(checkpoint_path, dataset_path, prediction_path):
    data = load_csv_data(dataset_path)
    input_smiles = data.smiles

    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]

    if config.model.type in ["gcn", "gat", "transformer"]:
        loader = prepare_dataloader(
            input_smiles, [None] * len(input_smiles), config, shuffle=False
        )

        model = get_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        preds, smiles = [], []
        for batch in loader:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds += pred.detach().squeeze().tolist()
            smiles += batch.smile

        df = pd.DataFrame({"smiles": smiles, "pred_labels": preds})
        df.to_csv(prediction_path)

    else:
        raise f"Currently not supported to load checkpoint for {config.model.type}."
