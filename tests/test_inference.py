#!/usr/bin/env python

"""Tests for `molproperty_prediction` package."""
import pathtools.path
import os
import torch
import numpy as np

from molproperty_prediction.utils.data_utils import load_csv_data, prepare_dataloader
from molproperty_prediction.utils.train_utils import get_model


def test_inference():
    base_path = pathtools.path.absolute_path(".")
    checkpoint_path = os.path.join(base_path, "checkpoints/model_checkpoint_ep_66_loss_0.33.pt")

    smiles = ["CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12",
              "CCOc1cc2nn(CCC(C)(C)O)cc2cc1NC(=O)c1cccc(C(F)F)n1",
              "CC(C)(Oc1ccc(-c2cnc(N)c(-c3ccc(Cl)cc3)c2)cc1)C(=O)O",
              "CC#CC(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1",
              "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1"]
    labels = np.array([-4.366646766662598,
              -4.481037139892578,
              -5.053138256072998,
              -4.036594390869141,
              -4.4794535636901855])

    input_smiles = smiles

    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]

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

    preds = np.array(preds)
    assert np.all(np.isclose(preds, labels))

