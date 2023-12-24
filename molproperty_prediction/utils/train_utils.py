import os

import numpy as np
import torch
from scipy.stats import pearsonr

from molproperty_prediction.model.gnn import GNN


def get_model(config):
    model = GNN(
        type=config.model.type,
        output_features=config.model.architecture.output_dim,
        model_config=config.model.architecture,
    )
    return model


def get_lr_scheduler(opt_config, optimizer):
    scheduler = None
    if opt_config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=opt_config.patience_lr_scheduler,
            min_lr=opt_config.lr / opt_config.order_mag_reduce,
        )

    return scheduler


def eval_nn(model, loader, loss_type="mse"):
    model.eval()

    preds, label = [], []
    for batch in loader:
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(pred.detach().numpy())
        label.append(batch.y.reshape(-1, 1).numpy())

    preds, label = np.concatenate(preds), np.concatenate(label)
    if loss_type == "mse":
        loss = np.sum((preds - label) ** 2)
        loss /= label.size
    elif loss_type == "pearson":
        loss = pearsonr(preds.reshape(-1), label.reshape(-1)).statistic
    else:
        raise ValueError("Loss type not defined!")
    return loss


def eval_gbm(model, descriptor, label, loss_type="mse"):
    preds = model.predict(descriptor)

    if loss_type == "mse":
        loss = np.sum((preds - label) ** 2)
        loss /= label.size
    elif loss_type == "pearson":
        loss = pearsonr(preds, label).statistic
    else:
        raise ValueError("Loss type not defined!")
    return loss


def save_checkpoint(epoch, loss, model_state_dict, checkpoint_path, config):
    all_checkpoints = os.listdir(checkpoint_path)

    if all_checkpoints == []:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "loss": loss,
                "config": config,
            },
            os.path.join(
                checkpoint_path, f"model_checkpoint_ep_{epoch}_loss_{loss:.2f}.pt"
            ),
        )
    else:
        for f in all_checkpoints:
            loss_old = float(f[:-3].split("_")[-1])

            if loss < loss_old:
                os.remove(os.path.join(checkpoint_path, f))
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_state_dict,
                        "loss": loss,
                        "config": config,
                    },
                    os.path.join(
                        config.optimization.checkpoint_path,
                        f"model_checkpoint_ep_{epoch}_loss_{loss:.2f}.pt",
                    ),
                )
