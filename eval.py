import numpy as np
from scipy.stats import pearsonr


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
