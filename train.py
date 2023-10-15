import torch
from tqdm import tqdm
import numpy as np
from lightgbm import LGBMRegressor

from eval import eval_nn
from model.gnn import GNN


def get_model(config):
    if config.model.name in ["GCN", "GAT", "Transformer"]:
        model = GNN(
            output_features=config.model.output_dim,
            model_config=config.model,
        )
    else:
        raise ValueError(f"Model {config.model.name} not yet defined!")

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


def train_nn(loader, config, logger=None):
    model = get_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.optimization.lr, weight_decay=5e-4
    )  # Define optimizer.
    scheduler = get_lr_scheduler(config.optimization, optimizer)

    loss_fct = torch.nn.MSELoss()

    nb_epochs = config.optimization.n_epochs
    with tqdm(range(nb_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            loss_eval = eval_nn(model, loader)
            tepoch.set_description(f"Epoch {epoch}")

            for batch in loader:
                model.train()
                optimizer.zero_grad()  # Clear gradients.
                out = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )  # Perform a single forward pass.
                loss = loss_fct(out, batch.y.reshape(-1, 1))
                loss.backward()  # Derive gradients.
                optimizer.step()

            if scheduler is not None:
                scheduler.step(loss_eval)

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
