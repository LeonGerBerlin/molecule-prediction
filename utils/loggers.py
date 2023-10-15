import wandb


class Logger:
    def __init__(self, config, raw_dict):
        self.run = wandb.init(
            project=config.logging.project_name,
            name=config.logging.experiment_name,
            config=raw_dict,
        )

    def log(self, metrics):
        wandb.log(metrics)

    def update(self, config):
        wandb.config.update(config)
