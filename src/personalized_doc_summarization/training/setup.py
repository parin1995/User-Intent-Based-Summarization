import torch
from torch import Tensor, LongTensor
import wandb


def train_setup(config: dict):

    if config["wandb"]:
        wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config
        run_dir = Path(wandb.run.dir)
    else:
        run_dir = Path(".")