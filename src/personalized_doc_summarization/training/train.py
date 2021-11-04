import torch
from torch import Tensor, LongTensor
import wandb
from pathlib import Path
from ..utils.train_utils import cuda_if_available


def train_setup(config: dict):

    if config["wandb"]:
        wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config
        run_dir = Path(wandb.run.dir)
    else:
        run_dir = Path(".")

    device = cuda_if_available(use_cuda=config["cuda"])
