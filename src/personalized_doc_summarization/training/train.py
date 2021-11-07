import torch
from torch import Tensor, LongTensor
import wandb
import numpy as np
import random
from pathlib import Path
from ..utils.train_utils import cuda_if_available
from dataset import DatasetUniformNegatives, DocumentTestSet
from loss import BCEWithWeightedLoss, BCELoss
from loopers import TrainLooper, TestLooper, EvalLooper
from ..models.bert import FineTuneBert
from ..utils.tensordataloader import TensorDataLoader
from ..utils.loggers import WandBLogger, Logger

import transformers as ppb  # pytorch transformers
from transformers import AdamW, get_linear_schedule_with_warmup

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train_setup(config: dict):
    if config["wandb"]:
        wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config
        run_dir = Path(wandb.run.dir)
        logger = WandBLogger
        summary_func = wandb.run.summary.update
    else:
        logger = Logger
        summary_func = None
        run_dir = Path(".")

    # For reproducibility
    set_seed(config["seed"])

    device = cuda_if_available(use_cuda=config["cuda"])

    bert_model, tokenizer = setup_bert(config, device)

    train_dataset = DatasetUniformNegatives.from_csv(
        train_path=config["train_file"],
        neg_ratio=config["neg_ratio"],
        device=device,
        tokenizer=tokenizer
    )
    val_dataset = DatasetValidation.from
    print("Loaded Training Data")


    model = FineTuneBert(bert_model=bert_model,
                         input_dim=768,
                         hidden_dim=config["hidden_dim"],
                         output_dim=1)

    model.to(device)
    print("Loaded model")

    if config["loss_fn"] == "UnweightedBCE":
        loss_function = BCELoss()

    train_dataloader = TensorDataLoader(
        train_dataset, batch_size=2**config["log_batch_size"], shuffle=True
    )

    # Creating Optimizer
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], eps=1e-8
    )

    # Set up the learning rate scheduler
    total_steps = config["epochs"] * len(train_dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    eval_loopers = []
    eval_loopers.append(
        EvalLooper(
            model=model,
            batchsize=2**config["log_batch_size"],
            logger=logger,
            summary_func=summary_func,
            dataset=val_dataset
        )
    )




def setup_bert(config: dict):
    if config["model"] == "distilbert-base-uncased":
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)
    return bert_model, tokenizer
