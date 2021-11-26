import torch
import wandb
import numpy as np
import uuid
import random
import json
from pathlib import Path
from ..utils.train_utils import cuda_if_available, EarlyStopping, ModelCheckpoint
from .dataset import DatasetUniformNegatives, DatasetValidation
from .loss import BCEWithWeightedLoss, BCELoss
from .loopers import TrainLooper, EvalLooper
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
        logger = WandBLogger()
        summary_func = wandb.run.summary.update
    else:
        logger = Logger()
        summary_func = None
        run_dir = Path(".")

    # For reproducibility
    set_seed(config["seed"])

    device = cuda_if_available(use_cuda=config["cuda"])

    bert_model, tokenizer = setup_bert(config)

    train_dataset = DatasetUniformNegatives.from_csv(
        train_path=config["train_file"],
        neg_ratio=config["neg_ratio"],
        device=device,
        tokenizer=tokenizer
    )
    print("Loaded Training Data")

    val_dataset = DatasetValidation.from_csv(
        val_dir=config["val_dir"],
        tokenizer=tokenizer,
        device=device
    )
    print("Loaded Val Data")

    model = FineTuneBert(bert_model=bert_model,
                         input_dim=768,
                         hidden_dim=config["hidden_dim"],
                         output_dim=1)

    model.to(device)
    print("Loaded model")

    if config["loss_fn"] == "UnweightedBCE":
        loss_function = BCELoss()

    train_dataloader = TensorDataLoader(
        train_dataset, batch_size=2 ** config["log_batch_size"], shuffle=True
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

    eval_loopers = [
        EvalLooper(
            model=model,
            batchsize=2 ** config["log_batch_size"],
            logger=logger,
            summary_func=summary_func,
            loss_fn=loss_function,
            dataset=val_dataset
        )
    ]

    train_looper = TrainLooper(
        model=model,
        opt=optimizer,
        loss_func=loss_function,
        dl=train_dataloader,
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        eval_looper=eval_loopers,
        logger=logger,
        save_model=ModelCheckpoint(run_dir),
        early_stopping=EarlyStopping("Loss", config["patience"]),
        summary_func=summary_func,
        scheduler=scheduler
    )

    best_model, best_metrics = train_looper.loop()

    save_model_metrics(best_model, best_metrics, config)

    print("Training Complete!!!")


def save_model_metrics(best_model, best_metrics, config):
    best_model_state_dict = {
        k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()
    }
    if config["wandb"]:
        random_hex = wandb.run.id
    else:
        random_hex = uuid.uuid4().hex

    output_dir = f'models/{random_hex}/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.save(best_model_state_dict, output_dir + "trained_model.pt")

    with open(f"{output_dir}/best_metrics.metric", "w", ) as f:
        output = json.dumps(dict(config))
        f.write(f"{output}\n")
        f.write(json.dumps(best_metrics))

    print(f"Output_Dir: {output_dir}")


def setup_bert(config: dict):
    if config["model"] == "distilbert-base-uncased":
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)
    return bert_model, tokenizer
