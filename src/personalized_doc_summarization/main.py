import typer
from typing import List, Optional
from .training.train import train_setup

app = typer.Typer(help="Training Parameters for Personalized Document Summarization")


@app.command()
def train(
        data_dir: str = typer.Option(...),
        train_file_name: str = typer.Option(...),
        epoch: int = typer.Option(10),
        learning_rate: float = typer.Option(0.01),
        cuda: bool = typer.Option(True, "--cuda/--no-cuda"),
        wandb: bool = typer.Option(False, "--wandb/--no-wandb"),
        neg_ratio: int = typer.Option(2),
        model: str = typer.Option("distilbert-base-uncased"),
        freeze_bert: bool = typer.Option(False, "--freeze/--no-freeze"),
        seed: int = typer.Option(42),
        loss_fn: str = typer.Option("UnweightedBCE"),
        hidden_dim: int = typer.Option(50),
        log_batch_size: int = typer.Option(3),
        patience: int = typer.Option(10),
        pos_weight: float = typer.Option(0.75)
):
    config = {}

    # Data Paths
    config["data_dir"] = data_dir
    config["train_file"] = data_dir + "/" + train_file_name
    config["test_dir"] = data_dir + "/test/"
    config["val_dir"] = data_dir + "/validation/"

    # Setup Params
    config["cuda"] = cuda
    config["wandb"] = wandb
    config["seed"] = seed

    # Training Params
    config["epochs"] = epoch
    config["learning_rate"] = learning_rate
    config["neg_ratio"] = neg_ratio
    config["model"] = model
    config["freeze_bert"] = freeze_bert
    config["loss_fn"] = loss_fn
    config["hidden_dim"] = hidden_dim
    config["log_batch_size"] = log_batch_size
    config["patience"] = patience
    config["positive_weight"]= pos_weight
    # Setup Training
    train_setup(config)

@app.command()
def test(todo: str):
    print("This is testing script")

def main():
    app()
