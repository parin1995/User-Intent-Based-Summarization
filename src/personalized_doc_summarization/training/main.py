import typer
from typing import List, Optional
from .setup import train_setup

app = typer.Typer(help="Training Parameters for Personalized Document Summarization")

@app.command()
def train(
        data_dir: str = typer.Option(...),
        train_file_name: str = typer.Option(...),
        epoch: int = typer.Option(5),
        learning_rate: float = typer.Option(0.01),
        cuda: bool = typer.Option(False, "--cude/--no-cuda"),
        wandb: bool = typer.Option(False, "--wandb/--no-wandb")
):
    config = {}
    config["data_dir"] = data_dir
    config["train_file"] = train_file_name
    config["epoch"] = epoch
    config["learning_rate"] = learning_rate
    config["cuda"] = cuda
    config["wandb"] = wandb

    train_setup(config)

if __name__ == "__main__":
    app()