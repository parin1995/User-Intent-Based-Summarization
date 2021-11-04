import warnings

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Callable, Iterable, Optional, Union
from ..utils.loggers import Logger
from tqdm.autonotebook import trange, tqdm
import time

from ..utils.exceptions import StopLoopingException, EarlyStoppingException
from ..utils.train_utils import IntervalConditional
from .metrics import calculate_optimal_F1, calculate_rouge_scores

import torch
from torch.nn import Module
import numpy as np


class EvalLooper(object):

    def __init__(self,
                 model: Module,
                 batchsize: int,
                 logger: Logger,
                 summary_func: Callable,
                 dataset: Module = None):

        self.model = model
        self.batch_size = batchsize
        self.logger = logger
        self.summary_func = summary_func
        self.dataset = dataset


    def loop(self, eval_run: str):
        self.model.eval()
        if eval_run == "validation":
            sent_ids = self.dataset.val_sent_ids
            attention_masks = self.dataset.val_attention_masks

        ground_truth = self.dataset.get_ground_truth()
        prediction_scores = np.zeros(ground_truth.shape)

        number_of_entries = sent_ids.shape[0]

        with torch.no_grad():
            pbar = tqdm(
                desc=f"[{self.name}] Evaluating", leave=False, total=number_of_entries
            )
            cur_pos = 0
            while cur_pos < number_of_entries:
                last_pos = cur_pos
                cur_pos += self.batchsize
                if cur_pos > number_of_entries:
                    cur_pos = number_of_entries
                    out_probs = self.model(sent_ids[last_pos:cur_pos], attention_masks[last_pos:cur_pos])
                    prediction_scores[last_pos:cur_pos] = out_probs.cpu().numpy()
                    pbar.update(self.batchsize)

            val_loss = self.loss(prediction_scores, labels=ground_truth, eval=True)
            metrics = calculate_optimal_F1(
                ground_truth.flatten(), prediction_scores.flatten()
            )
            metrics["val_loss"] = val_loss
            predictions = prediction_scores > metrics["threshold"]
            # TODO: rouge scores by joining these predictions from the eval_dataset

class TrainLooper(object):

    def __init__(self,
                 model: Module,
                 opt: Optimizer,
                 loss_func: Callable,
                 dl: DataLoader,
                 learning_rate: float,
                 epochs: int,
                 eval_looper: Iterable[EvalLooper],
                 logger: Logger,
                 save_model: Callable,
                 summary_func: Callable,
                 log_interval: Optional[Union[IntervalConditional, int]] = None
                 ):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.dl = dl
        self.learning_rate = learning_rate
        self.logger = logger
        self.epochs = epochs
        self.eval_looper = eval_looper
        self.log_interval = log_interval
        self.save_model = save_model
        self.summary_func = summary_func

        self.running_loss = []
        self.looper_metrics = {"Total Examples": 0}
        self.best_metrics_comparison_functions = {"F1": max}
        self.best_metrics = {}
        self.previous_best = None
        self.evaluation_runs = ["train_eval", "validation"]

        if self.log_interval is None:
            # by default, log every batch
            self.log_interval = IntervalConditional(0)

    def loop(self):
        try:
            self.running_loss = []
            for epoch in trange(self.epochs, desc="[Train] Epochs"):
                self.model.train()
                with torch.enable_grad():
                    self.train_loop(epoch)
        except StopLoopingException as e:
            warnings.warn(str(e))
        finally:
            self.logger.commit()

            # load in the best model
            previous_device = next(iter(self.model.parameters())).device
            self.model.load_state_dict(self.save_model.best_model_state_dict)
            self.model.to(previous_device)

            return self.model

    def train_loop(self, epoch: int):
        examples_this_epoch = 0
        last_time_stamp = time.time()
        num_batch_passed = 0
        examples_in_single_epoch = len(self.dl.dataset)
        for iteration, batch in enumerate(
                tqdm(self.dl, desc="[Train] Batch", leave=False)
        ):
            self.opt.zero_grad()
            input_sent_ids, input_attention_masks, num_pos_in_batch = batch

            out_probs = self.model(input_sent_ids, input_attention_masks)
            loss = self.loss_func(out_probs, num_pos_in_batch)
            self.running_losses.append(loss.detach().item())

            examples_this_epoch += num_pos_in_batch
            num_batch_passed += 1

            loss.backward()
            self.opt.step()

            last_log = self.log_interval.last

            if self.log_interval(self.looper_metrics["Total Examples"]):
                current_time_stamp = time.time()
                time_spend = (current_time_stamp - last_time_stamp) / num_batch_passed
                last_time_stamp = current_time_stamp
                num_batch_passed = 0
                self.logger.collect({"avg_time_per_batch": time_spend})

                self.logger.collect(self.looper_metrics)
                mean_loss = sum(self.running_losses) / (
                        self.looper_metrics["Total Examples"] - last_log
                )
                metrics = {}

                for eval_run in self.evaluation_runs:
                    for eval_looper in self.eval_loopers:
                        metrics[eval_run] = eval_looper.loop(eval_run)

                metrics["Mean_Train_Loss"] = mean_loss

                self.logger.collect(
                    {
                        **{
                            f"[Train_Eval] {metric_name}": value
                            for metric_name, value in metrics["train_eval"].items()
                        },
                        **{
                            f"[Train] Mean_Train_Loss": metrics["Mean_Train_Loss"]
                        },
                        **{
                            f"[Validation] {metric_name}": value
                            for metric_name, value in metrics["validation"].items()
                        },
                        "Epoch": epoch + examples_this_epoch / examples_in_single_epoch,
                    }
                )

                self.logger.commit()
                self.running_losses = []
                self.update_best_metrics(metrics)
                self.save_if_best_(self.best_metrics["F1"])
                self.early_stopping(self.best_metrics["F1"])

    def update_best_metrics(self, metrics: dict[str, float]) -> None:
        for name, comparison in self.best_metrics_comparison_functions.items():
            if name not in self.best_metrics:
                for metric, val in metrics['validation'].items():
                    self.best_metrics[metric] = val
            else:
                best_metric = comparison(
                    metrics['validation'][name], self.best_metrics[name]
                )
                if metrics['validation'][name] == best_metric:
                    for metric, val in metrics['validation'].items():
                        self.best_metrics[metric] = val

        self.summary_func(
            {
                f"[Validation] Best Model {name}": val
                for name, val in self.best_metrics.items()

            }
        )

    def save_if_best_(self, best_metric) -> None:
        if best_metric != self.previous_best:
            self.save_model(self.model)
            self.previous_best = best_metric


class TestLooper(object):

    def __init__(self):
        pass

    def loop(self, eval_run):
        pass
