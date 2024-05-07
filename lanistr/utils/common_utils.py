"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
import datetime
import logging
import os
import re
import shutil
import time

import omegaconf
import pytz
import torch
import torchmetrics
from utils.parallelism_utils import is_main_process


logger = logging.getLogger(__name__)


def pretty_print(num):
  magnitude = 0
  while abs(num) >= 1000:
    magnitude += 1
    num /= 1000.0
  return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


def print_time(tstart):
  """Print the time elapsed since the start time and the current time and the current time.

  Args:
    tstart: start time
  """
  print("*" * 100)
  print("All Done! ")
  print("[Elapsed time = {:.1f} min]".format((time.time() - tstart) / (60)))

  # dd/mm/YY H:M:S
  fmt = "%d/%m/%Y %H:%M:%S"

  # Current time in UTC
  now_utc = datetime.datetime.now(pytz.timezone("UTC"))

  # Convert to US/Pacific time zone
  now_pacific = now_utc.astimezone(pytz.timezone("US/Pacific"))

  print("Job finished at =", now_pacific.strftime(fmt))
  logger.info("Job finished at %s", now_pacific.strftime(fmt))


def get_metrics(args):
  """Get metrics for pretraining and finetuning.

  Args:
    args: config file

  Returns:
    metrics: dictionary of metrics
    metric_names: list of metric names
  """
  metrics = {"train": {}, "test": {}}

  # we always have loss in the metrics during both pretraining and fine tuning
  metric_names = ["Loss"]

  if args.task == "pretrain":
    metrics["train"]["Loss"] = torchmetrics.aggregation.MeanMetric().to(
        args.device
    )
    loss_names = ["MLM", "MIM", "MTM", "MFM", "MMM"]
    for loss_name in loss_names:
      metrics["train"][loss_name] = torchmetrics.aggregation.MeanMetric().to(
          args.device
      )
    metric_names += loss_names

  elif args.task == "finetune":
    for phase in ["train", "test"]:
      metrics[phase]["Loss"] = torchmetrics.aggregation.MeanMetric().to(
          args.device
      )

    if args.dataset_name.startswith("amazon"):
      metric_names.append("ACCURACY")
      for phase in ["train", "test"]:
        metrics[phase]["ACCURACY"] = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        ).to(args.device)

    elif args.dataset_name.startswith("mimic"):
      metric_names.append("AUROC")
      for phase in ["test", "train"]:
        metrics[phase]["AUROC"] = torchmetrics.AUROC(
            num_classes=args.num_classes, task="binary"
        ).to(args.device)

  return metrics, metric_names


def print_model_size(model, model_name):
  """Print the number of parameters in the model.

  Args:
    model: model
    model_name: name of the model
  """
  if is_main_process():
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    print(f"*********** {model_name} *********************")
    print(
        f"Number of total parameters in the model: {pretty_print(total_params)}"
    )
    print(
        "Number of trainable parameters in the model:"
        f" {pretty_print(trainable_params)}"
    )
    print("**************************************************")
    logger.info("*********** %s *********************", model_name)
    logger.info(
        "Number of total parameters in the model: %s",
        pretty_print(total_params),
    )
    logger.info(
        "Number of trainable parameters in the model: %s",
        pretty_print(trainable_params),
    )
    logger.info("**************************************************")


def how_long(tstart, what="load data"):
  if is_main_process():
    print("*" * 100)
    print(
        " >>>>>> Elapsed time to {} = {:.1f} min ({:.1f} seconds)".format(
            what, (time.time() - tstart) / (60), (time.time() - tstart)
        )
    )
    print("*" * 100)
    logger.info(
        " >>>>>> Elapsed time to {} = {:.1f} min".format(
            what, (time.time() - tstart) / (60)
        )
    )


def print_only_by_main_process(to_print):
  if is_main_process():
    print("*" * 100)
    print(f"{to_print}")


def print_df_stats(df, name):
  if is_main_process():
    print(f"{name} shape: {df.shape}")
    print(f"{name} columsn: {df.columns}")
    print("***" * 20)


def print_pretrain_performance_by_main_process(
    epoch,
    num_epochs,
    m_t,
    train_results,
    valid_results,
    is_best,
    best_perf,
    metric_name,
):
  """Print the performance of the model during pretraining.

  Args:
    epoch: current epoch
    num_epochs: total number of epochs
    m_t: metrics logger
    train_results: dictionary of training results
    valid_results: dictionary of validation results
    is_best: whether the current model is the best model
    best_perf: best performance of the model
    metric_name: name of the metric

  Returns:
    m_t: metrics logger
  """
  if torch.is_tensor(best_perf):
    best_perf = best_perf.item()
  if is_main_process():
    print(f"PreTraining Epoch {epoch+1}/{num_epochs}, Train: ", end="  ")
    for k, v in train_results.items():
      # if k == "Loss":
      if torch.is_tensor(v):
        print(f"{k}: {v.item():.4f}", end="  ")
        train_results[k] = v.item()
      else:
        print(f"{k}: {v:.4f}", end="  ")

      m_t.update(f"{k}", v)

    print("Valid: ", end="")

    for k, v in valid_results.items():
      if torch.is_tensor(v):
        print(f"{k}: {v.item():.4f}", end="  ")
        valid_results[k] = v.item()
      else:
        print(f"{k}: {v:.4f}", end="  ")

      m_t.update(f"valid_{k}", v)

    print(f"    | Best {metric_name} :{best_perf:.4f}", end="  ")

    if is_best:
      print(" **")
      m_t.update("best_epoch", epoch)
      m_t.update("best_perf", best_perf)

    else:
      print()
    logger.info(
        "PreTraining Epoch %d/%d || Train Loss: %.6f, T-MLM: %.3f"
        " T-MIM: %.3f T-MTM: %.3f T-MFM: %.3f T-MMM: %.3f || Valid"
        " Loss: %.6f V-MLM: %.3f V-MIM: %.3f V-MTM: %.3f V-MFM: %.3f"
        " V-MMM: %.3f",
        epoch + 1,
        num_epochs,
        train_results["Loss"],
        train_results["MLM"],
        train_results["MIM"],
        train_results["MTM"],
        train_results["MFM"],
        train_results["MMM"],
        valid_results["Loss"],
        valid_results["MLM"],
        valid_results["MIM"],
        valid_results["MTM"],
        valid_results["MFM"],
        valid_results["MMM"],
    )

  return m_t


def save_checkpoint_optimizer(
    epoch,
    optimizer,
    scheduler,
    is_best,
    file_dir,
    filename,
    best_filename="pretrain_optimizer_best.pth.tar",
):
  """Save the optimizer.

  Args:
    epoch: current epoch
    optimizer: optimizer
    scheduler: scheduler
    is_best: whether the current model is the best model
    file_dir: directory to save the model
    filename: name of the file to save the model
    best_filename: name of the file to save the best model
  """
  filename = os.path.join(file_dir, filename)
  torch.save(
      {
          "epoch": epoch,
          "optimizer": optimizer.state_dict(),
          "scheduler": scheduler.state_dict(),
      },
      filename,
  )
  if is_best:
    shutil.copyfile(filename, os.path.join(file_dir, best_filename))


def print_performance_by_main_process(
    epoch,
    num_epochs,
    m_t,
    train_results,
    valid_results,
    is_best,
    best_perf,
    metric_name,
):
  """Print the performance of the model during finetuning.

  Args:
    epoch: current epoch
    num_epochs: total number of epochs
    m_t: metrics logger
    train_results: dictionary of training results
    valid_results: dictionary of validation results
    is_best: whether the current model is the best model
    best_perf: best performance of the model
    metric_name: name of the metric

  Returns:
    m_t: metrics logger
  """
  if torch.is_tensor(best_perf):
    best_perf = best_perf.item()
  if is_main_process():
    print(f"Epoch {epoch+1}/{num_epochs}, Train: ", end="  ")
    for k, v in train_results.items():
      if torch.is_tensor(v):
        print(f"{k}: {v.item():.4f}", end="  ")
        train_results[k] = v.item()
      else:
        print(f"{k}: {v:.4f}", end="  ")

      m_t.update(f"train_{k}", v)

    print("Valid: ", end="")

    for k, v in valid_results.items():
      if torch.is_tensor(v):
        print(f"{k}: {v.item():.4f}", end="  ")
        valid_results[k] = v.item()
      else:
        print(f"{k}: {v:.4f}", end="  ")

      m_t.update(f"valid_{k}", v)

    print(f"    | Best {metric_name} :{best_perf:.4f}", end="  ")

    if is_best:
      print(" **")
      m_t.update("best_epoch", epoch)
      m_t.update("best_perf", best_perf)

    else:
      print()

    if metric_name == "Loss":
      logger.info(
          "Epoch %d/%d || Train Loss: %.6f || Valid Loss: %.6f",
          epoch + 1,
          num_epochs,
          train_results["Loss"],
          valid_results["Loss"],
      )
    else:
      logger.info(
          (
              "Epoch %d/%d || Train Loss: %.6f, Train Perf: %.6f || Valid"
              " Loss: %.6f Valid Perf: %.6f ||| Best %s: %.6f"
          ),
          epoch + 1,
          num_epochs,
          train_results["Loss"],
          train_results[metric_name],
          valid_results["Loss"],
          valid_results[metric_name],
          metric_name,
          best_perf,
      )

  return m_t


def print_config(args):
  """Print the config file.

  Args:
    args: config file
  """
  logger.info("Training parameters: ")
  if is_main_process():
    for k, v in args.items():
      print(f"{k}  --> {v}")
      logger.info("%s --> %s", k, v)

    print(omegaconf.OmegaConf.to_yaml(args))
    save_a_copy_at = os.path.join(args.output_dir, f"config_{args.task}.yaml")
    with open(save_a_copy_at, "w") as fp:
      omegaconf.OmegaConf.save(config=args, f=fp.name)


class MetricsLogger:
  """Metrics logger class.

  Attributes:
    logs: dictionary of logs
    list_inq: list of keys in the logs
  """

  def __init__(self):

    self.logs = dict()
    self.logs["valid_loss"] = []
    self.logs["train_loss"] = []
    self.logs["train_acc"] = []

    self.logs["train_loss_mlm"] = []
    self.logs["train_loss_mim"] = []
    self.logs["train_loss_mtm"] = []
    self.logs["train_loss_mfm"] = []
    self.logs["train_loss_mmm"] = []

    self.logs["valid_loss_mlm"] = []
    self.logs["valid_loss_mim"] = []
    self.logs["valid_loss_mtm"] = []
    self.logs["valid_loss_mfm"] = []
    self.logs["valid_loss_mmm"] = []

    self.logs["test_loss"] = None
    self.logs["test_perf"] = None
    self.logs["test_acc"] = None
    self.logs["test_auroc"] = None
    self.logs["best_acc"] = None
    self.logs["best_auroc"] = None
    self.logs["best_epoch"] = None

    self.list_inq = [
        "valid_loss",
        "train_loss",
        "train_acc",
        "valid_loss_mlm",
        "valid_loss_mim",
        "valid_loss_mtm",
        "valid_loss_mfm",
        "valid_loss_mmm",
        "train_loss_mlm",
        "train_loss_mim",
        "train_loss_mtm",
        "train_loss_mfm",
        "train_loss_mmm",
    ]

  def update(self, key, val):
    if key in self.list_inq:
      self.logs[key].append(val)
    else:
      self.logs[key] = val

  def save(self, file_dir):
    filename = os.path.join(file_dir, "results.log")
    torch.save(self.logs, filename)

  def update_lists(self, dicts):
    for d in dicts:
      for k, v in d.items():
        self.update(k, v)


def save_checkpoint(
    model, is_best, file_dir, filename, best_filename="model_best.pth.tar"
):
  """Save the model.

  Args:
    model: model
    is_best: whether the current model is the best model
    file_dir: directory to save the model
    filename: name of the file to save the model
    best_filename: name of the file to save the best model
  """
  model_to_save = (
      model.module if hasattr(model, "module") else model
  )  # Take care of distributed/parallel training
  filename = os.path.join(file_dir, filename)
  torch.save(model_to_save.state_dict(), filename)
  if is_best:
    shutil.copyfile(filename, os.path.join(file_dir, best_filename))


def load_checkpoint(current_model, best_checkpoint, different_datasets=False):
  """Load the best model.

  Args:
    current_model: current model
    best_checkpoint: best model
    different_datasets: whether the current dataset is different from the
      dataset used to train the best model

  Returns:
    current_model: current model
  """
  new_state_dict = collections.OrderedDict()
  for k, v in best_checkpoint.items():
    if (
        k.startswith("tabular_encoder.embedder.embeddings")
        and different_datasets
    ):
      pass
    else:
      new_state_dict[k] = v

  current_model.load_state_dict(new_state_dict, strict=False)

  return current_model


def load_checkpoint_with_module(current_model, best_checkpoint):
  """Load the best model.

  Args:
    current_model: current model
    best_checkpoint: best model

  Returns:
    current_model: current model
  """
  new_state_dict = collections.OrderedDict()
  for k, v in best_checkpoint.items():
    if "module" not in k:
      k = "module." + k
    new_state_dict[k] = v

  current_model.load_state_dict(new_state_dict)

  return current_model


# To control logging level for various modules used in the application:
def set_global_logging_level(level=logging.ERROR, prefices=None):
  """Override logging levels of different modules based on their name as a prefix.

  It needs to be invoked after the modules have been loaded so that their
  loggers have been initialized.

  Args:
    level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
    prefices: list of one or more str prefices to match (e.g. [ "transformers",
      "torch"]). Optional. Default is `None` to match all active loggers. The
      match is a case-sensitive `module_name.startswith(prefix)`.
  """
  if prefices is None:
    prefices = [""]
  prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
  for name in logging.root.manager.loggerDict:
    if re.match(prefix_re, name):
      logging.getLogger(name).setLevel(level)
