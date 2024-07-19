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

import logging
import os
from typing import Mapping, Tuple

import numpy as np
import omegaconf
import torch
import tqdm
import transformers
from utils.common_utils import get_metrics
from utils.common_utils import load_checkpoint_with_module
from utils.common_utils import MetricsLogger
from utils.common_utils import print_only_by_main_process
from utils.common_utils import print_performance_by_main_process
from utils.common_utils import print_pretrain_performance_by_main_process
from utils.common_utils import save_checkpoint
from utils.common_utils import save_checkpoint_optimizer
from utils.parallelism_utils import is_main_process


logger = logging.getLogger(__name__)


class Trainer:
  """Basic training pipeline for LANISTR."""

  def __init__(
      self,
      model: torch.nn.Module,
      args: omegaconf.DictConfig,
  ):
    """Initialize the trainer.

    Args:
      model: The model to train.
      args: The arguments.
    """
    super().__init__()
    self.model = model
    self.args = args
    self.device = args.device

    self.num_epochs = args.scheduler.num_epochs
    self.lr = self.args.optimizer.learning_rate
    self.wd = self.args.optimizer.weight_decay

    self.distributed = args.distributed
    self.local_rank = args.local_rank
    self.multiprocessing_distributed = args.multiprocessing_distributed
    self.ngpus_per_node = args.ngpus_per_node
    self.world_size = args.world_size

    # pretrain metric logger
    self.m_p = MetricsLogger()
    # train metric logger
    self.m_t = MetricsLogger()

    self.mimic = args.dataset_name.startswith("mimic")
    self.amazon = args.dataset_name.startswith("amazon")
    self.metrics, self.metric_names = get_metrics(args)

  def get_optimizer(
      self,
  ) -> Tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Get optimizer and scheduler for pretraining."""
    optimizer = torch.optim.AdamW(
        [p for p in self.model.module.parameters() if p.requires_grad],
        lr=self.lr,
        weight_decay=self.wd,
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.args.scheduler.warmup_epochs,
        num_training_steps=self.args.scheduler.num_epochs,
        num_cycles=0.5,
        last_epoch=-1,
    )
    return optimizer, scheduler

  def pretrain(self, dataloaders: Mapping[str, torch.utils.data.DataLoader]):
    """Pretrain the model."""

    logger.info("Pretraining starts: ")
    train_loader = dataloaders["train"]

    self.optimizer, self.scheduler = self.get_optimizer()
    init_epoch = 0

    # if resume,load optimizer and scheduler from pretrain checkpoint
    if self.args.pretrain_resume:
      latest_checkpoint_path = os.path.join(
          self.args.output_dir,
          f"pretrain_multimodal_checkpoint_optimizer_{self.args.pretrain_initialize_from_epoch}.pth.tar",
      )
      if os.path.exists(latest_checkpoint_path):
        print_only_by_main_process(
            "Initializing the entire model from previous pretrain"
        )
        loc = "cuda:{}".format(self.args.device)
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=loc)
      else:
        raise FileNotFoundError(
            f"Pretrained checkpoint {latest_checkpoint_path} not found."
            " Pretrain first by passing task=pretrain as an argument"
        )
      self.optimizer.load_state_dict(latest_checkpoint["optimizer"])
      self.scheduler.load_state_dict(latest_checkpoint["scheduler"])
      init_epoch = latest_checkpoint["epoch"]
    else:
      print_only_by_main_process("Randomly initializing the entire model")

    best_epoch = 0
    best_perf = np.inf
    for epoch in range(init_epoch, self.args.scheduler.num_epochs):
      if self.distributed:
        train_loader.sampler.set_epoch(epoch)

      if epoch > 0:
        self.scheduler.step()  # Update learning rate schedule

      # train for one epoch
      train_results = self.pretrain_epoch(train_loader)

      is_best = train_results["Loss"] < best_perf
      best_perf = min(train_results["Loss"], best_perf)
      if is_best:
        best_epoch = epoch

      print_pretrain_performance_by_main_process(
          epoch,
          self.args.scheduler.num_epochs,
          self.m_t,
          train_results,
          train_results,
          is_best,
          best_perf,
          metric_name="Loss",
      )

      if not self.multiprocessing_distributed or (
          self.multiprocessing_distributed
          and self.local_rank % self.ngpus_per_node == 0
      ):
        save_checkpoint(
            self.model,
            is_best,
            file_dir=self.args.output_dir,
            filename=f"pretrain_multimodal_checkpoint_{epoch}.pth.tar",
            best_filename="pretrain_multimodal_model_best.pth.tar",
        )
        save_checkpoint_optimizer(
            epoch,
            self.optimizer,
            self.scheduler,
            is_best,
            file_dir=self.args.output_dir,
            filename=(
                f"pretrain_multimodal_checkpoint_optimizer_{epoch}.pth.tar"
            ),
            best_filename="pretrain_multimodal_optimizer_best.pth.tar",
        )

      for metric_name in self.metric_names:
        self.metrics["train"][metric_name].reset()

    logger.info(
        "Pretraining ends for lanistr. Best epoch was at epoch=%d", best_epoch
    )
    print_only_by_main_process(
        "Pretraining ends for lanistr. Best epoch was at epoch=%d", best_epoch
    )

  def pretrain_epoch(
      self, train_loader: torch.utils.data.DataLoader
  ) -> Mapping[str, float]:
    """Single epoch of pretraining."""

    self.model.train()
    for _, batch in enumerate(
        tqdm.tqdm(
            train_loader,
            desc="Training LANISTR",
            disable=not is_main_process(),
        )
    ):

      inputs = {}
      if self.args.time:
        inputs["padding_mask"] = batch["padding_mask"].cuda(
            self.device, non_blocking=True
        )
        inputs["timeseries"] = batch["timeseries"].cuda(
            self.device, non_blocking=True
        )
        inputs["noise_mask"] = batch["noise_mask"].cuda(
            self.device, non_blocking=True
        )

      if self.args.image:
        inputs["pixel_values"] = batch["pixel_values"].cuda(
            self.device, non_blocking=True
        )
        inputs["bool_masked_positions"] = batch["bool_masked_positions"].cuda(
            self.device, non_blocking=True
        )

      if self.args.text:
        inputs["input_ids"] = batch["input_ids"].cuda(
            self.device, non_blocking=True
        )
        inputs["attention_mask"] = batch["attention_mask"].cuda(
            self.device, non_blocking=True
        )

      if self.args.tab:
        inputs["features"] = batch["features"].cuda(
            self.device, non_blocking=True
        )

      # compute output
      output = self.model(inputs)
      loss = output.loss

      # compute gradient and do gradient update step
      if self.distributed:
        self.metrics["train"]["Loss"].update(loss)
        self.metrics["train"]["MLM"].update(output.loss_mlm)
        self.metrics["train"]["MIM"].update(output.loss_mim)
        self.metrics["train"]["MTM"].update(output.loss_mtm)
        self.metrics["train"]["MFM"].update(output.loss_mfm)
        self.metrics["train"]["MMM"].update(output.loss_mmm)
        self.optimizer.zero_grad()
        loss.backward()
      else:
        self.metrics["train"]["Loss"].update(loss.sum())
        self.metrics["train"]["MLM"].update(output.loss_mlm.sum())
        self.metrics["train"]["MIM"].update(output.loss_mim.sum())
        self.metrics["train"]["MTM"].update(output.loss_mtm.sum())
        self.metrics["train"]["MFM"].update(output.loss_mfm.sum())
        self.metrics["train"]["MMM"].update(output.loss_mmm.sum())
        self.optimizer.zero_grad()
        loss.sum().backward()

      torch.nn.utils.clip_grad_norm_(
          self.model.parameters(), self.args.optimizer.clip_value
      )
      self.optimizer.step()

    results = {}
    for metric_name in self.metric_names:
      results[metric_name] = self.metrics["train"][metric_name].compute()

    return results

  def train(self, dataloaders: Mapping[str, torch.utils.data.DataLoader]):
    """Train the model."""
    train_dataloader = dataloaders["train"]
    valid_dataloader = dataloaders["valid"]

    self.optimizer, self.scheduler = self.get_optimizer()

    best_perf = 0
    for epoch in range(self.num_epochs):
      if self.distributed:
        train_dataloader.sampler.set_epoch(epoch)

      if epoch > 0:
        self.scheduler.step()  # Update learning rate schedule

      # train for one epoch
      train_results = self.train_epoch(train_dataloader)

      # evaluate on validation set
      valid_results = self.validate(valid_dataloader)

      is_best = valid_results[self.args.perf_metric.upper()] > best_perf
      best_perf = max(valid_results[self.args.perf_metric.upper()], best_perf)

      print_performance_by_main_process(
          epoch,
          self.num_epochs,
          self.m_t,
          train_results,
          valid_results,
          is_best,
          best_perf,
          metric_name=self.args.perf_metric.upper(),
      )

      if not self.multiprocessing_distributed or (
          self.multiprocessing_distributed
          and self.local_rank % self.ngpus_per_node == 0
      ):
        save_checkpoint(
            self.model,
            is_best,
            file_dir=self.args.output_dir,
            filename="checkpoint.pth.tar",
        )

      for metric_name in self.metric_names:
        self.metrics["train"][metric_name].reset()
        self.metrics["test"][metric_name].reset()

  def train_epoch(
      self, train_loader: torch.utils.data.DataLoader
  ) -> Mapping[str, float]:
    """Single epoch of training."""

    # switch to train mode
    self.model.train()
    for _, batch in enumerate(
        tqdm.tqdm(train_loader, desc="Training ", disable=not is_main_process())
    ):
      inputs = {}
      if self.args.time:
        inputs["padding_mask"] = batch["padding_mask"].cuda(
            self.device, non_blocking=True
        )
        inputs["timeseries"] = batch["timeseries"].cuda(
            self.device, non_blocking=True
        )
        inputs["noise_mask"] = batch["noise_mask"].cuda(
            self.device, non_blocking=True
        )
      if self.args.image:
        inputs["pixel_values"] = batch["pixel_values"].cuda(
            self.device, non_blocking=True
        )
        inputs["bool_masked_positions"] = batch["bool_masked_positions"].cuda(
            self.device, non_blocking=True
        )
      if self.args.text:
        inputs["input_ids"] = batch["input_ids"].cuda(
            self.device, non_blocking=True
        )
        inputs["attention_mask"] = batch["attention_mask"].cuda(
            self.device, non_blocking=True
        )
      if self.args.tab:
        inputs["features"] = batch["features"].cuda(
            self.device, non_blocking=True
        )
      if batch["labels"].ndim > 1:
        inputs["labels"] = (
            batch["labels"].squeeze(1).cuda(self.device, non_blocking=True)
        )
      else:
        inputs["labels"] = batch["labels"].cuda(self.device, non_blocking=True)

      # compute output
      output = self.model(inputs)
      loss = output.loss

      # measure accuracy and record loss
      self.metrics["train"][self.args.perf_metric.upper()].update(
          preds=output.logits, target=inputs["labels"]
      )

      # compute gradient and do gradient update step
      if self.distributed:
        self.metrics["train"]["Loss"].update(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
      else:
        self.metrics["train"]["Loss"].update(loss.sum().item())
        self.optimizer.zero_grad()
        loss.sum().backward()

      torch.nn.utils.clip_grad_norm_(
          self.model.parameters(), self.args.optimizer.clip_value
      )
      self.optimizer.step()

    results = {}
    for metric_name in self.metric_names:
      results[metric_name] = self.metrics["train"][metric_name].compute()

    return results

  def validate(
      self, val_loader: torch.utils.data.DataLoader, prefix="Validation  "
  ):
    """Single validation on validation set."""

    # switch to evaluate mode
    self.model.eval()

    with torch.no_grad():
      for _, batch in enumerate(
          tqdm.tqdm(val_loader, desc=prefix, disable=not is_main_process())
      ):

        inputs = {}
        if self.args.time:
          inputs["padding_mask"] = batch["padding_mask"].cuda(
              self.device, non_blocking=True
          )
          inputs["timeseries"] = batch["timeseries"].cuda(
              self.device, non_blocking=True
          )
          inputs["noise_mask"] = batch["noise_mask"].cuda(
              self.device, non_blocking=True
          )
        if self.args.image:
          inputs["pixel_values"] = batch["pixel_values"].cuda(
              self.device, non_blocking=True
          )
          inputs["bool_masked_positions"] = batch["bool_masked_positions"].cuda(
              self.device, non_blocking=True
          )
        if self.args.text:
          inputs["input_ids"] = batch["input_ids"].cuda(
              self.device, non_blocking=True
          )
          inputs["attention_mask"] = batch["attention_mask"].cuda(
              self.device, non_blocking=True
          )
        if self.args.tab:
          inputs["features"] = batch["features"].cuda(
              self.device, non_blocking=True
          )

        if batch["labels"].ndim > 1:
          inputs["labels"] = (
              batch["labels"].squeeze(1).cuda(self.device, non_blocking=True)
          )
        else:
          inputs["labels"] = batch["labels"].cuda(
              self.device, non_blocking=True
          )

        # compute output
        output = self.model(inputs)
        loss = output.loss

        # measure accuracy and record loss
        self.metrics["test"][self.args.perf_metric.upper()].update(
            preds=output.logits, target=inputs["labels"]
        )

        if self.distributed:
          self.metrics["test"]["Loss"].update(loss.item())
        else:
          self.metrics["test"]["Loss"].update(loss.sum().item())

      results = {}
      for metric_name in self.metric_names:
        results[metric_name] = self.metrics["test"][metric_name].compute()

      return results

  def test(self, test_dataloader: torch.utils.data.DataLoader):
    """Single test on test set using the best model on the main process.

    Args:
      test_dataloader: The test dataloader.
    """

    best_checkpoint = torch.load(
        os.path.join(self.args.output_dir, "model_best.pth.tar")
    )
    self.model = load_checkpoint_with_module(self.model, best_checkpoint)

    results = self.validate(test_dataloader, prefix="TEST ---")
    logger.info(
        "Final Test Perf: %f | Test Loss: %f",
        results[self.args.perf_metric.upper()].item(),
        results["Loss"],
    )
    for metric_name in self.metric_names:
      print("Test Best Perf %s : %f", metric_name, results[metric_name].item())
      logger.info(
          "Test Best Perf %s : %f", metric_name, results[metric_name].item()
      )

    self.m_t.update("test_perf", results[self.args.perf_metric.upper()].item())
    self.m_t.update("test_loss", results["Loss"])
    self.m_t.save(self.args.output_dir)
