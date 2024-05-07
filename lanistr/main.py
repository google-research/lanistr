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

from __future__ import absolute_import

import argparse
import logging
import os
import pathlib
import random
import time
import warnings

from dataset.amazon.load_data import load_amazon
from dataset.mimic_iv.load_data import load_mimic
import numpy as np
import omegaconf
import torch
from trainer import Trainer
import transformers
from utils.common_utils import how_long
from utils.common_utils import print_config
from utils.common_utils import print_only_by_main_process
from utils.common_utils import set_global_logging_level
from utils.data_utils import generate_loaders
from utils.model_utils import build_model
from utils.parallelism_utils import is_main_process
from utils.parallelism_utils import setup_model

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
set_global_logging_level(logging.ERROR, ["transformers"])


def main() -> None:
  # Arguments
  parser = argparse.ArgumentParser(
      description="Multimodal Learning with LANISTR"
  )
  parser.add_argument(
      "--config", type=str, default="./configs/mimic_pretrain.yaml"
  )
  parser.add_argument(
      "--local_rank",
      type=int,
      default=0,
      help=(
          "Comes from torch.distributed.launch; will be ignored if DDP is not"
          " used. don't touch this."
      ),
  )
  parser.add_argument(
      "overrides",
      nargs="*",
      help=(
          "Any key=svalue arguments to override config values "
          "(use dots for.nested=overrides)"
      ),
  )
  flags = parser.parse_args()
  overrides = omegaconf.OmegaConf.from_cli(flags.overrides)
  config = omegaconf.OmegaConf.load(flags.config)
  args = omegaconf.OmegaConf.merge(config, overrides)
  args.local_rank = flags.local_rank

  # Settings for multi-GPU training:
  # nodes - number of machines, ngpus_per_node - number of GPUs to use per
  # machine any world_size > 1 will lead to distributed training: either
  # DP or DDP. DDP is further enabled by args.multiprocessing_distributed = True
  args.distributed = args.world_size > 1 or args.multiprocessing_distributed
  if args.distributed:
    current_env = os.environ.copy()
    args.local_rank = int(current_env["LOCAL_RANK"])
    args.world_size = int(current_env["WORLD_SIZE"])
  else:
    args.local_rank = 0

  args.device = args.local_rank

  # Only when DDP is used; DP doesn't need this
  if args.distributed and args.multiprocessing_distributed:
    torch.cuda.set_device(args.device)
    torch.distributed.init_process_group(
        backend=args.dist_backend,  # default to nccl
    )

  if not args.ngpus_per_node:
    args.ngpus_per_node = torch.cuda.device_count()

  main_worker(args)


def main_worker(args: omegaconf.DictConfig) -> None:
  time.time()

  # Setting dataset loader
  if args.dataset_name == "mimic-iv":
    load_dataset = load_mimic
  elif args.dataset_name == "amazon":
    load_dataset = load_amazon
  else:
    raise NotImplementedError(f"{args.dataset_name} not implemented.")

  # Set seed
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  # Setup logging
  pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  log_name = (
      f"{args.task}.log"
      if not args.experiment_name
      else args.experiment_name + ".log"
  )
  logging.basicConfig(
      filename=os.path.join(args.output_dir, log_name)
      if args.local_rank in [-1, 0]
      else None,
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )

  logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
      args.local_rank,
      args.device,
      args.world_size,
      bool(args.local_rank != -1),
  )
  print_config(args)

  # Load tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_encoder_name)

  # Load dataset
  tic = time.time()
  print_only_by_main_process("Loading datasets ... ")
  dataset = load_dataset(args, tokenizer)
  how_long(tic)

  # Load model and parallelize it
  model = build_model(
      args,
      tabular_data_information=dataset["tabular_data_information"],
  )
  args, model = setup_model(args, model)

  # Create the trainer and generate data loaders
  trainer = Trainer(model, args)
  dataloaders = generate_loaders(args, dataset)

  # Pretrain or finetune
  if args.task == "pretrain":
    pretrain_start = time.time()
    trainer.pretrain(dataloaders)
    how_long(
        pretrain_start,
        f"Pre-training finished after {args.scheduler.num_epochs} epochs",
    )

  elif args.task == "finetune":
    if args.do_train:
      train_start = time.time()
      trainer.train(dataloaders)
      how_long(
          train_start, f"Train the model for {args.scheduler.num_epochs} epochs"
      )

    elif args.do_test:
      if is_main_process():
        test_start = time.time()
        trainer.test(dataloaders["test"])
        how_long(test_start, "testing the model ")

  else:
    raise ValueError(f"Task {args.task} not implemented.")


if __name__ == "__main__":
  main()
