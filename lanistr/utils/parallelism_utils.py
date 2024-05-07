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

from typing import Tuple

import omegaconf
import torch


def is_dist_avail_and_initialized() -> bool:
  """Checks if distributed training is available and initialized."""

  if not torch.distributed.is_available():
    return False

  if not torch.distributed.is_initialized():
    return False

  return True


def get_rank() -> int:

  if not is_dist_avail_and_initialized():
    return 0

  return torch.distributed.get_rank()


def is_main_process() -> bool:
  return get_rank() == 0


def setup_model(
    args: omegaconf.DictConfig,
    model: torch.nn.Module,
) -> Tuple[
    omegaconf.DictConfig,
    torch.nn.Module,
]:
  """Transfer the main model to single/multiple gpus, adjust batch and workers.

  Args:
    args: Config arguments.
    model (torch.nn.Module): Can be a secondary network architecture or the main
      model

  Returns:
    Updated arguments
    Parallelized model distributed over gpus or multiprocesses
  """
  ngpus_per_node = args.ngpus_per_node

  if not torch.cuda.is_available():
    raise ValueError(
        "using CPU at Google?!, this will be slow - re-install your nvidia"
        " drivers! "
    )

  if args.multiprocessing_distributed:  # use DDP
    print(
        "Setting up DistributedDataParallel model."
        f" {args.device + 1}/{args.world_size} process initialized. We are on"
        f" GPU #{args.device}"
    )
    model.cuda(args.device)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
    args.eval_batch_size = int(args.eval_batch_size / ngpus_per_node)
    args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # Convert BatchNorm to SyncBatchNorm.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.device],
        find_unused_parameters=True,
    )

  elif args.device and args.ngpus_per_node == 1:
    torch.cuda.set_device(args.device)
    model = model.cuda(args.device)

  else:  # use DP
    print(f"Using DataParallel on {ngpus_per_node} GPUs")
    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

  return args, model
