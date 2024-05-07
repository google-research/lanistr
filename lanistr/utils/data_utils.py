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
from typing import Dict, Tuple
import numpy as np
import omegaconf
import torch
from torchvision import transforms
import transformers
from utils.parallelism_utils import is_main_process


logger = logging.getLogger(__name__)


def generate_loaders(
    args: omegaconf.DictConfig, dataset: torch.utils.data.Dataset
) -> Dict[str, torch.utils.data.DataLoader]:
  """Generate the data loaders for the given dataset.

  Args:
    args: the arguments for the experiment
    dataset: the dataset to load

  Returns:
    A dictionary of data loaders
  """
  if args.task == 'pretrain':
    trainset = dataset['train']
    if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          trainset, shuffle=True, drop_last=True
      )
    else:
      train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if is_main_process():
      print(f'Number of training     examples: {len(trainset)}')
      logger.info('Number of training     examples: %d', len(trainset))

    data_loaders = {
        'train': train_dataloader,
    }
    return data_loaders

  elif args.task == 'finetune':
    trainset = dataset['train']
    testset = dataset['test']
    valset = dataset['valid']

    if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          trainset, shuffle=True, drop_last=True
      )
      valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valset, shuffle=False
      )
      test_sampler = None
    else:
      train_sampler = None
      valid_sampler = None
      test_sampler = None

    valid_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=valid_sampler,
    )

    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True,
    )
    if is_main_process():
      print(f'Number of training     examples: {len(trainset)}')
      print(f'Number of test         examples: {len(testset)}')
      logger.info('Number of training     examples: %d', len(trainset))
      logger.info(
          'Number of test         examples: %d', len(test_dataloader.dataset)
      )

    data_loaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloader,
    }
    return data_loaders


def get_image_transforms(
    args: omegaconf.DictConfig,
) -> Tuple[transforms.Compose, transforms.Compose]:
  """Get the image transforms for the given arguments.

  Args:
    args: the arguments for the experiment

  Returns:
    A tuple of the train and test transforms
  """

  image_processor = transformers.ViTImageProcessor.from_pretrained(
      args.image_encoder_name
  )
  train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(
          args.image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
      ),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=image_processor.image_mean, std=image_processor.image_std
      ),
  ])

  test_transforms = transforms.Compose([
      transforms.Resize(args.image_size),
      transforms.CenterCrop(args.image_crop),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=image_processor.image_mean, std=image_processor.image_std
      ),
  ])

  return train_transforms, test_transforms


class MaskGenerator:
  """A class to generate boolean masks for the pretraining task.

  A mask is a 1D tensor of shape (model_patch_size**2,) where the value is
  either 0 or 1, where 1 indicates "masked".
  """

  def __init__(
      self,
      input_size: int = 192,
      mask_patch_size: int = 32,
      model_patch_size: int = 4,
      mask_ratio: float = 0.6,
  ):
    """Initialize the MaskGenerator.

    Args:
      input_size: the size of the input image
      mask_patch_size: the size of the mask patch
      model_patch_size: the size of the model patch
      mask_ratio: the ratio of the mask patch to the model patch
    """
    self.input_size = input_size
    self.mask_patch_size = mask_patch_size
    self.model_patch_size = model_patch_size
    self.mask_ratio = mask_ratio

    if self.input_size % self.mask_patch_size != 0:
      raise ValueError('Input size must be divisible by mask patch size')
    if self.mask_patch_size % self.model_patch_size != 0:
      raise ValueError('Mask patch size must be divisible by model patch size')

    self.rand_size = self.input_size // self.mask_patch_size
    self.scale = self.mask_patch_size // self.model_patch_size

    self.token_count = self.rand_size**2
    self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

  def __call__(self):
    mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
    mask = np.zeros(self.token_count, dtype=int)
    mask[mask_idx] = 1

    mask = mask.reshape((self.rand_size, self.rand_size))
    mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

    return torch.tensor(mask.flatten(), dtype=torch.bool)
