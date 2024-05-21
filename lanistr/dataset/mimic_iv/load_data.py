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

import math
import os
from typing import Any, Dict

from dataset.mimic_iv.mimic_utils import Discretizer
from dataset.mimic_iv.mimic_utils import get_normalizer
from dataset.mimic_iv.mimic_utils import load_labeled_multimodal_data
from dataset.mimic_iv.mimic_utils import load_pretraining_multimodal_data
from dataset.mimic_iv.mimic_utils import load_unimodal_data
from dataset.mimic_iv.mimic_utils import padding_mask
from third_party.mvts_transformer.timeseries_encoder import noise_mask
import numpy as np
import omegaconf
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import torchvision
import transformers
from utils.data_utils import get_image_transforms
from utils.data_utils import MaskGenerator


class MimicImageTextTabTimeseries(data.Dataset):
  """Dataset class for MIMIC-IV multimodal data.

  This class is used to create PyTorch datasets for multimodal learning tasks
  using MIMIC-IV data. It supports the following modalities:

  - Image: Images are read from a specified directory and preprocessed using
    image transformations (e.g., resizing, cropping, etc.).

  - Text: Text is tokenized using a specified tokenizer (e.g., from Hugging Face
    Transformers).

  - Tabular: Tabular data is read from a specified CSV file

  - Timeseries: Timeseries data is read from a specified CSV file and
    preprocessed using a specified normalizer.

  Attributes:
    args: An object containing configuration parameters, including: - text
      (bool): Whether to include text data. - image (bool): Whether to include
      image data. - time (bool): Whether to include timeseries data. - (other
      relevant parameters for image transforms, tokenization, etc.)
    df: A DataFrame containing the data for this dataset.
    tokenizer: A tokenizer object (e.g., from Hugging Face Transformers) for
      text preprocessing.
    transform: A function that applies image transformations to the input image.
    num_patches: The number of patches in the image.
    labels: The labels for the data.
    mask_generator: A MaskGenerator object for generating masks for image
      masking.
    discretizer: A Discretizer object for discretizing timeseries data.
    features: The features for the data.
    normalizer: A normalizer object for normalizing tabular and timeseries data.
    image: Whether to include image data.
    text: Whether to include text data.
    time: Whether to include timeseries data.
    tab: Whether to include tabular data.
  """

  def __init__(
      self,
      args: omegaconf.DictConfig,
      df: pd.DataFrame,
      tokenizer: transformers.BertTokenizer,
      transform: torchvision.transforms.Compose,
      feature_names: Any = None,
      text: bool = True,
      image: bool = True,
      time: bool = True,
      tab: bool = False,
  ):
    """Initialize the dataset.

    Args:
        args: omegaconf.DictConfig
        df: A DataFrame containing the data for this dataset.
        tokenizer: A tokenizer object (e.g., from Hugging Face Transformers) for
          text preprocessing.
        transform: A function that applies image transformations to the input
          image.
        feature_names: The names of the features to include in the dataset.
        text: Whether to include text data.
        image: Whether to include image data.
        time: Whether to include timeseries data.
        tab: Whether to include tabular data.
    """
    self.args = args
    self.df = df
    self.tokenizer = tokenizer
    self.transform = transform
    self.num_patches = (args.image_size // args.mask_patch_size) ** 2
    if args.task == 'finetune':
      self.labels = df['y_true'].values

    self.mask_generator = MaskGenerator(
        input_size=args.image_size,
        mask_patch_size=args.mask_patch_size,
        model_patch_size=args.model_patch_size,
        mask_ratio=args.image_masking_ratio,
    )

    self.discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy=args.impute_strategy,
        start_time=args.start_time,
        config_path=args.discretizer_config_path,
    )

    if tab:
      self.features = self.df[feature_names].values

    self.normalizer = get_normalizer(args, self.discretizer)
    self.image = image
    self.text = text
    self.time = time
    self.tab = tab

  def __getitem__(self, index):
    """Get item from dataset.

    Args:
        index: index of the item

    Returns:
        item: item from dataset
    """
    row = self.df.iloc[index]

    item = {}

    if self.text:
      if isinstance(row['text'], str):
        item = self.tokenizer.encode_plus(
            row['text'][140:],
            max_length=self.args.max_token_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
      else:
        if math.isnan(row['text']):
          item = self.tokenizer.encode_plus(
              '',
              max_length=self.args.max_token_length,
              truncation=True,
              add_special_tokens=True,
              return_token_type_ids=False,
              padding='max_length',
              return_attention_mask=True,
              return_tensors='pt',
          )

    # image
    if self.image:
      image_filename = row['image']
      if isinstance(image_filename, str):
        image_path = os.path.join(self.args.image_data_dir, image_filename)
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        item['pixel_values'] = img
        item['bool_masked_pos'] = self.mask_generator()
      else:

        if math.isnan(row['image']):
          item['pixel_values'] = torch.zeros(
              size=(3, self.args.image_size, self.args.image_size),
              dtype=torch.float,
          )
          item['bool_masked_pos'] = torch.ones(
              self.num_patches, dtype=torch.bool
          )

    # tabular
    if self.tab:
      item['features'] = torch.tensor(
          np.vstack(self.features[index]).astype(np.float32)
      ).squeeze(1)

    # time
    if self.time:
      ts_filename = row['timeseries']

      if isinstance(ts_filename, str):

        ts_filename = ts_filename.replace('npy', 'csv')
        ts = self._read_timeseries(
            ts_filename, time_bound=self.args.timeseries_max_seq_len
        )
        item['timeseries'] = torch.tensor(ts).float()

        lengths = [x.shape[0] for x in item['timeseries'].unsqueeze(0)]
        # padding_mask (`torch.BoolTensor` of shape
        # `(batch_size, padded_length)`):
        # Indicates which time steps are masked (0) and which aren't (1).
        # "1" means keep
        item['padding_mask'] = padding_mask(
            torch.LongTensor(lengths),
            max_len=self.args.timeseries_max_seq_len,
        ).squeeze(0)
        item['noise_mask'] = noise_mask(
            timeseries=item['timeseries'],
            masking_ratio=self.args.timeseries_masking_ratio,
            lm=self.args.timeseries_mean_mask_length,
            mode=self.args.timeseries_mask_mode,
            distribution=self.args.timeseries_mask_distribution,
            exclude_feats=None,
        ).to(dtype=bool)

      else:
        if math.isnan(ts_filename):
          item['timeseries'] = torch.zeros(
              size=(
                  self.args.timeseries_max_seq_len,
                  self.args.timeseries_input_dim,
              ),
              dtype=torch.float,
          )
          item['padding_mask'] = torch.ones(
              self.args.timeseries_max_seq_len, dtype=torch.bool
          )
          item['noise_mask'] = noise_mask(
              timeseries=item['timeseries'],
              masking_ratio=self.args.timeseries_masking_ratio,
              lm=self.args.timeseries_mean_mask_length,
              mode=self.args.timeseries_mask_mode,
              distribution=self.args.timeseries_mask_distribution,
              exclude_feats=None,
          ).to(dtype=bool)

    # ground truth label if finetuning
    if self.args.task == 'finetune':
      item['labels'] = torch.tensor(row['y_true'], dtype=torch.long)

    return item

  def _read_timeseries(self, ts_filename, time_bound=None):

    ret = []
    with open(ts_filename, 'r') as tsfile:
      header = tsfile.readline().strip().split(',')
      assert header[0] == 'Hours'
      for line in tsfile:
        mas = line.strip().split(',')
        if time_bound is not None:
          t = float(mas[0])
          if t > time_bound + 1e-6:
            break
        ret.append(np.array(mas))

    ret = np.stack(ret)
    data_discretized, _ = self.discretizer.transform(
        ret, end=self.args.timeseries_max_seq_len
    )
    data_discretized = self.normalizer.transform(data_discretized)
    return data_discretized

  def __len__(self) -> int:
    """Returns the number of items in the dataset.

    Returns:
        len(self.df): number of items in the dataset
    """
    return len(self.df)


def load_mimic(
    args: omegaconf.DictConfig, tokenizer: transformers.AutoTokenizer
) -> Dict[str, Any]:
  """Load MIMIC dataset.

  Args:
      args: arguments
      tokenizer: tokenizer

  Returns:
      dataset: torch.utils.data.Dataset.
  """

  if args.task == 'pretrain':
    # data from all modalities is read in different csv files,
    # it will created if does not exist
    unimodal_dataframes = load_unimodal_data(args)
    # merge unimodal dataframes and construct the entire pretraining dataframe
    # and split into train/validation/test
    dataframes = load_pretraining_multimodal_data(args, unimodal_dataframes)
    # create a PyTorch dataset from data frames
    dataset = create_multimodal_dataset_from_dataframes(
        args, dataframes, tokenizer
    )
  elif args.task == 'finetune':
    dataframes = load_labeled_multimodal_data(args)
    dataset = create_multimodal_dataset_from_dataframes(
        args, dataframes, tokenizer
    )
    dataset['tabular_data_information']['cat_dims'] = [3, 10, 4, 3, 5, 34]
  else:
    raise ValueError('Invalid task: %s' % args.task)
  return dataset


def create_multimodal_dataset_from_dataframes(
    args: omegaconf.DictConfig,
    dataframes: Dict[str, pd.DataFrame],
    tokenizer: Any,
) -> Dict[str, Any]:
  """Creates datasets for training, validation, and testing from DataFrames.

  This function processes input DataFrames containing image, text, tabular, and
  timeseries data (as specified in `args`) to create PyTorch datasets suitable
  for multimodal learning tasks. It applies appropriate image transformations
  and tokenization.

  Args:
      args: omegaconf.DictConfig
      dataframes: A dictionary of DataFrames, with keys 'train', 'valid', and
        'test', each containing the respective data splits.
      tokenizer: A tokenizer object (e.g., from Hugging Face Transformers) for
        text preprocessing.

  Returns:
      A dictionary containing:
          - 'train': A MimicImageTextTabTimeseries dataset for training.
          - 'valid': A MimicImageTextTabTimeseries dataset for validation.
          - 'test': A MimicImageTextTabTimeseries dataset for testing.
          - 'tabular_data_information': A dictionary containing metadata about
          the tabular features (e.g., column names, data types).

  Raises:
      KeyError: If any of the required keys ('train', 'valid', 'test') are
        missing from the `dataframes` dictionary.
      ValueError: If any of the specified modalities (text, image, time) are
        not supported by the `MimicImageTextTabTimeseries` class.
  """
  train_transform, test_transform = get_image_transforms(args)

  mm_train = MimicImageTextTabTimeseries(
      args=args,
      df=dataframes['train'],
      tokenizer=tokenizer,
      transform=train_transform,
      text=args.text,
      image=args.image,
      time=args.time,
  )

  mm_test = MimicImageTextTabTimeseries(
      args=args,
      df=dataframes['test'],
      tokenizer=tokenizer,
      transform=test_transform,
      text=args.text,
      image=args.image,
      time=args.time,
  )

  mm_valid = MimicImageTextTabTimeseries(
      args=args,
      df=dataframes['valid'],
      tokenizer=tokenizer,
      transform=test_transform,
      text=args.text,
      image=args.image,
      time=args.time,
  )

  return {
      'train': mm_train,
      'valid': mm_valid,
      'test': mm_test,
      'tabular_data_information': dataframes['tabular_data_information'],
  }
