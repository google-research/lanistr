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

import gzip
import json
import os
from typing import List, Tuple

import numpy as np
import omegaconf
import pandas as pd
from sklearn import preprocessing
from torchvision import transforms
from utils.common_utils import print_df_stats
from utils.common_utils import print_only_by_main_process


def load_multimodal_data(args: omegaconf.DictConfig) -> pd.DataFrame:
  """Loads multimodal data from a specified file, handling pretraining and other tasks.

    This function reads data from either a JSON.GZ file (for pretraining) or a
    CSV file (for other tasks). It assumes the data has been prepared and saved
    to the correct location according to the provided arguments from an
    OmegaConf configuration.

  Args:
    args: An OmegaConf.DictConfig object

  Returns:
      A pandas DataFrame containing the loaded data.
  """
  if args.task == "pretrain":
    path = os.path.join(args.data_dir, f"{args.category}_total.json.gz")
    data = read_gzip(path)
  else:
    path_to_clean_data = os.path.join(
        args.data_dir, f"{args.category}_total.csv"
    )
    data = pd.read_csv(path_to_clean_data)
    data = data.reset_index(drop=True)
  try:
    data = data.drop(columns=["Unnamed: 0"], axis=1)
  except KeyError:
    pass
  return data


def read_gzip(path: str) -> pd.DataFrame:
  """Read gzip file.

  Args:
    path: Path to gzip file.

  Returns:
    data: pd.DataFrame.
  """
  data = []
  with gzip.open(path) as f:
    for l in f:
      data.append(json.loads(l.strip()))
  data = pd.DataFrame.from_dict(data[0])
  data = data.reset_index(drop=True)
  return data


def preprocess_amazon_tabular_features(
    data: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> Tuple[pd.DataFrame, List[int], List[int], int]:
  """Preprocess Amazon tabular features.

  Args:
    data: Dataframe.
    categorical_cols: List of categorical columns.
    numerical_cols: List of numerical columns.

  Returns:
    data: Dataframe.
    cat_idxs: List of categorical indices.
    cat_dims: List of categorical dimensions.
    input_dim: Input dimension.
  """
  data, cat_idxs, cat_dims, input_dim = encode_tabular_features(
      data, categorical_cols, numerical_cols
  )
  return data, cat_idxs, cat_dims, input_dim


def encode_tabular_features(
    data: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]
) -> Tuple[pd.DataFrame, List[int], List[int], int]:
  """Encodes tabular features for machine learning processing.

  This function handles both categorical and numerical features in a DataFrame:

  - Categorical features are label-encoded, with missing values filled as
  "VV_likely".
  - Numerical features have missing values imputed with their mean.

  Args:
      data: The pandas DataFrame containing the features.
      categorical_cols: A list of column names representing categorical
        features.
      numerical_cols: A list of column names representing numerical features.

  Returns:
      A tuple containing:
          - The modified DataFrame with encoded features.
          - A list of indices indicating the positions of categorical features.
          - A list of dimensions (number of unique values) for each categorical
          feature.
          - The total input dimension (number of features after encoding).
  """
  categorical_columns = []
  categorical_dims = {}
  for col in data.columns:
    if col in categorical_cols:
      print_only_by_main_process(f"{col} ==> {data[col].nunique()}")
      l_enc = preprocessing.LabelEncoder()
      data.loc[:, col] = data.loc[:, col].fillna("VV_likely")
      data.loc[:, col] = l_enc.fit_transform(data[col].values)
      categorical_columns.append(col)
      categorical_dims[col] = len(l_enc.classes_)
    elif col in numerical_cols:
      data[col] = data[col].fillna(data[col].mean())

  input_dim = len(categorical_cols) + len(numerical_cols)
  features = categorical_cols + numerical_cols
  cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
  cat_dims = [
      categorical_dims[f]
      for _, f in enumerate(features)
      if f in categorical_columns
  ]

  return data, cat_idxs, cat_dims, input_dim


def get_train_and_test_splits(
    args: omegaconf.DictConfig,
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Get train, test and validation splits.

  Args:
    args: An OmegaConf.DictConfig object
    data: Dataframe.

  Returns:
    train_data: Train data.
    test_data: Test data.
    valid_data: Validation data.
  """
  if args.test_ratio > 0:
    total_product_ids = list(set(data["asin"].values))
    np.random.seed(2022)  # Do not change this seed
    rand_idx = np.random.randint(
        low=0,
        high=len(total_product_ids),
        size=int(args.test_ratio * len(total_product_ids)),
    )
    test_rand_idx = rand_idx[: len(rand_idx) // 2 + 1]
    valid_rand_idx = rand_idx[len(rand_idx) // 2 + 1 :]
    print_only_by_main_process(
        f"# of validation indices: {len(valid_rand_idx)}, first 10 of them:"
        f" {valid_rand_idx[:10]}"
    )
    print_only_by_main_process(
        f"# of test indices: {len(test_rand_idx)}, first 10 of them:"
        f" {test_rand_idx[:10]}"
    )

    test_product_ids = [total_product_ids[idx] for idx in test_rand_idx]
    valid_product_ids = [total_product_ids[idx] for idx in valid_rand_idx]

    def categorise(row):
      if row["asin"] in test_product_ids:
        return "test"
      elif row["asin"] in valid_product_ids:
        return "valid"
      else:
        return "train"

    data["split"] = data.apply(categorise, axis=1)
    print_df_stats(data, "Loading the entire CSV data file")

    # Remove the test subjects from the pretraining data
    print_only_by_main_process(f"data all: {data.shape}")
    train_data = data[data["split"] == "train"]
    test_data = data[data["split"] == "test"]
    valid_data = data[data["split"] == "valid"]

    print_only_by_main_process(f"train data: {train_data.shape}")
    print_only_by_main_process(f"valid data: {valid_data.shape}")
    print_only_by_main_process(f"test data: {test_data.shape}")

    if args.sub_samples > 0:
      n_valid = int(args.eval_batch_size)
      n_test = int(args.test_batch_size)
      n_train = int(args.sub_samples)

      train_data = train_data.sample(n=n_train, random_state=args.seed)
      print_only_by_main_process(
          f"Dataset size for sub_sampled train: {train_data.shape}"
      )
      valid_data = valid_data.sample(n=n_valid, random_state=args.seed)
      print_only_by_main_process(
          f"Dataset size for sub_sampled valid: {valid_data.shape}"
      )  # pylint: disable=line-too-long
      test_data = test_data.sample(n=n_test, random_state=args.seed)
      print_only_by_main_process(
          f"Dataset size for sub_sampled test: {test_data.shape}"
      )

    train_data = drop_last(train_data, batch_size=args.train_batch_size)
    valid_data = drop_last(valid_data, batch_size=args.eval_batch_size)
    test_data = drop_last(test_data, batch_size=args.test_batch_size)

    print_only_by_main_process(f"train data: {train_data.shape}")
    print_only_by_main_process(f"valid data: {valid_data.shape}")
    print_only_by_main_process(f"test data: {test_data.shape}")

  else:
    train_data = drop_last(data, batch_size=args.train_batch_size)
    valid_data = drop_last(data, batch_size=args.eval_batch_size)
    test_data = drop_last(data, batch_size=args.test_batch_size)

  return train_data, test_data, valid_data


def drop_last(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
  """Drop last row to make sure the number of samples is divisible by the batch size.

  Args:
    df: Dataframe.
    batch_size: Batch size.

  Returns:
    df: Dataframe.
  """
  # hack to make sure the number of samples is divisible by the batch size
  number_of_rows_to_remove = len(df) % batch_size

  if number_of_rows_to_remove > 0:
    df = df.iloc[:-number_of_rows_to_remove]
  return df


def get_amazon_transforms(
    args: omegaconf.DictConfig,
) -> Tuple[transforms.Compose, transforms.Compose]:
  """Get Amazon transforms.

  Args:
    args: An OmegaConf.DictConfig object

  Returns:
    train_transform: Train transform.
    test_transform: Test transform.
  """
  normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(args.image_crop),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
  ])
  test_transform = transforms.Compose([
      transforms.Resize(args.image_size),
      transforms.CenterCrop(args.image_crop),
      transforms.ToTensor(),
      normalize,
  ])

  return train_transform, test_transform
