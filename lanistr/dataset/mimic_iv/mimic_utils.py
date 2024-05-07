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

import functools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import omegaconf
import pandas as pd
import torch
from utils.common_utils import print_df_stats
from utils.common_utils import print_only_by_main_process


# From https://github.com/nyuad-cai/MedFuse/blob/main/ehr_utils/preprocessing.py
class Discretizer:
  """Discretizer class.

  Attributes:
    _id_to_channel: id to channel
    _channel_to_id: channel to id
    _is_categorical_channel: is categorical channel
    _possible_values: possible values
    _normal_values: normal values
    _header: header
    _timestep: timestep
    _store_masks: store masks
    _start_time: start time
    _impute_strategy: impute strategy
    _done_count: done count
    _empty_bins_sum: empty bins sum
    _unused_data_sum: unused data sum
  """

  def __init__(
      self,
      timestep=0.8,
      store_masks=True,
      impute_strategy='zero',
      start_time='zero',
      config_path='./data/MIMIC-IV-V2.2/discretizer_config.json',
  ):

    with open(config_path) as f:
      config = json.load(f)
      self._id_to_channel = config['id_to_channel']
      self._channel_to_id = dict(
          zip(self._id_to_channel, range(len(self._id_to_channel)))
      )
      self._is_categorical_channel = config['is_categorical_channel']
      self._possible_values = config['possible_values']
      self._normal_values = config['normal_values']

    self._header = ['Hours'] + self._id_to_channel
    self._timestep = timestep
    self._store_masks = store_masks
    self._start_time = start_time
    self._impute_strategy = impute_strategy

    # for statistics
    self._done_count = 0
    self._empty_bins_sum = 0
    self._unused_data_sum = 0

  def transform(
      self, x_data: np.ndarray, header: List[str] = None, end: float = None
  ) -> Tuple[np.ndarray, List[str]]:
    """Transform the data.

    Args:
      x_data: data
      header: header
      end: end

    Returns:
      data: data
      new_header: new header
    """
    if header is None:
      header = self._header
    assert header[0] == 'Hours'
    eps = 1e-6

    n_channels = len(self._id_to_channel)
    ts = [float(row[0]) for row in x_data]
    for i in range(len(ts) - 1):
      assert ts[i] < ts[i + 1] + eps

    if self._start_time == 'relative':
      first_time = ts[0]
    elif self._start_time == 'zero':
      first_time = 0
    else:
      raise ValueError('start_time is invalid')

    if end is None:
      max_hours = max(ts) - first_time
    else:
      max_hours = end - first_time

    n_bins = int(max_hours / self._timestep + 1.0 - eps)

    cur_len = 0
    begin_pos = [0 for _ in range(n_channels)]
    end_pos = [0 for _ in range(n_channels)]
    for i in range(n_channels):
      channel = self._id_to_channel[i]
      begin_pos[i] = cur_len
      if self._is_categorical_channel[channel]:
        end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
      else:
        end_pos[i] = begin_pos[i] + 1
      cur_len = end_pos[i]

    data = np.zeros(shape=(n_bins, cur_len), dtype=float)
    mask = np.zeros(shape=(n_bins, n_channels), dtype=int)
    original_value = [['' for _ in range(n_channels)] for _ in range(n_bins)]
    total_data = 0
    unused_data = 0

    def write(data, bin_id, channel, value, begin_pos):
      channel_id = self._channel_to_id[channel]
      if self._is_categorical_channel[channel]:
        category_id = self._possible_values[channel].index(value)
        n_values = len(self._possible_values[channel])
        one_hot = np.zeros((n_values,))
        one_hot[category_id] = 1
        for pos in range(n_values):
          data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
      else:
        data[bin_id, begin_pos[channel_id]] = float(value)

    for row in x_data:
      t = float(row[0]) - first_time
      if t > max_hours + eps:
        continue
      bin_id = int(t / self._timestep - eps)
      assert 0 <= bin_id < n_bins

      for j in range(1, len(row)):
        if not row[j]:
          continue
        channel = header[j]
        channel_id = self._channel_to_id[channel]

        total_data += 1
        if mask[bin_id][channel_id] == 1:
          unused_data += 1
        mask[bin_id][channel_id] = 1

        write(data, bin_id, channel, row[j], begin_pos)
        original_value[bin_id][channel_id] = row[j]

    # impute missing values

    if self._impute_strategy not in [
        'zero',
        'normal_value',
        'previous',
        'next',
    ]:
      raise ValueError('impute strategy is invalid')

    if self._impute_strategy in ['normal_value', 'previous']:
      prev_values = [[] for _ in range(len(self._id_to_channel))]
      for bin_id in range(n_bins):
        for channel in self._id_to_channel:
          channel_id = self._channel_to_id[channel]
          if mask[bin_id][channel_id] == 1:
            prev_values[channel_id].append(original_value[bin_id][channel_id])
            continue
          if self._impute_strategy == 'normal_value':
            imputed_value = self._normal_values[channel]  # pytype: disable=name-error
            write(data, bin_id, channel, imputed_value, begin_pos)
          elif self._impute_strategy == 'previous':
            if not prev_values[channel_id]:
              imputed_value = self._normal_values[channel]
            else:
              imputed_value = prev_values[channel_id][-1]
            write(data, bin_id, channel, imputed_value, begin_pos)
        else:
          raise ValueError('impute strategy is invalid')

    if self._impute_strategy == 'next':
      prev_values = [[] for _ in range(len(self._id_to_channel))]
      for bin_id in range(n_bins - 1, -1, -1):
        for channel in self._id_to_channel:
          channel_id = self._channel_to_id[channel]
          if mask[bin_id][channel_id] == 1:
            prev_values[channel_id].append(original_value[bin_id][channel_id])
            continue
          if not prev_values[channel_id]:
            imputed_value = self._normal_values[channel]
          else:
            imputed_value = prev_values[channel_id][-1]
          write(data, bin_id, channel, imputed_value, begin_pos)

    empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(n_bins)])
    self._done_count += 1
    self._empty_bins_sum += empty_bins / (n_bins + eps)
    self._unused_data_sum += unused_data / (total_data + eps)

    if self._store_masks:
      data = np.hstack([data, mask.astype(np.float32)])

    # create new header
    new_header = []
    for channel in self._id_to_channel:
      if self._is_categorical_channel[channel]:
        values = self._possible_values[channel]
        for value in values:
          new_header.append(channel + '->' + value)
      else:
        new_header.append(channel)

    if self._store_masks:
      for _, channel in enumerate(self._id_to_channel):
        new_header.append('mask->' + channel)

    new_header = ','.join(new_header)

    return (data, new_header)

  def print_statistics(self) -> None:
    """Print statistics."""
    print('statistics of discretizer:')
    print('\tconverted {} examples'.format(self._done_count))
    print(
        '\taverage unused data = {:.2f} percent'.format(
            100.0 * self._unused_data_sum / self._done_count
        )
    )
    print(
        '\taverage empty  bins = {:.2f} percent'.format(
            100.0 * self._empty_bins_sum / self._done_count
        )
    )


# From https://github.com/nyuad-cai/MedFuse/blob/main/ehr_utils/preprocessing.py
class Normalizer:
  """Normalizer class.

  Attributes:
    _means: means
    _stds: stds
    _fields: fields
    _sum_x: sum x
    _sum_sq_x: sum sq x
    _count: count
  """

  def __init__(self, fields=None):
    """Initialize the normalizer.

    Args:
      fields: fields
    """
    self._means = None
    self._stds = None
    self._fields = None
    if fields is not None:
      self._fields = list(fields)

    self._sum_x = None
    self._sum_sq_x = None
    self._count = 0

  def _feed_data(self, x: np.ndarray) -> None:
    """Feed data.

    Args:
      x: data
    """
    x = np.array(x)
    self._count += x.shape[0]
    if self._sum_x is None:
      self._sum_x = np.sum(x, axis=0)
      self._sum_sq_x = np.sum(x**2, axis=0)
    else:
      self._sum_x += np.sum(x, axis=0)
      self._sum_sq_x += np.sum(x**2, axis=0)

  def _save_params(self, save_file_path: str) -> None:
    """Save parameters.

    Args:
      save_file_path: save file path
    """
    eps = 1e-7
    with open(save_file_path, 'wb') as save_file:
      n = self._count
      self._means = 1.0 / n * self._sum_x
      self._stds = np.sqrt(
          1.0
          / (n - 1)
          * (
              self._sum_sq_x
              - 2.0 * self._sum_x * self._means
              + n * self._means**2
          )
      )
      self._stds[self._stds < eps] = eps
      data = {'means': self._means, 'stds': self._stds}
      data = pd.DataFrame(data)
      data.to_csv(save_file, index=False)

  def load_params(self, load_file_path: str) -> None:
    """Load parameters.

    Args:
      load_file_path: load file path
    """
    df = pd.read_csv(
        load_file_path, dtype={'means': float, 'stds': float}, encoding='utf-8'
    )
    self._means = np.array(df['means'])
    self._stds = np.array(df['stds'])

  def transform(self, x: np.ndarray) -> np.ndarray:
    """Transform the data.

    Args:
      x: data

    Returns:
      ret: transformed data
    """
    if self._fields is None:
      fields = range(x.shape[1])
    else:
      fields = self._fields
    ret = 1.0 * x
    for col in fields:
      ret[:, col] = (x[:, col] - self._means[col]) / self._stds[col]
    return ret


def load_unimodal_data(
    args: omegaconf.DictConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Load unimodal data (image, text, time series).

  Args:
      args: arguments

  Returns:
      image_records: pd.Dataframe
      text_records: pd.Dataframe
      time_records: pd.Dataframe
  """

  if not os.path.exists(args.unimodal_data_dir):
    os.makedirs(args.unimodal_data_dir)

  image_records, text_records = None, None

  if not os.path.exists(os.path.join(args.unimodal_data_dir, 'time.csv')):
    print('Creating unimodal data')
    if args.image:
      image_records = pd.read_csv(
          os.path.join(
              args.root_data_dir, 'mimic-cxr', '2.0.0', 'cxr-record-list.csv'
          )
      )
      image_records['image'] = image_records['path'].str.replace('.dcm', '.jpg')
      image_records.to_csv(
          os.path.join(args.unimodal_data_dir, 'image.csv'), index=False
      )
      print_df_stats(image_records, 'image')

    if args.text:
      text_records = pd.read_csv(
          os.path.join(
              args.root_data_dir,
              'mimic-iv-note',
              '2.2',
              'note',
              'discharge.csv',
          )
      )
      text_columns_to_keep = ['note_id', 'subject_id', 'text']
      text_records = text_records[text_columns_to_keep]
      text_records['text'] = text_records['text'].apply(
          lambda x: x.replace('\n', ' ')
      )
      text_records.to_csv(
          os.path.join(args.unimodal_data_dir, 'text.csv'), index=False
      )
      print_df_stats(text_records, 'text')

    # We read time records at this stage regardless of whether args.time is True
    # or False because we need to create the train/val/splits based of MedFuse
    # paper
    # We preprocess the time series and save them as csv files
    time_records = preprocess_timeseries(args)
    time_records.to_csv(
        os.path.join(args.unimodal_data_dir, 'time.csv'), index=False
    )
    print_df_stats(time_records, 'time')

  else:
    print_only_by_main_process(
        'Loading already existing unimodal CSV files from'
        f' {args.unimodal_data_dir}'
    )
    if args.image:
      image_records = pd.read_csv(
          os.path.join(args.unimodal_data_dir, 'image.csv')
      )
      print_df_stats(image_records, 'image')
    if args.text:
      text_records = pd.read_csv(
          os.path.join(args.unimodal_data_dir, 'text.csv')
      )
      text_records['text'] = text_records['text'].apply(
          lambda x: x.replace('\n', ' ')
      )
      print_df_stats(text_records, 'text')

    # We read time records at this stage regardless of whether args.time is True
    # or False because we need to create the train/val/splits based of MedFuse
    # paper
    time_records = pd.read_csv(os.path.join(args.unimodal_data_dir, 'time.csv'))
    print_df_stats(time_records, 'time')

  return image_records, text_records, time_records


def find_timeseries(args: omegaconf.DictConfig, row: pd.DataFrame) -> str:
  """Find timeseries data for a given row.

  Args:
      args: arguments
      row: row of the dataframe

  Returns:
      split: split of the timeseries data
  """
  stay_filename = row.stay
  train_stay_filename = os.path.join(args.task_data_dir, 'train', stay_filename)
  test_stay_filename = os.path.join(args.task_data_dir, 'test', stay_filename)
  if os.path.exists(train_stay_filename):
    return os.path.join(args.task_data_dir, 'train')
  elif os.path.exists(test_stay_filename):
    return os.path.join(args.task_data_dir, 'test')


def get_normalizer_and_discritizer(
    args: omegaconf.DictConfig,
) -> Tuple[Normalizer, Discretizer]:
  """Get normalizer and discretizer.

  Args:
      args: arguments

  Returns:
      normalizer: normalizer
      discretizer: discretizer
  """
  discretizer = Discretizer(
      timestep=float(args.timestep),
      store_masks=True,
      impute_strategy=args.impute_strategy,
      start_time=args.start_time,
      config_path=args.discretizer_config_path,
  )
  normalizer = get_normalizer(args, discretizer)
  return normalizer, discretizer


def read_by_file_name(
    row: pd.DataFrame, dataset_dir: str, time_bound: Optional[float] = None
) -> Dict[str, Any]:
  """Read timeseries data by file name.

  Args:
      row: row of the dataframe
      dataset_dir: directory of the timeseries data
      time_bound: time bound of the timeseries data

  Returns:
      ret: timeseries data
  """

  ts_filename = row['stay']  # .values[0]
  t = row['period_length']  # .values[0]
  y = row['y_true']  # .values[0]
  stay_id = row['stay_id']  # .values[0]

  (seq, header) = _read_timeseries(
      ts_filename, dataset_dir, time_bound=time_bound
  )

  return {
      'X': seq,
      't': t,
      'y': y,
      'stay_id': stay_id,
      'header': header,
      'name': ts_filename,
  }


def _read_timeseries(
    ts_filename: str, dataset_dir: str, time_bound: Optional[float] = None
) -> Tuple[np.ndarray, List[str]]:
  """Read timeseries data from a file.

  Args:
      ts_filename: filename of the timeseries data
      dataset_dir: directory of the timeseries data
      time_bound: time bound of the timeseries data

  Returns:
      ret: timeseries data
      header: header of the timeseries data
  """
  ret = []
  with open(os.path.join(dataset_dir, ts_filename), 'r') as tsfile:
    header = tsfile.readline().strip().split(',')
    assert header[0] == 'Hours'
    for line in tsfile:
      mas = line.strip().split(',')
      if time_bound is not None:
        t = float(mas[0])
        if t > time_bound + 1e-6:
          break
      ret.append(np.array(mas))
  return (np.stack(ret), header)


def preprocess_timeseries(args: omegaconf.DictConfig) -> pd.DataFrame:
  """Preprocess timeseries data.

  Args:
      args: arguments

  Returns:
      time_records: pd.Dataframe
  """
  normalizer, discretizer = get_normalizer_and_discritizer(args)
  period_length = args.timeseries_max_seq_len

  def create_timeseries(row):
    dataset_dir = find_timeseries(args, row)
    ret = read_by_file_name(row, dataset_dir, time_bound=None)
    data = ret['X']
    ts = ret['t'] if ret['t'] > 0.0 else period_length
    data = discretizer.transform(data, end=ts)[0]
    if normalizer is not None:
      data = normalizer.transform(data)
    filename = os.path.join(dataset_dir, row['stay'].split('.')[0] + '.npy')
    np.save(filename, data)
    return filename

  # Adding subject_id to time records - all stays are from ICU stays (not ED)
  time_records_train = pd.read_csv(
      os.path.join(args.task_data_dir, 'train_listfile.csv')
  )
  time_records_train['split'] = 'train'
  time_records_test = pd.read_csv(
      os.path.join(args.task_data_dir, 'test_listfile.csv')
  )
  time_records_test['split'] = 'test'
  time_records_val = pd.read_csv(
      os.path.join(args.task_data_dir, 'val_listfile.csv')
  )
  time_records_val['split'] = 'valid'

  time_records = pd.concat(
      [time_records_train, time_records_val, time_records_test]
  )
  all_stays = pd.read_csv(f'{args.preprocessed_data_dir}/root/all_stays.csv')
  icu_stay_id_to_subject_id = dict(
      zip(all_stays['stay_id'], all_stays['subject_id'])
  )
  time_records['subject_id'] = time_records['stay_id'].map(
      icu_stay_id_to_subject_id
  )

  time_records['timeseries'] = time_records.apply(create_timeseries, axis=1)

  return time_records


def load_pretraining_multimodal_data(
    args: omegaconf.DictConfig,
    unimodal_dataframes: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Dict[str, Any]:
  """Load pretraining multimodal data.

  Args:
      args: arguments
      unimodal_dataframes: unimodal dataframes

  Returns:
      data: pd.Dataframe
  """
  image_records, text_records, time_records = unimodal_dataframes

  tabular_data_information = {
      'input_dim': None,
      'cat_idxs': None,
      'cat_dims': None,
      'feature_names': None,
  }

  columns = ['subject_id', 'split']
  df_list = []
  if args.image:
    columns.append('image')
    df_list.append(image_records)
  if args.text:
    columns.append('text')
    df_list.append(text_records)

  columns += ['timeseries', 'stay_id', 'stay']
  df_list.append(time_records)

  data = merge_and_remove_duplicates(df_list=df_list, columuns=columns)
  print_only_by_main_process(f"after merge: {data['split'].value_counts()}")

  if args.sub_samples > 0:
    data = data.sample(n=args.sub_samples, random_state=args.seed)
    data = data.reset_index(drop=True)

  train_data = data[data['split'] == 'train']
  test_data = data[data['split'] == 'test']

  train_data = drop_last(
      df=train_data, args=args, batch_size=args.train_batch_size
  )
  test_data = drop_last(
      df=test_data, args=args, batch_size=args.test_batch_size
  )

  print_df_stats(train_data, 'Final size of unlabeled multimodal training data')
  print_df_stats(test_data, 'Final size of unlabeled multimodal test data')

  return {
      'train': train_data,
      'valid': test_data,
      'test': test_data,
      'tabular_data_information': tabular_data_information,
  }


def merge_and_remove_duplicates(
    df_list: List[pd.DataFrame], columuns: List[str]
) -> pd.DataFrame:
  """Merge and remove duplicates.

  Args:
      df_list: list of pd.Dataframe
      columuns: columns to keep

  Returns:
      df_merged: pd.Dataframe
  """
  df_merged = merge_data(df_list, how='outer')
  df_merged = df_merged[columuns]
  print_df_stats(df_merged, 'df_merged after merging')
  df_merged = df_merged.drop_duplicates()
  print_df_stats(df_merged, 'df_merged after dropping duplicates')
  return df_merged


def merge_data(df: List[pd.DataFrame], how: str) -> pd.DataFrame:
  """Merge data using pandas.

  Args:
      df: list of pd.Dataframe
      how: how to merge

  Returns:
      df_merged: pd.Dataframe
  """
  df_merged = functools.reduce(
      lambda left, right: pd.merge(left, right, how=how), df
  )
  return df_merged


def drop_last(
    df: pd.DataFrame, args: omegaconf.DictConfig, batch_size: int
) -> pd.DataFrame:
  """Drop last rows if required.

  Args:
      df: pd.Dataframe
      args: arguments
      batch_size: batch size

  Returns:
      df: pd.Dataframe
  """
  # hack to make sure the number of samples is divisible by the batch size
  fixed_batch_size = int(batch_size // args.ngpus_per_node)
  number_of_rows_to_remove = len(df) % fixed_batch_size

  if number_of_rows_to_remove > 0:
    df = df.iloc[:-number_of_rows_to_remove]
  return df


def load_labeled_multimodal_data(args: omegaconf.DictConfig) -> Dict[str, Any]:
  """Load labeled multimodal data.

  Args:
      args: arguments

  Returns:
      data: pd.Dataframe
  """
  image_records, text_records, time_records = load_unimodal_data(args)

  cxr_metadata = pd.read_csv(
      f'{args.image_data_dir}/mimic-cxr-2.0.0-metadata.csv'
  )
  icu_stay_metadata = pd.read_csv(
      f'{args.preprocessed_data_dir}/root/all_stays.csv'
  )
  columns = ['subject_id', 'stay_id', 'intime', 'outtime']

  # # only common subjects with both icu stay and an xray
  cxr_merged_icustays = cxr_metadata.merge(
      icu_stay_metadata[columns], how='inner', on='subject_id'
  )

  # combine study date time
  cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(
      lambda x: f'{int(float(x)):06}'
  )
  cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(
      cxr_merged_icustays['StudyDate'].astype(str)
      + ' '
      + cxr_merged_icustays['StudyTime'].astype(str),
      format='%Y%m%d %H%M%S',
  )

  cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
  cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)

  end_time = cxr_merged_icustays.intime + pd.DateOffset(
      hours=args.timeseries_max_seq_len
  )
  cxr_merged_icustays_during = cxr_merged_icustays.loc[
      (cxr_merged_icustays.StudyDateTime >= cxr_merged_icustays.intime)
      & ((cxr_merged_icustays.StudyDateTime <= end_time))
  ]

  # select cxrs with the ViewPosition == 'AP
  cxr_merged_icustays_ap = cxr_merged_icustays_during[
      cxr_merged_icustays_during['ViewPosition'] == 'AP'
  ]

  groups = cxr_merged_icustays_ap.groupby('stay_id')

  groups_selected = []
  for group in groups:
    # select the latest cxr for the icu stay
    selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
    groups_selected.append(selected)
  data = pd.concat(groups_selected, ignore_index=True)
  print_df_stats(data, 'data')

  columns_to_keep = ['dicom_id', 'subject_id', 'study_id', 'stay_id']
  data = data[columns_to_keep]

  if args.image:
    dicom_id_to_path = dict(
        zip(image_records['dicom_id'], image_records['path'])
    )
    data['image'] = data['dicom_id'].apply(lambda x: dicom_id_to_path[x])
    data['image'] = data['image'].str.replace('.dcm', '.jpg')
    print_df_stats(data, "data['image']")

  if args.text:
    data = data[data['subject_id'].isin(text_records['subject_id'].values)]
    subject_id_to_text_note = dict(
        zip(text_records['subject_id'], text_records['text'])
    )
    data['text'] = data['subject_id'].apply(
        lambda x: subject_id_to_text_note[x]
    )
  print_df_stats(data, "data['text']")

  data = data[data['stay_id'].isin(time_records['stay_id'].values)]
  stay_id_to_time = dict(
      zip(time_records['stay_id'], time_records['timeseries'])
  )
  data['timeseries'] = data['stay_id'].apply(lambda x: stay_id_to_time[x])
  print_df_stats(data, "data['timeseries']")

  stay_id_to_split = dict(zip(time_records['stay_id'], time_records['split']))
  data['split'] = data['stay_id'].apply(lambda x: stay_id_to_split[x])
  print_df_stats(data, 'split')

  stay_id_to_label = dict(zip(time_records['stay_id'], time_records['y_true']))
  data['y_true'] = data['stay_id'].apply(lambda x: stay_id_to_label[x])
  print_df_stats(data, 'labeled_multimodal_data')

  if args.sub_samples > 0:
    data = data.sample(n=args.sub_samples, random_state=args.seed)
    data = data.reset_index(drop=True)

  train_data = data[data['split'] == 'train']
  print_df_stats(train_data, 'labeled_multimodal_train_data')
  test_data = data[data['split'] == 'test'][:617]
  print_df_stats(test_data, 'labeled_multimodal_test_data')

  tabular_data_information = {
      'input_dim': None,
      'cat_idxs': None,
      'cat_dims': None,
      'feature_names': None,
  }

  return {
      'train': train_data,
      'valid': test_data,
      'test': test_data,
      'tabular_data_information': tabular_data_information,
  }


def get_normalizer(
    args: omegaconf.DictConfig, discretizer: Discretizer
) -> Normalizer:
  """Load the normalizer.

  Args:
    args: arguments
    discretizer: Discretizer

  Returns:
    Normalizer.
  """
  discretizer_header = discretizer.transform(read_a_sample_file(args))[1]
  discretizer_header = discretizer_header.split(',')[1:]
  cont_channels = [
      i for (i, x) in enumerate(discretizer_header) if x.find('->') == -1
  ]

  normalizer = Normalizer(fields=cont_channels)
  normalizer.load_params(args.normalizer_file)
  return normalizer


def read_a_sample_file(args: omegaconf.DictConfig) -> np.ndarray:
  """Read a sample file.

  Args:
      args: arguments

  Returns:
      ret: timeseries data
  """
  path = f'{args.task_data_dir}/train/14991576_episode3_timeseries.csv'

  ret = []
  with open(path, 'r') as tsfile:
    header = tsfile.readline().strip().split(',')
    assert header[0] == 'Hours'
    for line in tsfile:
      mas = line.strip().split(',')
      ret.append(np.array(mas))
  return np.stack(ret)


def padding_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
  """Used to mask padded positions.

  Creates a (batch_size, max_len) boolean mask from a tensor of sequence
  lengths, where 1 means keep element at this position (time step)

  Args:
    lengths: tensor of sequence lengths
    max_len: maximum length of the sequence

  Returns:
    mask: boolean mask
  """
  batch_size = lengths.numel()
  max_len = (
      max_len or lengths.max_val()
  )  # trick works because of overloading of 'or' operator for non-boolean types
  return (
      torch.arange(0, max_len, device=lengths.device)
      .type_as(lengths)
      .repeat(batch_size, 1)
      .lt(lengths.unsqueeze(1))
  )
