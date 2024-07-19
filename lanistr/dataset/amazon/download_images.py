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

import argparse
import gzip
import json
import os
from typing import Dict, List, Optional

import omegaconf
import pandas as pd
import requests
import tqdm


def load_and_clean_meta_data(
    args: omegaconf.DictConfig,
) -> pd.DataFrame:  # pylint: disable=line-too-long
  """Loads and cleans product metadata from a compressed JSON file.

  This function performs the following steps:

  1. Loads metadata from a gzipped JSON file specified by `args.category`.
  2. Converts the loaded data into a pandas DataFrame.
  3. Fills missing values (NaN) with empty strings.
  4. Removes rows where the 'title' field contains unformatted HTML content
  (identified by the presence of "getTime").
  5. Drops duplicate rows based on the 'asin' column.
  6. Resets the DataFrame index.

  Args:
      args: An OmegaConf.DictConfig object

  Returns:
      A pandas DataFrame containing the cleaned metadata.
  """
  ### load the meta data
  meta_data = read_gzip(name=f'meta_{args.category}', args=args)

  df = pd.DataFrame.from_dict(meta_data)
  ### remove rows with unformatted title (i.e. some 'title' may still contain
  ### html style content)
  df_ = df.fillna('')
  metadata = df_[
      ~df_.title.str.contains('getTime')
  ]  # filter those unformatted rows
  metadata = metadata.drop_duplicates(subset=['asin']).reset_index(drop=True)
  return metadata


def read_gzip(name: str, args: omegaconf.DictConfig) -> List[Dict[str, str]]:
  """Reads a gzipped file and returns a list of JSON objects.

  Args:
      name: The name of the file to read.
      args: An OmegaConf.DictConfig object

  Returns:
      A list of JSON objects.
  """
  path = os.path.join(args.root_data_dir, args.category, f'{name}.json.gz')
  data = []
  with gzip.open(path) as f:
    for l in f:
      data.append(json.loads(l.strip()))
  return data


def load_data(args: omegaconf.DictConfig) -> pd.DataFrame:
  """Loads the main data from a compressed JSON file.

  This function performs the following steps:

  1. Loads the main data from a gzipped JSON file specified by `args.category`.
  2. Converts the loaded data into a pandas DataFrame.
  3. Resets the DataFrame index.

  Args:
      args: An OmegaConf.DictConfig object

  Returns:
      A pandas DataFrame containing the loaded data.
  """
  ### load the main data
  data = read_gzip(name=f'{args.category}', args=args)
  df = pd.DataFrame.from_dict(data)
  df = df.reset_index(drop=True)
  return df


def get_reviews(
    row: pd.Series,
    index: int,
    nan_indices_summary: List[int],
    nan_indices_review_text: List[int],
) -> Optional[str]:
  """Extracts and cleans the review text from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.
      index: The index of the row in the data.
      nan_indices_summary: A list of indices where the 'summary' field is NaN.
      nan_indices_review_text: A list of indices where the 'reviewText' field is
        NaN.

  Returns:
      - A string containing the extracted review text, if successful.
      - None if the review text cannot be extracted.
  """
  review = ''
  second_word = False
  if nan_indices_summary[index] == 0:
    review += row['summary'].values[0]
    second_word = True
  if nan_indices_review_text[index] == 0:
    if second_word:
      review += '. '
    review += row['reviewText'].values[0]

  if not review:
    review = None
  return review


def get_review_votes(
    row: pd.Series, index: int, nan_indices_votes: List[int]
) -> int:
  """Extracts and cleans the review vote from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.
      index: The index of the row in the data.
      nan_indices_votes: A list of indices where the 'vote' field is NaN.

  Returns:
      - An integer representing the extracted vote, if successful.
      - 0 if the vote field is empty or cannot be converted to an integer.
  """
  vote = 0
  if nan_indices_votes[index] == 0:
    vote = int(row['vote'].values[0])
  return vote


def get_product_brands(
    meta_row: pd.Series, meta_data_exists: bool
) -> Optional[str]:
  """Extracts and cleans the product brand from metadata, if available.

  Args:
      meta_row: A pandas Series containing the metadata for a product, including
        a 'brand' field if `meta_data_exists` is True.
      meta_data_exists: A boolean indicating whether metadata is present.

  Returns:
      - A string containing the extracted product brand, if successful.
      - None if the brand field is empty or cannot be converted to a string.
      - None if `meta_data_exists` is False (no metadata available).
  """
  if meta_data_exists:
    if meta_row['brand'].values[0]:
      brand_str = meta_row['brand'].values[0]
      product_brand = brand_str
    else:
      product_brand = None
  else:
    product_brand = None
  return product_brand


def download_and_save_image(
    image_data_dir: str, urls: List[str], index: int
) -> Optional[str]:
  """Downloads and saves the image from a URL.

  Args:
      image_data_dir: The directory where the image will be saved.
      urls: A list of URLs to download the image from.
      index: The index of the image in the list of URLs.

  Returns:
      The filename of the saved image, or None if the image could not be
      downloaded or saved.
  """
  image_filename = f"{index}_{urls[0].split('.jpg')[0].split('/')[-1]}.jpg"
  image_path = os.path.join(image_data_dir, image_filename)
  if not os.path.exists(image_path):
    with open(image_path, 'wb') as f:
      f.write(requests.get(urls[0]).content)

  if os.path.exists(image_path) and os.path.getsize(image_path) > 10:
    return image_filename
  else:
    return None


def get_product_prices(
    meta_data_exists: bool, meta_row: pd.Series  # pylint: disable=line-too-long
) -> Optional[float]:  # pylint: disable=line-too-long
  """Extracts and cleans the product price from metadata, if available.

  Args:
      meta_data_exists: A boolean indicating whether metadata is present.
      meta_row: A pandas Series containing the metadata for a product, including
        a 'price' field if `meta_data_exists` is True.

  Returns:
      - A float representing the extracted price, if successful.
      - 0 if the price field is empty or cannot be converted to a float.
      - None if `meta_data_exists` is False (no metadata available).
  """
  if meta_data_exists:
    if meta_row['price'].values[0]:
      price_str = meta_row['price'].values[0]
      try:
        price_flt = float(price_str.replace('$', ''))
        return price_flt
      except ValueError:
        return 0
    else:
      return 0
  else:
    return None


def get_review_names(
    row: pd.Series, index: int, nan_indices_reviewer_names: List[int]
) -> Optional[str]:
  """Extracts and cleans the reviewer name from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.
      index: The index of the row in the data.
      nan_indices_reviewer_names: A list of indices where the 'reviewerName'
        field is NaN.

  Returns:
      - A string containing the extracted reviewer name, if successful.
      - None if the reviewer name cannot be extracted.
  """
  review_name = None
  if nan_indices_reviewer_names[index] == 0:
    review_name = row['reviewerName'].values[0]
  return review_name


def get_review_years(row: pd.Series) -> int:
  """Extracts and cleans the review year from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.

  Returns:
      - An integer representing the extracted review year, if successful.
      - 0 if the review year field is empty or cannot be converted to an
      integer.
  """
  year = int(row['reviewTime'].values[0].split(',')[1])
  return year


def get_unix_review_time(row: pd.Series) -> int:
  """Extracts and cleans the unix review time from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.

  Returns:
      - An integer representing the extracted unix review time, if successful.
      - 0 if the unix review time field is empty or cannot be converted to an
      integer.
  """
  return int(row['unixReviewTime'].values[0])


def get_labels(row: pd.Series) -> int:
  """Extracts and cleans the label from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.

  Returns:
      - An integer representing the extracted label, if successful.
      - 0 if the label field is empty or cannot be converted to an integer.
  """
  return int(row['overall'].values[0]) - 1


def get_reviewer_ids(row: pd.Series) -> str:
  """Extracts and cleans the reviewer id from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.

  Returns:
      - A string representing the extracted reviewer id, if successful.
      - None if the reviewer id field is empty or cannot be converted to a
      string.
  """
  return row['reviewerID'].values[0]


def get_verified_reviews(row: pd.Series) -> int:
  """Extracts and cleans the verified review from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.

  Returns:
      - An integer representing the extracted verified review, if successful.
      - 0 if the verified review field is empty or cannot be converted to an
      integer.
  """
  return row['verified'].values[0].astype(int)


def get_product_ids(row: pd.Series) -> str:
  """Extracts and cleans the product id from a row of data.

  Args:
      row: A pandas Series containing the data for a single review.

  Returns:
      - A string representing the extracted product id, if successful.
      - None if the product id field is empty or cannot be converted to a
      string.
  """
  return row['asin'].values[0]


def main():

  parser = argparse.ArgumentParser(description='Downloading APR2018 Images')
  parser.add_argument('--category', type=str, default='Office_Products')
  parser.add_argument('--seed', type=int, default=2022)
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--root_data_dir', type=str, default='./data/APR2018/')

  args = parser.parse_args()

  image_data_dir = os.path.join(args.root_data_dir, args.category, 'images')

  data = load_data(args)
  meta_data = load_and_clean_meta_data(args)

  # correction on vote column having comma in the string values
  data['vote'] = data['vote'].str.replace(r',', '')

  nan_indices_summary = data.isna()['summary'].values.astype(int)
  nan_indices_review_text = data.isna()['reviewText'].values.astype(int)
  nan_indices_votes = data.isna()['vote'].values.astype(int)

  # text
  reviews = []

  # image
  image_filenames = []

  # categorical tabular features
  reviewer_ids = []
  verified_reviews = []
  product_ids = []
  review_years = []

  # numerical tabular features
  review_votes = []
  unix_review_times = []

  # class labels
  labels = []
  num_low_res, num_high_res, num_user_url = 0, 0, 0
  range_ = len(data) if args.category == 'Office_Products' else 50000

  for index in tqdm.tqdm(range(range_)):

    row = data.loc[data.index == index]
    # row = data.iloc[index]

    meta_row = meta_data.loc[meta_data['asin'] == row['asin'].item()]
    meta_data_exists = False if meta_row.empty else True

    amazon_image_exists = False
    user_image_exists = False

    if meta_data_exists:
      high_res_urls = meta_row['imageURLHighRes'].item()
      if len(high_res_urls) >= 1:

        image_filename = download_and_save_image(
            image_data_dir, high_res_urls, index
        )
        image_filenames.append(image_filename)
        num_high_res += 1
        amazon_image_exists = True

      else:
        low_res_urls = meta_row['imageURL'].item()
        if len(low_res_urls) >= 1:
          num_low_res += 1
          image_filename = download_and_save_image(
              image_data_dir, low_res_urls, index
          )
          image_filenames.append(image_filename)
          amazon_image_exists = True

    if not amazon_image_exists:
      user_urls = row['image'].item()

      if isinstance(row['image'].item(), list):
        image_filename = download_and_save_image(
            image_data_dir, user_urls, index
        )
        image_filenames.append(image_filename)
        num_user_url += 1
        user_image_exists = True

    if not amazon_image_exists and not user_image_exists:
      image_filenames.append(None)

    reviews.append(
        get_reviews(row, index, nan_indices_summary, nan_indices_review_text)
    )
    review_years.append(get_review_years(row))
    review_votes.append(get_review_votes(row, index, nan_indices_votes))
    unix_review_times.append(get_unix_review_time(row))
    labels.append(get_labels(row))

    reviewer_ids.append(get_reviewer_ids(row))
    verified_reviews.append(get_verified_reviews(row))
    product_ids.append(get_product_ids(row))

  categorical_cols = ['reviewerID', 'verified', 'asin', 'year']
  numerical_cols = ['vote', 'unixReviewTime']
  image_cols = ['ImageFileName']
  text_cols = ['Review']
  label_col = ['labels']

  d = pd.DataFrame()

  # text
  d['Review'] = reviews

  # image
  d['ImageFileName'] = image_filenames

  # categorical tabular features
  d['reviewerID'] = reviewer_ids
  d['verified'] = verified_reviews
  d['asin'] = product_ids
  d['year'] = review_years

  # numerical tabular features
  d['vote'] = review_votes
  d['unixReviewTime'] = unix_review_times

  # class labels
  d['labels'] = labels

  to_be_removed_cols = [
      item
      for item in d.columns
      if item
      not in categorical_cols
      + numerical_cols
      + image_cols
      + text_cols
      + label_col
  ]

  d = d.drop(columns=to_be_removed_cols, axis=1)
  d = d.reset_index(drop=True)

  datafile_with_image_paths = os.path.join(
      args.root_data_dir, args.category, f'{args.category}.csv'
  )
  d.to_csv(datafile_with_image_paths)
  print(f'CSV file of data is saved at {datafile_with_image_paths}')


if __name__ == '__main__':
  main()
