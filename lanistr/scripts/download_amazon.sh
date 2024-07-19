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

# Download Amazon Review Data from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
mkdir data/APR2018
mkdir data/APR2018/Office_Products
mkdir data/APR2018/Office_Products/images
mkdir data/APR2018/AMAZON_FASHION
mkdir data/APR2018/AMAZON_FASHION/images
mkdir data/APR2018/All_Beauty
mkdir data/APR2018/All_Beauty/images

cd data/APR2018/Office_Products
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Office_Products.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz

cd ../AMAZON_FASHION
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_AMAZON_FASHION.json.gz

cd ../All_Beauty
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Beauty.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz

cd ../../../

# the following take nearly 30 minutes each.
python dataset/amazon/download_images.py --category All_Beauty
python dataset/amazon/download_images.py --category AMAZON_FASHION

# this will take many hours but it goes by fast because there are not too many images
python dataset/amazon/download_images.py --category Office_Products
