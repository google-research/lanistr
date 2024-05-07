# LANISTR: Multimodal Learning from Structured and Unstructured Data

This repository contains the implementation of the paper:

> **LANISTR: Multimodal Learning from Structured and Unstructured Data**
> [[Paper]](https://arxiv.org/pdf/2305.16556.pdf)  <br>
> [Sayna Ebrahimi](https://saynaebrahimi.github.io/), [Sercan Arik](https://sites.google.com/corp/view/sercanarik/home), [Yihe Dong](https://yihedong.me/), [Tomas Pfister](https://tomas.pfister.fi/).
> <br>

Multimodal large-scale pretraining has shown impressive performance for
unstructured data such as language and image. However, a prevalent real-world
scenario involves structured data types, tabular and time-series, along with
unstructured data. Such scenarios have been understudied. To bridge this gap, we
propose LANISTR, an attention-based framework to learn from LANguage, Image, and
STRuctured data. The core of LANISTR's methodology is rooted in
\textit{masking-based} training applied across both unimodal and multimodal
levels. In particular, we introduce a new similarity-based multimodal masking
loss that enables it to learn cross-modal relations from large-scale multimodal
data with missing modalities. On two real-world datasets, MIMIC-IV (from
healthcare) and Amazon Product Review (from retail), LANISTR demonstrates
remarkable improvements, 6.6\% (in AUROC) and 14\% (in accuracy) when fine-tuned
with 0.1\% and 0.01\% of labeled data, respectively, compared to the
state-of-the-art alternatives. Notably, these improvements are observed even
with very high ratio of samples (35.7\% and 99.8\% respectively) not containing
all modalities, underlining the robustness of LANISTR to practical missing
modality challenge.

--------------------------------------------------------------------------------

<div  align="center">
<img src="lanistr/Figures/lanistr.gif" width="95%">
</div>


## Setup

### Install LANISTR

```bash
conda create -n lanistr python=3.8 -y
conda activate lanistr
pip install -e .
```

### Data Preparation

#### MIMIC-IV-v2.2

-   For [MIMIC-IV v2.2 dataset](https://mimic.mit.edu/), please follow the
    instructions to obtain access. Note that, prior to requesting access to
    MIMIC, you must become a credentialed user on PhysioNet where the MIMIC data
    is hosted
    ([see instructions](https://physionet.org/settings/credentialing/)).
-   Once you obtain access, log into your PhysioNet account and visit
    [MIMIC-IV project page](https://physionet.org/content/mimiciv/),
    [MIMIC-CXR project page](https://physionet.org/content/mimic-cxr/) and
    [MIMIC-ED project page](https://physionet.org/content/mimic-iv-ed/). Find
    the “Files” section in the project description and download the data after
    signing the user agreement. Place all the data under `physionet.org`
    directory as shown below.
-   Next, follow
    [these instructions](https://github.com/nyuad-cai/MedFuse/blob/main/mimic4extract/README.md)
    to extract and preprocess the data into `./lanistr/data/MIMIC-IV-V2.2/` directory.
    Note that two sub-directories will be generated in `./lanistr/data/MIMIC-IV-V2.2/`
    named as `root` and `in-hospital-mortality`. This path
    (`./lanistr/data/MIMIC-IV-V2.2/`) is called as `preprocessed_data_dir` in the config
    files in our code.
-   Download
    [discretizer_config.json](https://github.com/nyuad-cai/MedFuse/blob/main/ehr_utils/resources/discretizer_config.json)
    and place it in `./lanistr/data/MIMIC-IV-V2.2/discretizer_config.json`.

#### Amazon Product Review 2018

-   For
    [Amazon Product Review Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/),
    please use the provided download script to download the data from the
    original servers upon agreeing to their user agreements. Make sure you are in `./lanistr/` subdirectory.

```bash
bash scripts/download_amazon.sh
```

Once downloaded, the data directory in lanistr code should look like as below for
`./lanistr/data/APR2018`

```bash
lanistr/
  configs/
  datasets/
  model/
  Figures/
  scripts/
  utils/
  ......
  data/
    |-- MIMIC-IV-V2.2
      | -- discretizer_config.json
      | -- in-hospital-mortality
      | -- physionet.org
        | -- files
          | -- mimic-cxr
          | -- mimic-cxr-jpg
          | -- mimic-iv-ed
          | -- mimic-iv-note
          | -- mimiciv
        | -- robots.txt

    |-- APR2018
      | -- All_Beauty
        | -- images
        | -- All_Beauty.json.gz
        | -- meta_All_Beauty.json.gz
      | -- AMAZON_FASHION
        | -- images
        | -- AMAZON_FASHION.json.gz
        | -- meta_AMAZON_FASHION.json.gz
      | -- Office_Products
        | -- images
        | -- Office_Products.json.gz
        | -- meta_Office_Products.json.gz

          ......
```

## MIMIC-IV-v2.2


Experiments on MIMIC-IV dataset start by first pretraining the model followed by
finetuning on the labeled portion of the data. For both steps we have used
8xA100 with 40GB memory. For evaluation, we only use a single GPU. The following scripts run pretraining and fine-tuning experiments. Please make sure to run them from `./lanistr/` subdirectory.



### Pre-training

```bash
bash scripts/mimic_pretrain.sh
```

### Fine-tuning

```bash
bash scripts/mimic_finetune.sh
```

### Evaluation

```bash
bash scripts/mimic_eval.sh
```

## Amazon Product Review 2018

Experiments on APR2018 dataset start by first pretraining the model on the
Office category. We have finetuned the pretrained checkpoint on two distinct
categories of `All_Beauty` and `AMAZON_FASHION`. For all experiments we have
used 8xA100 with 40GB memory. For evaluation, we only use a single GPU. The following scripts run pretraining and fine-tuning experiments. Please make sure to run them from `./lanistr/` subdirectory.

### Pre-training

```bash
bash scripts/amazon_pretrain.sh
```

### Fine-tuning

For All_Beatuy and Fashion category use the following scripts, respectively.

```bash
bash scripts/amazon_finetune_beauty.sh
bash scripts/amazon_finetune_fashion.sh
```

### Evaluation

```bash
bash scripts/amazon_eval_beauty.sh
bash scripts/amazon_eval_fashion.sh
```

## FAQ

### How to use LANISTR on your own data

**Prepare your data:** Organize your data into a CSV file where each column
represents a different modality (image filenames, text, time series file paths,
and tabular features). Ensure categorical and numerical tabular features are in
separate columns.

**Adapt the dataset class:** Modify the dataset class located at
./dataset/amazon/load_data.py to correctly read and iterate through your CSV
data. This might involve adjusting column names, data types, and loading logic.

### How to skip pretraining and use LANISTR for supervised learning only

**Choose a finetuning config file:** Select one of the provided finetuning
configuration files (e.g., `./lanistr/configs/mimic_finetune.yaml`).

**Set finetune_initialize_from to random:** In the chosen config file, locate
the finetune_initialize_from parameter and set its value to `random. This will
initialize LANISTR's architecture with random weights, except for the image
encoder (initialized from a pretrained ViT on ImageNet), the text encoder
(pretrained BERT), and the multimodal fusion encoder (pretrained FLAVA on
ImageNet). The time series and TabNet encoders will still initialize randomly.

### What if I have a different combination of modalities than those in the paper?

In all configuration files, you can enable or disable individual modalities
using the following parameters:

```
# modalities
image: true  # Set to false if you don't have image data
text: true   # Set to false if you don't have text data
time: true   # Set to false if you don't have time series data
tab: true    # Set to false if you don't have tabular data
```

### What is the exact PyTorch installation command you used?
We used `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`


## Citation

If you find our code or paper helpful, please kindly cite LANISTR: ```BibTeX

@article{ebrahimi2023lanistr, title={LANISTR: Multimodal Learning from
Structured and Unstructured Data}, author={Ebrahimi, Sayna and Arik, Sercan O
and Dong, Yihe and Pfister, Tomas}, journal={arXiv preprint arXiv:2305.16556},
year={2023} } ```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Acknowledgement

This repo borrows some codes from
[MedFUse](https://github.com/nyuad-cai/MedFuse),
[pytorch_tabnet](https://github.com/dreamquark-ai/tabnet),
[Multivariate Time Series Transformer Framework](https://github.com/gzerveas/mvts_transformer/tree/master),
[SimSiam](https://github.com/facebookresearch/simsiam). Thanks for their great
works.

**This is not an officially supported Google product.**

