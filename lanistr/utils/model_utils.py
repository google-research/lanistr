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

import os

from external_modules.mvts_transformer.timeseries_encoder import TimeSeriesEncoder
from external_modules.tabnet.tabular_encoder import TabNet
from external_modules.tabnet.tabular_encoder import TabNetPretraining
from model.lanistr_utils import BertOnlyMLMHead as mlm_head
from model.lanistr_utils import build_projector
from model.lanistr_utils import ViTForMaskedImageModelingDecoder as mim_head
from model.modeling_lanistr import LANISTRMultiModalForPreTraining
from model.modeling_lanistr import LANISTRMultiModalModel
from model.modules.classifier import PredictionMLP
import omegaconf
import torch
import transformers
from utils.common_utils import load_checkpoint
from utils.common_utils import print_model_size
from utils.common_utils import print_only_by_main_process


def build_model(
    args: omegaconf.DictConfig, tabular_data_information
) -> torch.nn.Module:
  """Builds the model based on the arguments.

  Args:
    args: The arguments.
    tabular_data_information: The information about the tabular data.

  Returns:
    The model: torch.nn.Module.

  Raises:
    Exception: If the pretrained checkpoint is not found.
  """
  tabular_encoder, image_encoder, text_encoder, timeseries_encoder = (
      None,
      None,
      None,
      None,
  )
  image_proj, text_proj, tabular_proj, time_proj = None, None, None, None

  # Image encoder
  if args.image:
    image_encoder = transformers.ViTModel(
        transformers.ViTConfig(), use_mask_token=True
    ).from_pretrained(args.image_encoder_name)

    if args.task == "finetune":
      for p in image_encoder.parameters():
        p.requires_grad = args.image_encoder_trainable

    image_proj = build_projector(
        in_dim=args.image_embedding_dim,
        hidden_dim=None,
        out_dim=args.projection_dim,
        projection_type="SingleLayer",
    )
    print_model_size(image_encoder, "Image Encoder")
    print_model_size(image_proj, "Image Projector")
    assert args.image_size == image_encoder.config.image_size
    assert args.image_embedding_dim == image_encoder.config.hidden_size

  # Text Encoder and masked language modeling
  if args.text:
    # based on https://github.com/huggingface/transformers/blob/94b3f544a1f5e04b78d87a2ae32a7ac252e22e31/src/transformers/models/bert/modeling_bert.py#L1296  # pylint: disable=line-too-long
    bert_config = transformers.BertConfig()
    bert_config.is_decoder = False
    text_encoder = transformers.BertModel(
        bert_config, add_pooling_layer=False
    ).from_pretrained(args.text_encoder_name)

    if args.task == "finetune":
      for p in text_encoder.parameters():
        p.requires_grad = args.text_encoder_trainable

    text_proj = build_projector(
        in_dim=args.text_embedding_dim,
        hidden_dim=None,
        out_dim=args.projection_dim,
        projection_type="SingleLayer",
    )
    print_model_size(text_encoder, "Text Encoder")
    print_model_size(text_proj, "Text Projector")
    assert args.text_embedding_dim == text_encoder.config.hidden_size

  # Tabular encoder
  if args.tab:
    if args.task == "pretrain":
      tabular_encoder = TabNetPretraining(
          input_dim=tabular_data_information["input_dim"],
          pretraining_ratio=args.tabular_pretraining_ratio,
          cat_idxs=tabular_data_information["cat_idxs"],
          cat_dims=tabular_data_information["cat_dims"],
          cat_emb_dim=args.tabular_cat_emb_dim,
          mask_type=args.tabular_mask_type,
          n_d=args.tabular_n_d,
          n_a=args.tabular_n_a,
          epsilon=1e-12,
          virtual_batch_size=int(
              args.train_batch_size // args.ngpus_per_node // 2
          ),
      )
    elif args.task == "finetune":
      tabular_encoder = TabNet(
          input_dim=tabular_data_information["input_dim"],
          output_dim=args.tabular_output_dim,
          cat_idxs=tabular_data_information["cat_idxs"],
          cat_dims=tabular_data_information["cat_dims"],
          cat_emb_dim=args.tabular_cat_emb_dim,
          mask_type=args.tabular_mask_type,
          n_d=args.tabular_n_d,
          n_a=args.tabular_n_a,
          epsilon=1e-12,
          virtual_batch_size=int(
              args.train_batch_size // args.ngpus_per_node // 2
          ),
      )

      for p in tabular_encoder.parameters():
        p.requires_grad = args.tabular_encoder_trainable

    tabular_proj = build_projector(
        in_dim=args.tabular_embedding_dim,
        hidden_dim=None,
        out_dim=args.projection_dim,
        projection_type="SingleLayer",
    )
    print_model_size(tabular_encoder, "Tabular Encoder")
    print_model_size(tabular_proj, "Tabular Projector")

  if args.time:

    timeseries_encoder = TimeSeriesEncoder(
        feat_dim=args.timeseries_input_dim,
        max_len=args.timeseries_max_seq_len,
        d_model=args.timeseries_embedding_dim,
        n_heads=args.timeseries_n_heads,
        num_layers=args.timeseries_layers,
        dim_feedforward=args.timeseries_dim_feedforward,
        dropout=args.timeseries_dropout,
        activation=args.timeseries_activation,
        freeze=not args.timeseries_encoder_trainable,
    )

    time_proj = build_projector(
        in_dim=args.timeseries_embedding_dim,
        hidden_dim=None,
        out_dim=args.projection_dim,
        projection_type="SingleLayer",
    )
    print_model_size(timeseries_encoder, "Timeseries Encoder")
    print_model_size(time_proj, "Timeseries Projector")

  # Multimodal encoder (fusion mechanism)
  mm_fusion = transformers.FlavaMultimodalModel(
      transformers.FlavaMultimodalConfig(), add_pooling_layer=True
  ).from_pretrained("facebook/flava-full")
  for p in mm_fusion.parameters():
    p.requires_grad = args.mm_encoder_trainable
  print_model_size(mm_fusion, "Cross-attention Multimodal Fusion Encoder")

  mm_width = mm_fusion.config.hidden_size
  mm_proj = build_projector(
      in_dim=mm_width,
      hidden_dim=args.mm_hidden_dim,
      out_dim=args.mm_output_dim,
      projection_type="SimSiam",
  )
  print_model_size(mm_proj, "Multimodal Projector")

  mm_predictor = PredictionMLP(
      in_dim=args.mm_output_dim,
      hidden_dim=args.predictor_hidden_dim,
      out_dim=args.predictor_out_dim,
  )
  print_model_size(mm_predictor, "Predictor Head")

  multimodal_model = None
  if args.task == "pretrain":
    multimodal_model = LANISTRMultiModalForPreTraining(
        args=args,
        image_encoder=image_encoder,
        mim_head=mim_head,
        text_encoder=text_encoder,
        mlm_head=mlm_head,
        tabular_encoder=tabular_encoder,
        timeseries_encoder=timeseries_encoder,
        mm_fusion=mm_fusion,
        image_proj=image_proj,
        text_proj=text_proj,
        tabular_proj=tabular_proj,
        time_proj=time_proj,
        mm_proj=mm_proj,
        mm_predictor=mm_predictor,
    )

    if args.pretrain_resume:
      latest_checkpoint_path = os.path.join(
          args.output_dir,
          f"pretrain_multimodal_checkpoint_{args.pretrain_initialize_from_epoch}.pth.tar",
      )
      if os.path.exists(latest_checkpoint_path):
        loc = "cuda:{}".format(args.device)
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=loc)
      else:
        raise FileNotFoundError(
            f"Pretrained checkpoint {latest_checkpoint_path} not found."
            " Pretrain first by passing task=pretrain as an argument"
        )
      multimodal_model = load_checkpoint(
          multimodal_model,
          latest_checkpoint,
          different_datasets=True if args.dataset_name == "amazon" else False,
      )
    else:
      print_only_by_main_process("Randomly initializing the entire model")

    print_model_size(multimodal_model, "pretraining with LANISTR")

    if args.text:
      for p in multimodal_model.text_encoder.parameters():
        p.requires_grad = args.text_encoder_trainable
    if args.image:
      for p in multimodal_model.image_encoder.parameters():
        p.requires_grad = args.image_encoder_trainable
    if args.tab:
      for p in multimodal_model.tabular_encoder.parameters():
        p.requires_grad = args.tabular_encoder_trainable
    if args.time:
      for p in multimodal_model.timeseries_encoder.parameters():
        p.requires_grad = args.timeseries_encoder_trainable

    for p in multimodal_model.mm_fusion.parameters():
      p.requires_grad = args.mm_encoder_trainable

    print_model_size(multimodal_model, "pretraining with LANISTR")

  elif args.task == "finetune":
    pretrain_model = LANISTRMultiModalForPreTraining(
        args=args,
        image_encoder=image_encoder,
        mim_head=mim_head,
        text_encoder=text_encoder,
        mlm_head=mlm_head,
        tabular_encoder=tabular_encoder,
        timeseries_encoder=timeseries_encoder,
        mm_fusion=mm_fusion,
        image_proj=image_proj,
        text_proj=text_proj,
        tabular_proj=tabular_proj,
        time_proj=time_proj,
        mm_proj=mm_proj,
        mm_predictor=mm_predictor,
    )

    if args.finetune_initialize_from == "pretrain":
      best_checkpoint_path = os.path.join(
          args.output_dir, "pretrain_multimodal_model_best.pth.tar"
      )
      if os.path.exists(best_checkpoint_path):
        loc = "cuda:{}".format(args.device)
        best_checkpoint = torch.load(best_checkpoint_path, map_location=loc)
      else:
        raise FileNotFoundError(
            f"Pretrained checkpoint {best_checkpoint_path} not found. Pretrain"
            " first by passing task=pretrain as an argument"
        )
      pretrain_model = load_checkpoint(
          pretrain_model,
          best_checkpoint,
          different_datasets=True if args.dataset_name == "amazon" else False,
      )
    elif args.finetune_initialize_from == "random":
      print_only_by_main_process("Randomly initializing the entire model")
    else:
      raise ValueError(
          "finetune_initialize_from should be either pretrain or random"
      )

    classifier = build_projector(
        in_dim=mm_fusion.config.hidden_size,
        hidden_dim=args.classifier_hidden_dim,
        out_dim=args.num_classes,
        projection_type="MLP",
    )

    print_model_size(classifier, "Classifier module")

    multimodal_model = LANISTRMultiModalModel(
        args=args,
        image_encoder=pretrain_model.image_encoder,
        text_encoder=pretrain_model.text_encoder,
        tabular_encoder=pretrain_model.tabular_encoder,
        timeseries_encoder=pretrain_model.timeseries_encoder,
        mm_fusion=pretrain_model.mm_fusion,
        image_proj=pretrain_model.image_proj,
        text_proj=pretrain_model.text_proj,
        tabular_proj=pretrain_model.tabular_proj,
        time_proj=pretrain_model.time_proj,
        classifier=classifier,
    )

    print_model_size(
        multimodal_model, "Finetuning with LANISTR before possibly freezing"
    )

    if args.text:
      for p in multimodal_model.text_encoder.parameters():
        p.requires_grad = args.text_encoder_trainable
      for p in multimodal_model.text_proj.parameters():
        p.requires_grad = args.text_proj_trainable
    if args.image:
      for p in multimodal_model.image_encoder.parameters():
        p.requires_grad = args.image_encoder_trainable
      for p in multimodal_model.image_proj.parameters():
        p.requires_grad = args.image_proj_trainable
    if args.tab:
      for p in multimodal_model.tabular_encoder.parameters():
        p.requires_grad = args.tabular_encoder_trainable
      for p in multimodal_model.tabular_proj.parameters():
        p.requires_grad = args.tabular_proj_trainable
    if args.time:
      for p in multimodal_model.timeseries_encoder.parameters():
        p.requires_grad = args.timeseries_encoder_trainable
      for p in multimodal_model.time_proj.parameters():
        p.requires_grad = args.timeseries_proj_trainable

    for p in multimodal_model.mm_fusion.parameters():
      p.requires_grad = args.mm_encoder_trainable

    print_model_size(
        multimodal_model, "Finetuning with LANISTR after possibly freezing"
    )

  return multimodal_model
