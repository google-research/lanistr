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

from model.modules.projection import MLPProjectionHead
from model.modules.projection import OneLayerProjectionHead
from model.modules.projection import SimSiamProjectionHead
import torch
from torch import nn
import transformers
from transformers import activations


class BertPredictionHeadTransform(nn.Module):
  """Transforms the hidden states of a BERT model into a new set states."""

  def __init__(self, config):
    """Initializes the BertPredictionHeadTransform module.

    Args:
        config: A configuration object containing the model parameters.
    """
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    if isinstance(config.hidden_act, str):
      self.transform_act_fn = activations.ACT2FN[config.hidden_act]
    else:
      self.transform_act_fn = config.hidden_act
    self.layer_norm = nn.LayerNorm(
        config.hidden_size, eps=config.layer_norm_eps
    )

  def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    return hidden_states


class BertLMPredictionHead(nn.Module):
  """Predicts the next token in a sequence of tokens."""

  def __init__(self, config):
    """Initializes the BertLMPredictionHead module.

    Args:
        config: A configuration object containing the model parameters.
    """
    super().__init__()
    self.transform = BertPredictionHeadTransform(config)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    # Need a link between the two variables so that the bias is correctly
    # resized with `resize_token_embeddings`
    self.decoder.bias = self.bias

  def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    """Forward pass.

    Args:
      hidden_states: torch.Tensor.

    Returns:
      transformed hidden states: torch.Tensor.
    """
    hidden_states = self.transform(hidden_states)
    hidden_states = self.decoder(hidden_states)
    return hidden_states


class BertOnlyMLMHead(nn.Module):
  """BertOnlyMLMHead is a module that predicts the next token in a sequence of tokens."""

  def __init__(self, config):
    """Initializes the BertOnlyMLMHead module.

    Args:
        config: A configuration object containing the model parameters.
    """
    super().__init__()
    self.predictions = BertLMPredictionHead(config)

  def forward(self, sequence_output: torch.FloatTensor) -> torch.FloatTensor:
    prediction_scores = self.predictions(sequence_output)
    return prediction_scores


class ViTForMaskedImageModelingDecoder(nn.Module):
  """Decodes a sequence of tokens into a sequence of images."""

  def __init__(self, config: transformers.ViTConfig):
    """Initializes the ViTForMaskedImageModelingDecoder module.

    Args:
        config: A configuration object containing the model parameters.

    Returns:
        None.
    """
    super().__init__()

    self.decoder = nn.Sequential(
        nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.encoder_stride**2 * config.num_channels,
            kernel_size=1,
        ),
        nn.PixelShuffle(config.encoder_stride),
    )

  def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
    """Forward pass.

    Args:
      x: torch.FloatTensor. The input tensor.

    Returns:
      decoded images: torch.FloatTensor. The decoded images.
    """
    return self.decoder(x)


def build_projector(
    in_dim: int, hidden_dim: int, out_dim: int, projection_type: str
) -> nn.Module:
  """Builds a projector.

  Args:
      in_dim: The input dimension of the projector.
      hidden_dim: The hidden dimension of the projector.
      out_dim: The output dimension of the projector.
      projection_type: The type of the projector.

  Returns:
      A projector: nn.Module

  Raises:
      Exception: If the projection_type is not supported.
  """
  if projection_type == "SimSiam":
    projection_head = SimSiamProjectionHead
  elif projection_type == "SingleLayer":
    projection_head = OneLayerProjectionHead
  elif projection_type == "MLP":
    projection_head = MLPProjectionHead
    hidden_dim = [hidden_dim, hidden_dim]
  else:
    raise ValueError(f"Unsupported projection type: {projection_type}")

  return projection_head(
      in_dim=in_dim,
      hidden_dim=hidden_dim,
      out_dim=out_dim,
  )
