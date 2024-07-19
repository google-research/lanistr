"""Transformer encoder for time series."""

import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
  """Transformer encoder for time series."""

  def __init__(
      self,
      feat_dim: int,
      max_len: int,
      d_model: int,
      n_heads: int,
      num_layers: int,
      dim_feedforward: int,
      dropout: float = 0.1,
      activation: str = 'gelu',
      freeze: bool = False,
  ) -> None:
    """Initializes the transformer encoder.

    Args:
        feat_dim: dimension of input features
        max_len: maximum length of input sequences
        d_model: dimension of transformer embeddings
        n_heads: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: dimension of feedforward layer in transformer
        dropout: dropout rate
        activation: activation function for transformer feedforward layer
        freeze: whether to freeze the transformer parameters
    """
    super().__init__()  # super() is not needed in python 3

    self.max_len = max_len
    self.d_model = d_model
    self.n_heads = n_heads

    self.project_inp = nn.Linear(feat_dim, d_model)
    self.pos_enc = LearnablePositionalEncoding(
        d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
    )

    encoder_layer = nn.TransformerEncoderLayer(
        d_model,
        self.n_heads,
        dim_feedforward,
        dropout * (1.0 - freeze),
        activation=activation,
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    self.output_layer = nn.Linear(d_model, feat_dim)
    self.act = nn.GELU()
    self.dropout1 = nn.Dropout(dropout)
    self.feat_dim = feat_dim

  def forward(
      self,
      inp: torch.Tensor,
      padding_masks: torch.Tensor,
      noise_mask_single: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward pass of the transformer encoder.

    Args:
        inp: (batch_size, seq_length, feat_dim) torch tensor of masked features
        padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep
          vector at this position, 0 means padding
        noise_mask_single: (batch_size, seq_length, feat_dim) boolean tensor, 1
          means keep vector at this position, 0 means mask

    Returns:
        output: (batch_size, seq_length, feat_dim)
    """

    # permute because pytorch convention for transformers is [seq_length,
    # batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
    inp = inp.permute(1, 0, 2)
    inp = self.project_inp(inp) * math.sqrt(
        self.d_model
    )  # [seq_length, batch_size, d_model] project input vectors to d_model
    # dimensional space
    inp = self.pos_enc(inp)  # add positional encoding
    # NOTE: logic for padding masks is reversed to comply with definition in
    # MultiHeadAttention, TransformerEncoderLayer
    output = self.transformer_encoder(
        inp, src_key_padding_mask=~padding_masks
    )  # (seq_length, batch_size, d_model)
    output = self.act(
        output
    )  # the output transformer encoder/decoder embeddings don't include
    # non-linearity
    output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
    output = self.dropout1(output)
    if noise_mask_single is not None:
      output = output * noise_mask_single
    # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation
    # over (seq_length, batch_size).
    output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

    return output


class LearnablePositionalEncoding(nn.Module):
  """Learnable positional encoding.

  Attributes:
    dropout: dropout rate
    max_len: maximum length of input sequences
    d_model: dimension of transformer embeddings
  """

  def __init__(
      self, d_model: int, dropout: float = 0.1, max_len: int = 1024
  ) -> None:
    """Initializes the learnable positional encoding.

    Args:
        d_model: dimension of transformer embeddings
        dropout: dropout rate
        max_len: maximum length of input sequences
    """
    super().__init__()  # super() is not needed in python 3
    self.dropout = nn.Dropout(p=dropout)
    # Each position gets its own embedding
    # Since indices are always 0 ... max_len, we don't have to do a look-up
    self.pe = nn.Parameter(
        torch.empty(max_len, 1, d_model)
    )  # requires_grad automatically set to True
    nn.init.uniform_(self.pe, -0.02, 0.02)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the learnable positional encoding.

    Args:
        x:  [sequence length, batch size, embed dim] the sequence fed to the
          positional encoder model (required).

    Returns:
        output: [sequence length, batch size, embed dim]
    """
    x = x + self.pe[: x.size(0), :]
    return self.dropout(x)


def geom_noise_mask_single(
    length: int, lm: int, masking_ratio: float
) -> np.ndarray:
  """Randomly create a boolean mask of length `length`.

  The mask consists of subsequences of average length lm, masking with 0s a
  `masking_ratio` proportion of the sequence L. The length of masking
  subsequences and intervals follow a geometric distribution.

  Args:
      length: length of mask and sequence to be masked
      lm: average length of masking subsequences (streaks of 0s)
      masking_ratio: proportion of L to be masked

  Returns:
      (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of
      length L
  """
  keep_mask = np.ones(length, dtype=bool)
  p_m = (
      1 / lm
  )  # probability of each masking sequence stopping. parameter of geometric
  # distribution.
  p_u = (
      p_m * masking_ratio / (1 - masking_ratio)
  )  # probability of each unmasked sequence stopping. parameter of geometric
  # distribution.
  p = [p_m, p_u]

  # Start in state 0 with masking_ratio probability
  state = int(
      np.random.rand() > masking_ratio
  )  # state 0 means masking, 1 means not masking
  for i in range(length):
    keep_mask[i] = (
        state  # here it happens that state and masking value corresponding to
        # state are identical
    )
    if np.random.rand() < p[state]:
      state = 1 - state

  return keep_mask


def noise_mask(
    timeseries: np.ndarray,
    masking_ratio: float,
    lm: int = 3,
    mode: str = 'separate',
    distribution: str = 'random',
    exclude_feats: Optional[List[int]] = None,
) -> torch.Tensor:
  """Creates a random boolean mask of the same shape as X.

  The mask has 0s at places where a feature should be masked.

  Args:
      timeseries: (seq_length, feat_dim) numpy array of features corresponding
        to a single sample
      masking_ratio: proportion of seq_length to be masked. At each time step,
        will also be the proportion of feat_dim that will be masked on average
      lm: average length of masking subsequences (streaks of 0s). Used only when
        `distribution` is 'geometric'.
      mode: whether each variable should be masked separately ('separate'), or
        all variables at a certain positions should be masked concurrently
        ('concurrent')
      distribution: whether each mask sequence element is sampled independently
        at random, or whether sampling follows a markov chain (and thus is
        stateful), resulting in geometric distributions of masked squences of a
        desired mean length `lm`
      exclude_feats: iterable of indices corresponding to features to be
        excluded from masking (i.e. to remain all 1s)

  Returns:
      boolean numpy array with the same shape as sample, with 0s at places where
      a
      feature should be masked
  """
  if exclude_feats is not None:
    exclude_feats = set(exclude_feats)

  if distribution == 'geometric':  # stateful (Markov chain)
    if mode == 'separate':  # each variable (feature) is independent
      mask = np.ones(timeseries.shape, dtype=bool)
      for m in range(timeseries.shape[1]):  # feature dimension
        if exclude_feats is None or m not in exclude_feats:
          mask[:, m] = geom_noise_mask_single(
              timeseries.shape[0], lm, masking_ratio
          )  # time dimension
    else:  # replicate across feature dimension (mask all variables at the same
      # positions concurrently)
      mask = np.tile(
          np.expand_dims(
              geom_noise_mask_single(timeseries.shape[0], lm, masking_ratio), 1
          ),
          timeseries.shape[1],
      )
  else:  # each position is independent Bernoulli with p = 1 - masking_ratio
    if mode == 'separate':
      mask = np.random.choice(
          np.array([True, False]),
          size=timeseries.shape,
          replace=True,
          p=(1 - masking_ratio, masking_ratio),
      )
    else:
      mask = np.tile(
          np.random.choice(
              np.array([True, False]),
              size=(timeseries.shape[0], 1),
              replace=True,
              p=(1 - masking_ratio, masking_ratio),
          ),
          timeseries.shape[1],
      )

  return torch.from_numpy(mask)
