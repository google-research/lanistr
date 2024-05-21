"""TabNet encoder and decoder."""

import dataclasses
import math
from typing import Any, List, Optional

import numpy as np
import torch
from torch import nn
import transformers


@dataclasses.dataclass
class TabNetModelOutput(transformers.utils.ModelOutput):
  """Base class for pretraining tabnet model s outputs."""

  loss: Optional[torch.FloatTensor] = None
  last_hidden_state: Optional[torch.Tensor] = None
  output: Optional[torch.Tensor] = None


class TabNetEncoder(nn.Module):
  """TabNet encoder.

  This is the encoder part of the TabNet model. It takes the input and passes it
  through the TabNet encoder.
  """

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      n_d: int = 128,
      n_a: int = 128,
      n_steps: int = 6,
      gamma: float = 1.8,
      n_independent: int = 2,
      n_shared: int = 2,
      epsilon: float = 1e-15,
      virtual_batch_size: int = 256,
      momentum: float = 0.95,
      mask_type: str = "sparsemax",
  ):
    """Defines main part of the TabNet network without the embedding layers.

    Args:
        input_dim (int): Number of features.
        output_dim (int or list): Dimension of network output (one for
          regression, two for binary classification, etc...).
        n_d (int): Dimension of the prediction layer (usually between 4 and 64).
        n_a (int): Dimension of the attention layer (usually between 4 and 64).
        n_steps (int): Number of successive steps in the network (usually
          between 3 and 10).
        gamma (float): Float above 1, scaling factor for attention updates
          (usually between 1.0 to 2.0).
        n_independent (int): Number of independent GLU layers in each GLU block
          (default 2).
        n_shared (int): Number of independent GLU layers in each GLU block
          (default 2).
        epsilon (float): Avoid log(0), this should be kept very low.
        virtual_batch_size (int): Batch size for Ghost Batch Normalization.
        momentum (float): Float value between 0 and 1 which will be used for
          momentum in all batch norm.
        mask_type (str): Either "sparsemax" or "entmax": this is the masking
          function to use.
    """
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.is_multi_task = isinstance(output_dim, list)
    self.n_d = n_d
    self.n_a = n_a
    self.n_steps = n_steps
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_independent = n_independent
    self.n_shared = n_shared
    self.virtual_batch_size = virtual_batch_size
    self.mask_type = mask_type
    self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    if self.n_shared > 0:
      shared_feat_transform = torch.nn.ModuleList()
      for i in range(self.n_shared):
        if i == 0:
          shared_feat_transform.append(
              nn.Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
          )
        else:
          shared_feat_transform.append(
              nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
          )

    else:
      shared_feat_transform = None

    self.initial_splitter = FeatTransformer(
        input_dim=self.input_dim,
        output_dim=n_d + n_a,
        shared_layers=shared_feat_transform,
        n_glu_independent=self.n_independent,
        virtual_batch_size=self.virtual_batch_size,
        momentum=momentum,
    )

    self.feat_transformers = torch.nn.ModuleList()
    self.att_transformers = torch.nn.ModuleList()

    for _ in range(n_steps):
      transformer = FeatTransformer(
          input_dim=self.input_dim,
          output_dim=n_d + n_a,
          shared_layers=shared_feat_transform,
          n_glu_independent=self.n_independent,
          virtual_batch_size=self.virtual_batch_size,
          momentum=momentum,
      )
      attention = AttentiveTransformer(
          input_dim=n_a,
          output_dim=self.input_dim,
          virtual_batch_size=self.virtual_batch_size,
          momentum=momentum,
          mask_type=self.mask_type,
      )
      self.feat_transformers.append(transformer)
      self.att_transformers.append(attention)

  def forward(
      self, x: torch.Tensor, prior: Optional[torch.Tensor] = None
  ) -> Any:
    """Forward pass of the TabNet encoder.

    Args:
      x: input features of shape [batch_size, num_features]
      prior: prior for the attention mechanism. If None, will be initialized to
        ones.

    Returns:
      steps_output: list of outputs of the GLU activations of all transformer
        blocks.
      m_loss: attention coefficient loss.
    """

    x = self.initial_bn(x)

    if prior is None:
      prior = torch.ones(x.shape).to(x.device)

    m_loss = 0
    att = self.initial_splitter(x)[:, self.n_d :]

    steps_output = []
    for step in range(self.n_steps):
      m = self.att_transformers[step](prior, att)
      m_loss += torch.mean(
          torch.sum(torch.mul(m, torch.log(m + self.epsilon)), dim=1)
      )
      # update prior
      prior = torch.mul(self.gamma - m, prior)
      # output
      masked_x = torch.mul(m, x)
      out = self.feat_transformers[step](masked_x)
      d = torch.nn.ReLU()(out[:, : self.n_d])
      steps_output.append(d)
      # update attention
      att = out[:, self.n_d :]

    m_loss /= self.n_steps
    return steps_output, m_loss

  def forward_masks(self, x: torch.Tensor) -> Any:
    """Forward masks.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      m_explain: output of the TabNet encoder
      masks: list of attention masks
    """
    x = self.initial_bn(x)

    prior = torch.ones(x.shape).to(x.device)
    m_explain = torch.zeros(x.shape).to(x.device)
    att = self.initial_splitter(x)[:, self.n_d :]
    masks = {}

    for step in range(self.n_steps):
      m = self.att_transformers[step](prior, att)
      masks[step] = m
      # update prior
      prior = torch.mul(self.gamma - m, prior)
      # output
      masked_x = torch.mul(m, x)
      out = self.feat_transformers[step](masked_x)
      d = torch.nn.ReLU()(out[:, : self.n_d])
      # explain
      step_importance = torch.sum(d, dim=1)
      m_explain += torch.mul(m, step_importance.unsqueeze(dim=1))
      # update attention
      att = out[:, self.n_d :]

    return m_explain, masks


class TabNetDecoder(torch.nn.Module):
  """TabNet decoder.

  This is the decoder part of the TabNet model. It takes the output of the
  TabNet encoder and reconstructs the input.
  """

  def __init__(
      self,
      input_dim: int,
      n_d: int = 64,
      n_steps: int = 6,
      n_independent: int = 2,
      n_shared: int = 2,
      virtual_batch_size: int = 256,
      momentum: float = 0.98,
  ):
    """TabNet decoder.

    Args:
      input_dim: int Number of features
      n_d: int Dimension of the prediction  layer (usually between 4 and 64)
      n_steps: int Number of successive steps in the network (usually between 3
        and 10)
      n_independent: int Number of independent GLU layer in each GLU block
        (default 1)
      n_shared: int Number of independent GLU layer in each GLU block (default
        1)
      virtual_batch_size: int Batch size for Ghost Batch Normalization
      momentum: float Float value between 0 and 1 which will be used for
        momentum in all batch norm
    """
    super().__init__()
    self.input_dim = input_dim
    self.n_d = n_d
    self.n_steps = n_steps
    self.n_independent = n_independent
    self.n_shared = n_shared
    self.virtual_batch_size = virtual_batch_size

    self.feat_transformers = torch.nn.ModuleList()

    if self.n_shared > 0:
      shared_feat_transform = torch.nn.ModuleList()
      for i in range(self.n_shared):
        if i == 0:
          shared_feat_transform.append(nn.Linear(n_d, 2 * n_d, bias=False))
        else:
          shared_feat_transform.append(nn.Linear(n_d, 2 * n_d, bias=False))

    else:
      shared_feat_transform = None

    for _ in range(n_steps):
      transformer = FeatTransformer(
          n_d,
          n_d,
          shared_feat_transform,
          n_glu_independent=self.n_independent,
          virtual_batch_size=self.virtual_batch_size,
          momentum=momentum,
      )
      self.feat_transformers.append(transformer)

    self.reconstruction_layer = nn.Linear(n_d, self.input_dim, bias=False)
    initialize_non_glu(self.reconstruction_layer, n_d, self.input_dim)

  def forward(self, steps_output: Any) -> torch.Tensor:
    """Forward pass of the TabNet decoder.

    Args:
      steps_output: list of outputs of the GLU activations of all transformer
        blocks.

    Returns:
      res: reconstructed input.
    """
    res = 0
    for step_nb, step_output in enumerate(steps_output):
      x = self.feat_transformers[step_nb](step_output)
      res = torch.add(res, x)
    res = self.reconstruction_layer(res)
    return res


@dataclasses.dataclass
class TabNetPretrainingModelOutput(transformers.utils.ModelOutput):
  """Base class for pretraining tabnet model outputs."""

  masked_loss: Optional[torch.FloatTensor] = None
  unmasked_loss: Optional[torch.FloatTensor] = None
  masked_last_hidden_state: Optional[torch.Tensor] = None
  unmasked_last_hidden_state: Optional[torch.Tensor] = None
  embeds: torch.FloatTensor = None


class TabNetPretraining(torch.nn.Module):
  """TabNet pretraining model.

  This is the pretraining model for TabNet. It takes the input and randomly
  obfuscates a percentage of the features. It then uses the TabNet encoder to
  reconstruct the input.
  """

  def __init__(
      self,
      input_dim: int,
      pretraining_ratio: float = 0.2,
      n_d: int = 64,
      n_a: int = 64,
      n_steps: int = 6,
      gamma: float = 1.8,
      cat_idxs: Optional[List[int]] = None,
      cat_dims: Optional[List[int]] = None,
      cat_emb_dim: int = 1,
      n_independent: int = 2,
      n_shared: int = 2,
      epsilon: float = 1e-15,
      virtual_batch_size: int = 128,
      momentum: float = 0.98,
      mask_type: str = "sparsemax",
      n_shared_decoder: int = 1,
      n_indep_decoder: int = 1,
  ):
    """Defines main part of the TabNet network without the embedding layers.

    Args:
        input_dim: Number of features.
        pretraining_ratio: Ratio of features to randomly discard for
          reconstruction.
        n_d: Dimension of the prediction layer (usually between 4 and 64).
        n_a: Dimension of the attention layer (usually between 4 and 64).
        n_steps: Number of successive steps in the network (usually between 3
          and 10).
        gamma: Float above 1, scaling factor for attention updates (usually
          between 1.0 to 2.0).
        cat_idxs: Index of each categorical column in the dataset.
        cat_dims: Number of categories in each categorical column.
        cat_emb_dim: Size of the embedding of categorical features. If int, all
          categorical features will have the same embedding size. If list of
          int, every corresponding feature will have a specific size.
        n_independent: Number of independent GLU layers in each GLU block
          (default 2).
        n_shared: Number of shared GLU layers in each GLU block (default 2).
        epsilon: Avoid log(0), this should be kept very low.
        virtual_batch_size: Batch size for Ghost Batch Normalization.
        momentum: Value between 0 and 1 used for momentum in all batch
          normalizations.
        mask_type: Either "sparsemax" or "entmax", the masking function to use.
        n_shared_decoder: Number of shared GLU layers in each decoder GLU block
          (default 1).
        n_indep_decoder: Number of independent GLU layers in each decoder GLU
          block (default 1).

    Returns:
        None.
    """
    super().__init__()

    self.cat_idxs = cat_idxs if cat_idxs is not None else []
    self.cat_dims = cat_dims if cat_dims is not None else []
    self.cat_emb_dim = cat_emb_dim

    self.input_dim = input_dim
    self.n_d = n_d
    self.n_a = n_a
    self.n_steps = n_steps
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_independent = n_independent
    self.n_shared = n_shared
    self.mask_type = mask_type
    self.pretraining_ratio = pretraining_ratio
    self.n_shared_decoder = n_shared_decoder
    self.n_indep_decoder = n_indep_decoder

    if self.n_steps <= 0:
      raise ValueError("n_steps should be a positive integer.")
    if self.n_independent == 0 and self.n_shared == 0:
      raise ValueError("n_shared and n_independent can't be both zero.")

    self.virtual_batch_size = virtual_batch_size
    self.embedder = EmbeddingGenerator(
        input_dim, cat_dims, cat_idxs, cat_emb_dim
    )
    self.post_embed_dim = self.embedder.post_embed_dim

    self.masker = RandomObfuscator(self.pretraining_ratio)
    self.encoder = TabNetEncoder(
        input_dim=self.post_embed_dim,
        output_dim=self.post_embed_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        epsilon=epsilon,
        virtual_batch_size=self.virtual_batch_size,
        momentum=momentum,
        mask_type=mask_type,
    )
    self.decoder = TabNetDecoder(
        self.post_embed_dim,
        n_d=n_d,
        n_steps=n_steps,
        n_independent=self.n_indep_decoder,
        n_shared=self.n_shared_decoder,
        virtual_batch_size=self.virtual_batch_size,
        momentum=momentum,
    )

  def forward(self, x: torch.Tensor) -> TabNetPretrainingModelOutput:
    """Forward pass.

    Args:
        x: torch.Tensor

    Returns:
        res : output of reconstruction
        embedded_x : embedded input
        obf_vars : which variable where obfuscated
    """
    embedded_x = self.embedder(x)
    if self.training:
      masked_x, masked_obf_vars = self.masker(embedded_x)
      # set prior of encoder with obf_mask
      prior = 1 - masked_obf_vars
      masked_steps_out, _ = self.encoder(masked_x, prior=prior)
      res = self.decoder(masked_steps_out)
      masked_last_hidden_state = torch.sum(masked_steps_out, dim=0)
      masked_loss = self.compute_loss(res, embedded_x, masked_obf_vars)

      unmasked_steps_out, _ = self.encoder(embedded_x)
      unmasked_res = self.decoder(unmasked_steps_out)
      unmasked_obf_vars = torch.ones(embedded_x.shape).to(x.device)
      unmasked_last_hidden_state = torch.sum(unmasked_steps_out, dim=0)
      unmasked_loss = self.compute_loss(
          unmasked_res, embedded_x, unmasked_obf_vars
      )

      return TabNetPretrainingModelOutput(
          masked_loss=masked_loss,
          unmasked_loss=unmasked_loss,
          masked_last_hidden_state=masked_last_hidden_state,
          unmasked_last_hidden_state=unmasked_last_hidden_state,
          embeds=embedded_x,
      )
    else:
      steps_out, _ = self.encoder(embedded_x)
      res = self.decoder(steps_out)
      obf_vars = torch.ones(embedded_x.shape).to(x.device)

      last_hidden_state = torch.sum(torch.stack(steps_out, dim=0), dim=0)

      loss = self.compute_loss(res, embedded_x, obf_vars)

      return TabNetPretrainingModelOutput(
          masked_loss=None,
          unmasked_loss=loss,
          masked_last_hidden_state=None,
          unmasked_last_hidden_state=last_hidden_state,
          embeds=embedded_x,
      )

  def forward_masks(self, x: torch.Tensor) -> Any:
    """Forward masks.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      m_explain: output of the TabNet encoder
      masks: list of attention masks
    """
    embedded_x = self.embedder(x)
    return self.encoder.forward_masks(embedded_x)

  def compute_loss(
      self,
      y_pred: torch.Tensor,
      embedded_x: torch.Tensor,
      obf_vars: torch.Tensor,
      eps: float = 1e-9,
  ) -> torch.Tensor:
    """Compute the loss for the pretraining model.

    Args:
        y_pred: output of the model
        embedded_x: input to the model
        obf_vars: which variables were obfuscated
        eps: epsilon to avoid division by zero

    Returns:
        loss: loss of the model
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1

    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss


class TabNetNoEmbeddings(nn.Module):
  """TabNet encoder without embeddings.

  This is the encoder part of the TabNet model. It takes the input and passes it
  through the TabNet encoder.
  """

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      n_d: int = 8,
      n_a: int = 8,
      n_steps: int = 3,
      gamma: float = 1.3,
      n_independent: int = 2,
      n_shared: int = 2,
      epsilon: float = 1e-15,
      virtual_batch_size: int = 256,
      momentum: float = 0.95,
      mask_type: str = "sparsemax",
  ):
    """Defines main part of the TabNet network without the embedding layers.

    Args:
        input_dim (int): Number of features.
        output_dim (int or list): Dimension of network output (one for
          regression, two for binary classification, etc...).
        n_d (int): Dimension of the prediction layer (usually between 4 and 64).
        n_a (int): Dimension of the attention layer (usually between 4 and 64).
        n_steps (int): Number of successive steps in the network (usually
          between 3 and 10).
        gamma (float): Float above 1, scaling factor for attention updates
          (usually between 1.0 to 2.0).
        n_independent (int): Number of independent GLU layers in each GLU block
          (default 2).
        n_shared (int): Number of independent GLU layers in each GLU block
          (default 2).
        epsilon (float): Avoid log(0), this should be kept very low.
        virtual_batch_size (int): Batch size for Ghost Batch Normalization.
        momentum (float): Value between 0 and 1 used for momentum in all batch
          norm.
        mask_type (str): Either "sparsemax" or "entmax": the masking function to
          use.
    """
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.is_multi_task = isinstance(output_dim, list)
    self.n_d = n_d
    self.n_a = n_a
    self.n_steps = n_steps
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_independent = n_independent
    self.n_shared = n_shared
    self.virtual_batch_size = virtual_batch_size
    self.mask_type = mask_type
    self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

    self.encoder = TabNetEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        epsilon=epsilon,
        virtual_batch_size=virtual_batch_size,
        momentum=momentum,
        mask_type=mask_type,
    )

    if self.is_multi_task:
      self.multi_task_mappings: torch.nn.ModuleList[nn.Linear] = (
          torch.nn.ModuleList()
      )
      for task_dim in output_dim:
        task_mapping = nn.Linear(n_d, task_dim, bias=False)
        initialize_non_glu(task_mapping, n_d, task_dim)
        self.multi_task_mappings.append(task_mapping)
    else:
      self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
      initialize_non_glu(self.final_mapping, n_d, output_dim)

  def forward(self, x: torch.Tensor) -> Any:
    """Forward pass of the TabNet encoder.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      output: output of the TabNet encoder
      m_loss: attention coefficient loss
      res: last hidden state of the TabNet encoder
    """
    steps_output, m_loss = self.encoder(x)
    res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

    if self.is_multi_task:
      # Result will be in list format
      out = []
      for task_mapping in self.multi_task_mappings:
        out.append(task_mapping(res))
    else:
      out = self.final_mapping(res)
    return out, m_loss, res

  def forward_masks(self, x: torch.Tensor) -> Any:
    """Forward masks.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      m_explain: output of the TabNet encoder.
      masks: list of attention masks.
    """
    return self.encoder.forward_masks(x)


class TabNet(torch.nn.Module):
  """TabNet model.

  This is the TabNet model. It takes the input and passes it through the
  TabNet encoder.
  """

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      n_d: int = 64,
      n_a: int = 64,
      n_steps: int = 6,
      gamma: float = 1.8,
      cat_idxs: Optional[List[int]] = None,
      cat_dims: Optional[List[int]] = None,
      cat_emb_dim: int = 1,
      n_independent: int = 2,
      n_shared: int = 2,
      epsilon: float = 1e-15,
      virtual_batch_size: int = 256,
      momentum: float = 0.95,
      mask_type: str = "sparsemax",
  ):
    """Defines TabNet network.

    Args:
        input_dim (int): Initial number of features.
        output_dim (int): Dimension of network output (e.g., one for regression,
          two for binary classification).
        n_d (int): Dimension of the prediction layer (usually between 4 and 64).
        n_a (int): Dimension of the attention layer (usually between 4 and 64).
        n_steps (int): Number of successive steps in the network (usually
          between 3 and 10).
        gamma (float): Float above 1, scaling factor for attention updates
          (usually between 1.0 to 2.0).
        cat_idxs (list of int): Indices of categorical columns in the dataset.
        cat_dims (list of int): Number of categories in each categorical column.
        cat_emb_dim (int or list of int): Size of categorical feature embeddings
          (int for all, list for specific).
        n_independent (int): Number of independent GLU layers in each GLU block
          (default 2).
        n_shared (int): Number of independent GLU layers in each GLU block
          (default 2).
        epsilon (float): Avoid log(0), keep very low.
        virtual_batch_size (int): Batch size for Ghost Batch Normalization.
        momentum (float): Value between 0 and 1 used for momentum in all batch
          norm.
        mask_type (str): Either "sparsemax" or "entmax": the masking function to
          use.
    """
    super().__init__()
    self.cat_idxs = cat_idxs if cat_idxs is not None else []
    self.cat_dims = cat_dims if cat_dims is not None else []
    self.cat_emb_dim = cat_emb_dim

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.n_d = n_d
    self.n_a = n_a
    self.n_steps = n_steps
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_independent = n_independent
    self.n_shared = n_shared
    self.mask_type = mask_type

    if self.n_steps <= 0:
      raise ValueError("n_steps should be a positive integer.")
    if self.n_independent == 0 and self.n_shared == 0:
      raise ValueError("n_shared and n_independent can't be both zero.")

    self.virtual_batch_size = virtual_batch_size
    self.embedder = EmbeddingGenerator(
        input_dim, cat_dims, cat_idxs, cat_emb_dim
    )
    self.post_embed_dim = self.embedder.post_embed_dim
    self.tabnet = TabNetNoEmbeddings(
        self.post_embed_dim,
        output_dim,
        n_d,
        n_a,
        n_steps,
        gamma,
        n_independent,
        n_shared,
        epsilon,
        virtual_batch_size,
        momentum,
        mask_type,
    )

  def forward(self, x: torch.Tensor) -> TabNetModelOutput:
    """Forward pass of the TabNet model.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      TabNetModelOutput: output of the TabNet model.
    """
    x = self.embedder(x)
    output, m_loss, last_hidden_state = self.tabnet(x)

    return TabNetModelOutput(
        loss=m_loss,
        last_hidden_state=last_hidden_state,
        output=output,
    )

  def forward_masks(self, x: torch.Tensor) -> Any:
    """Forward masks.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      m_explain: output of the TabNet encoder.
      masks: list of attention masks.
    """
    x = self.embedder(x)
    return self.tabnet.forward_masks(x)


def initialize_non_glu(
    module: torch.nn.Module, input_dim: int, output_dim: int
) -> None:
  """Initialize non-glu layers.

  Args:
      module: The module to initialize (e.g., a Linear layer).
      input_dim: The input dimension of the module.
      output_dim: The output dimension of the module.

  Returns:
      None.
  """
  gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
  torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
  return


def initialize_glu(
    module: torch.nn.Module, input_dim: int, output_dim: int
) -> None:
  """Initialize glu layers.

  Args:
      module: The module to initialize (e.g., a Linear layer).
      input_dim: The input dimension of the module.
      output_dim: The output dimension of the module.

  Returns:
    None.
  """
  gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
  torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
  return


class GBN(torch.nn.Module):
  """Ghost Batch Normalization.

  https://arxiv.org/abs/1705.08741
  """

  def __init__(
      self,
      input_dim: int,
      virtual_batch_size: int = 128,
      momentum: float = 0.01,
  ):
    super().__init__()

    self.input_dim = input_dim
    self.virtual_batch_size = virtual_batch_size
    self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
    res = [self.bn(x_) for x_ in chunks]

    return torch.cat(res, dim=0)


class AttentiveTransformer(torch.nn.Module):
  """Attentive transformer.

  This is the attention transformer part of the TabNet model. It takes the
  output of the previous transformer and computes the attention coefficient.
  """

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      virtual_batch_size: int = 256,
      momentum: float = 0.98,
      mask_type: str = "sparsemax",
  ) -> None:
    """Initialize an attention transformer."""
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim, bias=False)
    initialize_non_glu(self.fc, input_dim, output_dim)
    self.bn = GBN(
        output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
    )

    if mask_type == "sparsemax":
      self.selector = Sparsemax(dim=-1)
    elif mask_type == "softmax":
      self.selector = torch.nn.Softmax(dim=-1)
    else:
      raise NotImplementedError(
          "Please choose either sparsemax" + "or entmax as masktype"
      )

  def forward(
      self, priors: torch.Tensor, processed_feat: torch.Tensor
  ) -> torch.Tensor:
    """Forward pass of the TabNet encoder.

    Args:
      priors: prior for the attention mechanism. If None, will be initialized to
        ones.
      processed_feat: output of the previous transformer.

    Returns:
      output: output of the TabNet encoder
    """
    x = self.fc(processed_feat)
    x = self.bn(x)
    x = torch.mul(x, priors)
    x = self.selector(x)
    return x


class FeatTransformer(torch.nn.Module):
  """Feature transformer.

  This is the feature transformer part of the TabNet model. It takes the
  output of the previous transformer and computes the attention coefficient.
  """

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      shared_layers: Optional[torch.nn.ModuleList] = None,
      n_glu_independent: int = 2,
      virtual_batch_size: int = 256,
      momentum: float = 0.98,
  ) -> None:
    """Initialize an attention transformer.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension (after transformation).
        shared_layers: Optional shared layers (ModuleList) across steps.
        n_glu_independent: (int) Number of independent GLU layers.
        virtual_batch_size: Batch size for Ghost Batch Normalization.
        momentum: Momentum factor for batch normalization.

    Returns:
      None.
    """
    super().__init__()
    params = {
        "n_glu": n_glu_independent,
        "virtual_batch_size": virtual_batch_size,
        "momentum": momentum,
    }

    if shared_layers is None:
      # no shared layers
      self.shared = torch.nn.Identity()
      is_first = True
    else:
      self.shared = GLUBlock(
          input_dim,
          output_dim,
          n_glu=len(shared_layers),
          first=True,
          shared_layers=shared_layers,
          virtual_batch_size=virtual_batch_size,
          momentum=momentum,
      )
      is_first = False

    if n_glu_independent == 0:
      # no independent layers
      self.specifics = torch.nn.Identity()
    else:
      spec_input_dim = input_dim if is_first else output_dim
      self.specifics = GLUBlock(
          spec_input_dim, output_dim, first=is_first, **params
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
      x: torch.Tensor: input features of shape [batch_size, num_features]

    Returns:
      output: torch.Tensor output of the feature transformer.
    """
    x = self.shared(x)
    x = self.specifics(x)
    return x


class GLUBlock(torch.nn.Module):
  """Independent GLU block, specific to each step."""

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      n_glu: int = 2,
      first: bool = False,
      shared_layers: Optional[torch.nn.ModuleList] = None,
      virtual_batch_size: int = 128,
      momentum: float = 0.95,
  ):
    """Initialize an independent GLU block.

    Args:
        input_dim (int): Number of features coming as input.
        output_dim (int): Number of features going as output.
        n_glu (int): Number of independent GLU layers.
        first (bool): Whether this is the first layer of the block.
        shared_layers (torch.nn.ModuleList): List of shared layers.
        virtual_batch_size (int): Batch size for Ghost Batch Normalization.
        momentum (float): Value between 0 and 1 used for momentum in all batch
          norm.
    """
    super().__init__()
    self.n_glu = n_glu
    self.first = first
    self.shared_layers = shared_layers
    self.glu_layers = torch.nn.ModuleList()

    params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

    fc = shared_layers[0] if shared_layers else None
    self.glu_layers.append(GLULayer(input_dim, output_dim, fc=fc, **params))
    for glu_id in range(1, self.n_glu):
      fc = shared_layers[glu_id] if shared_layers else None
      self.glu_layers.append(GLULayer(output_dim, output_dim, fc=fc, **params))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
    scale = math.sqrt(0.5)
    if self.first:  # the first layer of the block has no scale multiplication
      x = self.glu_layers[0](x)
      layers_left = range(1, self.n_glu)
    else:
      layers_left = range(self.n_glu)

    for glu_id in layers_left:
      x = torch.add(x, self.glu_layers[glu_id](x))
      x = x * scale
    return x


class GLULayer(torch.nn.Module):
  """GLU layer."""

  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      fc: Optional[nn.Linear] = None,
      virtual_batch_size: int = 128,
      momentum: float = 0.02,
  ):
    super().__init__()

    self.output_dim = output_dim
    if fc:
      self.fc = fc
    else:
      self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
    initialize_glu(self.fc, input_dim, 2 * output_dim)

    self.bn = GBN(
        2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
    )
    # self.bn = BatchNorm1d(2*output_dim, momentum=momentum)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc(x)
    x = self.bn(x)
    out = torch.mul(
        x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :])
    )
    return out


class EmbeddingGenerator(torch.nn.Module):
  """Classical embeddings generator."""

  def __init__(
      self,
      input_dim: int,
      cat_dims: List[int],
      cat_idxs: List[int],
      cat_emb_dim: int,
  ):
    """This is an embedding module for an entire set of features.

    Args:
        input_dim: Number of features coming as input (number of columns).
        cat_dims: Number of modalities for each categorical feature. If the list
          is empty, no embeddings will be done.
        cat_idxs: Positional index for each categorical feature in inputs.
        cat_emb_dim: Embedding dimension for each categorical feature. If int,
          the same embedding dimension will be used for all categorical
          features.
    """
    super().__init__()
    if not cat_dims and not cat_idxs:
      self.skip_embedding = True
      self.post_embed_dim = input_dim
      return
    elif (not cat_dims) ^ (not cat_idxs):
      if not cat_dims:
        msg = (
            "If cat_idxs is non-empty, cat_dims must be defined as a list of"
            " same length."
        )
      else:
        msg = (
            "If cat_dims is non-empty, cat_idxs must be defined as a list of"
            " same length."
        )
      raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
      msg = "The lists cat_dims and cat_idxs must have the same length."
      raise ValueError(msg)

    self.skip_embedding = False
    if isinstance(cat_emb_dim, int):
      self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
      self.cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(self.cat_emb_dims) != len(cat_dims):
      msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                        and {len(cat_dims)}"""
      raise ValueError(msg)
    self.post_embed_dim = int(
        input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
    )

    self.embeddings: torch.nn.ModuleList = torch.nn.ModuleList()

    # Sort dims by cat_idx
    sorted_idxs = np.argsort(cat_idxs)
    cat_dims = [cat_dims[i] for i in sorted_idxs]
    self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

    for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
      self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

    # record continuous indices
    self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
    self.continuous_idx[cat_idxs] = 0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply embeddings to inputs.

    Inputs should be (batch_size, input_dim)
    Outputs will be of size (batch_size, self.post_embed_dim)

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      post_embeddings: output features of shape [batch_size, post_embed_dim]
    """
    if self.skip_embedding:
      # no embeddings required
      return x

    cols = []
    cat_feat_counter = 0
    for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
      # Enumerate through continuous idx boolean mask to apply embeddings
      if is_continuous:
        cols.append(x[:, feat_init_idx].float().view(-1, 1))
      else:
        cols.append(
            self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
        )
        cat_feat_counter += 1
    # concat
    post_embeddings = torch.cat(cols, dim=1)
    return post_embeddings


class RandomObfuscator(torch.nn.Module):
  """Create and applies obfuscation masks."""

  def __init__(self, pretraining_ratio: float):
    """This create random obfuscation for self suppervised pretraining.

    Args:
        pretraining_ratio: Ratio of features to randomly discard for
          reconstruction.

    Returns:
        None.
    """
    super().__init__()
    self.pretraining_ratio = pretraining_ratio

  def forward(self, x: torch.Tensor) -> Any:
    """Generate random obfuscation mask.

    Args:
      x: input features of shape [batch_size, num_features]

    Returns:
      masked input and obfuscated variables.
    """
    obfuscated_vars = torch.bernoulli(
        self.pretraining_ratio * torch.ones(x.shape)
    ).to(x.device)
    masked_input = torch.mul(1 - obfuscated_vars, x)
    return masked_input, obfuscated_vars


class Sparsemax(nn.Module):
  """Sparsemax function."""

  def __init__(self, dim: Optional[int] = None):
    """Initialize sparsemax activation.

    Args:
        dim (int, optional): The dimension over which to apply the sparsemax
          function.
    """
    super().__init__()

    self.dim = -1 if dim is None else dim

  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Forward function.

    Args:
        input_tensor (torch.Tensor): Input tensor. First dimension should be the
          batch size

    Returns:
        torch.Tensor: [batch_size x number_of_logits] Output tensor
    """
    # Sparsemax currently only handles 2-dim tensors,
    # so we reshape to a convenient shape and reshape back after sparsemax
    input_tensor = input_tensor.transpose(0, self.dim)
    original_size = input_tensor.size()
    input_tensor = input_tensor.reshape(input_tensor.size(0), -1)
    input_tensor = input_tensor.transpose(0, 1)
    dim = 1

    number_of_logits = input_tensor.size(dim)

    # Translate input by max for numerical stability
    input_tensor = input_tensor - torch.max(
        input_tensor, dim=dim, keepdim=True
    )[0].expand_as(input_tensor)

    # Sort input in descending order.
    # (NOTE: Can be replaced with linear time selection method described here:
    # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
    zs = torch.sort(input=input_tensor, dim=dim, descending=True)[0]
    range_tensor = torch.arange(
        start=1,
        end=number_of_logits + 1,
        step=1,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    ).view(1, -1)
    range_tensor = range_tensor.expand_as(zs)

    # Determine sparsity of projection
    bound = 1 + range_tensor * zs
    cumulative_sum_zs = torch.cumsum(zs, dim)
    is_gt = torch.gt(bound, cumulative_sum_zs).type(input_tensor.type())
    k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]

    # Compute threshold function
    zs_sparse = is_gt * zs

    # Compute taus
    taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
    taus = taus.expand_as(input_tensor)

    # Sparsemax output
    self.output = torch.max(torch.zeros_like(input_tensor), input_tensor - taus)

    # Reshape back to original shape
    output = self.output
    output = output.transpose(0, 1)
    output = output.reshape(original_size)
    output = output.transpose(0, self.dim)

    return output

  def backward(self, grad_output: torch.FloatTensor) -> torch.FloatTensor:
    """Backward function.

    Args:
        grad_output (torch.Tensor): Gradient of outputs of forward function

    Returns:
        torch.Tensor: Gradient of inputs of forward function
    """
    super().backward(grad_output)
    dim = 1

    nonzeros = torch.ne(self.output, 0).type(grad_output.type())
    summation = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(
        nonzeros, dim=dim
    )
    self.grad_input = nonzeros * (
        grad_output - summation.expand_as(grad_output)
    )

    return self.grad_input
