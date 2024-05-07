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

from typing import Callable, List, Optional, Union
import torch
from torch import nn


class SimSiamProjectionHead(nn.Module):
  """SimSiam projection head.

  This is the projection head used in SimSiam.
  """

  def __init__(
      self,
      in_dim: int = 768,
      hidden_dim: int = 256,
      out_dim: int = 256,
  ) -> None:
    """SimSiam projection head.

    Args:
        in_dim: input dimension
        hidden_dim: hidden dimension
        out_dim: output dimension
    """
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)
    )
    self.layer3 = nn.Sequential(
        nn.Linear(hidden_dim, out_dim),
    )
    self.num_layers = 3

  def set_layers(self, num_layers: int) -> None:
    """Set the number of layers in the projection head.

    Args:
        num_layers: number of layers in the projection head
    """
    self.num_layers = num_layers

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
        x: input tensor

    Returns:
        output tensor
    """
    if self.num_layers == 3:
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
    elif self.num_layers == 2:
      x = self.layer1(x)
      x = self.layer3(x)
    else:
      raise ValueError(
          f"num_layers {self.num_layers} is not supported in"
          " SimSiamProjectionHead"
      )
    return x


class OneLayerProjectionHead(nn.Module):
  """One layer projection head.

  This is the projection head with one linear layer followed by ReLU.
  """

  def __init__(
      self,
      in_dim: int,
      hidden_dim: Optional[int] = None,
      out_dim: int = 256,
  ) -> None:
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
        x: input tensor

    Returns:
        output tensor
    """
    x = self.layer1(x)
    return x


class MLPProjectionHead(nn.Module):
  """MLP projection head.

  This is the projection head with one linear layer followed by ReLU.
  """

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      hidden_dim: Optional[Union[int, List[int]]] = None,
      dropout: float = 0.5,
      activation: Callable[..., nn.Module] = nn.ReLU,
      normalization: Optional[Callable[..., nn.Module]] = None,
  ) -> None:
    super().__init__()

    layers = nn.ModuleList()

    if hidden_dim is None:
      hidden_dim = []

    if isinstance(hidden_dim, int):
      hidden_dim = [hidden_dim]

    for dim in hidden_dim:
      layers.append(nn.Linear(in_dim, dim))
      if normalization:
        layers.append(normalization(dim))
      layers.append(activation())
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      in_dim = dim
    layers.append(nn.Linear(in_dim, out_dim))
    self.model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Args:
        x: input tensor

    Returns:
        output tensor
    """
    return self.model(x)
