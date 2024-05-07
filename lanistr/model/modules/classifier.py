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

import torch


class PredictionMLP(torch.nn.Module):
  """A Multi-Layer Perceptron (MLP) for prediction tasks with a bottleneck structure."""

  def __init__(
      self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 2048
  ):
    """Initializes the PredictionMLP.

    Args:
      in_dim: The input dimension of the MLP.
      hidden_dim: The hidden dimension of the MLP.
      out_dim: The output dimension of the MLP.
    """
    super().__init__()
    self.layer1 = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(inplace=True)
    )
    self.layer2 = torch.nn.Linear(hidden_dim, out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the PredictionMLP.

    Args:
      x: The input tensor.

    Returns:
      The output tensor.
    """
    x = self.layer1(x)
    x = self.layer2(x)
    return x
