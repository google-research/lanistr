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
from torch import nn
from torch.nn import functional as F


def NegativeCosineSimilarityLoss(
    p: torch.FloatTensor,
    z: torch.FloatTensor,
    version: str = 'simplified',
) -> torch.FloatTensor:
  """Negative cosine similarity loss.

  Args:
      p: predicted values
      z: target values
      version: 'original' or 'simplified'

  Returns:
      scalar loss.
  """
  if version == 'original':
    z = z.detach()  # stop gradient
    p = F.normalize(p, dim=1)  # l2-normalize
    z = F.normalize(z, dim=1)  # l2-normalize
    return -(p * z).sum(dim=1).mean()

  elif version == 'simplified':  # same thing, much faster.
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
  else:
    raise ValueError('Invalid version')


class MaskedMSELoss(nn.Module):
  """Masked MSE Loss."""

  def __init__(self, reduction: str = 'mean'):

    super().__init__()

    self.reduction = reduction
    self.mse_loss = nn.MSELoss(reduction=self.reduction)

  def forward(
      self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor
  ) -> torch.Tensor:
    """Compute the loss between a target value and a prediction.

    Args:
        y_pred: Estimated values
        y_true: Target values
        mask: boolean tensor with 0s at places where values should be ignored
          and 1s where they should be considered

    Returns:
    if reduction == 'none':
        (num_active,) Loss for each active batch element as a tensor with
        gradient attached.
    if reduction == 'mean':
        scalar mean loss over batch as a tensor with gradient attached.
    """

    # for this particular loss, one may also elementwise multiply y_pred and
    # y_true with the inverted mask
    masked_pred = torch.masked_select(y_pred, mask)
    masked_true = torch.masked_select(y_true, mask)

    return self.mse_loss(masked_pred, masked_true)
