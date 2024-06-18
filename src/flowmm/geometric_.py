"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch


def mask_2d_to_batch(mask_2d: torch.BoolTensor) -> torch.LongTensor:
    ids = torch.arange(mask_2d.size(0), device=mask_2d.device)
    return ids.unsqueeze(-1).expand(-1, mask_2d.size(1))[mask_2d]
