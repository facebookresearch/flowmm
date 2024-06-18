"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
from geoopt import Euclidean
from geoopt.utils import size2shape


class EuclideanWithLogProb(Euclidean):
    def random_normal(
        self, *size, mean=0.0, std=1.0, device=None, dtype=None
    ) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
        return torch.randn(*size, device=mean.device, dtype=mean.dtype) * std + mean

    def random_base(self, *args, **kwargs):
        return self.random_normal(*args, **kwargs)

    random = random_base

    def normal_logprob(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.ndim == 0:
            raise NotImplementedError()
        else:
            dist = torch.distributions.normal.Normal(
                torch.zeros_like(x[-self.ndim :]),
                torch.ones_like(x[-self.ndim :]),
                validate_args=False,  # doesn't work with vmap yet https://github.com/pytorch/functorch/issues/257
            )
            dist = torch.distributions.independent.Independent(dist, 1)
        return dist.log_prob(x)

    def base_logprob(self, *args, **kwargs) -> torch.Tensor:
        return self.normal_logprob(*args, **kwargs)

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)
