"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import torch
from geoopt import Manifold, ManifoldTensor
from geoopt.utils import broadcast_shapes, size2shape


class NullManifold(Manifold):
    name = "NullManifold"
    reversible = False

    def __init__(self, ndim: int = 1):
        super().__init__()
        self.ndim = ndim

    def _check_point_on_manifold(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[bool, str | None]:
        return True, None

    def _check_vector_on_tangent(
        self, _: torch.Tensor, u: torch.Tensor, **kwargs
    ) -> tuple[bool, str | None]:
        if torch.equal(u, torch.zeros_like(u)):
            return True, None
        else:
            return False, f"The vector {u} is not 0."

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            inner = torch.zeros_like(u)
        else:
            inner = torch.zeros_like(u * v)
        if self.ndim > 0:
            inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
            x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim
        else:
            x_shape = x.shape
        i_shape = inner.shape
        target_shape = broadcast_shapes(x_shape, i_shape)
        return inner.expand(target_shape)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return torch.zeros_like(u).expand(target_shape)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, y.shape, v.shape)
        return torch.zeros_like(v).expand(target_shape)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return torch.zeros_like(u).expand(target_shape)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y - x)

    def random_zero(self, *size, dtype=None, device=None) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        return torch.zeros(*size, device=device, dtype=dtype)

    def random_base(self, *args, **kwargs):
        return self.random_zero(*args, **kwargs)

    random = random_base

    @staticmethod
    def delta_logprob(x: torch.Tensor) -> torch.Tensor:
        """this gives prob = 1 for anything"""
        return torch.full_like(x[..., 0], 0.0)

    def base_logprob(self, *args, **kwargs):
        return self.delta_logprob(*args, **kwargs)

    def metric_normalized(self, _: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)


class NullManifoldWithDeltaRandom(NullManifold):
    name = "NullManifoldWithDeltaRandom"
    reversible = False

    def __init__(self, point: torch.Tensor, ndim: int = 1):
        super().__init__()
        self.ndim = ndim
        self.register_buffer("point", point)

    def random_delta(self, *size, dtype=None, device=None) -> ManifoldTensor:
        self._assert_check_shape(size2shape(*size), "x")

        # use the default locations to generate so it works in a product manifold
        zeros = torch.zeros(*size, device=device, dtype=dtype)
        if dtype is None:
            dtype = zeros.dtype
        if device is None:
            device = zeros.device

        out = self.point.to(dtype=dtype, device=device) + zeros  # for broadcasting
        return out

    def random_base(self, *args, **kwargs):
        return self.random_delta(*args, **kwargs)

    random = random_base
