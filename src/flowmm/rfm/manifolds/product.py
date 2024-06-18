"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from functools import partial

import geoopt
import torch
from geoopt import Manifold, ProductManifold
from geoopt.manifolds.product import _calculate_target_batch_dim
from torch.func import jvp

from manifm.manifolds.utils import geodesic


class ProductManifoldWithLogProb(ProductManifold):
    def random_combined(
        self, *size, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        shape = geoopt.utils.size2shape(*size)
        self._assert_check_shape(shape, "x")
        batch_shape = shape[:-1]
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            points.append(
                manifold.random(*(batch_shape + shape), dtype=dtype, device=device)
            )
        tensor = self.pack_point(*points)
        return tensor

    random = random_combined

    def base_logprob(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = []
        begin_ind = 0
        for manifold, shape in zip(self.manifolds, self.shapes):
            assert len(shape) == 1
            end_ind = begin_ind + shape[0]
            log_probs.append(manifold.base_logprob(x[..., begin_ind:end_ind]))
            begin_ind = end_ind
        if len(log_probs) != len(self.manifolds):
            raise ValueError(
                "{} tensors expected, got {}".format(
                    len(self.manifolds), len(log_probs)
                )
            )
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def metric_normalized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.metric_normalized(point, tangent)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        return torch.cat(mapped_tensors, -1)

    def abits_clamp(self, x: torch.Tensor) -> torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim())
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            mapped = manifold.abits_clamp(point)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        return torch.cat(mapped_tensors, -1)

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        logdetG = torch.zeros_like(x).sum(-1)
        begin_ind = 0
        for manifold, shape in zip(self.manifolds, self.shapes):
            assert len(shape) == 1
            end_ind = begin_ind + shape[0]
            logdetG += manifold.logdetG(x[..., begin_ind:end_ind])
            begin_ind = end_ind
        return logdetG

    @staticmethod
    def _cond_u(
        manifold: Manifold, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(manifold, "geodesic"):
            # specific geodesic when the implementation calls for it
            mani_geo = partial(manifold.geodesic, x0, x1)
            x_t, u_t = jvp(mani_geo, (t,), (torch.ones_like(t).to(t),))
        else:
            # generic geodesic with expmap and logmap
            path = geodesic(manifold, x0, x1)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
        return x_t, u_t

    def cond_u(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_batch_dim = _calculate_target_batch_dim(x0.dim(), x1.dim(), t.dim())
        x_ts, u_ts = [], []
        for i, manifold in enumerate(self.manifolds):
            x0p = self.take_submanifold_value(x0, i)
            x1p = self.take_submanifold_value(x1, i)
            x_t, u_t = self._cond_u(manifold, x0p, x1p, t)

            x_t = x_t.reshape((*x_t.shape[:target_batch_dim], -1))
            u_t = u_t.reshape((*u_t.shape[:target_batch_dim], -1))
            x_ts.append(x_t)
            u_ts.append(u_t)
        return torch.cat(x_ts, -1), torch.cat(u_ts, -1)
