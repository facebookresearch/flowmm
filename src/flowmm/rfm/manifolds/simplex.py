"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math

import geoopt
import torch
from geoopt import Euclidean, Manifold, ManifoldTensor
from geoopt.utils import size2shape

from flowmm.rfm.manifolds.masked import MaskedManifold


class FlatDirichletSimplex(Euclidean):
    """Represents a simplex with a Dirichlet distribution.
    x = [x_1, x_2, ..., x_D], \sum_i x_i = 1 for i \in 1, 2, ... D.
    """

    name = "FlatDirichletSimplex"
    reversible = False

    def __init__(self):
        super().__init__(ndim=1)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        ok = torch.all(x >= 0.0)
        if not ok:
            return False, f"not all elements were greater or equal to 0"

        s = x.sum(dim=-1)
        ok = torch.allclose(s, torch.ones_like(s), atol=atol, rtol=rtol)
        if not ok:
            return False, f"`x.sum(dim=-1) != 1` with {atol=}, {rtol=}"
        return True, None

    def _check_vector_on_tangent(
        self, _: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        s = u.sum(dim=-1)
        target = torch.zeros_like(s)  # already all zeros, no need for mask
        ok = torch.allclose(s, target, atol=atol, rtol=rtol)
        if not ok:
            return False, f"`u.sum(dim=-1) != 0` with {atol=}, {rtol=}"
        return True, None

    @staticmethod
    def _project_to_z_simplex(x: torch.Tensor, z: float) -> torch.Tensor:
        """Project x's last dimension onto the unit simplex:
            P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2

        allows batching in the first+ dimensions

        Adapted from:
        https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        and
        https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
        """
        k = x.size(-1)

        x_sort_rev, _ = torch.sort(x, descending=True, dim=-1)
        cumsum_x_sort_rev = torch.cumsum(x_sort_rev, dim=-1) - z * torch.ones_like(x)

        ind = torch.arange(k, device=x.device) + 1
        cond = x_sort_rev - (cumsum_x_sort_rev / ind) > 0
        rho = torch.count_nonzero(cond, dim=-1)
        # theta = cumsum_x_sorted_rev[torch.arange(x.size(-2)), rho - 1] / rho  # doesn't work batched
        theta = (
            torch.gather(
                cumsum_x_sort_rev, dim=-1, index=rho.unsqueeze(-1) - 1
            ).squeeze(-1)
            / rho
        )
        return torch.maximum(x - theta.unsqueeze(-1), torch.zeros_like(x))

    @classmethod
    def projx(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._project_to_z_simplex(x, z=1.0)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # one reference: https://math.stackexchange.com/questions/3593688/calculating-the-tangent-space-of-the-manifold-of-probability-measures-on-a-finit
        # how to compute the tangent space at a point generally:
        # https://math.stackexchange.com/questions/2311754/tangent-space-of-circle-at-a-point
        # furthermore...
        # we need to make sure that our vectors don't go outside the boundary
        # https://math.ucr.edu/~res/math260s10/manwithbdy.pdf
        # i.e. our tangent space is a manifold with a boundary
        # Although, we handle this by projecting back to x
        u = u - u.mean(-1, keepdims=True)
        return super().proju(x, u)

    @staticmethod
    def random_base(*size, dtype=None, device=None) -> torch.Tensor:
        d = torch.distributions.Dirichlet(
            torch.ones(size[-1], dtype=dtype, device=device)
        )
        return d.sample(size[:-1])

    random = random_base

    @staticmethod
    def base_logprob(x: torch.Tensor) -> torch.Tensor:
        size = x.shape
        concentration = torch.ones(size[-1], dtype=x.dtype, device=x.device)
        d = torch.distributions.Dirichlet(concentration)
        return d.log_prob(x)

    # not implemented
    def egrad2rgrad(self, *args, **kwargs):
        raise NotImplementedError()

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        raise NotImplementedError()

    def abits_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MultiAtomFlatDirichletSimplex(MaskedManifold, FlatDirichletSimplex):
    """Represents a simplex with a Dirichlet distribution.
    x = [x_1, x_2, ..., x_D], \sum_i x_i = 1 for i \in 1, 2, ... D.

    This manifold can be thought of as having an extra dimension
    [..., NUM_ATOMS, NUM_CATEGORIES]
    """

    name = "MultiAtomFlatDirichletSimplex"
    reversible = False

    def __init__(self, num_categories: int, num_atoms: int, max_num_atoms: int):
        FlatDirichletSimplex.__init__(self)
        self.max_num_atoms = max_num_atoms
        self.num_categories = num_categories
        self.register_buffer("mask", self._get_mask(num_atoms, max_num_atoms))

    @property
    def dim_m1(self) -> int:
        return self.num_categories

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        ok = (x.shape[-1] % self.num_categories) == 0
        if not ok:
            return False, f"{x.shape[-1]=} % {self.num_categories=} !=0"

        x = self.reshape_and_mask(x.shape, x)

        ok = torch.all(x >= 0.0)
        if not ok:
            return False, f"not all elements were greater or equal to 0"

        s = x.sum(dim=-1)
        target = self.mask.to(s).squeeze() * torch.ones_like(s)
        ok = torch.allclose(s, target, atol=atol, rtol=rtol)
        if not ok:
            return False, f"`x.sum(dim=-1) != 1` with {atol=}, {rtol=}"
        return True, None

    def _check_vector_on_tangent(
        self, _: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        u = self.reshape_and_mask(u.shape, u)
        return super()._check_vector_on_tangent(_, u)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        return Euclidean(ndim=self.ndim).inner(
            x, u, v, keepdim=keepdim
        ) / self.mask.sum().to(u)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """project to closest point on simplex"""
        initial_shape = x.shape
        x = self.reshape_and_mask(initial_shape, x)
        px = self._project_to_z_simplex(x, z=1.0)
        return self.mask_and_reshape(initial_shape, px)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """last dim sums to zero"""
        # one reference: https://math.stackexchange.com/questions/3593688/calculating-the-tangent-space-of-the-manifold-of-probability-measures-on-a-finit
        # how to compute the tangent space at a point generally:
        # https://math.stackexchange.com/questions/2311754/tangent-space-of-circle-at-a-point
        # furthermore...
        # we need to make sure that our vectors don't go outside the boundary
        # https://math.ucr.edu/~res/math260s10/manwithbdy.pdf
        # i.e. our tangent space is a manifold with a boundary
        # Although, we handle this by projecting back to x
        initial_shape = u.shape
        u = self.reshape_and_mask(initial_shape, u)
        out = u - u.mean(-1, keepdims=True)
        out = self.mask_and_reshape(initial_shape, out)
        return Euclidean(self.ndim).proju(x, out)

    def _get_dirichlet(self, dtype, device) -> torch.distributions.Dirichlet:
        concentration = torch.ones(self.num_categories, dtype=dtype, device=device)
        return torch.distributions.Dirichlet(concentration, validate_args=False)

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        assert (
            size[-1] == self.max_num_atoms * self.num_categories
        ), "last dimension must be compatible with max_num_atoms and num_categories"

        d = self._get_dirichlet(dtype, device)
        # out = d.sample((*size[:-1], self.max_num_atoms))  # This does not work with vmap yet!
        # replace with...
        shape = d._extended_shape((*size[:-1], self.max_num_atoms))
        concentration = d.concentration.expand(shape)

        out = torch._sample_dirichlet(concentration)
        return self.mask_and_reshape(size, out)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        d = self._get_dirichlet(x.dtype, x.device)
        initial_shape = x.shape
        x = self.reshape_and_mask(initial_shape, x)
        not_mask = self.mask.logical_not().to(x)
        # these are simply values on the simplex (to avoid support issues), later masked out
        non_atoms = not_mask * torch.nn.functional.softmax(torch.rand_like(x), dim=-1)
        xp = x + non_atoms
        lps = d.log_prob(xp)
        return torch.sum(self.mask.to(x).squeeze() * lps, dim=-1)

    def extra_repr(self):
        return f"""
        num_atoms={self.num_atoms},
        max_num_atoms={self.max_num_atoms},
        num_categories={self.num_categories},
        """

    def metric_normalized(self, _: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)


def usinc(theta: torch.Tensor) -> torch.Tensor:
    """unnormalized sinc function"""
    out = theta.sin().div(theta)
    out2 = torch.where(
        theta == 0.0,
        torch.ones_like(theta),
        out,
    )
    out3 = torch.where(
        theta.isinf(),
        torch.zeros_like(theta),
        out2,
    )
    return out3


if __name__ == "__main__":
    s = MultiAtomFlatDirichletSimplex(num_categories=3, num_atoms=5, max_num_atoms=9)
    eg = s.random(10, 9 * 3)
    assert s.check_point_on_manifold(eg)
    lp = s.base_logprob(eg)

    u = s.proju(eg, eg)
    assert s.check_vector_on_tangent(eg, u)

    s = MultiAtomFlatDirichletSimplex(1, 1, 2)
    x = s.projx(torch.zeros(2))

    import numpy as np

    # messing around with the sphere
    sphere = geoopt.Sphere()
    scale = 2.0
    radius_2_sphere = geoopt.Scaled(sphere, scale)
    p1 = torch.tensor([-1.0, 0.0])
    p2 = torch.tensor([0.0, 1.0])
    np.testing.assert_allclose(sphere.dist(p1, p2), np.pi / 2)
    np.testing.assert_allclose(radius_2_sphere.dist(p1, p2), np.pi)

    # messing around with the probability simplex
    dim = 1
    # tangent vectors
    v = torch.randn(dim + 1)
    v = v - v.sum() / v.shape[0]
    u = torch.randn(dim + 1)
    u = u - u.sum() / u.shape[0]
    # points on manifold
    b = torch.zeros(dim + 1)
    b[0] = 1
    p = torch.rand(dim + 1)
    p[-1] = 1 - p[:-1].sum()
    q = torch.rand(dim + 1)
    q[-1] = 1 - q[:-1].sum()

    print("")
