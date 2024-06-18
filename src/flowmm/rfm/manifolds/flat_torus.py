"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math

import torch
from geoopt import Euclidean

from flowmm.rfm.manifolds.masked import MaskedManifold


class FlatTorus01(Euclidean):
    """Represents a flat torus on the [0, 1]^D subspace.

    Isometric to the product of 1-D spheres."""

    name = "FlatTorus01"
    reversible = False

    def __init__(self, ndim=1):
        super().__init__(ndim=ndim)

    @staticmethod
    def expmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (x + u) % 1.0

    @staticmethod
    def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = 2 * math.pi * (y - x)
        return torch.atan2(torch.sin(z), torch.cos(z)) / (2 * math.pi)

    @staticmethod
    def projx(x: torch.Tensor) -> torch.Tensor:
        return x % 1.0

    @staticmethod
    def random_uniform(*size, dtype=None, device=None) -> torch.Tensor:
        return torch.rand(*size, dtype=dtype, device=device)

    @staticmethod
    def uniform_logprob(x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[..., 0], 0.0)

    def random_base(self, *args, **kwargs):
        return self.random_uniform(*args, **kwargs)

    random = random_base

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)

    def extra_repr(self):
        return "ndim={}".format(self.ndim)

    def metric_normalized(self, _: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def abits_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)

    # not implemented
    def component_inner(self, *args, **kwargs):
        raise NotImplementedError()

    def norm(self, *args, **kwargs):
        raise NotImplementedError()

    def dist(self, *args, **kwargs):
        raise NotImplementedError()

    def dist2(self, *args, **kwargs):
        raise NotImplementedError()

    def egrad2rgrad(self, *args, **kwargs):
        raise NotImplementedError()


class MaskedNoDriftFlatTorus01(MaskedManifold, FlatTorus01):
    """Represents a flat torus on the [0, 1]^((N-1)*D) subspace.

    The first atom (n=1) gets fixed to the origin.

    Isometric to the product of (N-1) 1-D spheres."""

    name = "MaskedNoDriftFlatTorus01"
    reversible = False

    def __init__(
        self,
        dim_coords: int,
        num_atoms: int,
        max_num_atoms: int,
    ):
        super().__init__(ndim=1)
        self.dim_coords = dim_coords
        self.max_num_atoms = max_num_atoms
        mask = torch.zeros(max_num_atoms, dtype=torch.bool)
        mask[:num_atoms] = torch.ones(num_atoms, dtype=torch.bool)
        self.register_buffer("mask", mask.unsqueeze(-1))

    @property
    def dim_m1(self) -> int:
        return self.dim_coords

    def _get_mean_velocity(
        self, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> torch.Tensor:
        u = self.reshape_and_mask(u.shape, u)
        return torch.sum(
            self.mask * u,
            dim=-2,
            keepdim=True,
        ) / self.mask.sum(dim=[-2, -1])

    def _check_vector_on_tangent(
        self, _: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        mean = self._get_mean_velocity(u)
        target = torch.zeros_like(mean)  # already all zeros, no need for mask
        ok = torch.allclose(mean, target, atol=atol, rtol=rtol)
        if not ok:
            return False, f"`mean(u) != 0` with {atol=}, {rtol=}"
        return True, None

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        return Euclidean(ndim=self.ndim).inner(
            x, u, v, keepdim=keepdim
        ) / self.mask.sum().to(u)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        initial_shape = x.shape
        x = self.reshape_and_mask(initial_shape, x)
        x = self.mask_and_reshape(initial_shape, x)
        return super().projx(x)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """removes drift velocity"""
        initial_shape = u.shape
        u = self.reshape_and_mask(initial_shape, u)

        # remove the drift
        # regular minus is okay since the tangent space is euclidean
        mean = torch.sum(
            u,
            dim=-2,
            keepdim=True,
        ) / self.mask.to(
            u
        ).sum(dim=[-2, -1])
        out = u - mean
        out = self.mask_and_reshape(initial_shape, out)
        return super().proju(x, out)

    def random_uniform(self, *size, dtype=None, device=None) -> torch.Tensor:
        assert (
            size[-1] == self.max_num_atoms * self.dim_coords
        ), "last dimension must be compatible with max_num_atoms and dim_coords"
        out = torch.rand(*size, dtype=dtype, device=device)
        return self.projx(out)

    random = random_uniform

    def extra_repr(self):
        return f"dim_coords={self.dim_coords}"


class MaskedNoDriftFlatTorus01WrappedNormal(MaskedNoDriftFlatTorus01):
    name = "MaskedNoDriftFlatTorus01WrappedNormal"

    def __init__(
        self,
        dim_coords: int,
        num_atoms: int,
        max_num_atoms: int,
        scale: float = 0.1,
    ):
        super().__init__(dim_coords, num_atoms, max_num_atoms)
        self.scale = scale

    def random_wrapped_normal(self, *size, dtype=None, device=None) -> torch.Tensor:
        assert (
            size[-1] == self.max_num_atoms * self.dim_coords
        ), "last dimension must be compatible with max_num_atoms and dim_coords"
        out = torch.randn(*size, dtype=dtype, device=device) * self.scale
        return self.projx(out)

    random = random_wrapped_normal

    def base_logprob(self, *args, **kwargs):
        raise NotImplementedError()

    def extra_repr(self):
        return f"{super().extra_repr}, scale={self.scale}"


class FlatTorus01FixFirstAtomToOrigin(MaskedNoDriftFlatTorus01):
    """Represents a flat torus on the [0, 1]^((N-1)*D) subspace.

    Isometric to the product of (N-1) 1-D spheres."""

    name = "FlatTorus01FixFirstCoordToOrigin"
    reversible = False

    def _check_first_is_zero(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        ok = (x.shape[-1] % self.dim_coords) == 0
        if not ok:
            return False, f"{x.shape[-1]=} % {self.dim_coords=} !=0"

        x = self.reshape_and_mask(x.shape, x)
        first_coord = x[..., : self.dim_coords]
        ok = torch.allclose(
            first_coord, torch.zeros_like(first_coord), atol=atol, rtol=rtol
        )
        if not ok:
            return False, f"`first_coord(x) != 0` with {atol=}, {rtol=}"
        return True, None

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> tuple[bool, str | None] | bool:
        return self._check_first_is_zero(x, atol=atol, rtol=rtol)

    def _translate_first_coord_to_zero(self, x: torch.Tensor) -> torch.Tensor:
        initial_shape = x.shape
        x = self.reshape_and_mask(initial_shape, x)
        first_coord = x[..., 0:1, :]
        offset = self.logmap(first_coord, torch.zeros_like(first_coord))
        out = self.expmap(x, offset)
        return self.mask_and_reshape(initial_shape, out)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """translates everything so that the first coordinate is zero"""
        x = self._translate_first_coord_to_zero(x)
        return x % 1.0


class FlatTorus01FixFirstAtomToOriginWrappedNormal(FlatTorus01FixFirstAtomToOrigin):
    name = "FlatTorus01FixFirstCoordToOriginWrappedNormal"

    def __init__(
        self,
        dim_coords: int,
        num_atoms: int,
        max_num_atoms: int,
        scale: float = 0.1,
    ):
        super().__init__(dim_coords, num_atoms, max_num_atoms)
        self.scale = scale

    def random_wrapped_normal(self, *size, dtype=None, device=None) -> torch.Tensor:
        assert (
            size[-1] == self.max_num_atoms * self.dim_coords
        ), "last dimension must be compatible with max_num_atoms and dim_coords"
        out = torch.randn(*size, dtype=dtype, device=device) * self.scale
        return self.projx(out)

    random = random_wrapped_normal

    def base_logprob(self, *args, **kwargs):
        raise NotImplementedError()

    def extra_repr(self):
        return f"{super().extra_repr}, scale={self.scale}"


if __name__ == "__main__":
    m = FlatTorus01FixFirstAtomToOrigin(2, 5, 9)
    m.random(10, 2 * 9)
    print("")
