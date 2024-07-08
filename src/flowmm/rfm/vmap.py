"""Copyright (c) Meta Platforms, Inc. and affiliates."""
from __future__ import annotations

from typing import Any

import torch
from geoopt import Manifold

from flowmm.rfm.manifolds.null import NullManifoldWithDeltaRandom


class MethodToForward(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        method: str,
    ) -> None:
        super().__init__()
        self.module = module
        self.method = method

    def forward(self, *args, **kwargs) -> Any:
        method = getattr(self.module, self.method)
        return method(*args, **kwargs)


class Projx(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="projx")


class Proju(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="proju")


class Random(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="random")


class Inner(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="inner")


class BaseLogProb(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="base_logprob")


class MetricNormalized(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="metric_normalized")


class LogDetG(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="logdetG")


class CondU(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="cond_u")


class ABitsClamp(MethodToForward):
    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__(module=module, method="abits_clamp")


class VMapManifolds(torch.nn.Module):
    def __init__(self, manifolds: list[Manifold]) -> None:
        super().__init__()
        self.manifolds = torch.nn.ModuleList(manifolds)
        # creating the params and buffers is slow
        # we use an abstract Module with the same params/buffer hierarchy
        # to create them at initialization, but not at "method time"
        self._params, self._buffers = torch.func.stack_module_state(
            [MethodToForward(m, method="abstract") for m in self.manifolds]
        )

    def vmap(
        self,
        method_module: torch.nn.Module,
        data: torch.Tensor | tuple[torch.Tensor, ...] | tuple[int],
        data_in_dim: int | None,
        **kwargs,
    ) -> torch.Tensor:
        def wrapper(params, buffers, data, **kwargs):
            return torch.func.functional_call(
                method_module(self.manifolds[0]),
                (params, buffers),
                data,
                kwargs=kwargs,
            )

        return torch.func.vmap(wrapper, (0, 0, data_in_dim), randomness="different")(
            self._params, self._buffers, data, **kwargs
        )

    def check_point_on_manifold(
        self,
        x: torch.Tensor,
        *,
        explain: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> tuple[bool, None | str] | bool:
        assert x.size(0) == len(self.manifolds)
        for i, m in enumerate(self.manifolds):
            out = m.check_point_on_manifold(
                x[i : i + 1], explain=explain, atol=atol, rtol=rtol
            )
            if explain:
                if not out[0]:
                    raise ValueError(out[1] + f" at batch {i}")
            else:
                if not out:
                    raise ValueError(f"error at batch {i}")

        # if you got here without error, return True
        if explain:
            return True, ""
        else:
            return True

    def check_vector_on_tangent(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        explain: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> tuple[bool, None | str] | bool:
        assert x.size(0) == len(self.manifolds)
        for i, m in enumerate(self.manifolds):
            out = m.check_vector_on_tangent(
                x[i : i + 1], u[i : i + 1], explain=explain, atol=atol, rtol=rtol
            )
            if explain:
                if not out[0]:
                    raise ValueError(out[1] + f"at batch {i}")
            else:
                if not out:
                    raise ValueError(f"error at batch {i}")

        # if you got here without error, return True
        if explain:
            return True, ""
        else:
            return True

    def projx(self, x: torch.Tensor, *, data_in_dim: int | None = 0) -> torch.Tensor:
        return self.vmap(Projx, x, data_in_dim=data_in_dim)

    def proju(
        self, x: torch.Tensor, u: torch.Tensor, *, data_in_dim: int | None = 0
    ) -> torch.Tensor:
        return self.vmap(Proju, (x, u), data_in_dim=data_in_dim)

    def random(self, *size, dtype=None, device=None) -> torch.Tensor:
        if len(size) == 1:
            assert isinstance(
                size[0], int
            ), "wasn't a tuple of ints, try using the v_map_manifolds.random(*size) operator"
        elif len(size) == 2:
            assert size[-2] == len(self.manifolds)
        else:
            raise NotImplementedError("shape must be in {[F], [B, F]} ")
        return self.vmap(Random, size[-1], data_in_dim=None, dtype=dtype, device=device)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor | None = None,
        *,
        keepdim: bool = False,
        data_in_dim: int | None = 0,
    ) -> torch.Tensor:
        if v is None:
            v = u
        return self.vmap(Inner, (x, u, v), data_in_dim=data_in_dim, keepdim=keepdim)

    def base_logprob(
        self, x: torch.Tensor, *, data_in_dim: int | None = 0
    ) -> torch.Tensor:
        return self.vmap(BaseLogProb, x, data_in_dim=data_in_dim)

    def metric_normalized(
        self, x: torch.Tensor, u: torch.Tensor, *, data_in_dim: int | None = 0
    ) -> torch.Tensor:
        return self.vmap(MetricNormalized, (x, u), data_in_dim=data_in_dim)

    def logdetG(self, x: torch.Tensor, *, data_in_dim: int | None = 0) -> torch.Tensor:
        return self.vmap(LogDetG, x, data_in_dim=data_in_dim)

    def cond_u(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        *,
        data_in_dim: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vmap(CondU, (x0, x1, t), data_in_dim=data_in_dim)

    def abits_clamp(
        self, x: torch.Tensor, *, data_in_dim: int | None = 0
    ) -> torch.Tensor:
        self.vmap(ABitsClamp, (x,), data_in_dim=data_in_dim)


if __name__ == "__main__":
    from flowmm.rfm.manifolds.euclidean import EuclideanWithLogProb
    from flowmm.rfm.manifolds.flat_torus import FlatTorus01FixFirstAtomToOrigin
    from flowmm.rfm.manifolds.product import ProductManifoldWithLogProb

    dc = 2
    ns = 2
    mns = 3

    ms_without = [
        FlatTorus01FixFirstAtomToOrigin(dc, ns, mns),
        FlatTorus01FixFirstAtomToOrigin(dc, ns, mns),
        FlatTorus01FixFirstAtomToOrigin(dc, ns + 1, mns),
    ]

    b = len(ms_without)

    # For just the FlatTorus
    u = torch.zeros(b, mns * dc)
    u[:, :dc] = 1.0
    data = (torch.ones((b, mns * dc)), u)  # x, u

    for proj in [Projx, Proju, Random]:
        ms = [proj(m) for m in ms_without]
        params, buffers = torch.func.stack_module_state(
            ms
        )  # This is slow! Fixed below!

        def wrapper(params, buffers, data):
            return torch.func.functional_call(
                ms[0],
                (params, buffers),
                data,
            )

        if proj == Projx:
            outs = [m(data[0][i : i + 1]) for i, m in enumerate(ms)]
            output = torch.func.vmap(wrapper, (0, 0, 0))(params, buffers, data[0])
        elif proj == Proju:
            outs = [m(data[0][i : i + 1], data[1][i : i + 1]) for i, m in enumerate(ms)]
            output = torch.func.vmap(wrapper, (0, 0, 0))(params, buffers, data)
        elif proj == Random:
            outs = [m(*data[0][i : i + 1].shape) for i, m in enumerate(ms)]
            output = torch.func.vmap(wrapper, (0, 0, None), randomness="different")(
                params, buffers, data[0].shape[-1]
            )

        if proj == Random:
            assert torch.cat(outs).shape == output.shape
        else:
            assert torch.allclose(
                torch.cat(outs),
                output,
            )

    # For product manifolds
    u = torch.zeros(b, mns * dc + dc**2)
    u[:, :dc] = 1.0
    u[:, mns * dc :] = torch.eye(dc).reshape(dc**2).repeat(b, 1)

    num_atomic_types = 5
    a = torch.arange(num_atomic_types).repeat(10)[: mns * b]
    a = torch.nn.functional.one_hot(a, num_classes=num_atomic_types).reshape(b, mns, -1)
    a[0:2, 2, :] = torch.zeros_like(a[0:2, 2, :])
    a = a.to(u)
    x = torch.cat(
        [a.reshape(b, -1), torch.ones((b, mns * dc + dc**2))],
        dim=-1,
    )

    u = torch.cat([torch.zeros_like(a.reshape(b, -1)), u], dim=-1)
    data = (x, u)
    data_inner = data + (u.clone(),)

    ms = [
        ProductManifoldWithLogProb(
            (NullManifoldWithDeltaRandom(a[0, 0, :]), num_atomic_types),
            (NullManifoldWithDeltaRandom(a[0, 1, :]), num_atomic_types),
            (NullManifoldWithDeltaRandom(a[0, 2, :]), num_atomic_types),
            (FlatTorus01FixFirstAtomToOrigin(dc, ns, mns), mns * dc),
            (EuclideanWithLogProb(ndim=1), (dc**2,)),
        ),
        ProductManifoldWithLogProb(
            (NullManifoldWithDeltaRandom(a[1, 0, :]), num_atomic_types),
            (NullManifoldWithDeltaRandom(a[1, 1, :]), num_atomic_types),
            (NullManifoldWithDeltaRandom(a[1, 2, :]), num_atomic_types),
            (FlatTorus01FixFirstAtomToOrigin(dc, ns, mns), mns * dc),
            (EuclideanWithLogProb(ndim=1), (dc**2,)),
        ),
        ProductManifoldWithLogProb(
            (NullManifoldWithDeltaRandom(a[2, 0, :]), num_atomic_types),
            (NullManifoldWithDeltaRandom(a[2, 1, :]), num_atomic_types),
            (NullManifoldWithDeltaRandom(a[2, 2, :]), num_atomic_types),
            (FlatTorus01FixFirstAtomToOrigin(dc, ns + 1, mns), mns * dc),
            (EuclideanWithLogProb(ndim=1), (dc**2,)),
        ),
    ]

    assert len(ms) == b

    params, buffers = torch.func.stack_module_state(
        [MethodToForward(m, method="abstract") for m in ms]
    )

    for proj in [Projx, Proju, Random, Inner, BaseLogProb]:

        def wrapper(params, buffers, data):
            return torch.func.functional_call(proj(ms[0]), (params, buffers), data)

        if proj == Projx:
            outs = [proj(m)(data[0][i : i + 1]) for i, m in enumerate(ms)]
            output = torch.func.vmap(wrapper, (0, 0, 0))(params, buffers, data[0])
        elif proj == Proju:
            outs = [
                proj(m)(data[0][i : i + 1], data[1][i : i + 1])
                for i, m in enumerate(ms)
            ]
            output = torch.func.vmap(wrapper, (0, 0, 0))(params, buffers, data)
        elif proj == Random:
            outs = [proj(m)(*data[0][i : i + 1].shape) for i, m in enumerate(ms)]
            output = torch.func.vmap(wrapper, (0, 0, None), randomness="different")(
                params, buffers, data[0].shape[-1]
            )

            # Testing the VMap object
            vm = VMapManifolds(ms)
            output_random = vm.random(*data[0].shape)
        elif proj == Inner:
            outs = [
                proj(m)(
                    data_inner[0][i : i + 1],
                    data_inner[1][i : i + 1],
                    data_inner[2][i : i + 1],
                )
                for i, m in enumerate(ms)
            ]
            output = torch.func.vmap(wrapper, (0, 0, 0))(params, buffers, data_inner)

            # Testing the VMap object
            vm = VMapManifolds(ms)
            output_inner = vm.inner(*data_inner)  # with two vecs
            assert torch.allclose(output_inner, output)

            output_inner = vm.inner(*data_inner[:-1], None)  # with one vec
            assert torch.allclose(output_inner, output)

            # more testing, one manifold per batch
            xx = vm.projx(data[0][: len(ms)])
            uu = vm.proju(xx, data[0][: len(ms)])
            assert vm.check_point_on_manifold(xx)
            assert vm.check_vector_on_tangent(xx, uu)
        elif proj == BaseLogProb:
            outs = [proj(m)(data[0][i : i + 1]) for i, m in enumerate(ms)]
            output = torch.func.vmap(wrapper, (0, 0, 0))(params, buffers, data[0])

        if proj == Random:
            outs = torch.cat(outs)

            # test delta manifold
            get_num_atoms = lambda x: x[:, : mns * num_atomic_types].sum(dim=-1)
            assert torch.all(get_num_atoms(outs) == get_num_atoms(output))
            assert torch.all(get_num_atoms(outs) == get_num_atoms(output_random))

            # test shape
            assert outs.shape == output.shape
            assert outs.shape == output_random.shape
        else:
            assert torch.allclose(
                torch.cat(outs),
                output,
            )
