"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import torch
from tqdm import tqdm

from rfm_docking.manifm.solvers import get_step_fn
from flowmm.rfm.vmap import VMapManifolds


def projx_integrate_xt_to_x1(
    manifold: VMapManifolds,
    odefunc: callable | None,
    xt: torch.Tensor,
    t: torch.Tensor,
    vt: torch.Tensor | None = None,
    method="euler",
) -> torch.Tensor:
    step_fn = get_step_fn(method)
    dt = 1.0 - t
    if vt is None:
        vt = odefunc(t, xt)
    xt = step_fn(odefunc, xt, vt, t, dt)
    xt = manifold.projx(xt)
    return xt


@torch.no_grad()
def projx_cond_integrator_return_last(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = get_step_fn(method)

    xt = x0
    x1_tm1 = None  # init cond

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt, x1_tm1)
        x1_tm1 = projx_integrate_xt_to_x1(
            manifold, odefunc, xt, t0, vt, method=method
        )  # new cond
        x1_tm1 = manifold.abits_clamp(x1_tm1)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
    return xt
