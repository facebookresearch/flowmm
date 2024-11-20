"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from copy import deepcopy
import warnings
from functools import partial
from typing import Any, Literal

import hydra
import pytorch_lightning as pl
import torch
from scipy.spatial.transform import Rotation as R
from geoopt.manifolds.euclidean import Euclidean
from geoopt.manifolds.product import ProductManifold
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.func import jvp, vjp
from torch_geometric.data import Data, Batch, HeteroData
from torchmetrics import MeanMetric, MinMetric

from diffcsp.common.data_utils import (
    lattices_to_params_shape,
    lattice_params_to_matrix_torch,
)
from manifm.ema import EMA
from manifm.model_pl import ManifoldFMLitModule, div_fn, output_and_div
from rfm_docking.manifm.solvers import projx_integrator, projx_integrator_return_last
from flowmm.model.solvers import (
    projx_cond_integrator_return_last,
    projx_integrate_xt_to_x1,
)
from flowmm.model.standardize import get_affine_stats
from rfm_docking.manifold_getter import DockingManifoldGetter, Dims
from rfm_docking.dock.arch import DockingCSPNet, ProjectedConjugatedCSPNet
from rfm_docking.reassignment import ot_reassignment
from rfm_docking.utils import sample_mol_in_frac
from flowmm.rfm.manifolds.spd import SPDGivenN, SPDNonIsotropicRandom
from flowmm.rfm.vmap import VMapManifolds
from flowmm.rfm.manifolds.flat_torus import FlatTorus01


def output_and_div(
    vecfield: callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor | None = None,
    div_mode: Literal["exact", "rademacher"] = "exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    if div_mode == "exact":
        dx = vecfield(x)
        div = div_fn(vecfield)(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class DockingRFMLitModule(ManifoldFMLitModule):
    def __init__(self, cfg: DictConfig):
        pl.LightningModule.__init__(self)
        self.cfg = cfg
        self.save_hyperparameters()

        self.manifold_getter = DockingManifoldGetter(
            coord_manifold=cfg.model.manifold_getter.coord_manifold,
            dataset=cfg.data.dataset_name,
        )

        self.costs = {
            "loss_f": cfg.model.cost_coord,
            "loss_be": cfg.model.cost_be,
        }
        if cfg.model.affine_combine_costs:
            total_cost = sum([v for v in self.costs.values()])
            self.costs = {k: v / total_cost for k, v in self.costs.items()}

        model: DockingCSPNet = hydra.utils.instantiate(
            self.cfg.vectorfield, _convert_="partial"
        )
        # Model of the vector field.
        cspnet = ProjectedConjugatedCSPNet(
            cspnet=model,
            manifold_getter=self.manifold_getter,
            coord_affine_stats=get_affine_stats(
                cfg.data.dataset_name, self.manifold_getter.coord_manifold
            ),
        )
        if cfg.optim.get("ema_decay", None) is None:
            self.model = cspnet
        else:
            self.model = EMA(
                cspnet,
                cfg.optim.ema_decay,
            )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = {
            "loss": MeanMetric(),
            "loss_f": MeanMetric(),
            "unscaled/loss_f": MeanMetric(),
            "be_loss": MeanMetric(),
        }
        self.val_metrics = {
            "loss": MeanMetric(),
            "loss_f": MeanMetric(),
            "unscaled/loss_f": MeanMetric(),
            "be_loss": MeanMetric(),
        }
        self.test_metrics = {
            "loss": MeanMetric(),
            "loss_f": MeanMetric(),
            "unscaled/loss_f": MeanMetric(),
            "nll": MeanMetric(),
            "be_loss": MeanMetric(),
        }
        # for logging best so far validation accuracy
        self.val_metrics_best = {
            "loss": MinMetric(),
            "loss_f": MinMetric(),
            "unscaled/loss_f": MinMetric(),
            "be_loss": MinMetric(),
        }
        if self.cfg.val.compute_nll:
            self.val_metrics["nll"] = MeanMetric()
            self.val_metrics_best["nll"] = MinMetric()

    @staticmethod
    def _annealing_schedule(
        t: torch.Tensor, slope: float, intercept: float
    ) -> torch.Tensor:
        return slope * torch.nn.functional.relu(t - intercept) + 1.0

    @torch.no_grad()
    def sample(
        self,
        batch: HeteroData,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
        guidance_strength: float = 0.0, 
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            x1,
            manifold,
            f_manifold,
            dims,
            mask_f,
        ) = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            split_manifold=True,
        )
        if x0 is None:
            x0 = manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)
        else:
            x0 = x0.to(x1)

        return self.finish_sampling(
            batch=batch,
            x0=x0,
            manifold=manifold,
            f_manifold=f_manifold,
            dims=dims,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def gen_sample(
        self,
        batch: HeteroData,
        dim_coords: int,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shape,
            manifold,
            f_manifold,
            dims,
            mask_f,
        ) = self.manifold_getter.from_empty_batch(
            batch.osda.batch, dim_coords, split_manifold=True
        )
        num_atoms = self.manifold_getter._get_num_atoms(mask_f)

        x0 = batch.x0

        return self.finish_sampling(
            batch=batch,
            x0=x0,
            manifold=manifold,
            f_manifold=f_manifold,
            dims=dims,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def pred_sample(
        self,
        batch: HeteroData,
        atom_types: torch.LongTensor,
        dim_coords: int,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shape,
            manifold,
            f_manifold,
            dims,
            mask_f,
        ) = self.manifold_getter.from_only_atom_types(
            batch.osda.batch, atom_types, dim_coords, split_manifold=True
        )
        num_atoms = self.manifold_getter._get_num_atoms(mask_f)

        if x0 is None:
            x0 = manifold.random(*shape, device=batch.osda.batch.device)
        else:
            x0 = x0.to(device=batch.osda.batch.device)

        return self.finish_sampling(
            batch=batch,
            x0=x0,
            manifold=manifold,
            f_manifold=f_manifold,
            dims=dims,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def finish_sampling(
        self,
        batch: HeteroData,
        x0: torch.Tensor,
        manifold: VMapManifolds,
        f_manifold: VMapManifolds,
        dims: Dims,
        num_steps: int,
        entire_traj: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        guidance_strength = self.cfg.integrate.get("guidance_strength", 0.0)
        print("Guidance strength is", guidance_strength)

        vecfield = partial(
            self.vecfield,
            batch=batch, # NOTE assumes batch carries non-zero conditions
            guidance_strength=guidance_strength,
        )

        compute_traj_velo_norms = self.cfg.integrate.get(
            "compute_traj_velo_norms", False
        )

        c = self.cfg.integrate.get("inference_anneal_slope", 0.0)
        b = self.cfg.integrate.get("inference_anneal_offset", 0.0)

        anneal_coords = self.cfg.integrate.get("inference_anneal_coords", True)

        print(
            "Anneal Coords:",
            anneal_coords,
        )

        def scheduled_fn_to_integrate(
            t: torch.Tensor, 
            x: torch.Tensor, 
            cond_coords: torch.Tensor | None = None, 
            cond_be: torch.Tensor | None = None, 
        ) -> torch.Tensor:
            anneal_factor = self._annealing_schedule(t, c, b)
            out = vecfield(
                t=torch.atleast_2d(t),
                x=torch.atleast_2d(x),
                manifold=manifold,
                cond_coords=torch.atleast_2d(cond_coords) if isinstance(cond_coords, torch.Tensor) else cond_coords,
                cond_be=torch.atleast_2d(cond_be) if isinstance(cond_be, torch.Tensor) else cond_be,
            )
            if anneal_coords:
                out[0][:, 0 : dims.f].mul_(anneal_factor) # NOTE anneal only the coordinates, not the binding energy
            
            return out

        if self.cfg.model.get("self_cond", False):  # TODO mrx ignoring for now
            print("finish_sampling, self_cond True")
            x1 = projx_cond_integrator_return_last(
                manifold,
                scheduled_fn_to_integrate,
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
                guidance_strength=guidance_strength,
            )
            return x1

        elif entire_traj or compute_traj_velo_norms: 
            print(f"finish_sampling, {entire_traj} or {compute_traj_velo_norms}") # TODO mrx True, False
            # breakpoint()
            xs, vs = projx_integrator(
                manifold,
                scheduled_fn_to_integrate, # NOTE odefunc
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                pbar=True,
            )
        else:
            print("finish_sampling, else") # TODO mrx ignore for now
            x1 = projx_integrator_return_last(
                manifold,
                scheduled_fn_to_integrate,
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )
            return x1

        if compute_traj_velo_norms: # TODO mrx ignore for now
            print("finish_sampling, compute_traj_velo_norms True")
            s = 0
            e = dims.f
            norm_f = f_manifold.inner(
                xs[..., s:e], vs[..., s:e], vs[..., s:e], data_in_dim=1
            )

        print("finish_sampling entire_traj", entire_traj, "compute_traj_velo_norms", compute_traj_velo_norms)
        if entire_traj and compute_traj_velo_norms:
            # return xs, norm_a, norm_f, norm_l
            return xs, norm_f
        elif entire_traj and not compute_traj_velo_norms:
            return xs
        elif not entire_traj and compute_traj_velo_norms:
            # return xs[0], norm_a, norm_f, norm_l
            return xs[0], norm_f
        else:
            # this should happen due to logic above
            return xs[0]

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: Data,
        stage: str,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps: int = 1_000,
    ):
        """Computes the negative log-likelihood of a batch of data."""
        x1, manifold, dims, mask_f = self.manifold_getter(
            batch.osda.batch,
            batch.osda.atom_types,
            batch.osda.frac_coords,
            batch.osda.lengths,
            batch.osda.angles,
            split_manifold=False,
        )
        dim = sum(dims)

        nfe = [0]

        div_mode = self.cfg.integrate.get(
            "div_mode", "rademacher"
        )  # alternative: exact

        v = None
        if div_mode == "rademacher":
            v = torch.randint(low=0, high=2, size=x1.shape).to(x1) * 2 - 1

        vecfield = partial(self.vecfield, batch=batch)

        def odefunc(t, tensor):
            nfe[0] += 1
            t = t.to(tensor)
            x = tensor[..., :dim]

            def l_vecfield(x):
                return vecfield(
                    t=torch.atleast_2d(t), x=torch.atleast_2d(x), manifold=manifold
                )

            dx, div = output_and_div(l_vecfield, x, v=v, div_mode=div_mode)

            if hasattr(manifold, "logdetG"):
                corr = jvp(manifold.logdetG, (x,), (dx,))[1]
                div = div + 0.5 * corr.to(div)

            div = div.reshape(-1, 1)
            del t, x
            return torch.cat([dx, div], dim=-1)

        # Solve ODE on the product manifold of data manifold x euclidean.
        prod_manis = [
            ProductManifold((m, dim), (Euclidean(), 1)) for m in manifold.manifolds
        ]
        product_man = VMapManifolds(prod_manis)
        state1 = torch.cat([x1, torch.zeros_like(x1[..., :1])], dim=-1)

        with torch.no_grad():
            state0 = projx_integrator_return_last(
                product_man,
                odefunc,
                state1,
                t=torch.linspace(t1, 0, num_steps + 1).to(x1),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )

        x0, logdetjac = state0[..., :dim], state0[..., -1]

        x0_ = x0
        x0 = manifold.projx(x0)

        # log how close the final solution is to the manifold.
        integ_error = (x0[..., :dim] - x0_[..., :dim]).abs().max()
        # self.log(f"{stage}/integ_error", integ_error, batch_size=batch.batch_size)

        logp0 = manifold.base_logprob(x0)
        logp1 = logp0 + logdetjac

        if self.cfg.get("normalize_loglik", True):
            logp1 = logp1 / batch.num_atoms

        # Mask out those that left the manifold
        masked_logp1 = logp1
        if isinstance(manifold, (SPDGivenN, SPDNonIsotropicRandom)):
            mask = integ_error < 1e-5
            masked_logp1 = logp1[mask]

        if return_projx_error:
            return logp1, integ_error
        else:
            return masked_logp1

    def loss_fn(self, batch: Data, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.rfm_loss_fn(batch, *args, **kwargs)

    def rfm_loss_fn(
        self, batch: Data, x0: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        vecfield = partial(
            self.vecfield,
            batch=batch,
        )

        x0 = batch.x0
        x1 = batch.x1

        osda_manifold = batch.osda.manifold

        N = x1.shape[0]

        t = torch.rand(N, dtype=x1.dtype, device=x1.device).reshape(-1, 1)

        x1 = osda_manifold.projx(x1)

        x_t, u_t = osda_manifold.cond_u(x0, x1, t)
        x_t = x_t.reshape(N, x0.shape[-1])
        u_t = u_t.reshape(N, x0.shape[-1])

        # this projects out the mean from the tangent vectors
        # our model cannot predict it, so keeping it in inflates the loss
        u_t = osda_manifold.proju(x_t, u_t)

        cond_coords = None
        cond_be = None
        if self.cfg.model.self_cond:
            with torch.no_grad():
                if torch.rand((1)) < 0.5:
                    cond_coords, cond_be = projx_integrate_xt_to_x1(
                        osda_manifold,
                        lambda t, x: vecfield(
                            t=torch.atleast_2d(t),
                            x=torch.atleast_2d(x),
                            manifold=osda_manifold,
                        ),
                        x_t,
                        t,
                    ).detach_()

        u_t_pred, be_pred = vecfield(
            t=t,
            x=x_t,
            manifold=osda_manifold,
            cond_coords=cond_coords,
            cond_be=cond_be,
        )
        # maybe adjust the target
        diff = u_t_pred - u_t
        # diff = FlatTorus01.logmap(u_t_pred, x1).abs()  # NOTE predict coordinate
        # loss_f = diff.mean()

        max_num_atoms = batch.osda.mask_f.size(-1)
        dim_f_per_atom = batch.osda.dims.f / max_num_atoms

        s = 0
        e = batch.osda.dims.f
        # loss for each example in batch
        loss_f = (
            batch.osda.f_manifold.inner(x_t[:, s:e], diff[:, s:e], diff[:, s:e])
            / dim_f_per_atom
        )
        loss_f = loss_f.mean()
        # per dim, already per atom

        loss = self.costs["loss_f"] * loss_f

        # binding energy loss TODO mrx add more flexibility to try different losses 
        be_loss = torch.nn.functional.mse_loss(be_pred[s:e], batch.y['bindingatoms']) 
        return {
            "loss": loss,
            "loss_f": self.costs["loss_f"] * loss_f,
            "unscaled/loss_f": loss_f,
            "be_loss": self.costs["loss_be"] * be_loss,
        }

    def training_step(self, batch: Data, batch_idx: int):
        loss_dict = self.loss_fn(batch)

        if torch.isfinite(loss_dict["loss"]):
            # log train metrics

            for k, v in loss_dict.items():
                self.log(
                    f"train/{k}",
                    v,
                    batch_size=self.cfg.data.datamodule.batch_size.train,
                )
                self.train_metrics[k].update(v.cpu())
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss_dict['loss'].item()}.")
            return None

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return loss_dict

    def training_epoch_end(self, outputs: list[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for train_metric in self.train_metrics.values():
            train_metric.reset()

    def shared_eval_step(
        self,
        batch: Data,
        batch_idx: int,
        stage: Literal["val", "test"],
        compute_loss: bool,
        compute_nll: bool,
    ) -> dict[str, torch.Tensor]:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        out = {}

        if compute_loss:
            loss_dict = self.loss_fn(batch)
            if stage == "val":
                batch_size = self.cfg.data.datamodule.batch_size.val
            else:
                batch_size = self.cfg.data.datamodule.batch_size.test

            for k, v in loss_dict.items():
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )
                metrics[k].update(v.cpu())
            out.update(loss_dict)

        if compute_nll:
            nll_dict = {}
            logprob = self.compute_exact_loglikelihood(
                batch,
                stage,
                num_steps=self.cfg.integrate.get("num_steps", 1_000),
            )
            nll = -logprob.mean()
            nll_dict[f"{stage}/nll"] = nll
            nll_dict[f"{stage}/nll_num_steps"] = self.cfg.integrate.num_steps

            self.logger.log_metrics(nll_dict, step=self.global_step)
            metrics["nll"].update(nll.cpu())
            out.update(nll_dict)

        return out

    def compute_reconstruction(
        self,
        batch: HeteroData,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter(
            batch.osda.batch,
            batch.osda.atom_types,
            batch.osda.frac_coords,
            batch.osda.lengths,
            batch.osda.angles,
            split_manifold=False,
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.sample(batch, num_steps=num_steps)
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.sample(batch, num_steps=num_steps)
            norms = {}
        atom_types, frac_coords, lattices = self.manifold_getter.flatrep_to_crystal(
            recon, dims, mask_f
        )
        lengths, angles = lattices_to_params_shape(lattices)
        out = {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_recon_trajectory(
        self,
        batch: Data,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            split_manifold=False,
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.sample(
                batch, num_steps=num_steps, entire_traj=True
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.sample(batch, num_steps=num_steps, entire_traj=True)
            norms = {}

        atom_types, frac_coords, lattices = [], [], []
        lengths, angles = [], []
        for recon_step in recon:
            f = self.manifold_getter.flatrep_to_crystal(recon_step, dims, mask_f)
            # atom_types.append(batch.atom_types)
            frac_coords.append(f)
            lattices.append(batch.lattice)
            lengths.append(batch.lengths)
            angles.append(batch.angles)
        out = {
            "atom_types": torch.stack(atom_types, dim=0),
            "frac_coords": torch.stack(frac_coords, dim=0),
            "lattices": torch.stack(lattices, dim=0),
            "lengths": torch.stack(lengths, dim=0),
            "angles": torch.stack(angles, dim=0),
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_generation(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter.from_empty_batch(
            batch.osda.batch, dim_coords, split_manifold=False
        )

        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.gen_sample(
                batch, dim_coords, num_steps=num_steps
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.gen_sample(batch, dim_coords, num_steps=num_steps)
            norms = {}
        frac_coords = self.manifold_getter.flatrep_to_crystal(recon, dims, mask_f)
        out = {
            "atom_types": batch.osda.atom_types,
            "frac_coords": frac_coords,
            "lattices": batch.lattices,
            "lengths": batch.lattices[:, :3],
            "angles": batch.lattices[:, 3:],
            "num_atoms": batch.osda.num_atoms,
            "input_data_batch": batch.osda.batch,
        }
        out.update(norms)
        return out

    def compute_gen_trajectory(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter.from_empty_batch(
            batch.osda.batch, dim_coords, split_manifold=False
        )

        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.gen_sample(
                batch.osda.batch,
                dim_coords,
                num_steps=num_steps,
                entire_traj=True,
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.gen_sample(
                batch, dim_coords, num_steps=num_steps, entire_traj=True
            )
            norms = {}

        frac_coords = []
        for recon_step in recon:
            f = self.manifold_getter.flatrep_to_crystal(recon_step, dims, mask_f)
            frac_coords.append(f)

        osda_traj = {
            "atom_types": batch.osda.atom_types,
            "target_coords": batch.osda.frac_coords,
            "frac_coords": torch.stack(frac_coords, dim=0),
            "num_atoms": batch.osda.num_atoms,
            "batch": batch.osda.batch,
        }
        zeolite = {
            "atom_types": batch.zeolite.atom_types,
            "frac_coords": batch.zeolite.frac_coords,
            "batch": batch.zeolite.batch,
        }
        out = {
            "crystal_id": batch.crystal_id,
            "osda_traj": osda_traj,
            "zeolite": zeolite,
            "lattices": batch.lattices,
            "lengths": batch.lattices[:, :3],
            "angles": batch.lattices[:, 3:],
        }
        out.update(norms)
        return out

    def compute_prediction(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter.from_empty_batch(
            batch.batch, dim_coords, split_manifold=False
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.pred_sample(
                batch.batch, batch.atom_types, dim_coords, num_steps=num_steps
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.pred_sample(
                batch.batch, batch.atom_types, dim_coords, num_steps=num_steps
            )
            norms = {}
        frac_coords = self.manifold_getter.flatrep_to_crystal(recon, dims, mask_f)
        out = {
            "atom_types": batch.atom_types,
            "frac_coords": frac_coords,
            "lattices": batch.lattices,
            "lengths": batch.lengths,
            "angles": batch.angles,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def validation_step(self, batch: Data, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="val",
            compute_loss=True,
            compute_nll=self.cfg.val.compute_nll,
        )

    def validation_epoch_end(self, outputs: list[Any]):
        out = {}
        for key, val_metric in self.val_metrics.items():
            val_metric_value = (
                val_metric.compute()
            )  # get val accuracy from current epoch
            val_metric_best = self.val_metrics_best[key]
            val_metric_best.update(val_metric_value)
            self.log(
                f"val/best/{key}",
                val_metric_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )
            val_metric.reset()
            out[key] = val_metric_value
        return out

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="test",
            compute_loss=self.cfg.test.get("compute_loss", False),
            compute_nll=self.cfg.test.get("compute_nll", False),
        )

    def test_epoch_end(self, outputs: list[Any]):
        for test_metric in self.test_metrics.values():
            test_metric.reset()

    def predict_step(self, batch: Any, batch_idx: int):
        if not hasattr(batch, "frac_coords"):
            if self.cfg.integrate.get("entire_traj", False):
                print("predict_step compute_gen_trajectory") 
                return self.compute_gen_trajectory(
                    batch,
                    dim_coords=self.cfg.data.get("dim_coords", 3),
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
            else:
                print("predict_step compute_generation") # TODO mrx ignore for now
                return self.compute_generation(
                    batch,
                    dim_coords=self.cfg.data.get("dim_coords", 3),
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
        else:
            # not generating or predicting new structures
            if self.cfg.integrate.get("entire_traj", False):
                print("predict_step compute_recon_trajectory") # TODO mrx ignore for now
                return self.compute_recon_trajectory(
                    batch,
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
            else:
                print("predict_step compute_reconstruction") # TODO mrx ignore for now
                return self.compute_reconstruction(
                    batch,
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if self.cfg.optim.get("lr_scheduler", None) is not None:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.optim.lr_scheduler,
                optimizer,
            )
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "epoch",
                        "monitor": self.cfg.optim.monitor,
                        "frequency": self.cfg.optim.frequency,
                    },
                }
            elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": self.cfg.optim.interval,
                    },
                }
            else:
                raise NotImplementedError("unsuported lr_scheduler")
        else:
            return {"optimizer": optimizer}
