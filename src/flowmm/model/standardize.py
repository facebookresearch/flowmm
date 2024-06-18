"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from omegaconf import OmegaConf
from tqdm import tqdm


class AffineStatsParallelAlgorithm(torch.nn.Module):
    def __init__(
        self,
        shape: tuple | int,
        stable: bool = True,
        epsilon: float = 1e-10,
    ):
        """Accumulate mean and variance online using the "parallel algorithm" algorithm from [1].

        Args:
            shape: shape of mean, variance, and std array. do not include batch dimension!
            stable: compute using the stable version of the algorithm [1]
            epsilon: added to the computation of the standard deviation for numerical stability.

        References:
            [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        super().__init__()
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_M2", torch.zeros(shape))
        self.register_buffer("epsilon", torch.tensor(epsilon))
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
        self.stable = stable

    def _parallel_algorithm(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.shape[-len(self.shape) :] == self.shape
        na = self.n.clone()
        nb = x.shape[0]
        nab = na + nb

        xa = self._mean.clone()
        xb = x.mean(dim=0)
        delta = xb - xa
        if self.stable:
            xab = (na * xa + nb * xb) / nab
        else:
            xab = xa + delta * nb / nab

        m2a = self._M2.clone()
        m2b = (
            x.var(dim=(0,), unbiased=False) * nb
        )  # do not use bessel's correction then multiply by total number of items in batch.
        m2ab = m2a + m2b + delta**2 * na * nb / nab
        return nab, xab, m2ab

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.n, self._mean, self._M2 = self._parallel_algorithm(x)
        return self.mean, self.std

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def var(self) -> torch.Tensor:
        if self.n > 1:
            return self._M2 / (self.n - 1)
        else:
            return torch.zeros_like(self._M2)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.epsilon)


collect_stats_on_options = Literal["coord", "lattice"]


def compute_affine_stats(
    dataset: dataset_options,
    collect_stats_on: collect_stats_on_options,
    atom_type_manifold: atom_type_manifold_types,
    coord_manifold: coord_manifold_types,
    lattice_manifold: lattice_manifold_types,
    epochs: int = 5,
    analog_bits_scale: float = 1.0,
):
    from flowmm.cfg_utils import init_loaders
    from flowmm.rfm.manifold_getter import ManifoldGetter

    train_loader, *_ = init_loaders(dataset=dataset, batch_size=8192)

    manifold_getter = ManifoldGetter(
        atom_type_manifold=atom_type_manifold,
        coord_manifold=coord_manifold,
        lattice_manifold=lattice_manifold,
        analog_bits_scale=analog_bits_scale,
        dataset=dataset,
    )

    inited = False

    for _ in tqdm(range(epochs)):
        for batch in tqdm(train_loader):
            (
                x1,
                manifold,
                a_manifold,
                f_manifold,
                l_manifold,
                dims,
                mask_a_or_f,
            ) = manifold_getter(
                batch.batch,
                batch.atom_types,
                batch.frac_coords,
                batch.lengths,
                batch.angles,
                split_manifold=True,
            )
            x0 = manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)

            N = x1.shape[0]

            t = torch.rand(N, dtype=x1.dtype, device=x1.device).reshape(-1, 1)

            x1 = manifold.projx(x1)

            x_t, u_t = manifold.cond_u(x0, x1, t)
            x_t = x_t.reshape(N, x0.shape[-1])
            u_t = u_t.reshape(N, x0.shape[-1])

            # this projects out the mean from the tangent vectors
            # our model cannot predict it, so keeping it in inflates the loss
            u_t = manifold.proju(x_t, u_t)

            atom_types, frac_coords, lattices = manifold_getter.flatrep_to_georep(
                x_t,
                dims=dims,
                mask_a_or_f=mask_a_or_f,
            )
            vals_x_t = {
                "atom": atom_types,
                "coord": frac_coords,
                "lattice": lattices,
            }

            u_atom_types, u_frac_coords, u_lattices = manifold_getter.flatrep_to_georep(
                u_t,
                dims=dims,
                mask_a_or_f=mask_a_or_f,
            )
            vals_u_t = {
                "atom": u_atom_types,
                "coord": u_frac_coords,
                "lattice": u_lattices,
            }

            if not inited:
                aspa_x_t = AffineStatsParallelAlgorithm(
                    vals_x_t[collect_stats_on].shape[-1]
                )
                aspa_u_t = AffineStatsParallelAlgorithm(
                    vals_u_t[collect_stats_on].shape[-1]
                )
                inited = True

            aspa_x_t(vals_x_t[collect_stats_on])
            aspa_u_t(vals_u_t[collect_stats_on])

            print(
                f"{aspa_x_t.mean=}, {aspa_x_t.std=}; {aspa_u_t.mean=}, {aspa_u_t.std=}"
            )

    return {
        "x_t_mean": aspa_x_t.mean,
        "x_t_std": aspa_x_t.std,
        "u_t_mean": aspa_u_t.mean,
        "u_t_std": aspa_u_t.std,
    }


def get_affine_stats_filename(
    dataset: dataset_options,
    manifold_type: atom_type_manifold_types
    | coord_manifold_types
    | lattice_manifold_types,
) -> str:
    return f"stats_{dataset}_{manifold_type}.yaml"


def get_affine_stats(
    dataset: dataset_options,
    manifold_type: atom_type_manifold_types
    | coord_manifold_types
    | lattice_manifold_types,
    path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if path is None:
        file = Path(__file__).parent / get_affine_stats_filename(dataset, manifold_type)
        file = file.resolve()
    else:
        file = Path(path)
        file = file.resolve()

    if file.exists():
        stats = OmegaConf.load(str(file))
    else:
        raise FileNotFoundError(
            f"{file=} does not exist, have you computed the stats with `compute_affine_stats` already?"
        )
    return {k: torch.tensor(v) for k, v in stats.items()}


if __name__ == "__main__":
    import yaml

    from flowmm.cfg_utils import dataset_options
    from flowmm.rfm.manifold_getter import (
        atom_type_manifold_types,
        coord_manifold_types,
        lattice_manifold_types,
    )

    file = Path(__file__).parent / "spd_stats.yaml"
    file = file.resolve()

    # calculate the stats of each dataset
    for dataset in tqdm(list(dataset_options.__args__)):
        for collect_stats_on in collect_stats_on_options.__args__:
            # if collect_stats_on == "atom":
            #     pass  # categorical are not standardized

            if collect_stats_on == "coord":
                # flat_torus_01
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01",
                    lattice_manifold="non_symmetric",
                )
                # since translation is removed from the geodesic
                stats["u_t_mean"] = torch.zeros_like(stats["u_t_mean"])

                # we do not standardize these since they goes through a sinusoidal embedding
                stats["x_t_mean"] = torch.zeros_like(stats["x_t_mean"])
                stats["x_t_std"] = torch.ones_like(stats["x_t_std"])

                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "flat_torus_01"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

                # flat_torus_01_normal
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_normal",
                    lattice_manifold="non_symmetric",
                )
                # since translation is removed from the geodesic
                stats["u_t_mean"] = torch.zeros_like(stats["u_t_mean"])

                # we do not standardize these since they goes through a sinusoidal embedding
                stats["x_t_mean"] = torch.zeros_like(stats["x_t_mean"])
                stats["x_t_std"] = torch.ones_like(stats["x_t_std"])

                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "flat_torus_01_normal"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

                # flat_torus_01_fixfirst
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst",
                    lattice_manifold="non_symmetric",
                )
                # since translation is removed from the geodesic
                stats["u_t_mean"] = torch.zeros_like(stats["u_t_mean"])

                # we do not standardize these since they goes through a sinusoidal embedding
                stats["x_t_mean"] = torch.zeros_like(stats["x_t_mean"])
                stats["x_t_std"] = torch.ones_like(stats["x_t_std"])

                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "flat_torus_01_fixfirst"
                )
                file = file.resolve()

                # flat_torus_01_fixfirst_normal
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst_normal",
                    lattice_manifold="non_symmetric",
                )
                # since translation is removed from the geodesic
                stats["u_t_mean"] = torch.zeros_like(stats["u_t_mean"])

                # we do not standardize these since they goes through a sinusoidal embedding
                stats["x_t_mean"] = torch.zeros_like(stats["x_t_mean"])
                stats["x_t_std"] = torch.ones_like(stats["x_t_std"])

                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "flat_torus_01_fixfirst_normal"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)
            elif collect_stats_on == "lattice":
                # non_symmetric
                # this one is difficult because its multiplied by L
                # so we just leave it untransformed
                stats = {
                    "x_t_mean": torch.zeros(9),
                    "x_t_std": torch.ones(9),
                    "u_t_mean": torch.zeros(9),
                    "u_t_std": torch.ones(9),
                }
                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "non_symmetric"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

                # spd_euclidean_geo
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst",
                    lattice_manifold="spd_euclidean_geo",
                )
                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "spd_euclidean_geo"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

                # spd_riemanian_geo
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst",
                    lattice_manifold="spd_riemanian_geo",
                )
                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "spd_riemanian_geo"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

                # lattice_params
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst",
                    lattice_manifold="lattice_params",
                )

                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "lattice_params"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

                # lattice_params_normal_base
                stats = compute_affine_stats(
                    dataset=dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst",
                    lattice_manifold="lattice_params_normal_base",
                )

                file = Path(__file__).parent / get_affine_stats_filename(
                    dataset, "lattice_params_normal_base"
                )
                file = file.resolve()
                with open(file, "w") as f:
                    yaml.dump({k: v.tolist() for k, v in stats.items()}, f)
            else:
                raise ValueError()
