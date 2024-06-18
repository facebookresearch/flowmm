"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Literal

import numpy as np
import torch
from torch_geometric.utils import to_dense_batch

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from flowmm.cfg_utils import dataset_options
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from flowmm.geometric_ import mask_2d_to_batch
from flowmm.rfm.manifolds import (
    EuclideanWithLogProb,
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
    MultiAtomFlatDirichletSimplex,
    NullManifoldWithDeltaRandom,
    ProductManifoldWithLogProb,
)
from flowmm.rfm.manifolds.analog_bits import (
    MultiAtomAnalogBits,
    analog_bits_to_int,
    int_to_analog_bits,
)
from flowmm.rfm.manifolds.flat_torus import (
    MaskedNoDriftFlatTorus01,
    MaskedNoDriftFlatTorus01WrappedNormal,
)
from flowmm.rfm.manifolds.lattice_params import LatticeParams, LatticeParamsNormalBase
from flowmm.rfm.manifolds.spd import (
    SPDGivenN,
    lattice_params_to_spd_vector,
    spd_vector_to_lattice_matrix,
)
from flowmm.rfm.vmap import VMapManifolds

Dims = namedtuple("Dims", ["a", "f", "l"])
ManifoldGetterOut = namedtuple(
    "ManifoldGetterOut", ["flat", "manifold", "dims", "mask_a_or_f"]
)
SplitManifoldGetterOut = namedtuple(
    "SplitManifoldGetterOut",
    [
        "flat",
        "manifold",
        "a_manifold",
        "f_manifold",
        "l_manifold",
        "dims",
        "mask_a_or_f",
    ],
)
GeomTuple = namedtuple("GeomTuple", ["a", "f", "l"])
atom_type_manifold_types = Literal["null_manifold", "simplex", "analog_bits"]
coord_manifold_types = Literal[
    "flat_torus_01",
    "flat_torus_01_normal",
    "flat_torus_01_fixfirst",
    "flat_torus_01_fixfirst_normal",
]
lattice_manifold_types = Literal[
    "non_symmetric",
    "spd_euclidean_geo",
    "spd_riemanian_geo",
    "lattice_params",
    "lattice_params_normal_base",
]


class ManifoldGetter(torch.nn.Module):
    def __init__(
        self,
        atom_type_manifold: atom_type_manifold_types,
        coord_manifold: coord_manifold_types,
        lattice_manifold: lattice_manifold_types,
        dataset: dataset_options | None = None,
        analog_bits_scale: float | None = None,
        length_inner_coef: float | None = None,
    ) -> None:
        super().__init__()
        self.atom_type_manifold = atom_type_manifold
        self.coord_manifold = coord_manifold
        self.lattice_manifold = lattice_manifold
        if atom_type_manifold == "analog_bits":
            assert analog_bits_scale is not None
        self.analog_bits_scale = analog_bits_scale
        self.length_inner_coef = length_inner_coef
        self.dataset = dataset

    @property
    def predict_atom_types(self):
        return False if self.atom_type_manifold == "null_manifold" else True

    @staticmethod
    def _atomic_one_hot(a: torch.LongTensor) -> torch.LongTensor:
        return torch.nn.functional.one_hot(a - 1, num_classes=NUM_ATOMIC_TYPES)

    @staticmethod
    def _inverse_atomic_one_hot(
        a: torch.LongTensor | np.ndarray, dim: int = -1
    ) -> torch.LongTensor:
        if isinstance(a, np.ndarray):
            return np.argmax(a, axis=dim) + 1
        elif isinstance(a, torch.Tensor):
            return torch.argmax(a, dim=dim) + 1
        else:
            raise TypeError()

    @staticmethod
    def _atomic_bits(a: torch.LongTensor, scale: float) -> torch.LongTensor:
        return int_to_analog_bits(a - 1, NUM_ATOMIC_BITS, scale)

    @staticmethod
    def _inverse_atomic_bits(a: torch.LongTensor | np.ndarray) -> torch.LongTensor:
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
            return analog_bits_to_int(a).numpy() + 1
        elif isinstance(a, torch.Tensor):
            return analog_bits_to_int(a) + 1
        else:
            raise TypeError()

    @staticmethod
    def _get_max_num_atoms(mask_a_or_f: torch.BoolTensor) -> int:
        return int(mask_a_or_f.sum(dim=-1).max())

    @staticmethod
    def _get_num_atoms(mask_a_or_f: torch.BoolTensor) -> torch.LongTensor:
        return mask_a_or_f.sum(dim=-1)

    def _to_dense(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        a, mask_a_or_f = to_dense_batch(
            x=atom_types, batch=batch
        )  # B x N x NUM_ATOMIC_TYPES, B x N
        f, _ = to_dense_batch(x=frac_coords, batch=batch)  # B x N x 3
        return a, f, mask_a_or_f

    def _to_flat(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        l: torch.Tensor,
        dims: Dims,
    ) -> torch.Tensor:
        a_flat = a.reshape(a.size(0), dims.a)
        f_flat = f.reshape(f.size(0), dims.f)
        l_flat = l.reshape(l.size(0), dims.l)
        return torch.cat([a_flat, f_flat, l_flat], dim=1)

    def georep_to_flatrep(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts from georep to the manifold flatrep"""
        a, f, mask_a_or_f = self._to_dense(
            batch, atom_types=atom_types, frac_coords=frac_coords
        )
        num_atoms = self._get_num_atoms(mask_a_or_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            atom_types_dense_one_hot=a,
            dim_coords=frac_coords.shape[-1],
            split_manifold=split_manifold,
        )
        # manifolds = [manifold.to(device=batch.device) for manifold in manifolds]
        flat = self._to_flat(a, f, lattices, dims)
        if split_manifold:
            return SplitManifoldGetterOut(
                flat,
                *manifolds,
                dims,
                mask_a_or_f,
            )
        else:
            return ManifoldGetterOut(
                flat,
                *manifolds,
                dims,
                mask_a_or_f,
            )

    def forward(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        frac_coords: torch.Tensor,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts data from the loader into the georep, then to the manifold flatrep"""
        atom_types = self._convert_atom_types(atom_types)

        if "spd" in self.lattice_manifold:
            lattices = lattice_params_to_spd_vector(lengths, angles)
        elif self.lattice_manifold == "non_symmetric":
            lattices = lattice_params_to_matrix_torch(lengths, angles)
        elif self.lattice_manifold == "lattice_params":
            lattices_deg = LatticeParams.cat(lengths, angles)
            lattices = LatticeParams().deg2uncontrained(lattices_deg)
        elif self.lattice_manifold == "lattice_params_normal_base":
            lattices = LatticeParams.cat(lengths, angles)
        else:
            raise NotImplementedError()

        return self.georep_to_flatrep(
            batch, atom_types, frac_coords, lattices, split_manifold
        )

    def _convert_atom_types(self, atom_types: torch.Tensor) -> torch.Tensor:
        if atom_types.ndim == 1:  # the types are NOT one_hot already
            if self.atom_type_manifold in ["simplex", "null_manifold"]:
                atom_types = self._atomic_one_hot(atom_types)  # B x NUM_ATOMIC_TYPES
            elif self.atom_type_manifold == "analog_bits":
                atom_types = self._atomic_bits(atom_types, self.analog_bits_scale)
            else:
                raise TypeError()
        return atom_types

    def from_empty_batch(
        self,
        batch: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        _, mask_a_or_f = to_dense_batch(
            x=torch.zeros((*batch.shape, 1), device=batch.device), batch=batch
        )  # B x N
        num_atoms = self._get_num_atoms(mask_a_or_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms, None, dim_coords, split_manifold=split_manifold
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_a_or_f

    def from_only_atom_types(
        self,
        batch: torch.LongTensor,
        atom_types: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        atom_types = self._convert_atom_types(atom_types)
        atom_types_dense_one_hot, mask_a_or_f = to_dense_batch(
            x=atom_types, batch=batch
        )  # B x N
        num_atoms = self._get_num_atoms(mask_a_or_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            atom_types_dense_one_hot,
            dim_coords,
            split_manifold=split_manifold,
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_a_or_f

    @staticmethod
    def _from_dense(
        a: torch.Tensor,
        f: torch.Tensor,
        mask_a_or_f: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return a[mask_a_or_f], f[mask_a_or_f]

    def _from_flat(
        self,
        flat: torch.Tensor,
        dims: Dims,
        max_num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, f, l, _ = torch.tensor_split(flat, np.cumsum(dims).tolist(), dim=1)

        if self.atom_type_manifold in ["simplex", "null_manifold"]:
            a = a.reshape(-1, max_num_atoms, NUM_ATOMIC_TYPES)
        elif self.atom_type_manifold == "analog_bits":
            a = a.reshape(-1, max_num_atoms, NUM_ATOMIC_BITS)
        else:
            raise TypeError()

        if "spd" in self.lattice_manifold:
            l = l.reshape(-1, dims.l)
        elif self.lattice_manifold == "non_symmetric":
            l = l.reshape(-1, int(math.sqrt(dims.l)), int(math.sqrt(dims.l)))
        elif (
            self.lattice_manifold == "lattice_params"
            or self.lattice_manifold == "lattice_params_normal_base"
        ):
            l = l.reshape(-1, dims.l)
        else:
            raise NotImplementedError()

        return a, f.reshape(-1, max_num_atoms, dims.f // max_num_atoms), l

    def flatrep_to_georep(
        self, flat: torch.Tensor, dims: Dims, mask_a_or_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to the georep"""
        max_num_atoms = self._get_max_num_atoms(mask_a_or_f)
        a, f, l = self._from_flat(flat, dims, max_num_atoms)
        a, f = self._from_dense(a, f, mask_a_or_f)
        return GeomTuple(a, f, l)

    def georep_to_crystal(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the georep to (one_hot / bits, frac_coords, lattice matrix)"""
        if "spd" in self.lattice_manifold:
            lattices = spd_vector_to_lattice_matrix(lattices)
        elif self.lattice_manifold == "non_symmetric":
            pass
        elif self.lattice_manifold == "lattice_params":
            lattices_deg = LatticeParams().uncontrained2deg(lattices)
            lengths, angles_deg = LatticeParams.split(lattices_deg)
            lattices = lattice_params_to_matrix_torch(lengths, angles_deg)
        elif self.lattice_manifold == "lattice_params_normal_base":
            lengths, angles_deg = LatticeParamsNormalBase.split(lattices)
            lattices = lattice_params_to_matrix_torch(lengths, angles_deg)
        else:
            raise NotImplementedError()

        return atom_types, frac_coords, lattices

    def flatrep_to_crystal(
        self, flat: torch.Tensor, dims: Dims, mask_a_or_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to (one_hot / bits, frac_coords, lattice matrix)"""
        return self.georep_to_crystal(*self.flatrep_to_georep(flat, dims, mask_a_or_f))

    @staticmethod
    def mask_a_or_f_to_batch(mask_a_or_f: torch.BoolTensor) -> torch.LongTensor:
        return mask_2d_to_batch(mask_a_or_f)

    @staticmethod
    def _get_manifold(
        num_atom: int,
        atom_types_dense_one_hot: torch.Tensor | None,
        dim_coords: int,
        max_num_atoms: int,
        batch_idx: int,
        split_manifold: bool,
        atom_type_manifold: atom_type_manifold_types,
        coord_manifold: coord_manifold_types,
        lattice_manifold: lattice_manifold_types,
        dataset: dataset_options | None = None,
        analog_bits_scale: float | None = None,
        length_inner_coef: float | None = None,
    ) -> (
        tuple[ProductManifoldWithLogProb]
        | tuple[
            ProductManifoldWithLogProb,
            MultiAtomAnalogBits
            | MultiAtomFlatDirichletSimplex
            | NullManifoldWithDeltaRandom,
            FlatTorus01FixFirstAtomToOrigin
            | FlatTorus01FixFirstAtomToOriginWrappedNormal
            | MaskedNoDriftFlatTorus01
            | MaskedNoDriftFlatTorus01WrappedNormal,
            EuclideanWithLogProb | SPDGivenN | LatticeParams,
        ]
    ):
        if atom_type_manifold == "simplex":
            a_manifold = (
                MultiAtomFlatDirichletSimplex(
                    num_categories=NUM_ATOMIC_TYPES,
                    num_atoms=num_atom,
                    max_num_atoms=max_num_atoms,
                ),
                NUM_ATOMIC_TYPES * max_num_atoms,
            )
        elif atom_type_manifold == "null_manifold":
            a_manifold = (
                NullManifoldWithDeltaRandom(
                    atom_types_dense_one_hot[batch_idx].reshape(-1)
                ),
                NUM_ATOMIC_TYPES * max_num_atoms,
            )
        elif atom_type_manifold == "analog_bits":
            a_manifold = (
                MultiAtomAnalogBits(
                    analog_bits_scale,
                    num_bits=NUM_ATOMIC_BITS,
                    num_atoms=num_atom,
                    max_num_atoms=max_num_atoms,
                ),
                NUM_ATOMIC_BITS * max_num_atoms,
            )
        else:
            raise ValueError(
                f"{atom_type_manifold=} not in {atom_type_manifold_types=}"
            )

        if coord_manifold == "flat_torus_01":
            f_manifold = (
                MaskedNoDriftFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_normal":
            f_manifold = (
                MaskedNoDriftFlatTorus01WrappedNormal(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_fixfirst":
            f_manifold = (
                FlatTorus01FixFirstAtomToOrigin(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_fixfirst_normal":
            f_manifold = (
                FlatTorus01FixFirstAtomToOriginWrappedNormal(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        else:
            raise ValueError(f"{coord_manifold=} not in {coord_manifold_types=}")

        if lattice_manifold == "non_symmetric":
            l_manifold = (EuclideanWithLogProb(ndim=1), dim_coords**2)
        elif lattice_manifold == "spd_euclidean_geo":
            spd = SPDGivenN.from_dataset(
                num_atom,
                dataset,
                Riem_geodesic=False,
            )
            l_manifold = (spd, spd.vecdim(dim_coords))
        elif lattice_manifold == "spd_riemanian_geo":
            spd = SPDGivenN.from_dataset(
                num_atom,
                dataset,
                Riem_geodesic=True,
            )
            l_manifold = (spd, spd.vecdim(dim_coords))
        elif lattice_manifold == "lattice_params":
            lp = LatticeParams.from_dataset(dataset, length_inner_coef)
            l_manifold = (lp, LatticeParams.dim(dim_coords))
        elif lattice_manifold == "lattice_params_normal_base":
            lp = LatticeParamsNormalBase()
            l_manifold = (lp, LatticeParamsNormalBase.dim(dim_coords))
        else:
            raise ValueError(f"{lattice_manifold=} not in {lattice_manifold_types=}")

        manifolds = [a_manifold] + [f_manifold] + [l_manifold]
        if split_manifold:
            return (
                ProductManifoldWithLogProb(*manifolds),
                a_manifold[0],
                f_manifold[0],
                l_manifold[0],
            )
        else:
            return ProductManifoldWithLogProb(*manifolds)

    def get_dims(self, dim_coords: int, max_num_atoms: int) -> Dims:
        if self.atom_type_manifold in ["null_manifold", "simplex"]:
            dim_a = NUM_ATOMIC_TYPES * max_num_atoms
        elif self.atom_type_manifold == "analog_bits":
            dim_a = NUM_ATOMIC_BITS * max_num_atoms
        else:
            raise NotImplementedError("")

        dim_f = dim_coords * max_num_atoms

        if self.lattice_manifold == "non_symmetric":
            dim_l = dim_coords**2
        elif "spd" in self.lattice_manifold:
            dim_l = SPDGivenN.vecdim(dim_coords)
        elif (
            self.lattice_manifold == "lattice_params"
            or self.lattice_manifold == "lattice_params_normal_base"
        ):
            dim_l = LatticeParams.dim(dim_coords)
        else:
            raise NotImplementedError("")

        return Dims(dim_a, dim_f, dim_l)

    def get_manifolds(
        self,
        num_atoms: torch.LongTensor,
        atom_types_dense_one_hot: torch.Tensor | None,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[VMapManifolds, tuple[int, int, int]]
        | tuple[
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            tuple[int, int, int],
        ]
    ):
        max_num_atoms = num_atoms.amax(0).cpu().item()

        out_manifolds, a_manifolds, f_manifolds, l_manifolds = [], [], [], []
        for batch_idx, num_atom in enumerate(num_atoms):
            manis = self._get_manifold(
                num_atom,
                atom_types_dense_one_hot,
                dim_coords,
                max_num_atoms,
                batch_idx,
                split_manifold=split_manifold,
                atom_type_manifold=self.atom_type_manifold,
                coord_manifold=self.coord_manifold,
                lattice_manifold=self.lattice_manifold,
                dataset=self.dataset,
                analog_bits_scale=self.analog_bits_scale,
                length_inner_coef=self.length_inner_coef,
            )
            if split_manifold:
                out_manifolds.append(manis[0])
                a_manifolds.append(manis[1])
                f_manifolds.append(manis[2])
                l_manifolds.append(manis[3])
            else:
                out_manifolds.append(manis)

        if split_manifold:
            return (
                VMapManifolds(out_manifolds),
                VMapManifolds(a_manifolds),
                VMapManifolds(f_manifolds),
                VMapManifolds(l_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
        else:
            return (
                VMapManifolds(out_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
