"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from collections import namedtuple
from typing import Literal

import numpy as np
import torch
from torch_geometric.utils import to_dense_batch

from flowmm.cfg_utils import dataset_options
from flowmm.geometric_ import mask_2d_to_batch
from flowmm.rfm.manifolds import (
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
    ProductManifoldWithLogProb,
)
from flowmm.rfm.manifolds.flat_torus import (
    MaskedNoDriftFlatTorus01,
    MaskedNoDriftFlatTorus01WrappedNormal,
)
from flowmm.rfm.vmap import VMapManifolds

Dims = namedtuple("Dims", ["f"])
ManifoldGetterOut = namedtuple(
    "ManifoldGetterOut", ["flat", "manifold", "dims", "mask_f"]
)
SplitManifoldGetterOut = namedtuple(
    "SplitManifoldGetterOut",
    [
        "flat",
        "manifold",
        "f_manifold",
        "dims",
        "mask_f",
    ],
)
GeomTuple = namedtuple("GeomTuple", ["f"])
coord_manifold_types = Literal[
    "flat_torus_01",
    "flat_torus_01_normal",
    "flat_torus_01_fixfirst",
    "flat_torus_01_fixfirst_normal",
]


class DockingManifoldGetter(torch.nn.Module):
    """Only contains the coord manifold"""

    def __init__(
        self,
        coord_manifold: coord_manifold_types,
        dataset: dataset_options | None = None,
    ) -> None:
        super().__init__()
        self.coord_manifold = coord_manifold
        self.dataset = dataset

    @staticmethod
    def _get_max_num_atoms(mask_f: torch.BoolTensor) -> int:
        """Returns the maximum number of atoms in the batch"""
        return int(mask_f.sum(dim=-1).max())

    @staticmethod
    def _get_num_atoms(mask_f: torch.BoolTensor) -> torch.LongTensor:
        """Returns the number of atoms in each graph in the batch"""
        return mask_f.sum(dim=-1)

    def _to_dense(
        self,
        batch: torch.LongTensor,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Converts sparse batches of variable-sized graphs into dense tensors.
        Pads each graph in the batch with zeros so that all graphs have the same number of nodes.
        Returns a mask tensor that indicates the valid nodes for each graph.
        """
        f, mask_f = to_dense_batch(x=frac_coords, batch=batch)  # B x N x 3, B x N
        return f, mask_f

    def _to_flat(
        self,
        f: torch.Tensor,
        dims: Dims,
    ) -> torch.Tensor:
        f_flat = f.reshape(f.size(0), dims.f)
        return f_flat

    def georep_to_flatrep(
        self,
        batch: torch.LongTensor,
        frac_coords: torch.Tensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts from georep to the manifold flatrep"""
        f, mask_f = self._to_dense(batch, frac_coords=frac_coords)
        num_atoms = self._get_num_atoms(mask_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            dim_coords=frac_coords.shape[-1],
            split_manifold=split_manifold,
        )
        # manifolds = [manifold.to(device=batch.device) for manifold in manifolds]
        flat = self._to_flat(f, dims)
        if split_manifold:
            return SplitManifoldGetterOut(
                flat,
                *manifolds,
                dims,
                mask_f,
            )
        else:
            return ManifoldGetterOut(
                flat,
                *manifolds,
                dims,
                mask_f,
            )

    def forward(
        self,
        batch: torch.LongTensor,
        frac_coords: torch.Tensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """Converts data from the loader into the georep, then to the manifold flatrep"""
        return self.georep_to_flatrep(batch, frac_coords, split_manifold)

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
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """Builds the manifolds from an empty batch"""
        _, mask_f = to_dense_batch(
            x=torch.zeros((*batch.shape, 1), device=batch.device), batch=batch
        )  # B x N
        num_atoms = self._get_num_atoms(mask_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms, dim_coords, split_manifold=split_manifold
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_f

    @staticmethod
    def _from_dense(
        f: torch.Tensor,
        mask_f: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms the dense representation into its sparse representation"""
        return f[mask_f]

    def _from_flat(
        self,
        flat: torch.Tensor,
        dims: Dims,
        max_num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, _ = torch.tensor_split(flat, np.cumsum(dims).tolist(), dim=1)

        return f.reshape(-1, max_num_atoms, dims.f // max_num_atoms)

    def flatrep_to_georep(
        self, flat: torch.Tensor, dims: Dims, mask_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to the georep"""
        max_num_atoms = self._get_max_num_atoms(mask_f)
        f = self._from_flat(flat, dims, max_num_atoms)
        f = self._from_dense(f, mask_f)
        return GeomTuple(f)

    def georep_to_crystal(
        self,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the georep to (one_hot / bits, frac_coords, lattice matrix)"""

        return frac_coords

    def flatrep_to_crystal(
        self, flat: torch.Tensor, dims: Dims, mask_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to (one_hot / bits, frac_coords, lattice matrix)"""
        return self.georep_to_crystal(*self.flatrep_to_georep(flat, dims, mask_f))

    @staticmethod
    def mask_f_to_batch(mask_f: torch.BoolTensor) -> torch.LongTensor:
        return mask_2d_to_batch(mask_f)

    @staticmethod
    def _get_manifold(
        num_atom: int,
        dim_coords: int,
        max_num_atoms: int,
        split_manifold: bool,
        coord_manifold: coord_manifold_types,
    ) -> (
        tuple[ProductManifoldWithLogProb]
        | FlatTorus01FixFirstAtomToOrigin
        | FlatTorus01FixFirstAtomToOriginWrappedNormal
        | MaskedNoDriftFlatTorus01
        | MaskedNoDriftFlatTorus01WrappedNormal,
    ):  # type: ignore
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

        manifolds = [f_manifold]
        if split_manifold:
            return (
                ProductManifoldWithLogProb(*manifolds),
                f_manifold[0],
            )
        else:
            return ProductManifoldWithLogProb(*manifolds)

    def get_dims(self, dim_coords: int, max_num_atoms: int) -> Dims:
        dim_f = dim_coords * max_num_atoms

        return Dims(dim_f)

    def get_manifolds(
        self,
        num_atoms: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[VMapManifolds, tuple[int]]
        | tuple[
            VMapManifolds,
            VMapManifolds,
            tuple[int],
        ]
    ):
        max_num_atoms = num_atoms.amax(0).cpu().item()

        out_manifolds, f_manifolds = [], []
        for batch_idx, num_atom in enumerate(num_atoms):
            manis = self._get_manifold(
                num_atom,
                dim_coords,
                max_num_atoms,
                split_manifold=split_manifold,
                coord_manifold=self.coord_manifold,
            )
            if split_manifold:
                out_manifolds.append(manis[0])
                f_manifolds.append(manis[1])
            else:
                out_manifolds.append(manis)

        if split_manifold:
            return (
                VMapManifolds(out_manifolds),
                VMapManifolds(f_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
        else:
            return (
                VMapManifolds(out_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
