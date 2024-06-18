"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import torch
from pymatgen.core import (
    Composition,
    DummySpecies,
    Element,
    Lattice,
    Species,
    Structure,
)

from diffcsp.common.data_utils import lattices_to_params_shape
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from flowmm.geometric_ import mask_2d_to_batch
from flowmm.joblib_ import joblib_map
from flowmm.rfm.manifold_getter import ManifoldGetter

trap = io.StringIO()

COLUMNS_COMPUTATIONS = {
    "composition": lambda df: df["structure"].map(get_composition_dict),
    "chemsys": lambda df: df["composition"].map(get_chemsys),
    "nary": lambda df: df["chemsys"].map(get_nary),
}


def get_nary(
    chemsys: set[Element, Species, DummySpecies] | list[Element, Species, DummySpecies]
) -> int:
    return len(set(chemsys))


def get_composition(
    structure: Structure | dict,
) -> Composition:
    if structure == {}:
        return Composition()
    return to_structure(structure).composition


def get_composition_dict(
    structure: Structure | dict,
) -> dict[str, float]:
    return get_composition(structure).as_dict()


def get_chemsys(
    composition: Composition | dict,
) -> set[Element | Species | DummySpecies] | None:
    elements = list(elt.name for elt in to_composition(composition).elements)
    return set(elements)


def to_composition(composition: Composition | dict) -> Composition:
    if isinstance(composition, dict):
        return Composition.from_dict(composition)
    else:
        return composition


def to_structure(structure: Structure | dict | str) -> Structure:
    with redirect_stdout(trap):
        if isinstance(structure, dict):
            return Structure.from_dict(structure)
        elif isinstance(structure, str):
            return Structure.from_str(structure, fmt="cif")
        else:
            return structure


def cdvae_to_structure(gen: dict, i: int) -> Structure:
    indexes = torch.cumsum(gen["num_atoms"].squeeze(), dim=0)
    start, end = 0 if i == 0 else indexes[i - 1], indexes[i]
    lengths = gen["lengths"][0][i]
    angles = gen["angles"][0][i]
    atom_types = gen["atom_types"][0][start:end]
    frac_coords = gen["frac_coords"][0][start:end]
    # print(lengths, angles, atom_types.shape, frac_coords.shape)
    return Structure(
        lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def diffcsp_to_structure(gen: dict, i: int, clip_atom_types: bool) -> Structure:
    indexes = torch.cumsum(gen["num_atoms"].squeeze(), dim=0)
    start = 0 if i == 0 else int(indexes[i - 1])
    end = int(indexes[i])
    lengths = gen["lengths"][i]
    angles = gen["angles"][i]
    if gen["atom_types"].ndim == 1:
        atom_types = gen["atom_types"][start:end]
    elif gen["atom_types"].shape[-1] == NUM_ATOMIC_BITS:
        atom_types = ManifoldGetter._inverse_atomic_bits(gen["atom_types"][start:end])
    elif gen["atom_types"].shape[-1] == NUM_ATOMIC_TYPES:
        atom_types = ManifoldGetter._inverse_atomic_one_hot(
            gen["atom_types"][start:end]
        )
    else:
        raise ValueError("unrecognized shape")
    frac_coords = gen["frac_coords"][start:end]
    if any(atom_types > 118) and not clip_atom_types:
        raise ValueError(f"there is an atom type > 118, {atom_types=}")
    if clip_atom_types:
        atom_types = atom_types.clip(0, 118)
    return Structure(
        lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


# from diffcsp
def _torch_geometric_to_crystal_dicts(
    frac_coords: torch.Tensor,
    atom_types: torch.LongTensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    num_atoms: torch.LongTensor,
) -> list[dict[str, np.ndarray]]:
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_dicts = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_dicts.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_dicts


def _crystal_dict_to_structure(crystal: dict[str, np.ndarray]) -> Structure | None:
    frac_coords = crystal["frac_coords"]
    atom_types = crystal["atom_types"]
    lengths = crystal["lengths"]
    angles = crystal["angles"]
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        return structure
    except:
        return None


def torch_geometric_to_structures(
    atom_types: torch.Tensor,
    frac_coords: torch.Tensor,
    lattices: torch.Tensor,
    batch: torch.LongTensor,
    n_jobs: int = -4,
    inner_max_num_threads: int = 2,
) -> list[Structure]:
    if atom_types.ndim > 1:
        atom_types = atom_types.argmax(dim=-1)
    lengths, angles = lattices_to_params_shape(lattices)
    _, num_atoms = batch.unique_consecutive(return_counts=True)
    crystal_dicts = _torch_geometric_to_crystal_dicts(
        frac_coords, atom_types, lengths, angles, num_atoms
    )
    return joblib_map(
        _crystal_dict_to_structure,
        crystal_dicts,
        n_jobs=n_jobs,
        inner_max_num_threads=inner_max_num_threads,
    )


def torch_geometric_mask_to_structures(
    atom_types: torch.Tensor,
    frac_coords: torch.Tensor,
    lattices: torch.Tensor,
    mask_a_or_f: torch.BoolTensor,
) -> list[Structure]:
    batch = mask_2d_to_batch(mask_a_or_f)
    return torch_geometric_to_structures(atom_types, frac_coords, lattices, batch)


def get_get_structure(gen: dict, clip_atom_types: bool) -> callable:
    try:
        gen["eval_setting"].down_sample_traj_step
        if gen["frac_coords"].shape[0] == 1:
            get_structure = cdvae_to_structure
        else:
            get_structure = lambda *args, **kwargs: diffcsp_to_structure(
                *args, **kwargs, clip_atom_types=clip_atom_types
            )
    except AttributeError:
        get_structure = lambda *args, **kwargs: diffcsp_to_structure(
            *args, **kwargs, clip_atom_types=clip_atom_types
        )
    return get_structure
