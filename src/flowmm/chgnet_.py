"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property

from ase import Atoms
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import TrajectoryObserver
from chgnet.model.model import CHGNet
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch import Tensor

from flowmm.joblib_ import joblib_map


@dataclass
class UnrelaxedRelaxedStructurePair:
    """collector of endpoints of relaxations. values are in ase units."""

    structure_dicts: tuple[dict, dict]
    energies: tuple[float, float]
    n_steps_to_relax: int
    stol: float = 0.5
    angle_tol: int = 10
    ltol: float = 0.3

    def __post_init__(self):
        self.matcher = StructureMatcher(
            stol=self.stol,
            angle_tol=self.angle_tol,
            ltol=self.ltol,
        )

    @cached_property
    def structures(self) -> tuple[Structure, Structure]:
        return tuple(Structure.from_dict(sd) for sd in self.structure_dicts)

    @cached_property
    def atoms(self) -> tuple[Atoms, Atoms]:
        return tuple(
            AseAtomsAdaptor.get_atoms(structure) for structure in self.structures
        )

    @cached_property
    def match(self) -> bool:
        return False if self.rms_dist is None else True

    @cached_property
    def rms_dist(self) -> bool:
        out = self.matcher.get_rms_dist(self.structures[0], self.structures[1])
        if out is None:
            return out
        elif isinstance(out, tuple):
            return out[0]
        else:
            raise ValueError()

    @classmethod
    def from_chgnet(
        cls,
        initial_structure: Structure,
        prediction: dict[str, Tensor],
        relaxation: dict[str, Structure | TrajectoryObserver],
    ):
        initial_structure.add_site_property("magmom", prediction["m"])
        final_structure = relaxation["final_structure"]
        trajectory = relaxation["trajectory"]
        return cls(
            structure_dicts=(initial_structure.as_dict(), final_structure.as_dict()),
            energies=(
                prediction["e"] * initial_structure.num_sites,
                trajectory.energies[-1],
            ),
            n_steps_to_relax=len(trajectory.energies),
        )


@dataclasses.dataclass
class RelaxationData:
    index: list[int] = dataclasses.field(default_factory=list)
    e_gen: list[float] = dataclasses.field(default_factory=list)
    e_relax: list[float] = dataclasses.field(default_factory=list)
    n_to_relax: list[int] = dataclasses.field(default_factory=list)
    rms_dist: list[float] = dataclasses.field(default_factory=list)
    matched: list[bool] = dataclasses.field(default_factory=list)
    converged: list[bool] = dataclasses.field(default_factory=list)
    exception: list[bool] = dataclasses.field(default_factory=list)
    num_sites: list[int] = dataclasses.field(default_factory=list)
    structure: list[dict] = dataclasses.field(default_factory=list)


def prerelax_with_chgnet(
    structure: Structure, chgnet: CHGNet | None = None, steps: int = 500
) -> UnrelaxedRelaxedStructurePair:
    if chgnet is None:
        chgnet = CHGNet.load()
    prediction = chgnet.predict_structure(structure)
    relaxer = StructOptimizer(model=chgnet)
    relaxation = relaxer.relax(structure, steps=steps)
    return UnrelaxedRelaxedStructurePair.from_chgnet(structure, prediction, relaxation)


def structures_to_chgnet_graphs(
    converter: CrystalGraphConverter,
    structures: list[Structure],
    n_jobs: int = -4,
    inner_max_num_threads: int = 2,
) -> list[CrystalGraph]:
    return joblib_map(
        converter,
        structures,
        n_jobs=n_jobs,
        inner_max_num_threads=inner_max_num_threads,
    )
