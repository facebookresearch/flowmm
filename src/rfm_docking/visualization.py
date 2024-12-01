from ase.io import read, write
from ase import Atoms, Atom
from ase.visualize import view
from pymatgen.core.lattice import Lattice
import torch
import pandas as pd
import numpy as np

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import ot_reassignment
from rfm_docking.featurization import get_atoms_and_pos
from diffcsp.common.data_utils import (
    lattice_params_to_matrix,
)


class InMemoryTrajectory:
    def __init__(self):
        self.frames = []

    def write(self, atoms):
        self.frames.append(atoms)

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)


def create_gif_from_traj(
    traj_file,
    output_gif,
):
    data = torch.load(traj_file, map_location="cpu")

    lattice = data["lattices"]

    osda = data["osda"]
    osda_atoms = osda["atom_types"]

    zeolite = data["zeolite"]
    zeolite_atoms = zeolite["atom_types"]

    if zeolite["frac_coords"].shape[0] == 1:
        zeolite["frac_coords"] = zeolite["frac_coords"].repeat(
            osda["frac_coords"].shape[0], 1, 1
        )

    structures = []
    for osda_coords, zeolite_coords in zip(osda["frac_coords"], zeolite["frac_coords"]):
        zeolite_atoms_i = zeolite_atoms
        zeolite_coords_i = zeolite_coords

        osda_atoms_i = osda_atoms
        osda_coords_i = osda_coords

        """osda_target_coords = (
            osda["optimize_target_coords"] % 1.0
        )"""
        osda_target_coords = osda["target_coords"] % 1.0

        # atoms_i = torch.cat([osda_atoms_i, zeolite_atoms_i])
        # coords_i = torch.cat([osda_coords_i, zeolite_coords_i])
        atoms_i = torch.cat([osda_atoms_i, osda_atoms_i])
        coords_i = torch.cat([osda_coords_i, osda_target_coords])

        # atoms_i = osda_atoms_i
        # coords_i = osda_coords_i

        predicted = Atoms(
            atoms_i, scaled_positions=coords_i, cell=tuple(lattice.squeeze().tolist())
        )

        """# TODO add some distincting color for the target
        for i, target_atom in enumerate(osda_atoms):
            target = Atom(target_atom, osda["target_coords"][i])
            predicted.append(target)"""

        structures.append(predicted)

    traj = InMemoryTrajectory()
    for atoms in structures:
        traj.write(atoms)

    write(output_gif, traj, rotation="30x,30y,30z", interval=1)
    print(f"GIF saved as {output_gif}")


def show_ground_truth(crystal_id):
    data_path = "/home/malte/flowmm/data/original_data.csv"

    df = pd.read_csv(data_path)

    row = df[df["dock_crystal"] == crystal_id]

    lattice = eval(row.dock_lattice[0])
    lattice = np.array(lattice)

    # lattice has to conform to pymatgen's Lattice object, rotate data accordingly
    lattice_matrix_target = lattice_params_to_matrix(*Lattice(lattice).parameters)

    M = lattice.T @ lattice_matrix_target
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure R is a proper rotation matrix with determinant 1
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1  # Correct for reflection if necessary
        R = U @ Vt

    lattice = lattice_matrix_target

    dock_axyz = eval(row.dock_xyz[0])
    dock_axyz = np.array(dock_axyz)

    dock_atoms, dock_pos = get_atoms_and_pos(dock_axyz)

    non_hydrogen = torch.where(dock_atoms.squeeze() != 1, True, False)

    dock_atoms = dock_atoms[non_hydrogen]
    dock_pos = dock_pos[non_hydrogen]
    dock_pos = dock_pos @ R

    # dock pos to fractional
    dock_pos = torch.inverse(torch.tensor(lattice)) @ dock_pos.T
    dock_pos = dock_pos % 1.0

    dock_pos = dock_pos.T

    struc = Atoms(
        dock_atoms, scaled_positions=dock_pos, cell=tuple(lattice.squeeze().tolist())
    )

    write("RFM_EMM17_0_target.png", struc, rotation="30x,30y,30z")


def vis_struc(atoms, frac_coords, lattice, name):
    struc = Atoms(
        atoms, scaled_positions=frac_coords, cell=tuple(lattice.squeeze().tolist())
    )
    write(f"{name}.png", struc, rotation="30x,30y,30z")


if __name__ == "__main__":
    # show_ground_truth(536467517)
    traj_file = "/home/malte/flowmm/runs/trash/2024-12-01/11-34-47/docking_only_coords-dock_cspnet-te07cq7v/536399918_traj.pt"
    crystal_id = traj_file.split("/")[-1].split("_")[0]

    create_gif_from_traj(
        # traj_file="/home/malte/flowmm/runs/trash/2024-11-13/10-44-45/docking_only_coords-dock_and_optimize_cspnet-tsq861qh/traj.pt",
        traj_file=traj_file,  # "/home/malte/flowmm/runs/trash/2024-11-26/14-30-27/docking_only_coords-dock_cspnet-iohsk2ru/traj.pt",
        output_gif=f"{crystal_id}.gif",
    )
