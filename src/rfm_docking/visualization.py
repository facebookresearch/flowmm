from ase.io import read, write
from ase import Atoms, Atom
import torch

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import ot_reassignment


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
    data = torch.load(traj_file)

    index = 0
    data = data[0]
    lattice = data["lattices"][index]

    osda = data["osda_traj"]
    osda_atoms = osda["atom_types"]

    zeolite = data["zeolite"]
    zeolite_atoms = zeolite["atom_types"]

    structures = []
    for osda_coords, zeolite_coords in zip(osda["frac_coords"], zeolite["frac_coords"]):
        zeolite_atoms_i = zeolite_atoms[zeolite["batch"] == index]
        zeolite_coords_i = zeolite_coords[zeolite["batch"] == index]

        osda_atoms_i = osda_atoms[osda["batch"] == index]
        osda_coords_i = osda_coords[osda["batch"] == index]

        osda_target_coords = osda["target_coords"][osda["batch"] == index] % 1.0
        # atoms_i = torch.cat([osda_atoms_i, zeolite_atoms_i])
        # coords_i = torch.cat([osda_coords_i, zeolite_coords_i])
        atoms_i = torch.cat([osda_atoms_i, osda_atoms_i])
        coords_i = torch.cat([osda_coords_i, osda_target_coords])

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


if __name__ == "__main__":
    create_gif_from_traj(
        # traj_file="/home/malte/flowmm/runs/trash/2024-11-13/10-44-45/docking_only_coords-dock_and_optimize_cspnet-tsq861qh/traj.pt",
        traj_file="/home/malte/flowmm/runs/trash/2024-11-18/11-38-35/docking_only_coords-dock_and_optimize_cspnet-ubcbiysy/traj.pt",
        output_gif="RFM_EMM17_0.gif",
    )
