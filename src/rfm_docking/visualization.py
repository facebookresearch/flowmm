from ase.io import read, write
from ase import Atoms, Atom
import torch


class InMemoryTrajectory:
    def __init__(self):
        self.frames = []

    def write(self, atoms):
        self.frames.append(atoms)

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)


def create_gif_from_cif(
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
    zeolite_atoms = zeolite["atom_types"][zeolite["batch"] == index]
    zeolite_coords = zeolite["frac_coords"][zeolite["batch"] == index]

    structures = []
    for coords in osda["frac_coords"]:
        osda_atoms_i = osda_atoms[osda["batch"] == index]
        osda_atoms_ = torch.cat([osda_atoms_i, osda_atoms_i])  # , zeolite_atoms])

        coords_i = coords[osda["batch"] == index]
        target_coords_i = osda["target_coords"][osda["batch"] == index]

        coords = torch.cat([coords_i, target_coords_i])  # , zeolite_coords])
        predicted = Atoms(
            osda_atoms_, scaled_positions=coords, cell=tuple(lattice.squeeze().tolist())
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
    create_gif_from_cif(
        # traj_file="/home/malte/flowmm/runs/trash/2024-11-08/15-09-58/docking_only_coords-rfm_cspnet-at6o35i2/traj.pt",

        # initial getting to know the code - traj looks good
        # traj_file="/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-12/11-28-32/docking_only_coords-dock_cspnet-6g97h5qt/traj.pt",
        # output_gif="/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-12/11-28-32/docking_only_coords-dock_cspnet-6g97h5qt/traj.gif",

        # added be code without the training and inference part - check traj is still ok 
        # traj_file="/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-15/12-51-09/docking_only_coords-dock_cspnet-uw81j9t7/traj.pt",
        # output_gif="/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-15/12-51-09/docking_only_coords-dock_cspnet-uw81j9t7/traj.gif",

        # training code added
        traj_file="/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-15/16-45-05/docking_only_coords-dock_cspnet-s368tr0o/traj.pt",
        output_gif="/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-15/16-45-05/docking_only_coords-dock_cspnet-s368tr0o/traj.gif",

        # inference code added but it does not output traj - todo mrx figure this out
    )
