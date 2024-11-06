import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from diffcsp.common.data_utils import (
    cart_to_frac_coords,
)


def smiles_to_pos(smiles, forcefield="mmff", device="cpu"):
    """Convert smiles to 3D coordinates."""
    # Use RDKit to generate a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    # Generate 3D coordinates for the molecule
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    if forcefield == "mmff":
        AllChem.MMFFOptimizeMolecule(mol)
    elif forcefield == "uff":
        AllChem.UFFOptimizeMolecule(mol)
    else:
        raise ValueError("Unrecognised force field")

    mol = Chem.RemoveHs(mol)

    # Extract the atom symbols and coordinates
    atom_coords = []
    for i, _ in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        pos = torch.tensor(
            [positions.x, positions.y, positions.z], dtype=torch.float32, device=device
        )
        atom_coords.append(pos)

    # Return the atomic numbers and coordinates as a tuple
    return torch.stack(atom_coords)


def sample_mol_in_frac(
    smiles,
    lattice_lengths,
    lattice_angles,
    forcefield="mmff",
    regularized=True,
    device="cpu",
):
    """Sample a molecule in a fractional space."""

    all_pos = []
    num_atoms = []
    for smi in smiles:
        # Convert SMILES to 3D coordinates
        pos = smiles_to_pos(smi, forcefield, device)

        all_pos.append(pos)
        num_atoms.append(pos.shape[0])

    all_pos = torch.cat(all_pos, dim=0).reshape(-1, 3).to(device)
    num_atoms = torch.tensor(num_atoms).to(device)

    # Convert cartesian coordinates to fractional coordinates
    frac = cart_to_frac_coords(
        all_pos,
        lattice_lengths,
        lattice_angles,
        num_atoms=num_atoms,
        regularized=regularized,
    )

    return frac
