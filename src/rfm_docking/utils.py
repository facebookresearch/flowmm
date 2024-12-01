import torch
from rdkit import Chem
from rdkit.Chem import AllChem


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
