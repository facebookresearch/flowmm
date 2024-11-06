import torch
from rdkit import Chem
import numpy as np

from diffcsp.script_utils import chemical_symbols

atom_features_list = {
    "atomic_num": list(range(len(chemical_symbols))) + ["misc"],  # stop at tungsten
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    # 'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    "numring": [0, 1, 2, 3, 4, 5, 6, "misc"],
    # 'numring': [0, 1, 2, 'misc'],
    "implicit_valence": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "numH": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    # 'numH': [0, 1, 2, 3, 4, 'misc'],
    "number_radical_e": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "is_aromatic": [False, True],
    "is_in_ring3": [False, True],
    "is_in_ring4": [False, True],
    "is_in_ring5": [False, True],
    "is_in_ring6": [False, True],
    "is_in_ring7": [False, True],
    "is_in_ring8": [False, True],
}

bond_features_list = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "bond_stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}


def get_feature_dims():
    node_feature_dims = [
        len(atom_features_list["atomic_num"]),
        len(atom_features_list["chirality"]),
        len(atom_features_list["degree"]),
        len(atom_features_list["numring"]),
        len(atom_features_list["implicit_valence"]),
        len(atom_features_list["formal_charge"]),
        len(atom_features_list["numH"]),
        len(atom_features_list["hybridization"]),
        len(atom_features_list["is_aromatic"]),
        len(atom_features_list["is_in_ring5"]),
        len(atom_features_list["is_in_ring6"]),
    ]
    edge_attribute_dims = [
        len(bond_features_list["bond_type"]),
        len(bond_features_list["bond_stereo"]),
        len(bond_features_list["is_conjugated"]),
        # 1,  # for the distance edge attributes after we combine the distance and bond edges
        # NOTE: so, a scalar that represents some sort of distance? Maybe the bond distance?
    ]
    return (
        node_feature_dims,
        edge_attribute_dims,
    )


def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def featurize_atoms(mol):
    atom_features = []
    ringinfo = mol.GetRingInfo()

    def safe_index_(key, val):
        return safe_index(atom_features_list[key], val)

    for idx, atom in enumerate(mol.GetAtoms()):
        features = [
            safe_index_("atomic_num", atom.GetAtomicNum()),
            safe_index_("chirality", str(atom.GetChiralTag())),
            safe_index_("degree", atom.GetTotalDegree()),
            safe_index_("numring", ringinfo.NumAtomRings(idx)),
            safe_index_("implicit_valence", atom.GetImplicitValence()),
            safe_index_("formal_charge", atom.GetFormalCharge()),
            safe_index_("numH", atom.GetTotalNumHs()),
            safe_index_("hybridization", str(atom.GetHybridization())),
            safe_index_("is_aromatic", atom.GetIsAromatic()),
            safe_index_("is_in_ring5", ringinfo.IsAtomInRingOfSize(idx, 5)),
            safe_index_("is_in_ring6", ringinfo.IsAtomInRingOfSize(idx, 6)),
        ]
        atom_features.append(features)

    return torch.tensor(atom_features)


def featurize_bonds(bond: Chem.Bond):
    """Featurize a bond based on features defined above."""
    bond_feature = [
        safe_index(bond_features_list["bond_type"], str(bond.GetBondType())),
        safe_index(bond_features_list["bond_stereo"], str(bond.GetStereo())),
        safe_index(bond_features_list["is_conjugated"], bond.GetIsConjugated()),
    ]
    return bond_feature


def split_zeolite_and_osda_pos(xyz_data: list, smiles: str, loading: int) -> tuple:
    """
    Split the coordinates of the zeolite and ligands from the data.
    Returns a tuple of the zeolite coordinates and a list containing each ligand's coordinates.
    """

    xyz_data = np.array(xyz_data)
    num_atoms = xyz_data.shape[0]

    osda_mol = Chem.MolFromSmiles(smiles)
    osda_mol = Chem.AddHs(osda_mol)
    num_osda_atoms = osda_mol.GetNumAtoms() * loading

    zeolite_axyz = xyz_data[: num_atoms - num_osda_atoms]
    osdas_axyz = xyz_data[num_atoms - num_osda_atoms :]

    # split_osdas_xyz = np.split(osdas_axyz, loading)

    return zeolite_axyz, osdas_axyz


def get_atoms_and_pos(atom_list: np.array) -> tuple[torch.Tensor, torch.Tensor]:
    """Process axyz file to get atom symbols and positions."""

    atoms = atom_list[:, 0].astype(int)
    atoms = torch.tensor(atoms, dtype=torch.int32)

    pos = torch.tensor(atom_list[:, 1:], dtype=torch.float32)

    return atoms, pos


def get_bond_edges(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """From Stark's FlowSite. Use the RDKit molecule to get the edges."""
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_attr += [featurize_bonds(bond)]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr)
    edge_attr = torch.concat([edge_attr, edge_attr], 0)

    return edge_index, edge_attr.type(torch.uint8)


def featurize_osda(
    ligand_smiles: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Featurize ligand based on SMILES using RDKit"""

    mol = Chem.MolFromSmiles(ligand_smiles)

    node_feats = featurize_atoms(mol)
    edge_index, edge_feats = get_bond_edges(mol)

    return node_feats, edge_feats, edge_index
