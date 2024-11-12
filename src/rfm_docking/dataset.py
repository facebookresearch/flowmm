import os
import hydra
import omegaconf
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from omegaconf import ValueNode
from p_tqdm import p_umap

from diffcsp.common.utils import PROJECT_ROOT
from rfm_docking.utils import sample_mol_in_frac
from tqdm import tqdm
from multiprocessing import Pool
from diffcsp.common.data_utils import (
    preprocess,
    add_scaled_lattice_prop,
    build_crystal_graph,
    lattice_params_to_matrix,
)
from rfm_docking.featurization import (
    get_atoms_and_pos,
    split_zeolite_and_osda_pos,
    featurize_osda,
)


def process_one(args):
    row, graph_method, prop_list = args

    crystal_id = row.dock_crystal

    ### process the lattice
    lattice_matrix = eval(row.dock_lattice)
    lattice_matrix = np.array(lattice_matrix)

    # lattice has to conform to pymatgen's Lattice object, rotate data accordingly
    lattice_matrix_target = lattice_params_to_matrix(
        *Lattice(lattice_matrix).parameters
    )

    M = lattice_matrix.T @ lattice_matrix_target
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure R is a proper rotation matrix with determinant 1
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1  # Correct for reflection if necessary
        R = U @ Vt

    lattice = Lattice(lattice_matrix_target)

    smiles = row.smiles
    loading = int(row.loading)
    node_feats, edge_feats, edge_index = featurize_osda(smiles)

    ### process the docked structures
    dock_axyz = eval(row.dock_xyz)

    # zeolite and osda are separated
    dock_zeolite_axyz, dock_osda_axyz = split_zeolite_and_osda_pos(
        dock_axyz, smiles, loading
    )
    # process the zeolite
    dock_zeolite_atoms, dock_zeolite_pos = get_atoms_and_pos(dock_zeolite_axyz)
    dock_zeolite_pos = dock_zeolite_pos @ R
    dock_zeolite = Structure(
        lattice=lattice,
        coords=dock_zeolite_pos,
        species=dock_zeolite_atoms,
        coords_are_cartesian=True,
    )
    dock_zeolite_graph_arrays = build_crystal_graph(dock_zeolite, graph_method)

    # process the osda
    dock_osda_atoms, dock_osda_pos = get_atoms_and_pos(dock_osda_axyz)

    # remove hydrogens
    non_hydrogen = torch.where(dock_osda_atoms.squeeze() != 1, True, False)
    dock_osda_atoms = dock_osda_atoms[non_hydrogen]
    dock_osda_pos = dock_osda_pos[non_hydrogen]

    dock_osda_pos = dock_osda_pos @ R
    dock_osda = Structure(
        lattice=lattice,
        coords=dock_osda_pos,
        species=dock_osda_atoms,
        coords_are_cartesian=True,
    )
    dock_osda_graph_arrays = build_crystal_graph(dock_osda, graph_method)

    ### process the optimized structures
    opt_axyz = eval(row.opt_xyz)
    opt_zeolite_axyz, opt_ligand_axyz = split_zeolite_and_osda_pos(
        opt_axyz, smiles, loading
    )
    opt_zeolite_atoms, opt_zeolite_pos = get_atoms_and_pos(opt_zeolite_axyz)
    opt_zeolite_pos = opt_zeolite_pos @ R
    opt_zeolite = Structure(
        lattice=lattice,
        coords=opt_zeolite_pos,
        species=opt_zeolite_atoms,
        coords_are_cartesian=True,
    )
    opt_zeolite_graph_arrays = build_crystal_graph(opt_zeolite, graph_method)

    opt_osda_atoms, opt_osda_pos = get_atoms_and_pos(opt_ligand_axyz)

    # remove hydrogens
    non_hydrogen = torch.where(opt_osda_atoms != 1)[0]
    opt_osda_atoms = opt_osda_atoms[non_hydrogen]
    opt_osda_pos = opt_osda_pos[non_hydrogen]

    opt_osda_pos = opt_osda_pos @ R
    opt_osda = Structure(
        lattice=lattice,
        coords=opt_osda_pos,
        species=opt_osda_atoms,
        coords_are_cartesian=True,
    )
    opt_osda_graph_arrays = build_crystal_graph(opt_osda, graph_method)

    properties = {k: row[k] for k in prop_list if k in row.keys()}
    preprocessed_dict = {
        "crystal_id": crystal_id,
        "smiles": smiles,
        "loading": loading,
        "osda_feats": (node_feats, edge_feats, edge_index),
        "dock_zeolite_graph_arrays": dock_zeolite_graph_arrays,
        "dock_osda_graph_arrays": dock_osda_graph_arrays,
        "opt_zeolite_graph_arrays": opt_zeolite_graph_arrays,
        "opt_osda_graph_arrays": opt_osda_graph_arrays,
    }
    preprocessed_dict.update(properties)
    return preprocessed_dict


def custom_preprocess(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    df = pd.read_csv(input_file)  # .loc[:0]

    def parallelized():
        # Create a pool of workers
        with Pool(num_workers) as pool:
            for item in tqdm(
                pool.imap_unordered(
                    process_one,
                    iterable=[
                        (df.iloc[idx], graph_method, prop_list)
                        for idx in range(len(df))
                    ],
                    chunksize=1,
                ),
                total=len(df),
            ):
                yield item

    # Convert the unordered results to a list
    unordered_results = list(parallelized())

    # Create a dictionary mapping crystal_id to results
    mpid_to_results = {result["crystal_id"]: result for result in unordered_results}

    # Create a list of ordered results based on the original order of the dataframe
    ordered_results = [
        mpid_to_results[df.iloc[idx]["dock_crystal"]] for idx in range(len(df))
    ]

    return ordered_results


def custom_add_scaled_lattice_prop(
    data_list, lattice_scale_method, graph_arrays_key="dock_zeolite_graph_arrays"
):
    for dict in data_list:
        graph_arrays = dict[graph_arrays_key]
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == "scale_length":
            lengths = lengths / float(num_atoms) ** (1 / 3)

        dict["scaled_lattice"] = np.concatenate([lengths, angles])


class CustomCrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers, prop)

        # add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        # TODO not sure if how to scale osda lattice - should it be different from zeolite, or scale their combined cell?
        custom_add_scaled_lattice_prop(
            self.cached_data, lattice_scale_method, "dock_zeolite_graph_arrays"
        )
        custom_add_scaled_lattice_prop(
            self.cached_data, lattice_scale_method, "dock_osda_graph_arrays"
        )

        custom_add_scaled_lattice_prop(
            self.cached_data, lattice_scale_method, "opt_zeolite_graph_arrays"
        )
        custom_add_scaled_lattice_prop(
            self.cached_data, lattice_scale_method, "opt_osda_graph_arrays"
        )

        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = custom_preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )

            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,  # NOTE edge indices will be overwritten with rdkit featurization
            to_jimages,
            num_atoms,
        ) = data_dict["dock_osda_graph_arrays"]

        smiles = data_dict["smiles"]

        # node_feats, edge_feats, edge_indices = data_dict["osda_feats"]

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        osda_data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T
            ).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            conformer=data_dict["conformer"] if "conformer" in data_dict else None,
        )

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["dock_zeolite_graph_arrays"]

        zeolite_data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T
            ).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,  # NOTE edge indices will be overwritten with rdkit featurization
            to_jimages,
            num_atoms,
        ) = data_dict["opt_osda_graph_arrays"]

        osda_data_opt = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T
            ).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            conformer=data_dict["conformer"] if "conformer" in data_dict else None,
        )

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["opt_zeolite_graph_arrays"]

        zeolite_data_opt = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T
            ).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        data = HeteroData()
        data.crystal_id = data_dict["crystal_id"]
        data.smiles = smiles
        data.osda = osda_data
        data.osda_opt = osda_data_opt
        data.zeolite = zeolite_data
        data.zeolite_opt = zeolite_data_opt
        data.num_atoms = osda_data.num_atoms + zeolite_data.num_atoms
        data.lengths = data.zeolite.lengths
        data.angles = data.zeolite.angles

        return data

    def __repr__(self) -> str:
        return f"CustomCrystDataset({self.name=}, {self.path=})"


if __name__ == "__main__":
    path = "data/test_data.csv"

    data = pd.read_csv(path)
    data = data.dropna(
        subset=[
            "dock_crystal",
            "dock_lattice",
            "smiles",
            "dock_xyz",
            "opt_xyz",
            "loading",
        ]
    )

    crystal_id = data.dock_crystal.tolist()

    lattices = data.dock_lattice.apply(eval).tolist()
    smiles = data.smiles.tolist()

    dock_xyz = data.dock_xyz.apply(eval).tolist()
    opt_xyz = data.opt_xyz.apply(eval).tolist()

    cols = data.columns
