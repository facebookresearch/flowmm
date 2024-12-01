import torch
from torch_geometric.data import Batch, HeteroData, Data

from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import ot_reassignment
from rfm_docking.manifold_getter import DockingManifoldGetter
from rfm_docking.sampling import sample_harmonic_prior
from rfm_docking.featurization import get_feature_dims


class DockThenOptimizeBatch(Batch):
    """Simply redefines the len method so we don't get into trouble when logging"""

    def __len__(self):
        return len(self.smiles)


def dock_then_optimize_collate_fn(
    batch: list[HeteroData],
    manifold_getter: DockingManifoldGetter,
    do_ot: bool = False,
) -> Batch:
    """Where the magic happens"""
    batch_size = len(batch)
    batch = Batch.from_data_list(batch)

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    ####################################
    # First, we prepare the docking task
    osda_dock = Batch.from_data_list(batch.osda, exclude_keys=["to_jimages, num_bonds"])
    zeolite_dock = Batch.from_data_list(
        batch.zeolite, exclude_keys=["to_jimages, num_bonds"]
    )

    (
        dock_x1,
        dock_manifold,
        dock_f_manifold,
        dock_dims,
        dock_mask_f,
    ) = manifold_getter(
        osda_dock.batch,
        osda_dock.frac_coords,
        split_manifold=True,
    )

    osda_dock.dims = dock_dims
    osda_dock.mask_f = dock_mask_f

    """
    # Harmonic prior sampling
    dock_x0 = sample_harmonic_prior(osda_dock, sigma=0.1)
    dock_x0 = manifold_getter.georep_to_flatrep(
        osda_dock.batch, dock_x0, False
    ).flat
    dock_x0 = dock_manifold.projx(dock_x0)"""

    # sample osda in fractional space in georep (N, 3)
    dock_x0 = dock_manifold.random(
        *dock_x1.shape, dtype=dock_x1.dtype, device=dock_x1.device
    )

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    dock_lattices = torch.cat([osda_dock.lengths, osda_dock.angles], dim=-1)

    # potentially do ot
    if do_ot:
        # OT
        # in georep for reassignment:
        x0_geo = manifold_getter.flatrep_to_georep(dock_x0, dock_dims, dock_mask_f).f
        x1_geo = manifold_getter.flatrep_to_georep(dock_x1, dock_dims, dock_mask_f).f

        for i, num_atoms_osda in enumerate(osda_dock.num_atoms):
            x0_osda_geo = x0_geo[osda_dock.batch == i][:num_atoms_osda]
            x1_osda_geo = x1_geo[osda_dock.batch == i][:num_atoms_osda]
            osda_atom_types = osda_dock.atom_types[osda_dock.batch == i][
                :num_atoms_osda
            ]

            # reassign x0 to x1
            _, reassigned_idx = ot_reassignment(
                x0_osda_geo, x1_osda_geo, osda_atom_types, cost="geodesic"
            )
            x0_osda_geo = x0_osda_geo[reassigned_idx]

            # reassigned x0 back to flatrep
            dock_x0[i, : num_atoms_osda * 3] = x0_osda_geo.flatten()

    dock = Batch(
        x0=dock_x0,
        x1=dock_x1,
        osda=osda_dock,
        zeolite=zeolite_dock,
        atom_types=osda_dock.atom_types,
        num_atoms=osda_dock.num_atoms,
        manifold=dock_manifold,
        f_manifold=dock_f_manifold,
        mask_f=dock_mask_f,
        batch=osda_dock.batch,
        dims=dock_dims,
        lattices=dock_lattices,
    )

    ##########################################
    # Second, we prepare the optimization task
    osda_opt = Batch.from_data_list(
        batch.osda_opt, exclude_keys=["to_jimages, num_bonds"]
    )
    zeolite_opt = Batch.from_data_list(
        batch.zeolite_opt, exclude_keys=["edge_index, to_jimages, num_bonds"]
    )

    frac_coords = [
        torch.cat([osda_opt.frac_coords, zeolite_opt.frac_coords], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    atom_types = [
        torch.cat([osda_opt.atom_types, zeolite_opt.atom_types], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    edges = [osda_opt.edge_index for osda_opt in batch.osda_opt]
    edge_feats = [osda_opt.edge_feats for osda_opt in batch.osda_opt]

    node_feats = [
        torch.cat([osda_opt.node_feats, zeolite_opt.node_feats], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    lengths = [osda.lengths for osda in batch.osda_opt]
    angles = [osda.angles for osda in batch.osda_opt]

    num_atoms = [
        osda.num_atoms + zeolite_opt.num_atoms
        for osda, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    data_list = [
        Data(
            frac_coords=FlatTorus01.projx(frac_coords[i]),  # project to 0 to 1
            atom_types=atom_types[i],
            node_feats=node_feats[i],
            edge_index=edges[i],
            edge_feats=edge_feats[i],
            lengths=lengths[i],
            angles=angles[i],
            num_atoms=num_atoms[i],
            batch=torch.ones_like(frac_coords[i], dtype=torch.long) * i,
        )
        for i in range(batch_size)
    ]

    optimize_batch = Batch.from_data_list(data_list)

    (
        optimize_x1,
        optimize_manifold,
        optimize_f_manifold,
        optimize_dims,
        optimize_mask_f,
    ) = manifold_getter(
        optimize_batch.batch,
        optimize_batch.frac_coords,
        split_manifold=True,
    )

    # NOTE optimize_x0 will be set during training based on the docking results
    optimize_x0 = torch.zeros_like(optimize_x1)

    # set the docked (but unoptimized) zeolite structure as x0
    zeolite_x0 = zeolite_dock.frac_coords
    for i, (num_atoms_osda, num_atoms_zeolite) in enumerate(
        zip(osda_opt.num_atoms, zeolite_dock.num_atoms)
    ):
        optimize_x0[
            i, num_atoms_osda * 3 : (num_atoms_osda + num_atoms_zeolite) * 3
        ] = zeolite_x0[zeolite_dock.batch == i].flatten()

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    optimize_lattices = torch.cat(
        [optimize_batch.lengths, optimize_batch.angles], dim=-1
    )

    optimize = Batch(
        x0=optimize_x0,
        x1=optimize_x1,
        osda=osda_opt,
        zeolite=zeolite_opt,
        edge_index=optimize_batch.edge_index,
        edge_feats=optimize_batch.edge_feats,
        atom_types=optimize_batch.atom_types,
        node_feats=optimize_batch.node_feats,
        num_atoms=optimize_batch.num_atoms,
        manifold=optimize_manifold,
        f_manifold=optimize_f_manifold,
        mask_f=optimize_mask_f,
        batch=optimize_batch.batch,
        dims=optimize_dims,
        lattices=optimize_lattices,
    )

    batch = DockThenOptimizeBatch(
        crystal_id=crystal_id,
        smiles=smiles,
        dock=dock,
        optimize=optimize,
    )

    return batch


class DockThenOptimizeCollater:
    def __init__(self, manifold_getter: DockingManifoldGetter, do_ot: bool = False):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return dock_then_optimize_collate_fn(
            batch=batch,
            manifold_getter=self.manifold_getter,
            do_ot=self.do_ot,
        )
