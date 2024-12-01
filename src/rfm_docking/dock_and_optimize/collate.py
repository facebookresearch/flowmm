import torch
from torch_geometric.data import Batch, HeteroData, Data

from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import reassign_molecule
from rfm_docking.manifold_getter import DockingManifoldGetter


def dock_and_optimize_collate_fn(
    batch: list[HeteroData], manifold_getter: DockingManifoldGetter, do_ot: bool = False
) -> Batch:
    """Where the magic happens"""
    batch_size = len(batch)
    batch = Batch.from_data_list(batch)

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    osda = Batch.from_data_list(batch.osda_opt)
    zeolite_dock = Batch.from_data_list(batch.zeolite)
    zeolite_opt = Batch.from_data_list(batch.zeolite_opt)

    frac_coords = [
        torch.cat([osda_opt.frac_coords, zeolite_opt.frac_coords], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    atom_types = [
        torch.cat([osda_opt.atom_types, zeolite_opt.atom_types], dim=0)
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
            lengths=lengths[i],
            angles=angles[i],
            num_atoms=num_atoms[i],
            batch=torch.ones_like(frac_coords[i], dtype=torch.long) * i,
        )
        for i in range(batch_size)
    ]

    batch = Batch.from_data_list(data_list)

    (
        x1,
        manifold,
        f_manifold,
        dims,
        mask_f,
    ) = manifold_getter(
        batch.batch,
        batch.frac_coords,
        split_manifold=True,
    )

    # sample osda in fractional space in georep (N, 3)
    x0 = manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)

    # set the docked (but unoptimized) zeolite structure as x0
    zeolite_x0 = zeolite_dock.frac_coords
    for i, (num_atoms_osda, num_atoms_zeolite) in enumerate(
        zip(osda.num_atoms, zeolite_dock.num_atoms)
    ):
        x0[i, num_atoms_osda * 3 : (num_atoms_osda + num_atoms_zeolite) * 3] = (
            zeolite_x0[zeolite_dock.batch == i].flatten()
        )

        # potentially do ot
        if do_ot:
            # OT
            # in georep for reassignment:
            x0_geo = manifold_getter.flatrep_to_georep(x0, dims, mask_f).f
            x1_geo = manifold_getter.flatrep_to_georep(x1, dims, mask_f).f

            x0_osda_geo = x0_geo[batch.batch == i][:num_atoms_osda]
            x1_osda_geo = x1_geo[batch.batch == i][:num_atoms_osda]
            osda_atom_types = batch.atom_types[batch.batch == i][:num_atoms_osda]

            # reassign x0 to x1
            _, reassigned_idx = reassign_molecule(x0_osda_geo, x1_osda_geo)
            x1_osda_geo = x1_osda_geo[reassigned_idx]

            # reassigned x1 back to flatrep
            x1[i, : num_atoms_osda * 3] = x1_osda_geo.flatten()

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([batch.lengths, batch.angles], dim=-1)

    batch = Batch(
        crystal_id=crystal_id,
        smiles=smiles,
        osda=osda,
        zeolite=zeolite_opt,
        x0=x0,
        x1=x1,
        atom_types=batch.atom_types,
        lattices=lattices,
        num_atoms=batch.num_atoms,
        manifold=manifold,
        f_manifold=f_manifold,
        dims=dims,
        mask_f=mask_f,
        batch=batch.batch,
    )

    return batch


class DockAndOptimizeCollater:
    def __init__(self, manifold_getter: DockingManifoldGetter, do_ot: bool = False):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return dock_and_optimize_collate_fn(
            batch=batch,
            manifold_getter=self.manifold_getter,
            do_ot=self.do_ot,
        )
