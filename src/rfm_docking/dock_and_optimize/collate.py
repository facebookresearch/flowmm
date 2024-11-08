import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader.dataloader import Collater

from rfm_docking.reassignment import ot_reassignment
from rfm_docking.manifold_getter import DockingManifoldGetter


def collate_fn(
    batch: list[HeteroData], manifold_getter: DockingManifoldGetter, do_ot: bool = False
) -> HeteroData:
    """Where the magic happens"""
    batch_size = len(batch)
    batch = Batch.from_data_list(batch)

    # batch data
    osda = batch.osda
    zeolite_dock = batch.zeolite_dock
    zeolite_opt = batch.zeolite_opt

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    osda = Batch.from_data_list(osda)
    zeolite = Batch.from_data_list(zeolite_opt)
    zeolite_dock = Batch.from_data_list(zeolite_dock)

    (
        osda_x1,
        osda_manifold,
        osda_f_manifold,
        osda_dims,
        osda_mask_f,
    ) = manifold_getter(
        osda.batch,
        osda.frac_coords,
        split_manifold=True,
    )

    osda.manifold = osda_manifold
    osda.f_manifold = osda_f_manifold
    osda.dims = osda_dims
    osda.mask_f = osda_mask_f

    zeolite_x1, _, _, zeolite_dims, zeolite_mask_f = manifold_getter(
        zeolite.batch, zeolite.frac_coords, split_manifold=True
    )
    zeolite.dims = zeolite_dims
    zeolite.mask_f = zeolite_mask_f

    x1 = [
        torch.cat([osda_x1[osda.batch == i], zeolite_x1[zeolite.batch == i]], dim=0)
        for i in range(batch_size)
    ]
    x1 = torch.cat(x1, dim=0)

    # sample osda in fractional space
    osda_x0 = osda_manifold.random(
        *osda_x1.shape, dtype=osda_x1.dtype, device=osda_x1.device
    )

    zeolite_x0, _, _, _, _ = manifold_getter.georep_to_flatrep(
        zeolite_dock.batch, zeolite_dock.frac_coords, split_manifold=True
    )

    x0 = [
        torch.cat([osda_x0[osda.batch == i], zeolite_x0[zeolite.batch == i]], dim=0)
        for i in range(batch_size)
    ]
    x0 = torch.cat(x0, dim=0)

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    batch = torch.cat([osda.batch, zeolite.batch], dim=0)

    num_atoms = osda.num_atoms + zeolite.num_atoms

    # potentially do ot
    if do_ot:
        # OT
        # in georep for reassignment:
        x0_geo = manifold_getter.flatrep_to_georep(x0, osda_dims, osda_mask_f).f
        x1_geo = manifold_getter.flatrep_to_georep(x1, osda_dims, osda_mask_f).f

        # reassign x0 to x1
        reassigned_idx = ot_reassignment(x0_geo, x1_geo, osda.batch, cost="geodesic")
        x1_geo = x1_geo[reassigned_idx]
        x1 = manifold_getter.georep_to_flatrep(
            osda.batch, x1_geo, split_manifold=True
        ).flat

    batch = Batch(
        crystal_id=crystal_id,
        smiles=smiles,
        osda=osda,
        zeolite=zeolite,
        x0=x0,
        x1=x1,
        lattices=lattices,
        num_atoms=num_atoms,
        batch=batch,
    )

    return batch
