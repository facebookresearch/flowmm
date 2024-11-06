import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader.dataloader import Collater

from rfm_docking.reassignment import ot_reassignment
from rfm_docking.docking_manifold_getter import DockingManifoldGetter


def collate_fn(
    batch: list[HeteroData], manifold_getter: DockingManifoldGetter, do_ot: bool = False
) -> HeteroData:
    """"""
    batch = Batch.from_data_list(batch)

    # batch data
    osda = batch.osda
    zeolite = batch.zeolite

    # TODO put this into collate_fn
    osda = Batch.from_data_list(osda)
    zeolite = Batch.from_data_list(zeolite)

    (
        x1,
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

    _, _, _, zeolite_dims, zeolite_mask_f = manifold_getter(
        zeolite.batch, zeolite.frac_coords, split_manifold=True
    )
    zeolite.dims = zeolite_dims
    zeolite.mask_f = zeolite_mask_f

    # sample mol in fractional space
    x0 = osda_manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    # build edges

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

    batch = HeteroData()
    batch.osda = osda
    batch.zeolite = zeolite
    batch.x0 = x0
    batch.x1 = x1
    batch.lattices = lattices

    return batch
