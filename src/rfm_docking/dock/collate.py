import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader.dataloader import Collater

from rfm_docking.reassignment import ot_reassignment
from rfm_docking.manifold_getter import DockingManifoldGetter


def dock_collate_fn(
    batch: list[HeteroData], manifold_getter: DockingManifoldGetter, do_ot: bool = False
) -> HeteroData:
    """Where the magic happens"""

    batch = Batch.from_data_list(batch)

    # batch data
    osda = batch.osda
    zeolite = batch.zeolite

    smiles = batch.smiles
    crystal_id = batch.crystal_id

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
    batch.crystal_id = crystal_id
    batch.smiles = smiles
    batch.osda = osda
    batch.zeolite = zeolite
    batch.x0 = x0
    batch.x1 = x1
    batch.lattices = lattices

    batch.num_atoms = osda.num_atoms
    batch.manifold = osda_manifold
    batch.f_manifold = osda_f_manifold
    batch.dims = osda_dims
    batch.mask_f = osda_mask_f
    batch.batch = osda.batch

    return batch


class DockCollater:
    def __init__(self, manifold_getter: DockingManifoldGetter, do_ot: bool = False):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return dock_collate_fn(batch, self.manifold_getter, self.do_ot)
