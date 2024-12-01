import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader.dataloader import Collater

from rfm_docking.reassignment import reassign_molecule
from rfm_docking.manifold_getter import DockingManifoldGetter
from rfm_docking.sampling import (
    sample_harmonic_prior,
    sample_uniform_then_gaussian,
    sample_uniform,
    sample_uniform_then_conformer,
)


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
    # x0 = osda_manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)

    # harmonic prior
    # x0 = sample_uniform_then_gaussian(osda, batch.loading, sigma=0.05)
    x0 = sample_harmonic_prior(osda, sigma=0.2)
    # x0 = sample_uniform(osda, batch.loading)
    # x0 = sample_uniform_then_conformer(osda, smiles, batch.loading)
    x0 = manifold_getter.georep_to_flatrep(osda.batch, x0, False).flat
    x0 = osda_manifold.projx(x0)

    # x0 = (x1 - 0.47) % 1.0

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    # build edges

    # potentially do ot
    if do_ot:
        # OT
        # in georep for reassignment:
        x0_geo = manifold_getter.flatrep_to_georep(x0, osda_dims, osda_mask_f).f
        x1_geo = manifold_getter.flatrep_to_georep(x1, osda_dims, osda_mask_f).f

        # iterate over batch
        for i in range(len(batch)):
            loading = batch.loading[i]

            x0_i = x0_geo[osda.batch == i].view(loading, -1, 3)
            x1_i = x1_geo[osda.batch == i].view(loading, -1, 3)

            # reassign x0 to x1
            permuted_x1, _, _ = reassign_molecule(x0_i, x1_i)
            permuted_x1 = permuted_x1.view(-1, 3)
            x1_geo[osda.batch == i] = permuted_x1

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
        num_atoms=osda.num_atoms,
        manifold=osda_manifold,
        f_manifold=osda_f_manifold,
        dims=osda_dims,
        mask_f=osda_mask_f,
        batch=osda.batch,
    )
    return batch


class DockCollater:
    def __init__(self, manifold_getter: DockingManifoldGetter, do_ot: bool = False):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return dock_collate_fn(batch, self.manifold_getter, self.do_ot)
