"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import torch
from geoopt import Manifold
from torch import nn
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Batch
from pymatgen.core import Structure, Lattice
from torch_scatter import scatter
from copy import deepcopy

from diffcsp.common.data_utils import radius_graph_pbc
from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch,
)
from diffcsp.pl_modules.cspnet import CSPLayer as DiffCSPLayer
from diffcsp.pl_modules.cspnet import CSPNet as DiffCSPNet
from diffcsp.pl_modules.cspnet import SinusoidsEmbedding
from diffcsp.script_utils import chemical_symbols

from flowmm.data import NUM_ATOMIC_TYPES
from flowmm.rfm.manifold_getter import (
    Dims,
    ManifoldGetter,
    ManifoldGetterOut,
)
from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.featurization import get_feature_dims


class Encoder(torch.nn.Module):
    """Ripped off from FlowSite. Thanks Hannes"""

    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims is a list with the length of each categorical feature
        super(Encoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding


class CSPLayer(DiffCSPLayer):
    """Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim,
        act_fn,
        dis_emb,
        ln,
        n_space: int = 3,
        represent_num_atoms: bool = False,
        self_cond: bool = False,
    ):
        nn.Module.__init__(self)

        self.self_cond = self_cond
        self.n_space = n_space
        self.dis_emb = dis_emb

        if dis_emb is None:
            self.dis_dim = n_space
        else:
            self.dis_dim = dis_emb.dim
        if self_cond:
            self.dis_dim *= 2

        self.represent_num_atoms = represent_num_atoms
        if represent_num_atoms:
            self.one_hot_dim = 100  # largest cell of atoms that we'd represent, this is safe for a HACK
            self.num_atom_embedding = nn.Linear(
                self.one_hot_dim, hidden_dim, bias=False
            )
            num_hidden_dim_vecs = 3
        else:
            num_hidden_dim_vecs = 2

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim  # for edge_feats
                + hidden_dim * num_hidden_dim_vecs  # for hi, hj
                + 3 * 2  # for lattices_flat_edges
                + self.dis_dim,  # for frac_diff
                hidden_dim,
            ),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def get_edge_features(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        edge_feats,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        edge_features = []

        if self.dis_emb is not None:
            if self.self_cond:
                _frac_diff, _pred_frac_diff = torch.tensor_split(frac_diff, 2, dim=-1)
                _frac_diff = self.dis_emb(_frac_diff)
                _pred_frac_diff = (
                    torch.zeros_like(_frac_diff)
                    if (torch.zeros_like(_pred_frac_diff) == _pred_frac_diff).all()
                    else self.dis_emb(_pred_frac_diff)
                )
                frac_diff = torch.concat([_frac_diff, _pred_frac_diff], dim=-1)
            else:
                frac_diff = self.dis_emb(frac_diff)

        lattices_flat = lattices.reshape(lattices.shape[0], -1)
        lattices_flat_edges = lattices_flat[edge2graph]

        edge_features.extend([edge_feats, hi, hj, lattices_flat_edges, frac_diff])

        return torch.cat(edge_features, dim=1)

    def edge_model(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        edge_feats,
    ):
        edge_feats = self.get_edge_features(
            node_features, lattices, edge_index, edge2graph, frac_diff, edge_feats
        )

        return self.edge_mlp(edge_feats)

    def forward(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        edge_feats,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, lattices, edge_index, edge2graph, frac_diff, edge_feats
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(DiffCSPNet):
    def __init__(
        self,
        hidden_dim: int = 512,
        time_dim: int = 256,
        num_layers: int = 6,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        n_space: int = 3,
        num_freqs: int = 128,
        edge_style: str = "fc",
        cutoff: float = 7.0,
        max_neighbors: int = 20,
        ln: bool = True,
        use_log_map: bool = True,
        dim_atomic_rep: int = NUM_ATOMIC_TYPES,
        self_edges: bool = True,
        self_cond: bool = False,
        be_dim: int = 256, 
        drop_be_prob: float = 0.0, 
    ):
        nn.Module.__init__(self)

        self.n_space = n_space
        self.time_emb = nn.Linear(1, time_dim, bias=False)
        self.be_emb = nn.Linear(1, be_dim, bias=False)
        self.drop_be_prob = drop_be_prob
        
        self.hidden_dim = hidden_dim

        self.self_cond = self_cond
        if self_cond:
            coef = 2
        else:
            coef = 1

        node_feat_dims, edge_feat_dims = get_feature_dims()
        self.node_feat_dims = node_feat_dims
        self.edge_feat_dims = edge_feat_dims

        self.node_embedding = nn.Sequential(
            Encoder(emb_dim=hidden_dim, feature_dims=node_feat_dims),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_embedding = nn.Sequential(
            Encoder(emb_dim=hidden_dim, feature_dims=edge_feat_dims),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.atom_latent_emb = nn.Linear(
            hidden_dim + time_dim + be_dim,
            hidden_dim,
            bias=True,  # False
        )

        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=n_space)
        elif dis_emb == "none":
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i,
                CSPLayer(
                    hidden_dim,
                    self.act_fn,
                    self.dis_emb,
                    ln=ln,
                    n_space=n_space,
                    self_cond=self_cond,
                ),
            )
        self.num_layers = num_layers

        # it makes sense to have no bias here since p(F) is translation invariant
        self.coord_out = nn.Linear(hidden_dim, n_space, bias=False)

        # readout block for binding energy. TODO mrx add options: : osda only, zeolite only, osda concat zeolite, osda + zeolite. Check padding as well
        self.be_out = nn.Linear(hidden_dim, 1, bias=True)

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        self.use_log_map = use_log_map
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_edges = self_edges

    def gen_edges(
        self,
        num_atoms,
        frac_coords,
        lattices,
        node2graph,
        edge_style="knn",
        radius=7.0,
        max_neighbors=None,
    ):
        if edge_style == "fc":
            if self.self_edges:
                lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            else:
                lis = [
                    torch.ones(n, n, device=num_atoms.device)
                    - torch.eye(n, device=num_atoms.device)
                    for n in num_atoms
                ]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            fc_edges = fc_edges.to(frac_coords.device)

            if self.use_log_map:
                # this is the shortest torus distance, but DiffCSP didn't use it
                frac_diff = FlatTorus01.logmap(
                    frac_coords[fc_edges[0]], frac_coords[fc_edges[1]]
                )
            else:
                frac_diff = frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]
            return fc_edges, frac_diff

        elif edge_style == "knn":
            _lattices = lattice_params_to_matrix_torch(lattices[:, :3], lattices[:, 3:])
            lattice_nodes = _lattices[node2graph]
            cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

            # we are dealing with huge graphs, so we need to loop over each graph to reduce memory usage
            all_edges = []
            all_to_jimages = []
            all_num_bonds = []
            for i, lattice in enumerate(_lattices):
                edge_index_i, to_jimages, num_bonds = radius_graph_pbc(
                    cart_coords[node2graph == i],
                    None,
                    None,
                    num_atoms[i].view(-1).to(cart_coords.device),
                    radius,
                    max_neighbors if max_neighbors is not None else self.max_neighbors,
                    device=cart_coords.device,
                    lattices=lattice.view(1, 3, 3),
                )
                all_edges.append(edge_index_i)
                all_to_jimages.append(to_jimages)
                all_num_bonds.append(num_bonds)

            all_edges = [
                edges + num_atoms[:i].sum() for i, edges in enumerate(all_edges)
            ]
            edge_index = torch.cat(all_edges, dim=1)
            to_jimages = torch.cat(all_to_jimages, dim=0)
            num_bonds = torch.cat(all_num_bonds, dim=0)

            if self.use_log_map:
                # this is the shortest torus distance, but DiffCSP didn't use it
                # not sure it makes sense for the cartesian space version
                edge_vector = FlatTorus01.logmap(
                    frac_coords[edge_index[0]], frac_coords[edge_index[1]]
                )
            else:
                distance_vectors = (
                    frac_coords[edge_index[1]] - frac_coords[edge_index[0]]
                )
                distance_vectors += to_jimages.float()

                edge_index, _, _, edge_vector = self.reorder_symmetric_edges(
                    edge_index, to_jimages, num_bonds, distance_vectors
                )
                edge_vector = -edge_vector

            return edge_index, edge_vector

    def gen_edges_new(self, num_atoms, frac_coords, lattices, batch, radius=7.0):
        all_edges = []
        counter = 0
        for i, lattice in enumerate(lattices):
            frac_coords_i = frac_coords[batch == i].cpu()

            # Number of atoms
            num_atoms_i = num_atoms[i]

            # Create a pymatgen Lattice object from the given lattice matrix
            lattice_pmg = Lattice.from_parameters(*lattice.cpu().tolist())

            # Create a pymatgen Structure object
            structure = Structure(
                lattice=lattice_pmg,
                species=["H"] * num_atoms_i,  # set dummy species, not needed
                coords=frac_coords_i,
                coords_are_cartesian=False,
            )

            # Initialize lists for edges and distances
            edges = []

            # Loop over each site in the structure to find neighbors within the cutoff radius
            for i in range(num_atoms_i):
                # Get neighbors within the cutoff radius using get_neighbors method
                neighbors = structure.get_neighbors(structure[i], radius)

                # Sort neighbors by distance to ensure we respect the max_neighbors constraint
                neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[
                    : self.max_neighbors
                ]

                # Collect valid edges and distances
                for neighbor in neighbors:
                    j = neighbor.index  # Index of the neighboring atom
                    edges.append((i, j))

            # Convert lists to numpy arrays
            edges = torch.tensor(edges, dtype=torch.long, device=frac_coords.device).t()
            edges += counter

            # Append the edges to the list of all edges
            all_edges.append(edges)

            # Update the counter
            counter += num_atoms_i

        # Concatenate all edges
        edges = torch.cat(all_edges, dim=1)

        if self.use_log_map:
            # this is the shortest torus distance, but DiffCSP didn't use it
            # not sure it makes sense for the cartesian space version
            frac_diff = FlatTorus01.logmap(frac_coords[edges[0]], frac_coords[edges[1]])
        else:
            frac_diff = frac_coords[edges[1]] - frac_coords[edges[0]]

        return edges, -frac_diff


class DockCSPNet(CSPNet):
    def forward(
        self,
        batch: Batch,
        t,
    ):
        t_emb = self.time_emb(t)
        t_emb = t_emb.expand(
            batch.osda.num_atoms.shape[0], -1
        )  # if there is a single t, repeat for the batch

        be_in = batch.y["bindingatoms"].float() 
        with torch.no_grad(): 
            be_in = torch.where(
                torch.rand_like(be_in) < self.drop_be_prob,
                torch.zeros_like(be_in),
                be_in,
            )
        be_emb = self.be_emb(be_in.view(-1, 1))
        be_emb = be_emb.expand(
            batch.osda.num_atoms.shape[0], -1
        )  
        
        edge_feat_dims = get_feature_dims()[-1]
        dummy_edge_ids = (
            torch.tensor(edge_feat_dims, device=t_emb.device) - 1
        )  # last index is the dummy index

        # create graph
        # for osda
        new_osda_edges, osda_frac_diff = self.gen_edges(
            batch.osda.num_atoms,
            batch.osda.xt,
            batch.lattices,
            batch.osda.batch,
            edge_style=self.edge_style,  # NOTE fc = fully connected
            radius=self.cutoff,
        )
        new_osda_edge_feats = dummy_edge_ids.repeat(new_osda_edges.shape[1], 1)
        new_osda_edge_feats = self.edge_embedding(new_osda_edge_feats)

        osda_internal_edges = batch.osda.edge_index
        osda_internal_edge_feats = self.edge_embedding(batch.osda.edge_feats)

        osda_edges = torch.cat([new_osda_edges, osda_internal_edges], dim=1)
        osda_edge_feats = torch.cat(
            [new_osda_edge_feats, osda_internal_edge_feats], dim=0
        )

        distance_vectors = FlatTorus01.logmap(
            batch.osda.xt[batch.osda.edge_index[0]],
            batch.osda.xt[batch.osda.edge_index[1]],
        )
        osda_frac_diff = torch.cat([osda_frac_diff, distance_vectors], dim=0)

        osda_internal_edge2graph = batch.osda.batch[osda_internal_edges[0]]
        osda_edge2graph = batch.osda.batch[osda_edges[0]]

        # for zeolite
        zeolite_edges, zeolite_frac_diff = self.gen_edges(
            batch.zeolite.num_atoms,
            batch.zeolite.frac_coords,
            batch.lattices,
            batch.zeolite.batch,
            edge_style=self.edge_style,
            radius=2.5,
        )
        zeolite_edge2graph = batch.zeolite.batch[zeolite_edges[0]]

        zeolite_edge_feats = dummy_edge_ids.repeat(zeolite_edges.shape[1], 1)
        zeolite_edge_feats = self.edge_embedding(zeolite_edge_feats)

        # for cross graph
        batch_size = max(batch.osda.batch.max(), batch.zeolite.batch.max()) + 1
        cross_num_atoms = batch.osda.num_atoms + batch.zeolite.num_atoms
        cross_frac_coords = [
            torch.cat(
                [
                    batch.osda.xt[batch.osda.batch == i],
                    batch.zeolite.frac_coords[batch.zeolite.batch == i],
                ],
                dim=0,
            )
            for i in range(batch_size)
        ]
        cross_frac_coords = torch.cat(cross_frac_coords, dim=0)
        cross_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=cross_frac_coords.device), cross_num_atoms
        ).to(cross_frac_coords.device)

        # a mask indicating which nodes in the cross graph are osda nodes
        osda_nodes_mask = [
            torch.cat(
                [
                    torch.ones(
                        (batch.osda.batch == i).sum(),
                        device=batch.osda.xt.device,
                        dtype=torch.bool,
                    ),
                    torch.zeros(
                        (batch.zeolite.batch == i).sum(),
                        device=batch.osda.xt.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )
            for i in range(batch_size)
        ]
        osda_nodes_mask = torch.cat(osda_nodes_mask, dim=0)

        # cross edges
        cross_edges, cross_frac_diff = self.gen_edges(
            cross_num_atoms,
            cross_frac_coords,
            batch.lattices,
            cross_batch,
            edge_style=self.edge_style,
            radius=self.cutoff,
            max_neighbors=20,
        )

        # remove edges that are zeolite-zeolite or osda-osda
        is_cross_edge = osda_nodes_mask[cross_edges[0]] & torch.logical_not(
            osda_nodes_mask[cross_edges[1]]
        )
        cross_edges = cross_edges[:, is_cross_edge]
        cross_frac_diff = cross_frac_diff[is_cross_edge]

        cross_edge2graph = cross_batch[cross_edges[0]]

        cross_edge_feats = dummy_edge_ids.repeat(cross_edges.shape[1], 1)
        cross_edge_feats = self.edge_embedding(cross_edge_feats)

        # neural network
        # embed atom features
        osda_node_features = self.node_embedding(batch.osda.node_feats)
        t_per_atom = t_emb.repeat_interleave(
            batch.osda.num_atoms.to(t_emb.device), dim=0
        )
        be_per_atom = be_emb.repeat_interleave(
            batch.osda.num_atoms.to(t_emb.device), dim=0
        )      
        osda_node_features = torch.cat([osda_node_features, t_per_atom, be_per_atom,], dim=1,)
        osda_node_features = self.atom_latent_emb(osda_node_features)

        zeolite_node_features = self.node_embedding(batch.zeolite.node_feats)
        t_per_atom = t_emb.repeat_interleave(
            batch.zeolite.num_atoms.to(t_emb.device), dim=0
        )
        be_per_atom = be_emb.repeat_interleave(
            batch.zeolite.num_atoms.to(t_emb.device), dim=0
        )
        zeolite_node_features = torch.cat([zeolite_node_features, t_per_atom, be_per_atom], dim=1)
        zeolite_node_features = self.atom_latent_emb(zeolite_node_features)

        for i in range(0, self.num_layers):
            # update osda node feats based on internal bonds first
            osda_node_features = self._modules["csp_layer_%d" % i](
                osda_node_features,
                batch.lattices,
                osda_internal_edges,
                osda_internal_edge2graph,
                distance_vectors,
                osda_internal_edge_feats,
            )

            # update zeolite node feats
            zeolite_node_features = self._modules["csp_layer_%d" % i](
                zeolite_node_features,
                batch.lattices,
                zeolite_edges,
                zeolite_edge2graph,
                zeolite_frac_diff,
                zeolite_edge_feats,
            )

            # initialize cross node feats
            cross_node_features = [
                torch.cat(
                    [
                        osda_node_features[batch.osda.batch == i],
                        zeolite_node_features[batch.zeolite.batch == i],
                    ],
                    dim=0,
                )
                for i in range(batch_size)
            ]
            cross_node_features = torch.cat(cross_node_features, dim=0)

            # update cross node feats
            cross_node_features = self._modules["csp_layer_%d" % i](
                cross_node_features,
                batch.lattices,
                cross_edges,
                cross_edge2graph,
                cross_frac_diff,
                cross_edge_feats,
            )

            # extract osda node feats from cross node feats
            osda_node_features = cross_node_features[osda_nodes_mask]
            zeolite_node_features = cross_node_features[
                torch.logical_not(osda_nodes_mask)
            ]

            # update osda node feats
            osda_node_features = self._modules["csp_layer_%d" % i](
                osda_node_features,
                batch.lattices,
                osda_edges,
                osda_edge2graph,
                osda_frac_diff,
                osda_edge_feats,
            )

        if self.ln:
            osda_node_features = self.final_layer_norm(osda_node_features)

        # predict coords
        coord_out = self.coord_out(osda_node_features)

        # predict binding energy TODO mrx add more options
        osda_node_features = scatter(
            osda_node_features,
            batch.osda.batch,
            dim=0,
            reduce="mean",
            )
        be_out = self.be_out(osda_node_features)

        return coord_out, be_out


class OptimizeCSPNet(CSPNet):
    def forward(
        self,
        batch: Batch,
        t,
    ):
        t_emb = self.time_emb(t)
        t_emb = t_emb.expand(
            batch.num_atoms.shape[0], -1
        )  # if there is a single t, repeat for the batch

        edge_feat_dims = get_feature_dims()[-1]
        dummy_edge_ids = (
            torch.tensor(edge_feat_dims, device=t_emb.device) - 1
        )  # last index is the dummy index

        # create graph
        edges, frac_diff = self.gen_edges(
            batch.num_atoms,
            batch.frac_coords,
            batch.lattices,
            batch.batch,
            edge_style="knn",  # NOTE fc = fully connected
            radius=self.cutoff,
        )
        dummy_edge_feats = dummy_edge_ids.repeat(edges.shape[1], 1)

        edges = torch.cat([edges, batch.edge_index], dim=1)
        edge_feats = torch.cat([dummy_edge_feats, batch.edge_feats], dim=0)
        edge_feats = self.edge_embedding(edge_feats)

        distance_vectors = FlatTorus01.logmap(
            batch.frac_coords[batch.edge_index[0]],
            batch.frac_coords[batch.edge_index[1]],
        )
        frac_diff = torch.cat([frac_diff, distance_vectors], dim=0)

        edge2graph = batch.batch[edges[0]]

        # neural network
        # embed atom features
        node_features = self.node_embedding(batch.node_feats)
        t_per_atom = t_emb.repeat_interleave(batch.num_atoms.to(t_emb.device), dim=0)

        node_features = torch.cat(
            [
                node_features,
                t_per_atom,
            ],
            dim=1,
        )
        node_features = self.atom_latent_emb(node_features)
        node_features = self.act_fn(node_features)

        # do docking first
        for i in range(0, self.num_layers):
            # update osda node feats
            node_features = self._modules["csp_layer_%d" % i](
                node_features, batch.lattices, edges, edge2graph, frac_diff, edge_feats
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # predict coords
        coord_out = self.coord_out(node_features)

        return coord_out


class ProjectedConjugatedCSPNet(nn.Module):
    def __init__(
        self,
        cspnet: CSPNet,
        manifold_getter: ManifoldGetter,
        coord_affine_stats: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.cspnet = cspnet
        self.manifold_getter = manifold_getter
        self.self_cond = cspnet.self_cond
        self.metric_normalized = False

        if coord_affine_stats is not None:
            self.register_buffer(
                "coord_u_t_mean", coord_affine_stats["u_t_mean"].unsqueeze(0)
            )
            self.register_buffer(
                "coord_u_t_std", coord_affine_stats["u_t_std"].unsqueeze(0)
            )

    def _conjugated_forward(
        self,
        batch: Batch,
        t: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> ManifoldGetterOut:
        raise NotImplementedError

    def forward(
        self,
        batch: Batch,
        t: torch.Tensor,
        x: torch.Tensor,
        manifold: Manifold,
        cond_coords: torch.Tensor | None = None,
        cond_be: torch.Tensor | None = None,
        guidance_strength = 0.0, 
    ) -> torch.Tensor:
        """u_t: [0, 1] x M -> T M

        representations are mapped as follows:
        `flat -> flat_manifold -> pytorch_geom -(nn)-> pytorch_geom -> flat_tangent_estimate -> flat_tangent`
        """
        x = manifold.projx(x)
        if cond_coords is not None:
            cond_coords = manifold.projx(cond_coords)
        (v, *_), be = self._conjugated_forward(
            batch,
            t,
            x,
            cond_coords,
            cond_be
        )

        if guidance_strength == 0.0: 
            for prop in batch.y.keys():
                batch.y[prop] = torch.zeros_like(batch.y[prop]).to(x.device)
            (guid_v, *guid_), guid_be = self._conjugated_forward(
                batch,
                t,
                x,
                cond_coords,
                cond_be
            )
            v = v + guidance_strength * guid_v
            guid_strength_tot = guidance_strength + 1 
            be = be / guid_strength_tot + guid_be * guidance_strength / guid_strength_tot # TODO mrx add options


        # NOTE comment out to predict position directly
        v = manifold.proju(x, v)

        if self.metric_normalized and hasattr(manifold, "metric_normalized"):
            v = manifold.metric_normalized(x, v)
        return v, be


class DockProjectedConjugatedCSPNet(ProjectedConjugatedCSPNet):
    def _conjugated_forward(
        self,
        batch: Batch,
        t: torch.Tensor,
        x: torch.Tensor,
        cond_coords: torch.Tensor | None,
        cond_be: torch.Tensor | None,
    ) -> ManifoldGetterOut:
        # handle osda first
        osda_frac_coords = self.manifold_getter.flatrep_to_georep(
            x,
            dims=batch.osda.dims,
            mask_f=batch.osda.mask_f,
        )
        batch.osda.xt = osda_frac_coords.f

        if self.self_cond:
            if cond_coords is not None:
                fc_cond = self.manifold_getter.flatrep_to_georep(
                    cond_coords,
                    dims=batch.osda.dims,
                    mask_f=batch.osda.mask_f,
                )
                fc_cond = fc_cond.f

            else:
                fc_cond = torch.zeros_like(batch.osda.xt)
            
            if cond_be is None: 
                cond_be = torch.zeros_like(batch.y['bindingatoms'])

            batch.osda.xt = torch.cat([batch.osda.xt, fc_cond], dim=-1)
            batch.y['bindingatoms'] = torch.cat(
                [batch.y['bindingatoms'], cond_be], dim=-1
            )

        coord_out, be_out = self.cspnet(
            batch,
            t,
        )

        return self.manifold_getter.georep_to_flatrep(
            batch=batch.osda.batch,
            frac_coords=coord_out,
            split_manifold=False,
        ), be_out


class OptimizeProjectedConjugatedCSPNet(ProjectedConjugatedCSPNet):
    def _conjugated_forward(
        self,
        batch: Batch,
        t: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> ManifoldGetterOut:
        frac_coords = self.manifold_getter.flatrep_to_georep(
            x,
            dims=batch.dims,
            mask_f=batch.mask_f,
        )
        batch.xt = frac_coords.f

        if self.self_cond:
            if cond is not None:
                fc_cond = self.manifold_getter.flatrep_to_georep(
                    cond,
                    dims=batch.dims,
                    mask_f=batch.mask_f,
                )
                fc_cond = fc_cond.f

            else:
                fc_cond = torch.zeros_like(frac_coords)

            batch.xt = torch.cat([batch.xt, fc_cond], dim=-1)

        coord_out = self.cspnet(
            batch,
            t,
        )

        return self.manifold_getter.georep_to_flatrep(
            batch=batch.batch,
            frac_coords=coord_out,
            split_manifold=False,
        )
