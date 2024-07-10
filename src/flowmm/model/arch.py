"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math

import torch
from geoopt import Manifold
from torch import nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter

from diffcsp.common.data_utils import lattice_params_to_matrix_torch, radius_graph_pbc
from diffcsp.pl_modules.cspnet import CSPLayer as DiffCSPLayer
from diffcsp.pl_modules.cspnet import CSPNet as DiffCSPNet
from diffcsp.pl_modules.cspnet import SinusoidsEmbedding
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from flowmm.rfm.manifold_getter import (
    Dims,
    ManifoldGetter,
    ManifoldGetterOut,
    lattice_manifold_types,
)
from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from flowmm.rfm.manifolds.lattice_params import LatticeParams
from flowmm.rfm.manifolds.spd import SPDGivenN, spd_vector_to_lattice_matrix


class CSPLayer(DiffCSPLayer):
    """Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim,
        act_fn,
        dis_emb,
        ln,
        lattice_manifold: lattice_manifold_types,
        n_space: int = 3,
        represent_num_atoms: bool = False,
        represent_angle_edge_to_lattice: bool = False,
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

        self.lattice_manifold = lattice_manifold
        if "spd" in lattice_manifold:
            self.dim_l = SPDGivenN.vecdim(self.n_space)
        elif (
            lattice_manifold == "lattice_params"
            or lattice_manifold == "lattice_params_normal_base"
        ):
            self.dim_l = LatticeParams.dim(self.n_space)
        elif lattice_manifold == "non_symmetric":
            self.dim_l = self.n_space**2
        else:
            raise ValueError()
        if self_cond:
            self.dim_l *= 2

        self.represent_num_atoms = represent_num_atoms
        if represent_num_atoms:
            self.one_hot_dim = 100  # largest cell of atoms that we'd represent, this is safe for a HACK
            self.num_atom_embedding = nn.Linear(
                self.one_hot_dim, hidden_dim, bias=False
            )
            num_hidden_dim_vecs = 3
        else:
            num_hidden_dim_vecs = 2

        self.represent_angle_edge_to_lattice = represent_angle_edge_to_lattice
        angle_edge_dims = n_space if represent_angle_edge_to_lattice else 0
        if self_cond:
            angle_edge_dims *= 2

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                angle_edge_dims
                + hidden_dim * num_hidden_dim_vecs
                + self.dim_l
                + self.dis_dim,
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

    def _create_unit_dots_ltlf(
        self,
        non_zscored_lattice: torch.Tensor,
        edge2graph: torch.LongTensor,
        frac_diff: torch.Tensor,
    ) -> torch.Tensor:
        ltl = non_zscored_lattice @ non_zscored_lattice.transpose(-1, -2)
        dots = torch.einsum("...ij,...j->...i", ltl[edge2graph], frac_diff)
        unit_dots = dots / dots.norm(dim=-1).unsqueeze(-1)
        return unit_dots

    def edge_model(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        num_atoms,
        non_zscored_lattice: torch.Tensor | None,
        non_zscored_lattice_pred: torch.Tensor | None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        edge_features = []
        if self.represent_angle_edge_to_lattice:
            # recall: lattice is an array of 3 _row_ vectors
            # theta = cos^{1}( a / ||a|| . b / ||b||)
            # a = lattice_1, b = cart_diff, cart_diff = L @ f_diff
            # in the end we get L^T @ L @ f_diff are the three values of the dot product
            if self.self_cond:
                _frac_diff, _frac_diff_pred = torch.tensor_split(frac_diff, 2, dim=-1)
                unit_dots = self._create_unit_dots_ltlf(
                    non_zscored_lattice, edge2graph, _frac_diff
                )
                if non_zscored_lattice_pred is None:
                    unit_dots_pred = torch.zeros_like(unit_dots)
                else:
                    unit_dots_pred = self._create_unit_dots_ltlf(
                        non_zscored_lattice_pred, edge2graph, _frac_diff_pred
                    )
                edge_features.append(torch.cat([unit_dots, unit_dots_pred], dim=-1))
            else:
                unit_dots = self._create_unit_dots_ltlf(
                    non_zscored_lattice, edge2graph, frac_diff
                )
                edge_features.append(unit_dots)

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

        if "spd" in self.lattice_manifold:
            lattices = lattices
        elif (
            self.lattice_manifold == "lattice_params"
            or self.lattice_manifold == "lattice_params_normal_base"
        ):
            lattices = lattices
        elif self.lattice_manifold == "non_symmetric":
            if self.self_cond:
                raise NotImplementedError(
                    "this doesnt work in CSPNet.forward in this file line 389, cannot do self_cond with non_symmetric"
                )
                initial_shape = lattices.shape
                lattices = lattices.reshape(
                    initial_shape[:-2], 2, self.n_space, self.n_space
                )
                lattices = lattices @ lattices.transpose(-1, -2)
                lattices = lattices.reshape(*initial_shape)
            else:
                lattices = lattices @ lattices.transpose(-1, -2)
        else:
            raise ValueError()

        lattices_flat = lattices.view(-1, self.dim_l)
        lattices_flat_edges = lattices_flat[edge2graph]

        edge_features.extend([hi, hj, lattices_flat_edges, frac_diff])
        if self.represent_num_atoms:
            one_hot = torch.nn.functional.one_hot(
                num_atoms, num_classes=self.one_hot_dim
            ).to(dtype=hi.dtype)
            num_atoms_rep = self.num_atom_embedding(one_hot)[edge2graph]
            edge_features.append(num_atoms_rep)
        return self.edge_mlp(torch.cat(edge_features, dim=1))

    def forward(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        num_atoms: torch.LongTensor,
        non_zscored_lattice: torch.Tensor | None,
        non_zscored_lattice_pred: torch.Tensor | None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features,
            lattices,
            edge_index,
            edge2graph,
            frac_diff,
            num_atoms,
            non_zscored_lattice,
            non_zscored_lattice_pred,
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
        lattice_manifold: lattice_manifold_types = "non_symmetric",
        concat_sum_pool: bool = False,
        represent_num_atoms: bool = False,
        represent_angle_edge_to_lattice: bool = False,
        self_edges: bool = True,
        self_cond: bool = False,
    ):
        nn.Module.__init__(self)
        assert not (
            self_cond and lattice_manifold == "non_symmetric"
        ), "network cannot handle self_cond with non_symmetric."

        self.lattice_manifold = lattice_manifold
        self.n_space = n_space
        self.time_emb = nn.Linear(1, time_dim, bias=False)

        self.self_cond = self_cond
        if self_cond:
            coef = 2
        else:
            coef = 1

        self.node_embedding = nn.Linear(
            dim_atomic_rep * coef,
            hidden_dim,
            bias=False,  # diffcsp's version has a bias in the embedding
        )
        self.atom_latent_emb = nn.Linear(hidden_dim + time_dim, hidden_dim, bias=False)
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
                    lattice_manifold=self.lattice_manifold,
                    n_space=n_space,
                    represent_num_atoms=represent_num_atoms,
                    represent_angle_edge_to_lattice=represent_angle_edge_to_lattice,
                    self_cond=self_cond,
                ),
            )
        self.num_layers = num_layers
        self.concat_sum_pool = concat_sum_pool
        if concat_sum_pool:
            num_pools = 2
        else:
            num_pools = 1
        # it makes sense to have no bias here since p(F) is translation invariant
        self.coord_out = nn.Linear(hidden_dim, n_space, bias=False)
        if (
            ("spd" in lattice_manifold)
            or lattice_manifold == "lattice_params"
            or lattice_manifold == "lattice_params_normal_base"
        ):
            self.lattice_out = nn.Linear(
                num_pools * hidden_dim, SPDGivenN.vecdim(n_space)
            )
        elif lattice_manifold == "non_symmetric":
            # diffcsp doesn't have a bias on lattice outputs
            self.lattice_out = nn.Linear(
                num_pools * hidden_dim, n_space**2, bias=False
            )
        else:
            raise ValueError()

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        self.use_log_map = use_log_map
        self.type_out = nn.Linear(hidden_dim, dim_atomic_rep)
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.represent_angle_edge_to_lattice = represent_angle_edge_to_lattice
        self.self_edges = self_edges

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
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
            if self.use_log_map:
                # this is the shortest torus distance, but DiffCSP didn't use it
                frac_diff = FlatTorus01.logmap(
                    frac_coords[fc_edges[0]], frac_coords[fc_edges[1]]
                )
            else:
                frac_diff = frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]
            return fc_edges, frac_diff
        elif self.edge_style == "knn":
            _lattices = self._convert_lin_to_lattice(lattices)
            lattice_nodes = _lattices[node2graph]
            cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords,
                None,
                None,
                num_atoms,
                self.cutoff,
                self.max_neighbors,
                device=num_atoms.device,
                lattices=_lattices,
            )

            j_index, i_index = edge_index
            if self.use_log_map:
                # this is the shortest torus distance, but DiffCSP didn't use it
                # not sure it makes sense for the cartesian space version
                raise NotImplementedError()
            else:
                distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )

            return edge_index_new, -edge_vector_new

    def _convert_lin_to_lattice(self, lattice: torch.Tensor) -> torch.Tensor:
        if "spd" in self.lattice_manifold:
            l = spd_vector_to_lattice_matrix(lattice)
        elif self.lattice_manifold == "lattice_params":
            l_deg = LatticeParams().uncontrained2deg(lattice)
            lengths, angles_deg = LatticeParams.split(l_deg)
            l = lattice_params_to_matrix_torch(lengths, angles_deg)
        elif self.lattice_manifold == "lattice_params_normal_base":
            lengths, angles = LatticeParams.split(lattice)
            l = lattice_params_to_matrix_torch(lengths, angles)
        elif self.lattice_manifold == "non_symmetric":
            l = lattice
        else:
            raise ValueError()
        return l

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        non_zscored_lattice,
    ):
        t_emb = self.time_emb(t)
        t_emb = t_emb.expand(
            num_atoms.shape[0], -1
        )  # if there is a single t, repeat for the batch

        # create graph
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        # maybe identify angles between cartesian vecs and lattice
        if self.represent_angle_edge_to_lattice:
            if self.self_cond:
                # TODO this fails for the simplex where the lattice is already [..., n_space, n_space]
                nzsl, nzsl_pred = torch.tensor_split(non_zscored_lattice, 2, dim=-1)
                l = self._convert_lin_to_lattice(nzsl)
                if (torch.zeros_like(nzsl_pred) == nzsl_pred).all():
                    l_pred = None
                else:
                    l_pred = self._convert_lin_to_lattice(nzsl_pred)
            else:
                l = self._convert_lin_to_lattice(non_zscored_lattice)
                l_pred = None
        else:
            l = None
            l_pred = None

        # neural network
        node_features = self.node_embedding(atom_types)
        t_per_atom = t_emb.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)
        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features,
                lattices,
                edges,
                edge2graph,
                frac_diff,
                num_atoms,
                non_zscored_lattice=l,
                non_zscored_lattice_pred=l_pred,
            )
        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # predict coords
        coord_out = self.coord_out(node_features)

        # predict lattice
        if self.concat_sum_pool:
            graph_features = torch.concat(
                [
                    scatter(node_features, node2graph, dim=0, reduce="mean"),
                    scatter(node_features, node2graph, dim=0, reduce="sum"),
                ],
                dim=-1,
            )
        else:
            graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        lattice_out = self.lattice_out(graph_features)
        if self.lattice_manifold == "non_symmetric":
            lattice_out = lattice_out.view(-1, self.n_space, self.n_space)
            lattice_out = torch.einsum(
                "bij,bjk->bik", lattice_out, lattices
            )  # recall: lattices from pymatgen are 3 _row_ vectors!

        # predict types
        type_out = self.type_out(node_features)

        return lattice_out, coord_out, type_out


class ProjectedConjugatedCSPNet(nn.Module):
    def __init__(
        self,
        cspnet: CSPNet,
        manifold_getter: ManifoldGetter,
        lattice_affine_stats: dict[str, torch.Tensor] | None = None,
        coord_affine_stats: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.cspnet = cspnet
        self.manifold_getter = manifold_getter
        self.metric_normalized = manifold_getter.lattice_manifold == "spd_riemanian_geo"
        self.self_cond = cspnet.self_cond

        if lattice_affine_stats is not None:
            self.register_buffer(
                "lat_x_t_mean", lattice_affine_stats["x_t_mean"].unsqueeze(0)
            )
            self.register_buffer(
                "lat_x_t_std", lattice_affine_stats["x_t_std"].unsqueeze(0)
            )
            self.register_buffer(
                "lat_u_t_mean", lattice_affine_stats["u_t_mean"].unsqueeze(0)
            )
            self.register_buffer(
                "lat_u_t_std", lattice_affine_stats["u_t_std"].unsqueeze(0)
            )
            if manifold_getter.lattice_manifold == "non_symmetric":
                dim = int(math.sqrt(self.lat_x_t_mean.shape[-1]))
                self.lat_x_t_mean = self.lat_x_t_mean.reshape(1, dim, dim)
                self.lat_x_t_std = self.lat_x_t_std.reshape(1, dim, dim)
                self.lat_u_t_mean = self.lat_u_t_mean.reshape(1, dim, dim)
                self.lat_u_t_std = self.lat_u_t_std.reshape(1, dim, dim)

        if coord_affine_stats is not None:
            self.register_buffer(
                "coord_u_t_mean", coord_affine_stats["u_t_mean"].unsqueeze(0)
            )
            self.register_buffer(
                "coord_u_t_std", coord_affine_stats["u_t_std"].unsqueeze(0)
            )

    def _conjugated_forward(
        self,
        num_atoms: torch.LongTensor,
        node2graph: torch.LongTensor,  # known in DiffCSP as batch
        dims: Dims,
        mask_a_or_f: torch.BoolTensor,
        t: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> ManifoldGetterOut:
        atom_types, frac_coords, lattices = self.manifold_getter.flatrep_to_georep(
            x,
            dims=dims,
            mask_a_or_f=mask_a_or_f,
        )

        non_zscored_lattice = (
            lattices.clone() if self.cspnet.represent_angle_edge_to_lattice else None
        )

        # z-score inputs
        if hasattr(self, "lat_x_t_mean"):
            lattices = (lattices - self.lat_x_t_mean) / self.lat_x_t_std

        if self.self_cond:
            if cond is not None:
                at_cond, fc_cond, l_cond = self.manifold_getter.flatrep_to_georep(
                    cond,
                    dims=dims,
                    mask_a_or_f=mask_a_or_f,
                )
                non_zscored_l_cond = (
                    l_cond.clone()
                    if self.cspnet.represent_angle_edge_to_lattice
                    else None
                )
                if hasattr(self, "lat_x_t_mean"):
                    l_cond = (l_cond - self.lat_x_t_mean) / self.lat_x_t_std
            else:
                at_cond = torch.zeros_like(atom_types)
                fc_cond = torch.zeros_like(frac_coords)
                l_cond = torch.zeros_like(lattices)
                non_zscored_l_cond = torch.zeros_like(lattices)
            atom_types = torch.cat([atom_types, at_cond], dim=-1)
            frac_coords = torch.cat([frac_coords, fc_cond], dim=-1)
            lattices = torch.cat([lattices, l_cond], dim=-1)
            if self.cspnet.represent_angle_edge_to_lattice:
                non_zscored_lattice = torch.cat(
                    [non_zscored_lattice, non_zscored_l_cond], dim=-1
                )

        lattice_out, coord_out, types_out = self.cspnet(
            t,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            non_zscored_lattice,
        )

        # z-score outputs
        if hasattr(self, "lat_u_t_std"):
            lattice_out = lattice_out * self.lat_u_t_std + self.lat_u_t_mean
        if hasattr(self, "coord_u_t_std"):
            coord_out = coord_out * self.coord_u_t_std + self.coord_u_t_mean

        if not self.manifold_getter.predict_atom_types:
            if self.cspnet.self_cond:
                # remove the extra zeros we appended above
                types_out, _ = torch.tensor_split(atom_types, 2, dim=-1)
            else:
                types_out = atom_types
        return self.manifold_getter.georep_to_flatrep(
            batch=node2graph,
            atom_types=types_out,
            frac_coords=coord_out,
            lattices=lattice_out,
            split_manifold=False,
        )

    def forward(
        self,
        num_atoms: torch.LongTensor,
        node2graph: torch.LongTensor,
        dims: Dims,
        mask_a_or_f: torch.BoolTensor,
        t: torch.Tensor,
        x: torch.Tensor,
        manifold: Manifold,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """u_t: [0, 1] x M -> T M

        representations are mapped as follows:
        `flat -> flat_manifold -> pytorch_geom -(nn)-> pytorch_geom -> flat_tangent_estimate -> flat_tangent`
        """
        x = manifold.projx(x)
        if cond is not None:
            cond = manifold.projx(cond)
        v, *_ = self._conjugated_forward(
            num_atoms, node2graph, dims, mask_a_or_f, t, x, cond
        )
        v = manifold.proju(x, v)

        if self.metric_normalized and hasattr(manifold, "metric_normalized"):
            v = manifold.metric_normalized(x, v)
        return v
