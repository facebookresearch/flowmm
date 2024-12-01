import torch
from torch_geometric.data import Batch
import numpy as np
from diffcsp.common.data_utils import (
    cart_to_frac_coords,
)
from rfm_docking.utils import smiles_to_pos


def sample_mol_in_frac(
    smiles,
    lattice_lengths,
    lattice_angles,
    loading,
    forcefield="mmff",
    regularized=True,
    device="cpu",
):
    """Sample a molecule in a fractional space."""

    all_pos = []
    num_atoms = []
    for smi, load in zip(smiles, loading):
        # Convert SMILES to 3D coordinates
        pos = smiles_to_pos(smi, forcefield, device)

        all_pos.append(pos.repeat(load, 1))
        num_atoms.append(pos.shape[0] * load)

    all_pos = torch.cat(all_pos, dim=0).reshape(-1, 3).to(device)
    num_atoms = torch.tensor(num_atoms).to(device)

    # Convert cartesian coordinates to fractional coordinates
    frac = cart_to_frac_coords(
        all_pos,
        lattice_lengths,
        lattice_angles,
        num_atoms=num_atoms,
        regularized=regularized,
    )

    return frac


def sample_uniform(osda: Batch, loading: torch.Tensor) -> torch.Tensor:
    """Sample from a uniform distribution and then apply a Gaussian noise."""
    num_mols = loading.sum()
    atoms_per_mol = osda.num_atoms // loading

    # assignment for each molecule
    mol_ids = torch.arange(num_mols, device=osda.frac_coords.device)
    atom_ids = torch.repeat_interleave(atoms_per_mol, loading)

    ids = torch.repeat_interleave(mol_ids, atom_ids)

    uniform = torch.rand((num_mols, 3))

    prior = uniform[ids]

    return prior


def sample_uniform_then_gaussian(
    osda: Batch, loading: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Sample from a uniform distribution and then apply a Gaussian noise."""
    uniform = sample_uniform(osda, loading)
    gaussian = torch.randn_like(osda.frac_coords)

    prior = gaussian * sigma + uniform

    return prior


def sample_uniform_then_conformer(
    osda: Batch, smiles: list[str], loading: torch.Tensor
) -> torch.Tensor:
    """Sample from a uniform distribution and then set the molecular conformer."""
    uniform = sample_uniform(osda, loading)
    conformers = sample_mol_in_frac(
        smiles, osda.lengths, osda.angles, loading, device=osda.frac_coords.device
    )

    prior = conformers + uniform

    return prior


def sample_harmonic_prior(batch: Batch, sigma: float) -> torch.Tensor:
    """Modified from FlowSite by Stark. Sample from a harmonic prior."""
    bid = batch.batch
    sigma = torch.zeros(len(batch.num_atoms)) + sigma
    lamb = 1 / sigma**2

    edges = batch.edge_index
    edges = edges[:, edges[0] < edges[1]]  # de-duplicate
    try:
        D, P = HarmonicSDE.diagonalize(
            batch.num_nodes,
            edges=edges.T,
            lamb=lamb[bid],
            ptr=None,  # batch.target_osda.ptr,
        )
    except Exception as e:
        print("batch.num_nodes", batch.num_nodes)
        print("batch.size", batch.size)
        raise e
    noise = torch.randn_like(batch.frac_coords)
    prior = P @ (noise / torch.sqrt(D)[:, None])

    return prior


class HarmonicSDE:
    """Taken from FlowSite by Stark."""

    def __init__(
        self, N=None, edges=[], antiedges=[], a=0.5, b=0.3, J=None, diagonalize=True
    ):
        self.use_cuda = False
        self.l = 1
        if not diagonalize:
            return
        if J is not None:
            J = J
            self.D, P = np.linalg.eigh(J)
            self.P = P
            self.N = self.D.size
            return

    @staticmethod
    def diagonalize(N, edges=[], antiedges=[], a=1, b=0.3, lamb=0.0, ptr=None):
        J = torch.zeros((N, N), device=edges.device)  # temporary fix
        for i, j in edges:
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        for i, j in antiedges:
            J[i, i] -= b
            J[j, j] -= b
            J[i, j] = J[j, i] = b
        J += torch.diag(lamb)
        if ptr is None:
            return torch.linalg.eigh(J)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:]):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            Ds.append(D)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)

    def eigens(self, t):  # eigenvalues of sigma_t
        np_ = torch if self.use_cuda else np
        D = 1 / self.D * (1 - np_.exp(-t * self.D))
        t = torch.tensor(t, device="cuda").float() if self.use_cuda else t
        return np_.where(D != 0, D, t)

    def conditional(self, mask, x2):
        J_11 = self.J[~mask][:, ~mask]
        J_12 = self.J[~mask][:, mask]
        h = -J_12 @ x2
        mu = np.linalg.inv(J_11) @ h
        D, P = np.linalg.eigh(J_11)
        z = np.random.randn(*mu.shape)
        return (P / D**0.5) @ z + mu

    def A(self, t, invT=False):
        D = self.eigens(t)
        A = self.P * (D**0.5)
        if not invT:
            return A
        AinvT = self.P / (D**0.5)
        return A, AinvT

    def Sigma_inv(self, t):
        D = 1 / self.eigens(t)
        return (self.P * D) @ self.P.T

    def Sigma(self, t):
        D = self.eigens(t)
        return (self.P * D) @ self.P.T

    @property
    def J(self):
        return (self.P * self.D) @ self.P.T

    def rmsd(self, t):
        l = self.l
        D = 1 / self.D * (1 - np.exp(-t * self.D))
        return np.sqrt(3 * D[l:].mean())

    def sample(self, t, x=None, score=False, k=None, center=True, adj=False):
        l = self.l
        np_ = torch if self.use_cuda else np
        if x is None:
            if self.use_cuda:
                x = torch.zeros((self.N, 3), device="cuda").float()
            else:
                x = np.zeros((self.N, 3))
        if t == 0:
            return x
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        D = self.eigens(t)
        xx = self.P.T @ x
        if center:
            z[0] = 0
            xx[0] = 0
        if k:
            z[k + l :] = 0
            xx[k + l :] = 0

        out = np_.exp(-t * self.D / 2)[:, None] * xx + np_.sqrt(D)[:, None] * z

        if score:
            score = -(1 / np_.sqrt(D))[:, None] * z
            if adj:
                score = score + self.D[:, None] * out
            return self.P @ out, self.P @ score
        return self.P @ out

    def score_norm(self, t, k=None, adj=False):
        if k == 0:
            return 0
        l = self.l
        np_ = torch if self.use_cuda else np
        k = k or self.N - 1
        D = 1 / self.eigens(t)
        if adj:
            D = D * np_.exp(-self.D * t)
        return (D[l : k + l].sum() / self.N) ** 0.5

    def inject(self, t, modes):
        # Returns noise along the given modes
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        z[~modes] = 0
        A = self.A(t, invT=False)
        return A @ z

    def score(self, x0, xt, t):
        # Score of the diffusion kernel
        Sigma_inv = self.Sigma_inv(t)
        mu_t = (self.P * np.exp(-t * self.D / 2)) @ (self.P.T @ x0)
        return Sigma_inv @ (mu_t - xt)

    def project(self, X, k, center=False):
        l = self.l
        # Projects onto the first k nonzero modes (and optionally centers)
        D = self.P.T @ X
        D[k + l :] = 0
        if center:
            D[0] = 0
        return self.P @ D

    def unproject(self, X, mask, k, return_Pinv=False):
        # Finds the vector along the first k nonzero modes whose mask is closest to X
        l = self.l
        PP = self.P[mask, : k + l]
        Pinv = np.linalg.pinv(PP)
        out = self.P[:, : k + l] @ Pinv @ X
        if return_Pinv:
            return out, Pinv
        return out

    def energy(self, X):
        l = self.l
        return (self.D[:, None] * (self.P.T @ X) ** 2).sum(-1)[l:] / 2

    @property
    def free_energy(self):
        l = self.l
        return 3 * np.log(self.D[l:]).sum() / 2

    def KL_H(self, t):
        l = self.l
        D = self.D[l:]
        return -3 * 0.5 * (np.log(1 - np.exp(-D * t)) + np.exp(-D * t)).sum(0)
