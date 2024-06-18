"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.func import jacrev, vmap
from typing_extensions import Self

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from manifm.manifolds.spd import SPD as manifm_SPD
from manifm.manifolds.spd import matrix_logarithm, normal_logprob, sqrtmh
from flowmm.cfg_utils import dataset_options, init_loaders


class AbstractSPD:
    @property
    def dtype(self) -> None:
        """the manifold requires bool tensors, this disrupts introspection default"""
        return None

    @property
    def device(self) -> None:
        """gets mapped to the right device as necessary"""
        return None

    @staticmethod
    def vecdim(n):
        return n * (n + 1) // 2

    @staticmethod
    def matdim(d):
        return int((math.sqrt(8 * d + 1) - 1) / 2)

    @staticmethod
    def vectorize(A: torch.Tensor) -> torch.Tensor:
        """Vectorizes a symmetric matrix to a n(n+1)/2 vector."""
        n = A.shape[-1]
        mask = torch.triu(torch.ones(n, n)) == 1
        mask = mask.broadcast_to(A.shape).to(A.device)
        vec = A[mask].reshape(*A.shape[:-2], -1)
        return vec

    @classmethod
    def devectorize(cls, x):
        size = x.shape
        d = x.shape[-1]
        n = cls.matdim(d)
        x = x.reshape(-1, d)

        def create_symm(x):
            A = torch.zeros(n, n).to(x)
            triu_indices = torch.triu_indices(row=n, col=n, offset=0).to(A.device)
            A = torch.index_put(A, (triu_indices[0], triu_indices[1]), x.reshape(-1))
            A = torch.index_put(
                A.mT, (triu_indices[0], triu_indices[1]), x.reshape(-1)
            ).mT
            return A

        A = vmap(create_symm)(x)
        A = A.reshape(*size[:-1], n, n)
        return A

    @classmethod
    def assert_spd(cls, x: torch.Tensor) -> torch.Tensor:
        x = cls.devectorize(x)
        eigvals = torch.linalg.eigvals(x)
        assert (
            eigvals.imag.max() <= 1e-4
        ), f"{eigvals=} had an imaginary component greater than 1e-4"
        if eigvals.real.min() <= 0:
            raise ValueError(f"Matrix not SPD. Smallest eigval is {eigvals.real.min()}")

    # ours takes the same shape as the other _cond_u geodesics (as opposed to the default which squeezes last t dim)
    def geodesic(self, x, y, t):
        """Computes the Riemannian geodesic A exp(t log(A^{-1}B)).
        x: (..., D)
        y: (..., D)
        t: (..., 1)
        """
        if self.Riem_geodesic:
            A, B = self.devectorize(x), self.devectorize(y)
            dtype = A.dtype
            A, B = A.double(), B.double()
            Ainv_B = torch.linalg.solve(A, B)
            U = t.unsqueeze(-1) * matrix_logarithm(Ainv_B)
            G_t = torch.matmul(A, torch.matrix_exp(U))
            return self.vectorize(G_t).to(dtype)
        else:
            return x + t * (y - x)

    @classmethod
    def closest_isotropic(cls, spd: torch.Tensor, Riem_dist: bool) -> torch.Tensor:
        iso = torch.eye(cls.matdim(spd.shape[-1]))
        spd = cls.devectorize(spd)
        if Riem_dist:
            factor = torch.linalg.det(spd).pow(1 / 3)
        else:
            fn = torch.trace if spd.ndim == 2 else vmap(torch.trace)(spd)
            factor = fn(spd) / 3.0
        return cls.vectorize(torch.atleast_1d(factor)[:, None, None] * iso[None, ...])


class SPDGivenN(AbstractSPD, manifm_SPD):
    name = "SPDGivenN"

    def __init__(
        self,
        n: int,
        atom_density: float,
        std_coef: float,
        base_expmap=True,
        Riem_geodesic=True,
        Riem_norm=True,
    ):
        """`mean = (n * (1/atom_density)) ** (2/3) * torch.eye(dim)`
        `logmap_std = (n * std_coef) ** (2/3) * torch.eye(dim)`

        note: logmap_std is the std of the data evaluated at the mean:
            i.e. `logmap(mean, data).std()`
        """
        super().__init__(None, None, base_expmap, Riem_geodesic, Riem_norm)
        del self.scale_std
        del self.scale_Id
        self.register_buffer(
            "scale_Id", torch.Tensor(n * atom_density ** (-1)).pow(2 / 3)
        )
        self.register_buffer("scale_std", torch.Tensor(n * std_coef).pow(2 / 3))

    @classmethod
    def from_dataset(cls, n: int, dataset: dataset_options, **kwargs) -> Self:
        atom_density, _ = get_atom_density(dataset)
        std_coef, _ = get_spd_pLTL_given_n_std_coef(dataset)
        return cls(n=n, atom_density=atom_density, std_coef=std_coef, **kwargs)

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        bsz = int(math.prod(size[:-1]))
        d = size[-1]
        n = self.matdim(d)

        # Wrap a Gaussian centered at the identity matrix.
        Id = torch.eye(n, dtype=dtype, device=device)
        Id = Id * self.scale_Id.to(Id)
        c = self.vectorize(Id).reshape(1, -1).expand(bsz, d)

        # Construct symmetric matrix where elements are iid Normal.
        u = torch.randn(bsz, d, dtype=dtype, device=device)
        u = u * self.scale_std.to(u)

        if self.base_expmap:
            # Exponential map to the manifold.
            x = self.expmap(c, u)
        else:
            # Beware this can sample a non-SPD matrix unless scale is small enough.
            x = c + u
        return x.reshape(*size)

    random = random_base

    def base_logprob(self, x):
        size = x.shape
        d = x.shape[-1]
        n = self.matdim(d)
        x = x.reshape(-1, d)
        Id = torch.eye(n, dtype=x.dtype, device=x.device) * self.scale_Id
        c = self.vectorize(Id).reshape(1, -1).expand_as(x)

        if self.base_expmap:
            # original N(0, 1) samples
            # print("x finite", torch.isfinite(x).all())
            u = self.logmap(c, x)
            # print("u finite", torch.isfinite(u).all())
            logpu = normal_logprob(u, 0.0, torch.log(self.scale_std)).sum(-1)

            # print(u)
            # print("logpu", logpu.shape, logpu.mean())

            # Warning: For some reason, functorch doesn't play well with the sqrtmh implementation.
            with torch.inference_mode(mode=False):

                def logdetjac(f):
                    def _logdetjac(*args):
                        jac = jacrev(f, chunk_size=256)(*args)
                        return torch.linalg.slogdet(jac)[1]

                    return _logdetjac

                # Change of variables in Euclidean space
                ldjs = vmap(logdetjac(self.expmap))(c, u)
                logpu = logpu - ldjs

                # print("ldjs", ldjs.shape, ldjs.mean())
        else:
            u = x - c
            logpu = normal_logprob(u, 0.0, torch.log(self.scale_std)).sum(-1)

            # print("logpu", logpu.shape, logpu.mean())

        # Change of metric from Euclidean to Riemannian
        ldgs = self.logdetG(x)

        # print("ldG", ldgs.shape, ldgs.mean())

        logpx = logpu - 0.5 * ldgs
        return logpx.reshape(
            *size[:-1]
        )  # this caused an error, TypeError: reshape() missing 1 required positional arguments: "shape"


class SPDNonIsotropicRandom(AbstractSPD, manifm_SPD):
    name = "SPDNonIsotropicRandom"

    def __init__(
        self,
        mean: torch.Tensor,
        logmap_std: torch.Tensor,
        base_expmap=True,
        Riem_geodesic=True,
        Riem_norm=True,
    ):
        """mean and logmap_std should be in vectorized form

        note: logmap_std is the std of the data evaluated at the mean:
            i.e. `logmap(mean, data).std()`
        """
        super().__init__(None, None, base_expmap, Riem_geodesic, Riem_norm)
        del self.scale_std
        del self.scale_Id
        assert mean.size(-1) == logmap_std.size(-1)
        self.assert_spd(mean)
        self.register_buffer("mean", mean)
        self.register_buffer("std", logmap_std)

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        assert size[-1] == self.mean.size(-1)
        bsz = int(math.prod(size[:-1]))
        d = size[-1]

        rnd = torch.randn(bsz, d, dtype=dtype, device=device)
        rnd_dtype = rnd.dtype
        rnd_device = rnd.device

        # Wrap a Gaussian centered at the mean spd matrix.
        c = self.mean.to(dtype=rnd_dtype, device=rnd_device).expand(bsz, d)

        # Construct symmetric matrix where elements are iid Normal, scaled by logstd.
        u = rnd.mul(
            self.logmap_std.expand(bsz, d).to(dtype=rnd_dtype, device=rnd_device)
        )

        if self.base_expmap:
            # Exponential map to the manifold.
            x = self.expmap(c, u)
        else:
            # Beware this can sample a non-SPD matrix unless scale is small enough.
            x = c + u
        return x.reshape(*size)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape
        d = x.shape[-1]
        x = x.reshape(-1, d)
        c = self.mean.to(x).expand_as(x)
        s = self.logmap_std.to(x).expand_as(x)

        normal = torch.distributions.Normal(loc=x, scale=s, validate_args=False)
        if self.base_expmap:
            u = self.logmap(c, x)
            logpu = normal.log_prob(u).sum(-1)

            # Warning: For some reason, functorch doesn't play well with the sqrtmh implementation.
            with torch.inference_mode(mode=False):

                def logdetjac(f):
                    def _logdetjac(*args):
                        jac = jacrev(f, chunk_size=256)(*args)
                        return torch.linalg.slogdet(jac)[1]

                    return _logdetjac

                # Change of variables in Euclidean space
                ldjs = vmap(logdetjac(self.expmap))(c, u)
                logpu = logpu - ldjs
        else:
            u = x - c
            logpu = normal.log_prob(u).sum(-1)

        # Change of metric from Euclidean to Riemannian
        ldgs = self.logdetG(x)

        logpx = logpu - 0.5 * ldgs
        if len(size) <= 1:
            return logpx.squeeze()
        else:
            return logpx.reshape(*size[:-1])


def lattice_matrix_to_spd_matrix(lattice: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ji,...jk->...ik", lattice, lattice)  # L^T L


def lattice_params_to_spd_matrix(
    lengths: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    return lattice_matrix_to_spd_matrix(lattice_params_to_matrix_torch(lengths, angles))


def lattice_params_to_spd_vector(
    lengths: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    return SPDGivenN.vectorize(lattice_params_to_spd_matrix(lengths, angles))


def spd_vector_to_lattice_matrix(spd_vector: torch.Tensor) -> torch.Tensor:
    mat = SPDGivenN.devectorize(spd_vector)
    return sqrtmh(mat)


def get_spd_data(
    dataset: dataset_options, path: str | Path | None, stem: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if path is None:
        file = Path(__file__).parent / f"{stem}.yaml"
    else:
        file = Path(path)
    file = file.resolve()

    if file.exists():
        stats = OmegaConf.load(str(file))
    else:
        raise FileNotFoundError(
            f"{file=} does not exist, have you computed the mean and std already?"
        )

    mean = torch.tensor(stats[dataset]["mean"])
    std = torch.tensor(stats[dataset]["std"])
    return mean, std


def compute_atom_density(
    dataset: dataset_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    """assumes a constant spacing of atoms within the cell"""
    train_loader, *_ = init_loaders(dataset=dataset)
    num_atoms, vols = [], []
    for batch in train_loader:
        num_atoms.append(batch.num_atoms)
        lattice = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        vols.append(torch.linalg.det(lattice))
    num_atoms = torch.cat(num_atoms, dim=0)
    vols = torch.cat(vols, dim=0)
    densities = num_atoms / vols
    return densities.mean(), densities.std()


@cache
def get_atom_density(
    dataset: dataset_options, path: str | Path | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_spd_data(dataset, path, "atom_density")


def compute_spd_pLTL_mean_logmap_std(
    dataset: dataset_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_loader, *_ = init_loaders(dataset=dataset)
    lattices = []
    for batch in train_loader:
        lattices.append(lattice_params_to_matrix_torch(batch.lengths, batch.angles))
    lattices = torch.cat(lattices, dim=0)
    spd = lattice_matrix_to_spd_matrix(lattices)
    eigvals = torch.linalg.eigvals(spd)
    assert (
        eigvals.imag <= 1e-4
    ).all(), "there was a significant imaginary component in one of the targets"

    s = manifm_SPD()
    x = s.vectorize(spd)
    mean = x.mean(0)
    u = s.logmap(mean, x)
    std = u.std(0)
    return mean, std


def get_spd_pLTL_mean_logmap_std(
    dataset: dataset_options, path: str | Path | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_spd_data(dataset, path, "spd_pLTL_stats")


def compute_spd_pLTL_given_n_std_coef(
    atom_density: float,
    dataset: dataset_options,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_loader, *_ = init_loaders(dataset=dataset)
    num_atoms, lattices = [], []
    for batch in train_loader:
        num_atoms.append(batch.num_atoms)
        lattices.append(lattice_params_to_matrix_torch(batch.lengths, batch.angles))
    num_atoms = torch.cat(num_atoms, dim=0)
    lattices = torch.cat(lattices, dim=0)
    spd = lattice_matrix_to_spd_matrix(lattices)
    eigvals = torch.linalg.eigvals(spd)
    assert (
        eigvals.imag <= 1e-4
    ).all(), "there was a significant imaginary component in one of the targets"

    all_ns, _ = num_atoms.unique().sort()
    s = manifm_SPD()
    spd_vecs = s.vectorize(spd)

    std_coefs = []
    for n in all_ns:
        prior_mean = torch.eye(spd.shape[-1]) * ((1 / atom_density) * n).pow(2 / 3)
        prior_mean = s.vectorize(prior_mean)
        mask = num_atoms == n
        spd_vecs_given_n = spd_vecs[mask]

        # # this gives the empirical isotropic spd
        # mean = spd_vecs_given_n.mean(0)
        # mean_iso = SPDGivenN.closest_isotropic(mean, Riem_dist=True)

        iso = SPDGivenN.closest_isotropic(spd_vecs_given_n, Riem_dist=True)
        log_noise_samples = s.logmap(prior_mean, iso)[:, 0]
        std_coefs.append((log_noise_samples.std() ** (3 / 2)) / n)
    std_coefs = torch.stack(std_coefs, dim=0)
    return std_coefs.mean(), std_coefs.std()


@cache
def get_spd_pLTL_given_n_std_coef(
    dataset: dataset_options, path: str | Path | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_spd_data(dataset, path, "spd_std_coef")


if __name__ == "__main__":
    import yaml
    from tqdm import tqdm

    # removed because we're not using p(L) directly anymore so no need to fit it
    # # mean and std of p(L)
    # compute_stats = True
    # file = Path(__file__).parent / "spd_pLTL_stats.yaml"
    # file = file.resolve()
    # if compute_stats:
    #     print("calculate the overall stats of p(L) for each dataset")
    #     stats = {}
    #     pbar = tqdm(list(dataset_options.__args__))
    #     for dataset in pbar:
    #         pbar.set_description(f"{dataset=}")
    #         mean, std = compute_spd_pLTL_mean_logmap_std(dataset)
    #         stats[dataset] = {
    #             "mean": mean.cpu().tolist(),
    #             "std": std.cpu().tolist(),
    #         }
    #     with open(file, "w") as f:
    #         yaml.dump(stats, f)
    # else:
    #     stats = OmegaConf.load(str(file))
    # density
    compute_stats = False
    file = Path(__file__).parent / "atom_density.yaml"
    file = file.resolve()
    if compute_stats:
        print("calculate the density atoms to volume")
        stats = {}
        pbar = tqdm(list(dataset_options.__args__))
        for dataset in pbar:
            pbar.set_description(f"{dataset=}")
            mean, std = compute_atom_density(dataset)
            stats[dataset] = {
                "mean": mean.cpu().tolist(),
                "std": std.cpu().tolist(),
            }

        with open(file, "w") as f:
            yaml.dump(stats, f)
    else:
        stats = OmegaConf.load(str(file))

    # mean and std of p(L | N)
    compute_stats = True
    file = Path(__file__).parent / "spd_std_coef.yaml"
    file = file.resolve()
    if compute_stats:
        print("calculate the stats of p(L | N) for each dataset")
        stats = {}
        pbar = tqdm(list(dataset_options.__args__))
        for dataset in pbar:
            pbar.set_description(f"{dataset=}")
            atom_density, _ = get_atom_density(dataset)
            mean, std = compute_spd_pLTL_given_n_std_coef(
                atom_density,
                dataset,
            )
            stats[dataset] = {
                "mean": mean.cpu().tolist(),
                "std": std.cpu().tolist(),
            }

        with open(file, "w") as f:
            yaml.dump(stats, f)
    else:
        stats = OmegaConf.load(str(file))

    lengths = torch.tensor([4.2460, 4.2460, 4.2460]).unsqueeze(0)
    angles = torch.tensor([90.0, 90.0, 90.0]).unsqueeze(0)
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    spd_vector = lattice_params_to_spd_vector(lengths, angles)
    lp = spd_vector_to_lattice_matrix(spd_vector)

    # are SPD matrices invariant to upper triangular transformations?
    l = lp.transpose(1, 2)
    ltl = torch.matmul(l.transpose(1, 2), l)
    c = torch.zeros_like(l) + torch.ones_like(l).triu()
    lc = l @ c
    lctlc = torch.matmul(lc.transpose(1, 2), lc)

    d = torch.linalg.det(ltl).pow(1 / 3) * torch.eye(l.shape[-1])
    d = manifm_SPD().vectorize(d)
    f = torch.tensor(8).pow(1 / 3) * manifm_SPD().vectorize(torch.eye(l.shape[-1]))
    logdf = manifm_SPD().logmap(d, f)

    # do some testing for SPDNonIsotropicRandom
    for dataset in tqdm(list(dataset_options.__args__)):
        s = manifm_SPD(Riem_geodesic=True, Riem_norm=True)
        mean = torch.tensor(stats[dataset]["mean"])
        std = torch.tensor(stats[dataset]["std"])
        assert (torch.linalg.eigvals(s.devectorize(mean)).imag <= 1e-4).all()
        assert (torch.linalg.eigvals(s.devectorize(mean)).real > 0.0).all()

        spd = SPDNonIsotropicRandom(mean, std)
        r = spd.random_base(10, mean.size(-1))
        lp = spd.base_logprob(r)
        print(r, lp)

        r = spd.random_base(3, 10, mean.size(-1))
        lp = spd.base_logprob(r)
        print(r, lp)
