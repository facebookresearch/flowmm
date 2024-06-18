"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from geoopt.manifolds.euclidean import Euclidean

from flowmm.data import NUM_ATOMIC_BITS
from flowmm.rfm.manifolds.masked import MaskedManifold


def int2bits(
    x: torch.LongTensor, n: int, out_dtype: torch.dtype = None
) -> torch.Tensor:
    """Convert an integer x in (...) into bits in (..., n)."""
    x = torch.bitwise_right_shift(
        torch.unsqueeze(x, -1), torch.arange(n, dtype=x.dtype, device=x.device)
    )
    x = torch.fmod(x, 2)
    return x.to(dtype=out_dtype)


def bits2int(
    x: torch.Tensor | np.ndarray, out_dtype: torch.dtype | np.dtype
) -> torch.Tensor:
    """Converts bits x in (..., n) into an integer in (...)."""
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=out_dtype)
        n = x.size(-1)
        x = torch.sum(x * (2 ** torch.arange(n)), dim=-1)
        return x
    else:
        x = x.astype(out_dtype)
        n = x.shape[-1]
        x = np.sum(x * (2 ** np.arange(n)), axis=-1)
        return x


def int_to_analog_bits(
    x: torch.LongTensor, n_bits: int, b_scale: float
) -> torch.Tensor:
    bits = int2bits(x, n_bits)
    analog_bits = (bits * 2 - 1) * b_scale
    return analog_bits


def analog_bits_to_int(analog_bits: torch.Tensor | np.ndarray) -> torch.LongTensor:
    if isinstance(analog_bits, torch.Tensor):
        signs = analog_bits.sign()
        bits = (signs + 1) / 2
        return bits2int(bits, torch.long)
    else:
        signs = np.sign(analog_bits)
        bits = (signs + 1) / 2
        return bits2int(bits, np.int64)


class AnalogBits(Euclidean):
    """Represents an integer in an 'analog' euclidean space."""

    name = "AnalogBits"
    reversible = True

    def __init__(self, scale: float):
        super().__init__(ndim=1)
        self.scale = scale

    # def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
    #     bits_01 = torch.randint(0, 2, size, dtype=dtype, device=device)
    #     return (bits_01 * 2 - 1) * self.scale

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        return torch.randn(size, dtype=dtype, device=device)

    random = random_base

    @staticmethod
    def base_logprob(x: torch.Tensor) -> torch.Tensor:
        n = x.size(-1)
        ps = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device) * 0.5**n
        return torch.log(ps)

    def extra_repr(self):
        return f"scale={self.scale}"

    def metric_normalized(self, _: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def logdetG(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x).sum(-1)

    # extra stuff we can't / wont use
    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False):
        raise NotImplementedError()

    def norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError()

    def dist():
        raise NotImplementedError()

    def dist2():
        raise NotImplementedError()

    def abits_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_(-self.scale, self.scale)


class MultiAtomAnalogBits(MaskedManifold, AnalogBits):
    """Represents an integer in an 'analog' euclidean space.
    This one handles masking out extra dimensions due to padding to `max_num_atoms`.
    """

    name = "MultiAtomAnalogBits"
    reversible = True

    def __init__(self, scale: float, num_bits: int, num_atoms: int, max_num_atoms: int):
        AnalogBits.__init__(self, scale=scale)
        self.num_bits = num_bits
        self.max_num_atoms = max_num_atoms
        self.register_buffer("mask", self._get_mask(num_atoms, max_num_atoms))

    @property
    def dim_m1(self) -> int:
        return self.num_bits

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        return Euclidean(ndim=self.ndim).inner(
            x, u, v, keepdim=keepdim
        ) / self.mask.sum().to(u)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        initial_shape = x.shape
        mx = self.reshape_and_mask(initial_shape, x)
        return self.mask_and_reshape(initial_shape, mx)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        initial_shape = u.shape
        u = self.reshape_and_mask(initial_shape, u)
        u = self.mask_and_reshape(initial_shape, u)
        return Euclidean(self.ndim).proju(x, u)

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        assert (
            size[-1] == self.max_num_atoms * self.num_bits
        ), "last dimension must be compatible with max_num_atoms and num_bits"
        analog_bits = AnalogBits(self.scale).random_base(
            *size[:-1], self.max_num_atoms, self.num_bits, dtype=dtype, device=device
        )  # this is still a random normal
        return self.mask_and_reshape(size, analog_bits)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("")
        initial_shape = x.shape
        x = self.reshape_and_mask(initial_shape, x)

        n = torch.sum(self.mask).to(x)
        ps = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device) * 0.5**n
        lps = torch.log(ps)
        return torch.sum(self.mask.to(x).squeeze() * lps, dim=-1)

    def extra_repr(self):
        return f"""
        scale={self.scale},
        num_bits={self.num_bits},
        num_atoms={int(self.mask.sum().cpu().item())},
        max_num_atoms={self.max_num_atoms},
        """


if __name__ == "__main__":
    # path = Path("proof_of_concept/temp.pt")
    # with open(path, "rb") as f:
    #     batch = torch.load(path)
    atom_types = torch.randint(1, 98, (1, 10))

    bits = int2bits(atom_types, NUM_ATOMIC_BITS)
    ints = bits2int(bits, torch.long)
    assert torch.all(ints == atom_types)

    analog_bits = int_to_analog_bits(atom_types, NUM_ATOMIC_BITS, 0.5)
    ints = analog_bits_to_int(analog_bits)
    assert torch.all(ints == atom_types)

    b = 10
    mna = 5
    na = 3
    scale = 1.0

    ab = AnalogBits(scale)
    bits = ab.random(NUM_ATOMIC_BITS, na)
    print(bits[0])
    # lps = ab.base_logprob(bits)
    # print(bits[0], lps[0])

    maab = MultiAtomAnalogBits(
        scale,
        NUM_ATOMIC_BITS,
        na,
        mna,
    )
    bits_ma = maab.random(b, NUM_ATOMIC_BITS * mna)
    # lps_ma = maab.base_logprob(bits_ma)
    print(
        bits_ma[0],
        # lps_ma[0],
    )
    print("")
