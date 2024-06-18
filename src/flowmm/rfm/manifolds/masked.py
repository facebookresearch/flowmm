"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from abc import ABC, abstractmethod

import torch


class MaskedManifold(ABC):
    @staticmethod
    def _get_mask(num_atoms: int, max_num_atoms: int) -> torch.Tensor:
        mask = torch.zeros(max_num_atoms, dtype=torch.bool)
        mask[:num_atoms] = torch.ones(num_atoms, dtype=torch.bool)
        return mask.unsqueeze(-1)

    @property
    def dtype(self) -> None:
        """the manifold requires bool tensors, this disrupts introspection default"""
        return None

    @property
    def device(self) -> None:
        """gets mapped to the right device as necessary, this disrupts introspection default"""
        return None

    @property
    def dim_m2(self) -> int:
        return self.max_num_atoms

    @property
    @abstractmethod
    def dim_m1(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        raise NotImplementedError()

    def reshape_and_mask(
        self,
        initial_shape: tuple[int, ...],
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """takes in a [..., A * B] shaped vector
        returns [..., A, B] shaped, masked vector"""
        vec = vec.reshape(*initial_shape[:-1], self.dim_m2, self.dim_m1)
        return self.mask.to(vec) * vec

    def mask_and_reshape(
        self,
        initial_shape: tuple[int, ...],
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """takes in a [..., A, B] shaped vector
        returns [..., A * B] shaped, masked (output) vector"""
        vec = self.mask.to(vec) * vec
        return vec.reshape(*initial_shape)
