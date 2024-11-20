import random
from typing import Optional, Sequence
from pathlib import Path
from functools import partial

import hydra
from hydra.utils import get_class
import numpy as np
import omegaconf
import pytorch_lightning as pl

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import get_scaler_from_data_list
from rfm_docking.manifold_getter import DockingManifoldGetter


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        coord_manifold: str,
        dataset_name: str,
        collate: DictConfig,
        do_ot: bool,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.manifold_getter = DockingManifoldGetter(
            coord_manifold=coord_manifold,
            dataset=dataset_name,
        )

        # TODO add OT to options
        # ...
        collate_class = get_class(collate._target_)
        self.collate_fn = collate_class(
            manifold_getter=self.manifold_getter, do_ot=do_ot
        )

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        self.scaler = dict()

        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key="scaled_lattice"
            )
            for prop in train_dataset.prop:
                self.scaler[prop] = get_scaler_from_data_list(train_dataset.cached_data, key=prop)
        else:
            try:
                self.lattice_scaler = torch.load(
                    Path(scaler_path) / "lattice_scaler.pt"
                )
                for prop in train_dataset.prop:
                    self.scaler[prop] = torch.load(Path(scaler_path) / f"prop_{prop}_scaler.pt")
            except:
                train_dataset = hydra.utils.instantiate(self.datasets.train)
                self.lattice_scaler = get_scaler_from_data_list(
                    train_dataset.cached_data, key="scaled_lattice"
                )
                for prop in train_dataset.prop:
                    self.scaler[prop] = get_scaler_from_data_list(train_dataset.cached_data, key=prop)

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1"
)
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup("fit")
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
