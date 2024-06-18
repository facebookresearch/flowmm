"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import functools
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import flowmm
import wandb
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES


@functools.cache
def generate_id():
    return wandb.util.generate_id()


def register_omega_conf_resolvers():
    OmegaConf.register_new_resolver(
        "do_ip",
        lambda x: True if x == "non_symmetric" else False,
    )
    OmegaConf.register_new_resolver(
        "get_dim_atomic_rep",
        lambda x: NUM_ATOMIC_BITS if x == "analog_bits" else NUM_ATOMIC_TYPES,
    )
    OmegaConf.register_new_resolver("generate_id", generate_id)
    OmegaConf.register_new_resolver("get_flowmm_version", lambda: flowmm.__version__)


def get_wandb_directory(checkpoint_path: Path) -> Path:
    job_dir = Path(get_job_directory(checkpoint_path))
    wandb_dir = job_dir / "wandb"
    if wandb_dir.is_dir() and wandb_dir.exists():
        return wandb_dir
    else:
        raise FileNotFoundError("could not find the wandb folder.")


def load_id_from_wandb(checkpoint_path: Path) -> str:
    wandb_dir = get_wandb_directory(checkpoint_path)
    runs = list(wandb_dir.glob("run-*"))
    guess_id = runs[0].stem.split("-")[-1]
    for run in runs:
        assert guess_id == run.stem.split("-")[-1]
    return guess_id


def load_date_from_wandb(checkpoint_path: Path) -> str:
    wandb_dir = get_wandb_directory(checkpoint_path)
    runs = list(wandb_dir.glob("run-*"))

    # check all have the same id
    guess_id = runs[0].stem.split("-")[-1]
    for run in runs:
        assert guess_id == run.stem.split("-")[-1]

    # check all have the same datetime
    guess_datetime = runs[0].stem.split("-")[-2]
    for run in runs:
        assert guess_datetime == run.stem.split("-")[-2]

    date = guess_datetime.split("_")[0]
    assert len(date) == 8
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


def load_latest_run_from_wandb(checkpoint_path: Path) -> str:
    wandb_dir = get_wandb_directory(checkpoint_path)
    return OmegaConf.load(wandb_dir / "latest-run" / "files" / "config.yaml")


def load_project_from_wandb(checkpoint_path: Path) -> str:
    loaded = load_latest_run_from_wandb(checkpoint_path)
    return loaded["logging/wandb/project"].value


def load_group_from_wandb(checkpoint_path: Path) -> str:
    loaded = load_latest_run_from_wandb(checkpoint_path)
    return loaded["logging/wandb/group"].value


def get_job_directory(file_or_checkpoint: Union[str, Dict[str, Any]]) -> str:
    found = False
    if isinstance(file_or_checkpoint, dict):
        chkpnt = file_or_checkpoint
        key = [x for x in chkpnt["callbacks"].keys() if "Checkpoint" in x][0]
        file = chkpnt["callbacks"][key]["dirpath"]
    else:
        file = file_or_checkpoint

    hydra_files = []
    directory = os.path.dirname(file)
    count = 0
    while not found:
        hydra_files = glob(
            os.path.join(os.path.join(directory, ".hydra/config.yaml")),
            recursive=True,
        )
        if len(hydra_files) > 0:
            break
        directory = os.path.dirname(directory)
        if directory == "":
            raise ValueError("Failed to find hydra config!")
        count += 1
        if count > 10_000:
            raise ValueError(f"Failed to find hydra config!, we tried {count=} times.")
    assert len(hydra_files) == 1, "Found ambiguous hydra config files!"
    job_dir = os.path.dirname(os.path.dirname(hydra_files[0]))
    return job_dir


def load_cfg(
    checkpoint: str | Path,
) -> DictConfig:
    checkpoint = str(Path(checkpoint).resolve())
    job_dir = get_job_directory(checkpoint)
    return OmegaConf.load(os.path.join(job_dir, ".hydra/config.yaml"))


def load_model(
    checkpoint: str | Path,
    eval_projx: bool = None,
    atol: float = None,
    rtol: float = None,
) -> "flowmm.model.model_pl.MaterialsRFMLitModule":
    from flowmm.model.model_pl import MaterialsRFMLitModule

    checkpoint = str(Path(checkpoint).resolve())
    chkpnt = torch.load(checkpoint, map_location="cpu")
    cfg = load_cfg(checkpoint)

    if eval_projx is not None:
        cfg.eval_projx = eval_projx

    if atol is not None:
        cfg.model.atol = atol

    if rtol is not None:
        cfg.model.rtol = rtol

    model = MaterialsRFMLitModule(cfg)
    model.load_state_dict(chkpnt["state_dict"])
    return cfg, model


def get_loaders(
    cfg: DictConfig,
    job_directory: Path | str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False, scaler_path=job_directory
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader(shuffle=False)
    val_loader = datamodule.val_dataloader()[0]
    test_loader = datamodule.test_dataloader()[0]
    return train_loader, val_loader, test_loader


class CSPDataset(Dataset):
    def __init__(self, atom_types: Sequence[Sequence[int]]):
        super().__init__()
        self.atom_typess = atom_types

    def __len__(self) -> int:
        return len(self.atom_typess)

    def __getitem__(self, index: int) -> Data:
        atom_types = self.atom_typess[index]
        num_atoms = len(atom_types)
        data = Data(
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=num_atoms,
            atom_types=torch.LongTensor(atom_types),
        )
        return data
