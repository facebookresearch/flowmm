"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from flowmm.model.eval_utils import get_loaders

dataset_options = Literal["carbon", "mp_20", "mpts_52", "perov", "mp_20_llama"]


def init_cfg(
    overrides: list[str] = [],
) -> DictConfig:
    project_root = config_dir = Path(__file__).parents[2]
    os.environ["PROJECT_ROOT"] = str(project_root.resolve())
    config_dir = project_root / f"scripts_model/conf"

    with initialize_config_dir(str(config_dir.resolve()), version_base="1.1"):
        cfg = compose(config_name="default", overrides=overrides)
    return cfg


def init_loaders(
    dataset: dataset_options,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    overrides = [f"data={dataset}"]
    if batch_size is not None:
        overrides.extend(
            [
                f"data.datamodule.batch_size.train={batch_size}",
                f"data.datamodule.batch_size.val={batch_size}",
                f"data.datamodule.batch_size.test={batch_size}",
            ]
        )
    cfg = init_cfg(overrides=overrides)
    return get_loaders(cfg)
