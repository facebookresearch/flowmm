"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Sequence

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers.wandb import WandbLogger
from torch_geometric.data import Batch, Data, DataLoader

import wandb
from diffcsp.script_utils import GenDataset
from flowmm.model.eval_utils import (
    CSPDataset,
    get_loaders,
    load_cfg,
    load_date_from_wandb,
    load_group_from_wandb,
    load_id_from_wandb,
    load_model,
    load_project_from_wandb,
    register_omega_conf_resolvers,
)
from flowmm.old_eval.generation_metrics import compute_generation_metrics
from flowmm.old_eval.lattice_metrics import compute_lattice_metrics
from flowmm.old_eval.reconstruction_metrics import compute_reconstruction_metrics

TASKS_TYPE = Literal[
    "reconstruct", "recon_trajectory", "generate", "gen_trajectory", "pred"
]
TASKS = deepcopy(TASKS_TYPE.__args__)
STAGE_TYPE = Literal["train", "val", "test"]
STAGES = deepcopy(STAGE_TYPE.__args__)
register_omega_conf_resolvers()


class TorchPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path | str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions, self.output_dir / f"predictions_{trainer.global_rank:02d}.pt"
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            self.output_dir / f"batch_indices_{trainer.global_rank:02d}.pt",
        )


@click.group()
def cli():
    pass


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_evals", type=int, default=1)
@click.option("--limit_predict_batches", type=str, default="1.")
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--div_mode",
    type=click.Choice(["exact", "rademacher"], case_sensitive=False),
    default=None,
)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
def nll(
    checkpoint: Path,
    stage: STAGE_TYPE,
    batch_size: int,
    num_evals: int,
    limit_predict_batches: str,
    num_steps: int,
    div_mode: bool,
    single_gpu: bool,
) -> None:
    raise NotImplementedError(
        "there are currently base distributions which make this unappealing."
    )


def get_target_dir(checkpoint: Path, subdir: bool) -> Path:
    if subdir:
        target_dir = checkpoint.parent / subdir
    else:
        target_dir = checkpoint.parent
    return target_dir.resolve()


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_evals", type=int, default=1)
@click.option("--limit_predict_batches", type=str, default="1.")
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
def reconstruct(
    checkpoint: Path,
    stage: STAGE_TYPE,
    batch_size: int | None,
    num_evals: int,
    limit_predict_batches: str,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
) -> None:
    cfg, model = load_model(checkpoint)

    if "null" not in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do reconstruction with an unconditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    stage = stage.lower()
    if batch_size is None:  # this must be explicitly set since the default is int
        batch_size = getattr(cfg.data.datamodule.batch_size, stage)
        print(f"Using {batch_size=} from default cfg")
    else:
        setattr(cfg.data.datamodule.batch_size, stage, batch_size)
        print(f"Using custom {batch_size=}")

    # update cfg
    if num_steps is not None:
        cfg.integrate.num_steps = num_steps
    if inference_anneal_slope is not None:
        cfg.integrate.inference_anneal_slope = inference_anneal_slope
    if inference_anneal_offset is not None:
        assert (0 <= inference_anneal_offset) and (inference_anneal_offset < 1)
        cfg.integrate.inference_anneal_offset = inference_anneal_offset
    if compute_traj_velo_norms:
        cfg.integrate.compute_traj_velo_norms = compute_traj_velo_norms

    cfg.integrate.inference_anneal_types = inference_anneal_types
    cfg.integrate.inference_anneal_coords = inference_anneal_coords
    cfg.integrate.inference_anneal_lattice = inference_anneal_lattice

    loaders = get_loaders(cfg)
    loader = loaders[STAGES.index(stage)]

    target_dir = get_target_dir(checkpoint, subdir)

    assert num_evals > 0
    directories = [f"reconstruct_{i:02d}" for i in range(num_evals)]

    for directory in directories:
        pred_writer = TorchPredictionWriter(
            output_dir=target_dir / directory,
            write_interval="epoch",
        )
        # save num_steps
        (target_dir / directory / "num_steps.txt").write_text(
            str(cfg.integrate.num_steps)
        )

        if single_gpu:
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=[pred_writer],
                limit_predict_batches=eval(limit_predict_batches),
            )
        else:
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy="ddp",
                devices="auto",
                callbacks=[pred_writer],
                limit_predict_batches=eval(limit_predict_batches),
            )
        trainer.predict(
            model,
            dataloaders=loader,
            return_predictions=False,
            ckpt_path=checkpoint,
        )


@cli.command(name="recon_trajectory")
@click.argument("checkpoint", type=Path)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_evals", type=int, default=1)
@click.option("--limit_predict_batches", type=str, default="1.")
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
def recon_trajectory(
    checkpoint: Path,
    stage: STAGE_TYPE,
    batch_size: int | None,
    num_evals: int,
    limit_predict_batches: str,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
) -> None:
    cfg, model = load_model(checkpoint)

    if "null" not in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do reconstruction with an unconditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    stage = stage.lower()
    if batch_size is None:  # this must be explicitly set since the default is int
        batch_size = getattr(cfg.data.datamodule.batch_size, stage)
        print(f"Using {batch_size=} from default cfg")
    else:
        setattr(cfg.data.datamodule.batch_size, stage, batch_size)
        print(f"Using custom {batch_size=}")

    # update cfg
    if num_steps is not None:
        cfg.integrate.num_steps = num_steps
    if inference_anneal_slope is not None:
        cfg.integrate.inference_anneal_slope = inference_anneal_slope
    if inference_anneal_offset is not None:
        assert (0 <= inference_anneal_offset) and (inference_anneal_offset < 1)
        cfg.integrate.inference_anneal_offset = inference_anneal_offset
    if compute_traj_velo_norms:
        cfg.integrate.compute_traj_velo_norms = compute_traj_velo_norms

    cfg.integrate.inference_anneal_types = inference_anneal_types
    cfg.integrate.inference_anneal_coords = inference_anneal_coords
    cfg.integrate.inference_anneal_lattice = inference_anneal_lattice

    # THIS ADDS A NEW FIELD TO CFG
    cfg.integrate.entire_traj = True

    loaders = get_loaders(cfg)
    loader = loaders[STAGES.index(stage)]

    target_dir = get_target_dir(checkpoint, subdir)

    assert num_evals > 0
    directories = [f"recon_trajectory_{i:02d}" for i in range(num_evals)]

    for directory in directories:
        pred_writer = TorchPredictionWriter(
            output_dir=target_dir / directory,
            write_interval="epoch",
        )
        # save num_steps
        (target_dir / directory / "num_steps.txt").write_text(
            str(cfg.integrate.num_steps)
        )

        if single_gpu:
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=[pred_writer],
                limit_predict_batches=eval(limit_predict_batches),
            )
        else:
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy="ddp",
                devices="auto",
                callbacks=[pred_writer],
                limit_predict_batches=eval(limit_predict_batches),
            )
        trainer.predict(
            model,
            dataloaders=loader,
            return_predictions=False,
            ckpt_path=checkpoint,
        )


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option("--num_samples", type=int, default=10_000)
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--gen_id", type=int, default=0, help=r"folder name is generate_{gen_id}")
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
def generate(
    checkpoint: Path,
    num_samples: int,
    batch_size: int | None,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    gen_id: int,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
) -> None:
    cfg, model = load_model(checkpoint)

    if "null" in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do generation with a conditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    # update cfg
    if num_steps is not None:
        cfg.integrate.num_steps = num_steps
    if inference_anneal_slope is not None:
        cfg.integrate.inference_anneal_slope = inference_anneal_slope
    if inference_anneal_offset is not None:
        assert (0 <= inference_anneal_offset) and (inference_anneal_offset < 1)
        cfg.integrate.inference_anneal_offset = inference_anneal_offset
    if compute_traj_velo_norms:
        cfg.integrate.compute_traj_velo_norms = compute_traj_velo_norms

    cfg.integrate.inference_anneal_types = inference_anneal_types
    cfg.integrate.inference_anneal_coords = inference_anneal_coords
    cfg.integrate.inference_anneal_lattice = inference_anneal_lattice

    sample_set = GenDataset(dataset=cfg.data.dataset_name, total_num=num_samples)
    loader = DataLoader(sample_set, batch_size=batch_size)

    target_dir = get_target_dir(checkpoint, subdir)

    directories = [f"generate_{gen_id:02d}"]

    for directory in directories:
        pred_writer = TorchPredictionWriter(
            output_dir=target_dir / directory,
            write_interval="epoch",
        )
        # save num_steps
        (target_dir / directory / "num_steps.txt").write_text(
            str(cfg.integrate.num_steps)
        )

        if single_gpu:
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=[pred_writer],
            )
        else:
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy="ddp",
                devices="auto",
                callbacks=[pred_writer],
            )
        trainer.predict(
            model,
            dataloaders=loader,
            return_predictions=False,
            ckpt_path=checkpoint,
        )


@cli.command(name="gen_trajectory")
@click.argument("checkpoint", type=Path)
@click.option("--num_samples", type=int, default=256)
@click.option("--batch_size", type=int, default=256)
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--gen_id", type=int, default=0, help=r"folder name is generate_{gen_id}")
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--compute_traj_velo_norms",
    is_flag=True,
    show_default=True,
    default=False,
)
def gen_trajectory(
    checkpoint: Path,
    num_samples: int,
    batch_size: int | None,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    gen_id: int,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
    compute_traj_velo_norms: bool | None,
) -> None:
    cfg, model = load_model(checkpoint)

    if "null" in cfg.model.manifold_getter.atom_type_manifold:
        raise ValueError(
            f"you cannot do generation with a conditional atom_type_manifold {cfg.model.manifold_getter.atom_type_manifold=}"
        )

    # update cfg
    if num_steps is not None:
        cfg.integrate.num_steps = num_steps
    if inference_anneal_slope is not None:
        cfg.integrate.inference_anneal_slope = inference_anneal_slope
    if inference_anneal_offset is not None:
        assert (0 <= inference_anneal_offset) and (inference_anneal_offset < 1)
        cfg.integrate.inference_anneal_offset = inference_anneal_offset
    if compute_traj_velo_norms:
        cfg.integrate.compute_traj_velo_norms = compute_traj_velo_norms

    cfg.integrate.inference_anneal_types = inference_anneal_types
    cfg.integrate.inference_anneal_coords = inference_anneal_coords
    cfg.integrate.inference_anneal_lattice = inference_anneal_lattice

    # THIS ADDS A NEW FIELD TO CFG
    cfg.integrate.entire_traj = True

    sample_set = GenDataset(dataset=cfg.data.dataset_name, total_num=num_samples)
    loader = DataLoader(sample_set, batch_size=batch_size)

    target_dir = get_target_dir(checkpoint, subdir)

    directories = [f"gen_trajectory_{gen_id:02d}"]

    for directory in directories:
        pred_writer = TorchPredictionWriter(
            output_dir=target_dir / directory,
            write_interval="epoch",
        )
        # save num_steps
        (target_dir / directory / "num_steps.txt").write_text(
            str(cfg.integrate.num_steps)
        )

        if single_gpu:
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=[pred_writer],
            )
        else:
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy="ddp",
                devices="auto",
                callbacks=[pred_writer],
            )
        trainer.predict(
            model,
            dataloaders=loader,
            return_predictions=False,
            ckpt_path=checkpoint,
        )


@cli.command()
@click.argument("checkpoint", type=Path)
@click.argument("atom_types_path", type=Path)
@click.option("--batch_size", type=int, default=16384)
@click.option("--num_steps", type=int, default=None)
@click.option(
    "--single_gpu/--multi_gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="use one gpu, not ddp",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--pred_id", type=int, default=0, help=r"folder name is pred_{pred_id}")
@click.option(
    "--inference_anneal_slope",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_offset",
    type=float,
    default=None,
)
@click.option(
    "--inference_anneal_types/--no-inference_anneal_types",
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    "--inference_anneal_coords/--no-inference_anneal_coords",
    is_flag=True,
    show_default=True,
    default=True,
)
@click.option(
    "--inference_anneal_lattice/--no-inference_anneal_lattice",
    is_flag=True,
    show_default=True,
    default=False,
)
def predict(
    checkpoint: Path,
    atom_types_path: Path,
    batch_size: int | None,
    num_steps: int | None,
    single_gpu: bool,
    subdir: str,
    pred_id: int,
    inference_anneal_slope: float | None,
    inference_anneal_offset: float | None,
    inference_anneal_types: bool,
    inference_anneal_coords: bool,
    inference_anneal_lattice: bool,
) -> None:
    cfg, model = load_model(checkpoint)

    # update cfg
    if num_steps is not None:
        cfg.integrate.num_steps = num_steps
    if inference_anneal_slope is not None:
        cfg.integrate.inference_anneal_slope = inference_anneal_slope
    if inference_anneal_offset is not None:
        assert (0 <= inference_anneal_offset) and (inference_anneal_offset < 1)
        cfg.integrate.inference_anneal_offset = inference_anneal_offset

    cfg.integrate.inference_anneal_types = inference_anneal_types
    cfg.integrate.inference_anneal_coords = inference_anneal_coords
    cfg.integrate.inference_anneal_lattice = inference_anneal_lattice

    with open(atom_types_path, "r") as f:
        atom_types = f.read()
    atom_types = eval(atom_types)
    dataset = CSPDataset(atom_types)
    loader = DataLoader(dataset, batch_size=batch_size)

    target_dir = get_target_dir(checkpoint, subdir)

    directories = [f"pred_{pred_id:02d}"]

    for directory in directories:
        pred_writer = TorchPredictionWriter(
            output_dir=target_dir / directory,
            write_interval="epoch",
        )
        # save num_steps
        (target_dir / directory / "num_steps.txt").write_text(
            str(cfg.integrate.num_steps)
        )

        if single_gpu:
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=[pred_writer],
            )
        else:
            trainer = pl.Trainer(
                accelerator="gpu",
                strategy="ddp",
                devices="auto",
                callbacks=[pred_writer],
            )
        trainer.predict(
            model,
            dataloaders=loader,
            return_predictions=False,
            ckpt_path=checkpoint,
        )


def _get_consolidated_path(directory: Path, task: str) -> str:
    return directory / f"consolidated_{task}.pt"


def _list_of_dicts_to_dict_of_lists(
    lod: list[dict[str, torch.Tensor | Batch]],
    keys_to_ignore: tuple[str] = (),
) -> dict[str, list[torch.Tensor] | list[Data]]:
    out = {k: [] for k in lod[0].keys()}
    for key, val in out.items():
        for d in lod:
            if key in keys_to_ignore:
                continue
            elif isinstance(d[key], Batch):
                val.append(d[key].to_data_list())
            else:
                val.append(d[key])
    return out


def _consolidate(
    target_dir: Path, task: TASKS_TYPE
) -> dict[str, list[dict[str, torch.Tensor | list[Data] | list[int]]]]:
    pattern = f"{task}_??"
    directories = sorted(list(target_dir.glob(pattern)))

    if not directories:
        print(f"no! directories found with the pattern {pattern}")
        return None
    else:
        print(f"yes directories found with the pattern {pattern}")

    # to a single array
    num_stepss = []
    out_by_eval = []
    for _, directory in enumerate(directories):
        preds = sorted(list(directory.glob("predictions_??.pt")))
        batches = sorted(list(directory.glob("batch_indices_??.pt")))
        assert len(preds) == len(batches)

        # get and empty out
        keys = torch.load(preds[0], map_location="cpu")[0][0].keys()
        out = {k: [] for k in keys}
        order = []

        # collect from output, keeping track of order
        for pred, batch in zip(preds, batches):
            pred = torch.load(pred, map_location="cpu")  # 1, 1, dict
            batch = torch.load(batch, map_location="cpu")  # 1, 1, batch_size

            for pp, bb in zip(pred, batch):
                for p, b in zip(pp, bb):
                    for k, v in p.items():
                        if isinstance(v, torch.Tensor):
                            out[k].append(v)
                        elif isinstance(v, Data):
                            out[k].extend(v.to_data_list())
                        else:
                            raise TypeError("don't know what to do with that type.")
                    order.extend(b)

        # organize out into our datatype
        for k, v in out.items():
            if k == "input_data_batch":
                out[k] = Batch.from_data_list(v)
            elif k in ["atom_types", "frac_coords", "lattices", "lengths", "angles"]:
                if "trajectory" in task:
                    out[k] = torch.concat(v, dim=1)
                else:
                    out[k] = torch.concat(v, dim=0)
            elif k == "num_atoms":
                out[k] = torch.concat(v, dim=0)
            else:
                raise ValueError(f"don't know what to do with {k=}")
        num_stepss.append(eval((target_dir / directory / "num_steps.txt").read_text()))
        out["batch_indices"] = torch.tensor(order)
        out_by_eval.append(out)

    # make sure all num_steps are the same
    assert all(
        [num_stepss[0] == i for i in num_stepss]
    ), f"not all num_steps agreed, got {num_stepss=}"
    num_steps = num_stepss[0]

    # save num_steps
    (_get_consolidated_path(target_dir, task).parent / "num_steps.txt").write_text(
        str(num_steps)
    )
    out_by_eval = _list_of_dicts_to_dict_of_lists(out_by_eval)
    torch.save(out_by_eval, _get_consolidated_path(target_dir, task))
    return out_by_eval


def _create_eval_pt(
    consolidated: dict[str, torch.Tensor] | None,
    target_dir: Path,
    filename: str,
) -> Path:
    # consolidated = {k: v.reshape(-1, v.shape[-1]) if k != "lattices" else v.reshape(-1, *v.shape[-2:]) for k, v in r.items()}
    if consolidated is None:
        raise ValueError(
            f"you cannot try to save an eval_pt with no data, {consolidated=}"
        )
    consolidated = {k: v[0] for k, v in consolidated.items()}
    consolidated["eval_setting"] = None
    path = target_dir / filename
    torch.save(consolidated, path)
    return path


@cli.command()
@click.argument("checkpoint", type=Path)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--path_eval_pt",
    show_default=True,
    default=None,
    help="select a path to save the eval_pt, otherwise it is not saved",
)
@click.option(
    "--task_to_save",
    default=None,
    help="if it is ambiguous, you can select the task for path_eval_pt",
)
def consolidate(
    checkpoint: Path,
    subdir: str,
    path_eval_pt: str | None,
    task_to_save: TASKS_TYPE | None,
) -> None:
    target_dir = get_target_dir(checkpoint, subdir)
    r = _consolidate(target_dir, "reconstruct")
    rt = _consolidate(target_dir, "recon_trajectory")
    g = _consolidate(target_dir, "generate")
    gt = _consolidate(target_dir, "gen_trajectory")
    p = _consolidate(target_dir, "pred")

    consolidations = {k: v for k, v in zip(TASKS, [r, rt, g, gt, p])}
    did_consolidate = {k: v != None for k, v in consolidations.items()}

    if any(did_consolidate.values()):
        if task_to_save is not None:
            print(f"consolidating {task_to_save}")
            consolidated = consolidations[task_to_save]
        elif sum(did_consolidate.values()) == 1:
            consolidated_task = list(consolidations.keys())[
                list((did_consolidate.values())).index(True)
            ]
            print(f"only {consolidated_task} was consolidated")
            consolidated = consolidations[consolidated_task]
        else:
            raise ValueError(
                f"more than one task was consolidated, so you must specify which one to save as eval_pt. FYI: {did_consolidate=}"
            )
    else:
        raise ValueError(
            "nothing was consolidated, so the program cannot print an eval for dft"
        )

    if sum(did_consolidate.values()) > 0 and path_eval_pt is not None:
        path = _create_eval_pt(consolidated, target_dir, path_eval_pt)
        print("eval_pt:")
        print(path)


def _reconstruction_metrics_wandb(
    target_dir: Path,
    consolidated_reconstruction_path: Path,
    global_step: int,
    stage: STAGE_TYPE,
) -> dict[str, float]:
    recon_metrics = {}
    if consolidated_reconstruction_path.exists():
        # should be 20 evals
        tmp, num_evals = compute_reconstruction_metrics(
            consolidated_reconstruction_path,
            multi_eval=True,
            metrics_path=target_dir / f"old_eval_metrics_reconstruct_multi.json",
            ground_truth_path=None,  # unnecessary since we save this when we consolidate
        )
        recon_metrics.update(
            {f"{stage}/" + k + f"_{num_evals:02d}": v for k, v in tmp.items()}
        )

        # should be 01 eval
        tmp, num_evals = compute_reconstruction_metrics(
            consolidated_reconstruction_path,
            multi_eval=False,
            metrics_path=target_dir / f"old_eval_metrics_reconstruct_single.json",
            ground_truth_path=None,  # unnecessary since we save this when we consolidate
        )
        recon_metrics.update(
            {f"{stage}/" + k + f"_{num_evals:02d}": v for k, v in tmp.items()}
        )  # this will be 01 evals

        recon_num_steps = eval(
            (consolidated_reconstruction_path.parent / "num_steps.txt").read_text()
        )
        recon_metrics.update({f"{stage}/recon_num_steps": recon_num_steps})
        recon_metrics.update({"trainer/global_step": global_step})
    return recon_metrics


def _generation_metrics_wandb(
    target_dir: Path,
    consolidated_generation_path: Path,
    gt_dataset_path: Path,
    global_step: int,
    eval_model_name: Literal["carbon", "mp20", "perovskite"],
    n_subsamples: int,
    stage: STAGE_TYPE,
) -> dict[str, float]:
    gen_metrics = {}
    if consolidated_generation_path.exists():
        tmp = compute_generation_metrics(
            path=consolidated_generation_path,
            metrics_path=target_dir / f"old_eval_metrics_generate.json",
            ground_truth_path=gt_dataset_path,
            eval_model_name=eval_model_name,
            n_subsamples=n_subsamples,
        )
        gen_metrics.update({f"{stage}/" + k: v for k, v in tmp.items()})
        gen_num_steps = eval(
            (consolidated_generation_path.parent / "num_steps.txt").read_text()
        )
        gen_metrics.update({f"{stage}/gen_n_subsamples": n_subsamples})
        gen_metrics.update({f"{stage}/gen_num_steps": gen_num_steps})
        gen_metrics.update({"trainer/global_step": global_step})
    return gen_metrics


@cli.command(name="old_eval_metrics")
@click.argument("checkpoint", type=Path)
@click.option(
    "--do_not_log_wandb",
    is_flag=True,
    show_default=True,
    default=False,
    help="do not log results in the wandb training run",
)
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option(
    "--gen_subsamples",
    type=int,
    default=1_000,
    help="gen metrics are on this many subsamples",
)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
def old_eval_metrics(
    checkpoint: Path,
    do_not_log_wandb: bool,
    subdir: str,
    gen_subsamples: int,
    stage: STAGE_TYPE,
) -> None:
    log_wandb = not do_not_log_wandb
    target_dir = get_target_dir(checkpoint, subdir)

    chkp = torch.load(checkpoint)
    global_step = chkp["global_step"]

    print(f"")
    print(f"======= reconstruction =======")
    print(f"")
    consolidated_reconstruction_path = target_dir / _get_consolidated_path(
        target_dir, "reconstruct"
    )
    if consolidated_reconstruction_path.exists():
        recon_metrics = _reconstruction_metrics_wandb(
            target_dir, consolidated_reconstruction_path, global_step, stage
        )
    else:
        recon_metrics = {}
        print(f"{consolidated_reconstruction_path=} not found")
        print(f"Not performing reconstruction metrics!")

    print(f"")
    print(f"======= generation =======")
    print(f"")

    consolidated_generation_path = target_dir / _get_consolidated_path(
        target_dir, "generate"
    )
    if consolidated_generation_path.exists():
        cfg = load_cfg(checkpoint)
        gen_metrics = _generation_metrics_wandb(
            target_dir=target_dir,
            consolidated_generation_path=consolidated_generation_path,
            gt_dataset_path=cfg.data.datamodule.datasets[stage][0].save_path,
            global_step=global_step,
            eval_model_name=cfg.data.eval_model_name,
            n_subsamples=gen_subsamples,
            stage=stage,
        )
    else:
        gen_metrics = {}
        print(f"{consolidated_generation_path=} not found")
        print(f"Not performing generation metrics!")

        print(f"")

    if (
        not consolidated_reconstruction_path.exists()
        and not consolidated_generation_path.exists()
    ):
        raise FileNotFoundError(
            f"Neither {consolidated_reconstruction_path=} nor {consolidated_generation_path=} exists."
        )

    if log_wandb:
        cfg = load_cfg(checkpoint)
        wandb_config = cfg.logging.wandb
        wandb_config.project = load_project_from_wandb(checkpoint)
        wandb_config.group = load_group_from_wandb(checkpoint)
        wandb_config.job_type = "cdvae_metrics"
        wandb_config.tags = [
            load_date_from_wandb(checkpoint),
            load_id_from_wandb(checkpoint),
        ]
        wandb_config = dict(wandb_config)
        del wandb_config["log_model"]
        wandb.init(**wandb_config)
        wandb.log(recon_metrics, global_step)
        wandb.log(gen_metrics, global_step)
        wandb.finish()


@cli.command(name="lattice_metrics")
@click.argument("checkpoint", type=Path)
# @click.option(
#     "--do_not_log_wandb",
#     is_flag=True,
#     show_default=True,
#     default=False,
#     help="do not log results in the wandb training run",
# )
@click.option(
    "--subdir", type=str, default="", help="subdir name at level of checkpoint"
)
@click.option("--stage", type=click.Choice(STAGES, case_sensitive=False), default="val")
def lattice_metrics(
    checkpoint: Path,
    # do_not_log_wandb: bool,
    subdir: str,
    stage: STAGE_TYPE,
) -> None:
    # log_wandb = not do_not_log_wandb
    target_dir = get_target_dir(checkpoint, subdir)

    chkp = torch.load(checkpoint)
    global_step = chkp["global_step"]

    consolidated_reconstruction_path = target_dir / _get_consolidated_path(
        target_dir, "reconstruct"
    )
    if consolidated_reconstruction_path.exists():
        compute_lattice_metrics(
            consolidated_reconstruction_path,
            metrics_path=target_dir / f"lattice_metrics_reconstruct_single.json",
        )  # right now this returns nothing since it just plots the distribution

    consolidated_generation_path = target_dir / _get_consolidated_path(
        target_dir, "generate"
    )
    if consolidated_generation_path.exists():
        cfg = load_cfg(checkpoint)
        compute_lattice_metrics(
            consolidated_generation_path,
            metrics_path=target_dir / f"lattice_metrics_generate.json",
            ground_truth_path=cfg.data.datamodule.datasets[stage][0].save_path,
        )  # right now this returns nothing since it just plots the distribution

    if (
        not consolidated_reconstruction_path.exists()
        and not consolidated_generation_path.exists()
    ):
        raise FileNotFoundError(
            f"Neither {consolidated_reconstruction_path=} nor {consolidated_generation_path=} exist."
        )


if __name__ == "__main__":
    cli()
