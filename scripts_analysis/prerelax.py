"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import submitit
import torch
from chgnet.model import CHGNet

from flowmm.chgnet_ import RelaxationData, prerelax_with_chgnet
from flowmm.pymatgen_ import cdvae_to_structure, diffcsp_to_structure, get_get_structure

#
# Here's a really nice script for CHGNet
# https://github.com/janosh/matbench-discovery/blob/main/models/chgnet/test_chgnet.py
#


def wait_for_jobs_to_finish(jobs: list, sleep_time_s: int = 5) -> None:
    # wait for the job to be finished
    num_finished = 0
    print(f"number of jobs: {len(jobs):02d}")
    while num_finished < len(jobs):
        time.sleep(sleep_time_s)
        num_finished = sum(job.done() for job in jobs)
        print(f"number finished: {num_finished:02d}", flush=True, end="\r")
    print("")
    print("jobs done!")
    return None


def prerelax_multiple(
    gen_file: Path,
    steps: int,
    index: np.ndarray,
    json_file: Path,
) -> pd.DataFrame:
    """cdvae structures, max relaxation steps, list of indexes -> json file"""
    chgnet = CHGNet.load()
    device = next(chgnet.parameters()).device
    gen = torch.load(gen_file, map_location=device)

    for k, v in gen.items():
        if isinstance(v, torch.Tensor):
            gen[k] = v.squeeze(0)

    rd = RelaxationData()
    for i in index.tolist():
        get_structure = get_get_structure(gen, clip_atom_types=False)
        try:
            structure = get_structure(gen, i)
            pair = prerelax_with_chgnet(structure=structure, chgnet=chgnet, steps=steps)
            rd.index.append(i)
            rd.e_gen.append(pair.energies[0])
            rd.e_relax.append(pair.energies[1])
            rd.n_to_relax.append(pair.n_steps_to_relax)
            rd.rms_dist.append(pair.rms_dist)
            rd.matched.append(pair.match)
            rd.converged.append(True if pair.n_steps_to_relax < steps else False)
            rd.exception.append(False)
            rd.num_sites.append(structure.num_sites)
            rd.structure.append(pair.structure_dicts[-1])
        except Exception as exp:
            print(exp)
            rd.index.append(i)
            rd.e_gen.append(None)
            rd.e_relax.append(None)
            rd.n_to_relax.append(pd.NA)
            rd.rms_dist.append(None)
            rd.matched.append(False)
            rd.converged.append(False)
            rd.exception.append(True)
            rd.num_sites.append(pd.NA)
            rd.structure.append({})

    # dataclasses.fields(rd) produces blank :(
    # dataclasses.asdict(rd) also produces blank
    pre_pandas = dict(
        (field.name, getattr(rd, field.name))
        for field in getattr(rd, "__dataclass_fields__").values()
    )

    if "batch_indices" in gen.keys():
        pre_pandas["index"] = gen["batch_indices"][index].tolist()
    else:
        print("no batch_indices given so assuming order is correct")

    df = pd.DataFrame.from_dict(pre_pandas).set_index("index")
    df.loc[:, "e_delta"] = np.abs(df.loc[:, "e_relax"] - df.loc[:, "e_gen"])
    df.to_json(json_file)
    return df


def main(args: Namespace) -> None:
    # get file
    gen_file = Path(args.path_to_structures)
    print(f"reading file: {str(gen_file.resolve())}")

    # determine how many structures there are
    num_structures = torch.load(gen_file)["num_atoms"].squeeze().numel()
    if args.num_structures is not None:
        assert args.num_structures <= num_structures
        num_structures = args.num_structures
    print(f"{num_structures=}")

    # split structures into parts
    index = np.arange(num_structures)
    indexes = np.array_split(index, args.num_jobs)
    indexes = [i for i in indexes if i.size > 0]  # filter out blanks

    # compute the energies
    files = [
        Path(args.log_dir) / f"{index.min():07d}-{index.max():07d}.json"
        for index in indexes
    ]

    # set cluster
    cluster = "local" if args.slurm_partition is None else "slurm"
    if args.debug:
        cluster = "debug"

    executor = submitit.AutoExecutor(
        folder=args.log_dir,
        cluster=cluster,
    )
    executor.update_parameters(
        slurm_array_parallelism=args.num_jobs,
        nodes=1,
        slurm_ntasks_per_node=1,
        cpus_per_task=4,
        gpus_per_node=0,
        timeout_min=args.timeout_min,
        slurm_partition=args.slurm_partition,
    )
    doit = partial(prerelax_multiple, gen_file, args.steps)
    jobs = executor.map_array(doit, indexes, files)

    wait_for_jobs_to_finish(jobs)

    # write to file
    df = pd.concat([job.result() for job in jobs])
    df.to_json(Path(args.path_to_json))
    print(f"wrote file to: ")
    print(f"{args.path_to_json}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "path_to_structures",
        type=Path,
        help="path to the eval_pt, typically `eval_for_dft.pt`, file from cdvae / diffcsp / rfm",
    )
    parser.add_argument("path_to_json", type=Path, help="output")
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="number of jobs to divide structures between, only works when --slurm_partition is set",
    )
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument(
        "--steps", type=int, default=1_500, help="maximum number of steps in relaxation"
    )
    parser.add_argument("--timeout_min", type=int, default=300)
    parser.add_argument("--slurm_partition", type=str, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run the prerelaxations sequentially in the same process as this script.",
    )
    args = parser.parse_args()

    main(args)
