"""Copyright (c) Meta Platforms, Inc. and affiliates."""

"""
There are several relevant settings for DFT.
1. The MP20 settings. This is what we use to evaluate CDVAE and DiffCSP which were trained on that data.

2. The OCP settings. These will be used to evaluate models trained on larger datasets than MP20.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm

from flowmm.pandas_ import filter_prerelaxed, maybe_get_missing_columns
from flowmm.pymatgen_ import COLUMNS_COMPUTATIONS, get_nary, to_structure

CHGNET_STABILITY_THRESHOLD = 0.1


def parse_vasp(path: Path | str):
    path = Path(path)
    vrun = Vasprun(filename=path)
    vrun.final_structure
    vrun.final_energy


def write_vasp_directory(
    structure: Structure | dict,
    path: Path | str,
    potcar_spec: bool = False,
) -> None:
    path = Path(path)
    structure = to_structure(structure)
    relax_set = MPRelaxSet(structure=structure)
    try:
        relax_set.write_input(
            output_dir=str(path.resolve()),
            make_dir_if_not_present=True,
            potcar_spec=potcar_spec,
        )
    except (TypeError, KeyError) as exp:
        print(f"{exp}")
    return None


def subsample_by_nary(
    df: pd.DataFrame, num_structures: int
) -> tuple[pd.DataFrame, dict[int, float]]:
    assert num_structures <= len(df)
    weight_map = (df.groupby("nary").count()["e_relax"] / len(df)).to_dict()
    df["weights"] = df["nary"].map(weight_map)
    return df.sample(num_structures, weights=df["weights"]), weight_map


def subsample(df: pd.DataFrame, num_structures: int) -> pd.DataFrame:
    return df.sample(num_structures)


def get_equal_weights(df: pd.DataFrame) -> dict[int, float]:
    return {i: 1 / df["nary"].max() for i in range(1, df["nary"].max() + 1)}


def main(args: Namespace) -> None:
    print(f"{args.root=}")

    # easier to reason with positives...
    filter_converged = not args.do_not_filter_converged

    # load data
    print("reading json")
    df = pd.read_json(args.json_in)
    print(f"{len(df)=} loaded")
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
    print(f"{len(df)=} get missing cols")
    df = filter_prerelaxed(
        df,
        filter_converged=filter_converged,
        maximum_nary=args.maximum_nary,
    )
    print(f"{len(df)=} filtered prerelaxed")
    # no potential for these
    df = df[~df["chemsys"].map(lambda x: "Ra" in x)]
    df = df[~df["chemsys"].map(lambda x: "At" in x)]
    df = df[~df["chemsys"].map(lambda x: "Po" in x)]
    print(f"{len(df)=} removed structures without labels")
    df["nary"] = df["chemsys"].map(get_nary)

    if args.ehulls is not None:
        df_hull = pd.read_json(args.ehulls)
        df = df.join(df_hull, how="inner")
        # filter out high energy structures
        df = df[df["e_above_hull_per_atom_chgnet"] < CHGNET_STABILITY_THRESHOLD]

    print("finished")

    num_structures = min(args.max_num_structures, len(df))
    print(f"In the end, producing {num_structures=} dft input files")

    # choose subsampling stragegy
    if args.subsample is None:
        pass
    elif args.subsample == "first":
        df = df[:num_structures]
    elif args.subsample == "random":
        print("subsample randomly with equal weight")
        df = subsample(df, num_structures)
        weights = get_equal_weights(df)
    elif args.subsample == "nary":
        print("subsample weighted by nary")
        df, weights = subsample_by_nary(df, num_structures)
    else:
        raise ValueError()
    print(f"{len(df)=}")

    # save basics
    args.root.mkdir(parents=True, exist_ok=True)
    with open(args.root / "args.yaml", "w") as f:
        to_save = deepcopy(vars(args))
        to_save["json_in"] = str(to_save["json_in"])
        to_save["root"] = str(to_save["root"])
        yaml.dump(to_save, f)
    with open(args.root / "weights.yaml", "w") as f:
        yaml.dump(weights, f)
    df.drop("structure", axis=1).to_csv(args.root / "dataframe.csv")

    Parallel(n_jobs=args.n_jobs)(
        delayed(write_vasp_directory)(structure, args.root / f"{i:06d}")
        for i, structure in tqdm(df["structure"].items())
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_in", type=Path, help="prerelaxed dataframe")
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--subsample",
        type=str,
        choices=["first", "random", "nary"],
        default="random",
    )
    parser.add_argument("--max_num_structures", type=int, default=10_000)
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument(
        "--ehulls",
        type=Path,
        default=None,
        help="if given, filter to chgnet e_above_hull_per_atom < 0.01",
    )  # an alternative is to triage with chgnet instead of cutoff
    parser.add_argument("--do_not_filter_converged", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    main(args)
