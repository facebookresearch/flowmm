"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from ase.io.trajectory import Trajectory
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

from flowmm.pandas_ import filter_prerelaxed, maybe_get_missing_columns
from flowmm.pymatgen_ import COLUMNS_COMPUTATIONS, to_structure

EntriessType = list[list[ComputedStructureEntry]]
EntryDictssType = list[list[dict]]
PATH_PPD_MP = Path(__file__).parents[1] / "mp_02072023/2023-02-07-ppd-mp.pkl"


NAME_FN_FOR_PATH = {
    "original_index": lambda path: int(path.stem.split("_")[-1]),
    "method": lambda path: path.stem.rsplit("_", maxsplit=1)[0],
}

NAME_FN_FOR_TRAJ = {
    "energy_initial": lambda traj: traj[0].get_potential_energy(),
    "energy": lambda traj: traj[-1].get_potential_energy(),
    "forces": lambda traj: traj[-1].get_forces(),
    # "stress": lambda traj: traj[-1].get_stress(),
    # "mag_mom": lambda traj: traj[-1].get_magnetic_moment(),
    "num_sites": lambda traj: traj[-1].get_global_number_of_atoms(),
}


def apply_name_fn_dict(
    obj: any,
    name_fn: dict[str, callable],
) -> dict[str, any]:
    return {prop: fn(obj) for prop, fn in name_fn.items()}


def get_record(
    file: Path,
) -> dict[str, any]:
    record = {}
    record.update(apply_name_fn_dict(file, NAME_FN_FOR_PATH))
    traj = Trajectory(file)
    record.update(apply_name_fn_dict(traj, NAME_FN_FOR_TRAJ))
    return record


def get_dft_results(
    root: Path,
    # n_jobs: int = 1
) -> pd.DataFrame:
    files: list[Path] = list(root.glob("*.traj"))
    # records = Parallel(n_jobs=n_jobs)(delayed(get_record)(file) for file in files)
    records = [get_record(file) for file in tqdm(files)]
    df = pd.DataFrame.from_records(records)
    df["method"] = df["method"].map(
        {
            "cdvae": "cdvae",
            "diffcsp_mp20": "diffcsp_mp20",
            "diffscp_mp20": "diffcsp_mp20",  # spelling error
        }
    )
    df["e_per_atom_dft"] = df["energy"] / df["num_sites"]
    df["e_per_atom_dft_initial"] = df["energy_initial"] / df["num_sites"]
    return df


def get_patched_phase_diagram_mp(path: Path) -> PatchedPhaseDiagram:
    with open(path, "rb") as f:
        ppd_mp = pickle.load(f)
    return ppd_mp


def get_e_hull_from_phase_diagram(
    phase_diagram: PhaseDiagram | PatchedPhaseDiagram,
    structure: Structure | dict,
) -> float:
    """returns e_hull_per_atom"""
    structure = to_structure(structure)
    try:
        return phase_diagram.get_hull_energy_per_atom(structure.composition)
    except (ValueError, AttributeError, ZeroDivisionError):
        return float("nan")


def main(args: Namespace) -> None:
    # load the data to compare to the hull
    print("readying json_in")
    df = pd.read_json(args.json_in)
    print("potentially getting missing columns")
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
    print(f"filtering to those which are prerelaxed")
    if args.maximum_nary is not None:
        print(f"and maximum nary={args.maximum_nary}")
    df = filter_prerelaxed(
        df,
        args.num_structures,
        maximum_nary=args.maximum_nary,
    )

    print(f"loading the saved mp phase diagram at {PATH_PPD_MP=}")
    ppd_mp = get_patched_phase_diagram_mp(PATH_PPD_MP)
    e_hulls = [get_e_hull_from_phase_diagram(ppd_mp, s) for s in df["structure"]]
    out = pd.DataFrame(data={"e_hull_per_atom": e_hulls})
    out.index = df.index  # this works because we filtered out exceptions above!

    out["e_above_hull_per_atom_chgnet_gen"] = (df["e_gen"] / df["num_sites"]) - out[
        "e_hull_per_atom"
    ]
    out["e_above_hull_per_atom_chgnet"] = (df["e_relax"] / df["num_sites"]) - out[
        "e_hull_per_atom"
    ]
    if args.clean_outputs_dir is not None:
        df_dft = get_dft_results(args.clean_outputs_dir)
        if args.method is not None:
            df_dft["method"] = args.method
        df_dft = df_dft[df_dft["original_index"].isin(df.index)]
        df_dft = df_dft.set_index("original_index")
        out["e_above_hull_per_atom_dft"] = (
            df_dft["e_per_atom_dft"] - out["e_hull_per_atom"]
        )
        out["e_above_hull_per_atom_dft_initial"] = (
            df_dft["e_per_atom_dft_initial"] - out["e_hull_per_atom"]
        )

    # write to file
    out.to_json(Path(args.json_out))
    print(f"wrote file to: ")
    print(f"{args.json_out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_in", type=Path, help="prerelaxed dataframe")
    parser.add_argument("json_out", type=Path, help="new dataframe")
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument(
        "--clean_outputs_dir",
        type=Path,
        default=None,
        help="root dir for vasp clean_outputs",
    )
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument("--method", type=str, default=None)

    args = parser.parse_args()

    main(args)
