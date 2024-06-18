"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from ase.io.trajectory import Trajectory
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry, PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Poscar
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


def get_compat_params(dft_path):
    incar_file = os.path.join(dft_path, "INCAR")
    poscar_file = os.path.join(dft_path, "POSCAR")
    incar = Incar.from_file(incar_file)
    poscar = Poscar.from_file(poscar_file)
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"
    return param


def get_energy_correction(traj, dft_path):
    ase_atoms = traj[-1]
    params = get_compat_params(dft_path)
    struct = AseAtomsAdaptor.get_structure(ase_atoms)
    cse_d = {
        "structure": struct,
        "energy": ase_atoms.get_potential_energy(),
        "correction": 0.0,
        "parameters": params,
    }
    cse = ComputedStructureEntry.from_dict(cse_d)
    out = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        # verbose=True,
        clean=True,
    )
    pde = PDEntry(composition=cse.composition, energy=cse.energy)
    return (cse, cse.energy, pde)


def apply_name_fn_dict(
    obj: any,
    name_fn: dict[str, callable],
) -> dict[str, any]:
    return {prop: fn(obj) for prop, fn in name_fn.items()}


def get_record(
    file: Path,
    dft_path: Path,
) -> dict[str, any]:
    record = {}
    record.update(apply_name_fn_dict(file, NAME_FN_FOR_PATH))
    traj = Trajectory(file)
    record.update(apply_name_fn_dict(traj, NAME_FN_FOR_TRAJ))
    _, file_id = os.path.split(file)
    file_id = file_id.split(".")[0]
    dft_files = os.path.join(dft_path, file_id)
    cse, corrected_energy, pde = get_energy_correction(traj, dft_files)
    record["corrected_energy"] = corrected_energy
    record["computed_structure_entry"] = cse.as_dict()
    record["phase_diagram_entry"] = pde.as_dict()

    return record


def get_dft_results(
    root: Path,
    # n_jobs: int = 1
) -> pd.DataFrame:
    dft_path = os.path.join(root, "dft")
    output_path = Path(os.path.join(root, "clean_outputs"))
    files: list[Path] = list(output_path.glob("*.traj"))
    # records = Parallel(n_jobs=n_jobs)(delayed(get_record)(file) for file in files)
    records = [get_record(file, dft_path) for file in tqdm(files)]
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


def get_e_hull_per_atom_from_pymatgen(
    phase_diagram: PhaseDiagram | PatchedPhaseDiagram,
    pde: PDEntry | dict,
) -> tuple[dict, float]:
    """returns e_hull_per_atom"""
    pde = PDEntry.from_dict(pde)
    try:
        out = phase_diagram.get_decomp_and_e_above_hull(pde, allow_negative=True)
    except (ValueError, AttributeError, ZeroDivisionError):
        out = ({}, float("nan"))
    return out


def main(args: Namespace) -> None:
    # load the data to compare to the hull
    print("readying json_in")
    df = pd.read_json(args.json_in)
    print("potentially getting missing columns")
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
    print("filtering to those which are prerelaxed")
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

    df_dft = get_dft_results(args.root_dft_clean_outputs)
    decomp_and_e = [
        get_e_hull_per_atom_from_pymatgen(ppd_mp, p)
        for p in df_dft["phase_diagram_entry"]
    ]

    decomposition, e_above_hull_dft_per_atom_corrected = zip(*decomp_and_e)
    df_dft["e_above_hull_per_atom_dft_corrected"] = list(
        e_above_hull_dft_per_atom_corrected
    )
    df_dft["decomposition"] = list(decomposition)

    if args.method is not None:
        df_dft["method"] = args.method
    df_dft = df_dft[df_dft["original_index"].isin(df.index)]
    df_dft = df_dft.set_index("original_index")
    out["e_above_hull_per_atom_dft"] = df_dft["e_per_atom_dft"] - out["e_hull_per_atom"]
    out["e_above_hull_per_atom_dft_initial"] = (
        df_dft["e_per_atom_dft_initial"] - out["e_hull_per_atom"]
    )

    out["e_above_hull_per_atom_dft_corrected"] = df_dft[
        "e_above_hull_per_atom_dft_corrected"
    ]
    out["decomposition"] = df_dft["decomposition"]

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
        "--root_dft_clean_outputs",
        type=Path,
        default=None,
        help="root dir which contains a folder called `dft` and `clean_outputs`.",
        required=True,
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
