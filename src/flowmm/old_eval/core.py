"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

from diffcsp.eval_utils import (
    get_crystals_list,
    load_data,
    smact_validity,
    structure_validity,
)
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from flowmm.joblib_ import joblib_map
from flowmm.rfm.manifold_getter import ManifoldGetter
from flowmm.rfm.manifolds.analog_bits import analog_bits_to_int

CrysArrayListType = list[dict[str, np.ndarray]]

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")


class Crystal(object):
    def __init__(
        self,
        crys_array_dict,
        make_atom_types_discrete: bool = True,
    ):
        self.frac_coords = crys_array_dict["frac_coords"]
        self.atom_types = crys_array_dict["atom_types"]

        if make_atom_types_discrete and (self.frac_coords.ndim == self.atom_types.ndim):
            if self.atom_types.shape[-1] == NUM_ATOMIC_TYPES:
                self.atom_types = ManifoldGetter._inverse_atomic_one_hot(
                    self.atom_types
                )
            elif self.atom_types.shape[-1] == NUM_ATOMIC_BITS:
                self.atom_types = ManifoldGetter._inverse_atomic_bits(self.atom_types)
            else:
                raise ValueError()

        self.lengths = crys_array_dict["lengths"].squeeze()
        assert self.lengths.ndim == 1
        self.angles = crys_array_dict["angles"].squeeze()
        assert self.lengths.ndim == 1
        self.dict = crys_array_dict

        self.get_structure()
        if self.constructed:
            self.get_composition()
            self.get_validity()
            self.get_fingerprints()
        else:
            self.valid = False

    def get_structure(self):
        if min(self.lengths) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        # this catches validity failures down the line
        elif (1 > self.atom_types).any() or (self.atom_types > 104).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = "unrealistically_small_lattice"
            except TypeError:
                self.constructed = False
                self.invalid_reason = f"{self.atom_types=} are not possible"
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_file_paths(root_path, task, label="", suffix="pt"):
    if label == "":
        out_name = f"eval_{task}.{suffix}"
    else:
        out_name = f"eval_{task}_{label}.{suffix}"
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(
    file_path: Path,
    batch_idx: int = 0,
) -> CrysArrayListType:
    """batch_idx == -1, diffcsp format
    batch_idx == -2, cdvae format
    """
    data = load_data(str(file_path.resolve()))
    if batch_idx == -1:
        # batch_size = data["frac_coords"].shape[0]
        batch_size = len(data["frac_coords"])
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data["frac_coords"][i],
                data["atom_types"][i],
                data["lengths"][i],
                data["angles"][i],
                data["num_atoms"][i],
            )
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data["frac_coords"],
            data["atom_types"],
            data["lengths"],
            data["angles"],
            data["num_atoms"],
        )
    # elif batch_idx == -3:
    #     batch_size = data['frac_coords'].shape[0]
    #     crys_array_list = []
    #     for i in range(batch_size):
    #         tmp_crys_array_list = {k: v[i].detach().cpu().numpy() for k, v in data.items() if k != "input_data_batch"}
    #         crys_array_list.append(tmp_crys_array_list)
    else:
        crys_array_list = get_crystals_list(
            data["frac_coords"][batch_idx],
            data["atom_types"][batch_idx],
            data["lengths"][batch_idx],
            data["angles"][batch_idx],
            data["num_atoms"][batch_idx],
        )

    if "input_data_batch" in data:
        batch_by_eval = data["input_data_batch"]
        if isinstance(batch_by_eval, dict):
            true_crystal_array_list = get_crystals_list(
                batch_by_eval["frac_coords"],
                batch_by_eval["atom_types"],
                batch_by_eval["lengths"],
                batch_by_eval["angles"],
                batch_by_eval["num_atoms"],
            )
        elif hasattr(batch_by_eval, "frac_coords"):
            true_crystal_array_list = get_crystals_list(
                batch_by_eval.frac_coords,
                batch_by_eval.atom_types,
                batch_by_eval.lengths,
                batch_by_eval.angles,
                batch_by_eval.num_atoms,
            )
        elif isinstance(batch_by_eval, list) and not isinstance(batch_by_eval[0], Data):
            if isinstance(batch_by_eval[0][0], Data) and hasattr(
                batch_by_eval[0][0], "frac_coords"
            ):
                true_crystal_array_list = []
                features = ["frac_coords", "atom_types", "lengths", "angles"]

                # we collect crystals from the first eval
                crystals_in_first_eval = batch_by_eval[0]
                for crystal in crystals_in_first_eval:
                    true_crystal_array_list.append(
                        {k: crystal[k].detach().cpu().numpy() for k in features}
                    )

                # ... and make sure that every eval is in the same order
                for crystals_in_batch in batch_by_eval[1:]:
                    for i, crystal in enumerate(crystals_in_batch):
                        for k in features:
                            t = true_crystal_array_list[i][k]
                            other = crystal[k].detach().cpu().numpy()
                            assert np.allclose(
                                t, other
                            ), f"{t=} != {other=}, that means the first eval is not in the same order as the subsequent"
            else:
                true_crystal_array_list = None
                print("batch_by_eval[0][0]=", batch_by_eval[0][0])
                print("Testing order in generation mode is not supported.")
        else:
            true_crystal_array_list = None
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def cif_to_crys_array_dict(cif: str) -> dict[str, np.ndarray]:
    with HiddenPrints():
        structure = Structure.from_str(cif, fmt="cif", primitive=True)
    lattice = structure.lattice
    return {
        "frac_coords": structure.frac_coords,
        "atom_types": np.array([_.Z for _ in structure.species]),
        "lengths": np.array(lattice.abc),
        "angles": np.array(lattice.angles),
    }


def cif_to_crystal(cif: str) -> Crystal:
    return Crystal(cif_to_crys_array_dict(cif))


def load_gt_crystal_array_list(ground_truth_path: Path) -> CrysArrayListType:
    ground_truth_path = Path(ground_truth_path)
    if ground_truth_path.suffix == ".csv":
        csv = pd.read_csv(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crys_array_dict,
            csv["cif"],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(csv["cif"]),
        )
    elif ground_truth_path.suffix == ".pt":
        pkl = torch.load(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crys_array_dict,
            [p["cif"] for p in pkl],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(pkl),
        )
    else:
        raise ValueError(f"{ground_truth_path=} had an unrecognized suffix.")
    return gt_crys


def load_gt_crystals(ground_truth_path: Path) -> CrysArrayListType:
    ground_truth_path = Path(ground_truth_path)
    if ground_truth_path.suffix == ".csv":
        csv = pd.read_csv(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crystal,
            csv["cif"],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(csv["cif"]),
        )
    elif ground_truth_path.suffix == ".pt":
        pkl = torch.load(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crystal,
            [p["cif"] for p in pkl],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(pkl),
        )
    else:
        raise ValueError(f"{ground_truth_path=} had an unrecognized suffix.")
    return gt_crys


def safe_crystal(x: dict[str, np.ndarray]) -> Crystal:
    if np.all(50 < x["angles"]) and np.all(x["angles"] < 130):
        return Crystal(x)
    else:
        # returns an absurd crystal
        atom_types = np.zeros((1, 100))
        atom_types[0, -1] = 1
        return Crystal(
            {
                "frac_coords": np.zeros((1, 3)),
                "atom_types": atom_types,
                "lengths": 100 * np.ones((3,)),
                "angles": np.ones((3,)) * 90,
            }
        )


def get_Crystal_obj_lists(
    path: Path,
    multi_eval: bool,
    ground_truth_path: Path | None = None,
) -> tuple[list[Crystal], list[Crystal]]:
    batch_idx = -1 if multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        path,
        batch_idx=batch_idx,
    )

    if ground_truth_path is not None:
        gt_crys = load_gt_crystals(ground_truth_path)
    else:
        gt_crys = joblib_map(
            lambda x: Crystal(x),
            true_crystal_array_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(true_crystal_array_list),
        )

    # TIMEOUT: this MAYBE could be speed up (with timeout) using https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers
    # I could not find a way to have timeouts on single processes using joblib without crashing the whole array.
    # the issue is that if there's a crystal that does not process, the whole thing doesn't return...
    #
    # I tried this but it doesn't work
    # https://github.com/joblib/joblib/pull/366#issuecomment-267603530
    #
    # instead of using timeout, I just replace all crystals with unrealistic angles with a null crystal
    if multi_eval:
        pred_crys = []
        num_evals = len(crys_array_list)
        for i in range(num_evals):
            print(f"Processing batch {i}")
            pred_crys.append(
                joblib_map(
                    safe_crystal,  # instead I remove unrealistic crystals
                    crys_array_list[i],
                    n_jobs=-4,
                    inner_max_num_threads=1,
                    desc="pred_cryst",
                    total=len(crys_array_list),
                )
            )
    else:
        num_evals = 1
        pred_crys = joblib_map(
            safe_crystal,  # instead I remove unrealistic crystals
            crys_array_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="pred_cryst",
            total=len(crys_array_list),
        )

    return pred_crys, gt_crys, num_evals


def save_metrics_only_overwrite_newly_computed(
    path: Path, metrics: dict[str, float]
) -> None:
    # only overwrite metrics computed in the new run.
    if Path(path).exists():
        with open(path, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(metrics)
            else:
                with open(path, "w") as f:
                    json.dump(metrics, f)
        if isinstance(written_metrics, dict):
            with open(path, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(path, "w") as f:
            json.dump(metrics, f)
