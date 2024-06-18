"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import io
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from toolz import compose

from flowmm.joblib_ import joblib_map
from flowmm.old_eval.core import save_metrics_only_overwrite_newly_computed
from flowmm.pandas_ import (
    filter_prerelaxed,
    get_intersection,
    maybe_get_missing_columns,
)
from flowmm.pymatgen_ import COLUMNS_COMPUTATIONS, get_chemsys, to_structure
from flowmm.tabular import VALID_STAGES, VALID_TABULAR_DATASETS, get_tabular_dataset

trap = io.StringIO()


def get_matches(
    structure: Structure, alternatives: pd.Series, matcher: StructureMatcher
) -> tuple[list[int], list[float]]:
    with redirect_stdout(trap):
        structure = to_structure(structure)
    matches, rms_dists = [], []
    for ind, alt in alternatives.items():
        with redirect_stdout(trap):
            alt_structure = to_structure(alt)
        rms_dist = matcher.get_rms_dist(structure, alt_structure)
        if rms_dist is not None:
            rms_dist, *_ = rms_dist
            rms_dists.append(rms_dist)
            matches.append(ind)
    return matches, rms_dists


def main(args: Namespace) -> None:
    df = pd.read_json(args.json_in)
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)

    if args.ehulls is not None:
        df_hull = pd.read_json(args.ehulls)
        df = df.join(df_hull, how="inner")
        # filter out high energy structures
        df = df[df[args.e_above_hull_column] <= args.e_above_hull_maximum]

    df = filter_prerelaxed(
        df,
        args.num_structures,
        maximum_nary=args.maximum_nary,
        minimum_nary=args.minimum_nary - 1,
    )

    path_json_sun_count = args.json_out.parent / args.json_sun_count
    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_stable": len(df)}
    )

    matcher = StructureMatcher()  # MatterGen Novelty settings
    # matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)  # CDVAE settings

    # uniqueness
    matches_rms_dists_s = joblib_map(
        lambda structure: get_matches(structure, df["structure"], matcher),
        df["structure"].array,
        n_jobs=-4,
        inner_max_num_threads=1,
        desc="Matching for uniqueness",
        total=len(df),
    )
    # place those lists into a dataframe
    records = []
    for j, (matches, rms_dists) in enumerate(matches_rms_dists_s):
        assert len(matches) == len(rms_dists)
        ind_self = df.index[j]

        if len(matches) == 0:
            record = {
                f"uniq_match_ind-0": pd.NA,
                f"uniq_rms_dist_to-0": float("nan"),
            }
        elif len(matches) == 1:
            if ind_self == matches[0]:
                record = {
                    f"uniq_match_ind-0": pd.NA,
                    f"uniq_rms_dist_to-0": float("nan"),
                }
            else:
                print(
                    f"did not match self! Matched {ind_self} to {matches[0]} with RMSD {rms_dists[0]}"
                )
                record = {
                    f"uniq_match_ind-0": matches[0],
                    f"uniq_rms_dist_to-0": rms_dists[0],
                }
        else:
            record = {}
            for i, (match, rms_dist) in enumerate(zip(matches, rms_dists)):
                if ind_self == match:
                    record[f"uniq_match_ind-{i}"] = pd.NA
                    record[f"uniq_rms_dist_to-{i}"] = float("nan")
                else:
                    record[f"uniq_match_ind-{i}"] = pd.NA if np.isnan(match) else match
                    record[f"uniq_rms_dist_to-{i}"] = rms_dist
        records.append(record)
    uniq_out = pd.DataFrame.from_records(records, index=df.index)
    uniq_match_cols = [col for col in uniq_out.columns if col.startswith("uniq_match")]
    uniq_out[uniq_match_cols] = uniq_out[uniq_match_cols].astype("Int64")

    # load tabular data to compare to
    tds = get_tabular_dataset(args.tabular_dataset)

    # novelty
    outs = []
    for stage in VALID_STAGES:
        if args.reprocess:
            tds.process(stage)
        tabular: pd.DataFrame = getattr(tds, stage + "_df")
        # compositions must match to compare the resulting structure
        gen_chemsys = df["composition"].map(compose(tuple, sorted, get_chemsys))
        tab_chemsys = tabular["composition"].map(compose(tuple, sorted, get_chemsys))
        intersection = get_intersection(gen_chemsys, tab_chemsys)
        gen_to_compare = df["structure"][gen_chemsys.isin(intersection)]
        tab_to_compare = tab_chemsys.isin(intersection)

        # now do pairwise comparisons between these filtered groups
        matches_rms_dists_s = joblib_map(
            lambda structure: get_matches(
                structure, tabular["cif"][tab_to_compare], matcher
            ),
            gen_to_compare.array,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="Matching for novelty",
            total=len(gen_to_compare),
        )

        # place those lists into a dataframe
        records = []
        for matches, rms_dists in matches_rms_dists_s:
            assert len(matches) == len(rms_dists)
            if len(matches) == 0:
                record = {
                    f"match_ind_{stage}-0": pd.NA,
                    f"rms_dist_to_{stage}-0": float("nan"),
                }
            else:
                record = {}
                for i, (match, rms_dist) in enumerate(zip(matches, rms_dists)):
                    record[f"match_ind_{stage}-{i}"] = match
                    record[f"rms_dist_to_{stage}-{i}"] = rms_dist
            records.append(record)
        out = pd.DataFrame.from_records(records, index=gen_to_compare.index)
        outs.append(out)
    out = pd.concat(outs, axis=1)
    out = pd.concat([uniq_out, out], axis=1)

    print(f"{len(df)=}")
    print(f"{len(out)=}")
    not_in_train = out[out["match_ind_train-0"].isna()]
    print(f"{len(not_in_train)=}")

    # remove duplicates that are not in the training set
    has_a_generated_dupe = pd.concat(
        [
            ~not_in_train[col].isna()
            for col in not_in_train.columns
            if col.startswith("uniq_match")
        ],
        axis=1,
    ).any(axis=1)
    not_in_train_is_dupe = not_in_train[has_a_generated_dupe]
    # mark the duplicates, avoiding the first one that appears
    dupes = []
    cols = [col for col in not_in_train_is_dupe.columns if col.startswith("uniq_match")]
    for i, row in not_in_train_is_dupe[cols].iterrows():
        if i not in dupes:
            dupes.extend(row.array.dropna().tolist())

    sun_materials = not_in_train.drop(dupes)
    print(f"{len(sun_materials)=}")

    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_sun_materials": len(sun_materials)}
    )

    out["sun"] = False
    out.loc[sun_materials.index, "sun"] = True
    out.to_json(args.json_out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_in", type=Path, help="prerelaxed dataframe")
    parser.add_argument("json_out", type=Path, help="new dataframe")
    parser.add_argument(
        "--tabular_dataset",
        type=str,
        choices=VALID_TABULAR_DATASETS,
        default="diffcsp_mp20",
    )
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument("--slurm_partition", type=str, default="ocp")
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,  # we know there aren't structures in the dataset with more than this
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument(
        "--minimum_nary",
        type=int,
        default=2,
        help="Any queries to structures with lower nary are avoided.",
    )
    parser.add_argument("--ehulls", type=str, default=None)
    parser.add_argument(
        "--e_above_hull_column", type=str, default="e_above_hull_per_atom_dft_corrected"
    )
    parser.add_argument("--e_above_hull_maximum", type=float, default=0.0)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--json_sun_count", type=str, default="sun_count.json")
    args = parser.parse_args()

    main(args)
