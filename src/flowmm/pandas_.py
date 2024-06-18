"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import pandas as pd


def filter_prerelaxed(
    df: pd.DataFrame,
    num_structures: int | None = None,
    filter_exceptions: bool = True,
    filter_converged: bool = False,
    # maxima: dict[str, float | int] = {"e_delta": 15.0},
    maxima: dict[str, float | int] = {},
    maximum_nary: int | None = None,
    minimum_nary: int = 0,
) -> pd.DataFrame:
    if filter_exceptions:
        df = df[df["exception"] == False]

    if filter_converged:
        df = df[df["converged"] == True]
    for key, value in maxima.items():
        df = df[df[key] < value]

    if maximum_nary is not None:
        df = df[df["chemsys"].map(len) <= maximum_nary]
    df = df[df["chemsys"].map(len) > minimum_nary]

    if num_structures is not None:
        print(f"limiting to the first {num_structures} samples after filtering")
        df = df.iloc[:num_structures, :]

    return df


def maybe_get_missing_columns(
    df: pd.DataFrame, maps: dict[str, callable | dict]
) -> pd.DataFrame:
    for name, mapping in maps.items():
        if name not in df.columns:
            df[name] = mapping(df)
    return df


def get_intersection(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(list(set(a) & set(b)))
