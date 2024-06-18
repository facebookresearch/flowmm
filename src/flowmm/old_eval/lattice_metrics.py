"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from flowmm.old_eval.core import (
    CrysArrayListType,
    get_crystal_array_list,
    load_gt_crystal_array_list,
)


def get_lengths_angles(
    crys_array_list: CrysArrayListType,
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.stack([ca["lengths"] for ca in crys_array_list])
    angles = np.stack([ca["angles"] for ca in crys_array_list])
    return lengths, angles


def create_data_dict(
    measurement: np.ndarray, name: str, is_gt: bool
) -> dict[str, np.ndarray | list[bool]]:
    labels = [f"{name}_{i}" for i in range(1, 4)] * len(measurement)
    return {
        "measurement": labels,
        "value": measurement.flatten(),
        "is_gt": [is_gt for _ in range(len(measurement.flatten()))],
    }


def get_df(
    measurement: np.ndarray,
    name: str,
    is_gt: bool,
) -> pd.DataFrame:
    return pd.DataFrame.from_dict(create_data_dict(measurement, name, is_gt))


def get_dfs_lengths_angles(
    lengths: np.ndarray,
    angles: np.ndarray,
    gt_lengths: np.ndarray | None = None,
    gt_angles: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_length = get_df(lengths, "length", is_gt=False)
    if gt_lengths is not None:
        gt_df_length = get_df(gt_lengths, "length", is_gt=True)
        df_length = pd.concat([df_length, gt_df_length])

    df_angle = get_df(angles, "angle", is_gt=False)
    if gt_angles is not None:
        gt_df_angle = get_df(gt_angles, "angle", is_gt=True)
        df_angle = pd.concat([df_angle, gt_df_angle])
    return df_length, df_angle


def plot_lengths_angles_histograms(
    lengths: np.ndarray,
    angles: np.ndarray,
    gt_lengths: np.ndarray | None = None,
    gt_angles: np.ndarray | None = None,
) -> tuple[sns.FacetGrid, sns.FacetGrid]:
    df_length, df_angle = get_dfs_lengths_angles(lengths, angles, gt_lengths, gt_angles)

    g_length = sns.FacetGrid(df_length, row="measurement", hue="is_gt")
    g_length.map(sns.histplot, "value", alpha=0.7)

    angle_range = (40, 140.0)
    g_angle = sns.FacetGrid(df_angle, row="measurement", hue="is_gt", xlim=angle_range)
    g_angle.map(sns.histplot, "value", alpha=0.7, bins=50, binrange=angle_range)
    return g_length, g_angle


def pairplot(
    lengths: np.ndarray,
    angles: np.ndarray,
    gt_lengths: np.ndarray | None = None,
    gt_angles: np.ndarray | None = None,
) -> tuple[sns.FacetGrid, sns.FacetGrid]:
    df_length, df_angle = get_dfs_lengths_angles(lengths, angles, gt_lengths, gt_angles)
    df = pd.concat([df_length, df_angle])

    # if I ever want to break it up into more selected pieces
    # g = sns.PairGrid(penguins)
    # g.map_diag(sns.histplot)
    # g.map_offdiag(sns.scatterplot)

    # g_angle = sns.FacetGrid(df_angle, row="measurement", hue="is_gt", xlim=(0, 180.0))
    # g_angle.map(sns.histplot, "value", alpha=0.7, bins=18)
    return sns.pairplot(df, hue="is_gt", kind="hist", corner=True)


def compute_lattice_metrics(
    path: Path,
    metrics_path: Path,
    ground_truth_path: Path | None = None,
) -> dict[str, float]:
    # get list of structures
    crys_array_list, true_crystal_array_list = get_crystal_array_list(path, batch_idx=0)
    if ground_truth_path is not None:
        true_crystal_array_list = load_gt_crystal_array_list(ground_truth_path)
    pred_lengths, pred_angles = get_lengths_angles(crys_array_list)
    gt_lengths, gt_angles = get_lengths_angles(true_crystal_array_list)

    # plotting
    g_length, g_angle = plot_lengths_angles_histograms(
        pred_lengths, pred_angles, gt_lengths, gt_angles
    )
    g_length.savefig(metrics_path.parent / "lengths.png")
    g_angle.savefig(metrics_path.parent / "angles.png")

    # g = pairplot(pred_lengths, pred_angles, gt_lengths, gt_angles)
    # g.savefig(metrics_path.parent / "pairplot.png")

    # TODO make metrics
    return {}
