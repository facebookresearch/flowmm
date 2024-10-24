"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import wasserstein_distance

from diffcsp.eval_utils import compute_cov, prop_model_eval
from flowmm.old_eval.core import (
    get_Crystal_obj_lists,
    save_metrics_only_overwrite_newly_computed,
)

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.0},
    "mp20_llama": {"struc": 0.4, "comp": 10.0},
    "carbon": {"struc": 0.2, "comp": 4.0},
    "perovskite": {"struc": 0.2, "comp": 4},
}


class GenEval(object):
    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False
            )
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f"not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}"
            )

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {"comp_valid": comp_valid, "struct_valid": struct_valid, "valid": valid}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {"wdist_density": wdist_density}

    def make_density_plot(self):
        import matplotlib.pyplot as plt

        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]

        fig, ax = plt.subplots()
        ax.hist(
            pred_densities,
            bins=20,
            alpha=0.5,
            range=(0, 20),
            label="Estimate",
            color="blue",
            density=True,
        )
        ax.hist(
            gt_densities,
            bins=20,
            alpha=0.5,
            range=(0, 20),
            label="Ground Truth",
            color="orange",
            density=True,
        )
        ax.legend()
        ax.xaxis.set_label_text("Density (g/cm^3)")
        return fig, ax

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species)) for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {"wdist_num_elems": wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(
                self.eval_model_name, [c.dict for c in self.valid_samples]
            )
            gt_props = prop_model_eval(
                self.eval_model_name, [c.dict for c in self.gt_crys]
            )
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {"wdist_prop": wdist_prop}
        else:
            return {"wdist_prop": None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys,
            self.gt_crys,
            struc_cutoff=cutoff_dict["struc"],
            comp_cutoff=cutoff_dict["comp"],
        )
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        print("prop_wdist is not going to happen")
        # metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics


@dataclass
class InvalidCrystal:
    comp_valid: bool = False
    struct_valid: bool = False
    valid: bool = False


def compute_generation_metrics(
    path: Path,
    metrics_path: Path,
    ground_truth_path: Path,
    eval_model_name: Literal["carbon", "mp20", "perovskite"],
    n_subsamples: int,
) -> dict[str, float]:
    all_metrics = {}
    gen_crys, gt_crys, _ = get_Crystal_obj_lists(
        path,
        multi_eval=False,
        ground_truth_path=ground_truth_path,
    )

    # here we drop absurd crystals and crystals that didn't construct
    gen_crys_out = []
    for gc in gen_crys:
        if not (gc.lengths == 100).all() and gc.constructed:
            gen_crys_out.append(gc)
        else:
            gen_crys_out.append(InvalidCrystal())

    gen_evaluator = GenEval(
        gen_crys_out, gt_crys, n_samples=n_subsamples, eval_model_name=eval_model_name
    )
    gen_metrics = gen_evaluator.get_metrics()
    # fig, ax = gen_evaluator.make_density_plot()
    all_metrics.update(gen_metrics)

    gnary = [len(set(c.atom_types.tolist())) for c in gen_crys]
    tnary = [len(set(c.atom_types.tolist())) for c in gt_crys]
    all_metrics.update({"wdist_nary": wasserstein_distance(gnary, tnary)})

    print(all_metrics)

    save_metrics_only_overwrite_newly_computed(metrics_path, all_metrics)
    return all_metrics
