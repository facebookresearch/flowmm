"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from flowmm.old_eval.core import (
    get_Crystal_obj_lists,
    save_metrics_only_overwrite_newly_computed,
)


class RecEval(object):
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def process_one(self, pred, gt, is_valid):
        if not is_valid:
            return None
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return None

    def get_match_rate_and_rms(self):
        validity = [c1.valid and c2.valid for c1, c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(self.process_one(self.preds[i], self.gts[i], validity[i]))
        # # I think this is slower?
        # rms_dists = joblib_map(
        #     lambda xx: self.process_one(*xx),
        #     zip(self.preds, self.gts, validity),
        #     n_jobs=-4,
        #     inner_max_num_threads=1,
        # )
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(RecEval):
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = self.process_one(
                    self.preds[j][i], self.gts[i], self.preds[j][i].valid
                )
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}


def compute_reconstruction_metrics(
    path: Path,
    multi_eval: bool,
    metrics_path: Path,
    ground_truth_path: Path | None = None,
) -> tuple[dict[str, float], int]:
    all_metrics = {}
    pred_crys, gt_crys, num_evals = get_Crystal_obj_lists(
        path,
        multi_eval,
        ground_truth_path,
    )
    if multi_eval:
        rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
    else:
        rec_evaluator = RecEval(pred_crys, gt_crys)
    recon_metrics = rec_evaluator.get_metrics()
    all_metrics.update(recon_metrics)

    print(all_metrics)

    save_metrics_only_overwrite_newly_computed(metrics_path, all_metrics)
    return all_metrics, num_evals
