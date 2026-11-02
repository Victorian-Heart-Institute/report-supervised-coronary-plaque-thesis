"""CPU-only checks using fabricated data; no clinical files or checkpoints."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "analysis"))
from contracts import validate_manifest
from coronary_pooling import territory_for_target, weighted_pool
from metrics import summarize, validation_thresholds
from bootstrap_core import weighted_aucs
from bootstrap import calculate


class MetricTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "scan_id": ["toy_a", "toy_b", "toy_c", "toy_d", "toy_e"],
                "patient_id": ["group_a", "group_a", "group_b", "group_c", "group_d"],
                "label": [0.0, 0.0, 1.0, 1.0, np.nan],
                "p_label": [0.1, 0.2, 0.8, 0.9, 0.99],
            }
        )

    def test_unknowns_confusion_and_separation(self):
        row = summarize(self.frame(), ["label"]).iloc[0]
        self.assertEqual((row.TP, row.TN, row.FP, row.FN), (2, 2, 0, 0))
        self.assertEqual((row.known, row.unknown), (4, 1))
        self.assertEqual(row.AUROC, 1.0)
        self.assertAlmostEqual(row.delta_median, 0.7)
        self.assertEqual(row.balanced_accuracy, 1.0)

    def test_all_positive_is_not_discrimination(self):
        frame = self.frame()
        frame["p_label"] = 0.9
        row = summarize(frame, ["label"]).iloc[0]
        self.assertEqual(row.AUROC, 0.5)
        self.assertEqual(row.balanced_accuracy, 0.5)
        self.assertEqual(row.delta_median, 0.0)

    def test_validation_threshold(self):
        value = validation_thresholds(self.frame(), ["label"])["label"]
        self.assertGreater(value, 0.2)
        self.assertLessEqual(value, 0.8)

    def test_weighted_cluster_auc_matches_expansion(self):
        frame = self.frame()
        patients, inverse = np.unique(frame.patient_id, return_inverse=True)
        weights = np.array([[2, 1, 2, 0], [1, 2, 1, 1]])
        got = weighted_aucs(frame, ["label"], inverse, weights)[:, 0]
        for i in range(len(weights)):
            sample = frame.iloc[np.repeat(np.arange(len(frame)), weights[i, inverse])]
            known = sample.label.notna()
            expected = roc_auc_score(
                sample.loc[known, "label"], sample.loc[known, "p_label"]
            )
            self.assertAlmostEqual(got[i], expected)

    def test_paired_identical_has_zero_interval(self):
        result = calculate(self.frame(), ["label"], repeats=100, paired=self.frame())
        self.assertEqual(
            (result["point"], result["ci95_lower"], result["ci95_upper"]),
            (0.0, 0.0, 0.0),
        )
        self.assertGreater(result["excluded_one_class_replicates"], 0)


class PoolingTests(unittest.TestCase):
    def test_empty_roi_falls_back_to_global_mean(self):
        features = torch.arange(16.0).reshape(1, 2, 2, 2, 2)
        got = weighted_pool(features, torch.zeros(1, 1, 4, 4, 4))
        torch.testing.assert_close(got, features.mean((2, 3, 4)))

    def test_roi_changes_only_feature_aggregation(self):
        features = torch.tensor([0.0, 10.0]).reshape(1, 1, 1, 1, 2)
        roi = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 1, 2)
        torch.testing.assert_close(weighted_pool(features, roi), torch.tensor([[10.0]]))

    def test_anatomical_routing(self):
        self.assertEqual(territory_for_target("pLAD_calcified"), "LCA")
        self.assertEqual(territory_for_target("mRCA_non_calcified"), "RCA")
        self.assertEqual(territory_for_target("PDA_calcified"), "union")


class ContractTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "scan_id": ["toy_train", "toy_val", "toy_test"],
                "patient_id": ["group_train", "group_val", "group_test"],
                "split": ["train", "val", "test"],
                "cohort": ["fabricated"] * 3,
                "label": [1.0, 0.0, np.nan],
            }
        )

    def test_valid_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.csv"
            frame = self.frame()
            frame.to_csv(path, index=False)
            self.assertEqual(len(validate_manifest(frame, path, ["label"])), 64)
            frame.loc[1, "patient_id"] = "group_train"
            with self.assertRaises(ValueError):
                validate_manifest(frame, path, ["label"])

    def test_invalid_label_and_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.csv"
            frame = self.frame()
            frame.to_csv(path, index=False)
            frame.loc[0, "label"] = 2
            with self.assertRaises(ValueError):
                validate_manifest(frame, path, ["label"])
            frame = self.frame()
            frame.loc[0, "scan_id"] = "../escape"
            with self.assertRaises(ValueError):
                validate_manifest(frame, path, ["label"])


class PackagingTests(unittest.TestCase):
    def test_twenty_configs_and_target_order(self):
        targets = json.loads((ROOT / "docs/TARGETS.json").read_text())
        paths = list((ROOT / "configs").glob("*.json"))
        self.assertEqual(len(paths), 20)
        for path in paths:
            config = json.loads(path.read_text())
            self.assertEqual(config["targets"], targets[config["task"]])
            contract = json.loads((ROOT / config["contract"]).read_text())
            self.assertEqual(contract["targets"], config["targets"])

    def test_python_syntax(self):
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
