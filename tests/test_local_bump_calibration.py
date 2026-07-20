from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tortuosite_score.vessels_detection.local_bump_calibration import (
    PAIRWISE_COLUMNS,
    V2Parameters,
    bootstrap_mean_interval,
    evaluate_pairwise_scores,
    leave_one_patient_out_calibration,
    pairwise_label_template,
    select_best_parameters,
    validate_pairwise_labels,
    write_pairwise_label_template,
)


class LocalBumpCalibrationTests(unittest.TestCase):
    def test_template_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_pairwise_label_template(Path(tmp) / "pairs.xlsx")
            template = pd.read_excel(path, sheet_name="Expert_pairs")
            with pd.ExcelFile(path) as workbook:
                sheet_names = workbook.sheet_names

        self.assertEqual(template.columns.tolist(), PAIRWISE_COLUMNS)
        self.assertEqual(sheet_names, ["Expert_pairs", "Instructions"])
        self.assertEqual(pairwise_label_template().columns.tolist(), PAIRWISE_COLUMNS)

    def test_validation_rejects_unknown_judgment(self) -> None:
        labels = pd.DataFrame(
            [["1_OD", "a", "1_OD", "b", "maybe"]],
            columns=PAIRWISE_COLUMNS,
        )

        with self.assertRaisesRegex(ValueError, "Invalid judgments"):
            validate_pairwise_labels(labels)

    def test_pairwise_metrics_separate_strict_and_similar_pairs(self) -> None:
        labels = validate_pairwise_labels(
            pd.DataFrame(
                [
                    ["1_OD", "a", "1_OD", "b", "left"],
                    ["2_OD", "a", "2_OD", "b", "similar"],
                ],
                columns=PAIRWISE_COLUMNS,
            )
        )
        scores = {("1_OD", "a"): 3.0, ("1_OD", "b"): 1.0, ("2_OD", "a"): 2.0, ("2_OD", "b"): 2.2}

        metrics, details = evaluate_pairwise_scores(labels, scores)

        self.assertEqual(metrics["strict_concordance"], 1.0)
        self.assertEqual(metrics["similar_pair_count"], 1)
        self.assertGreater(metrics["mean_similar_relative_gap"], 0.0)
        self.assertEqual(len(details), 2)

    def test_parameter_selection_and_patient_held_out_report(self) -> None:
        labels = validate_pairwise_labels(
            pd.DataFrame(
                [
                    ["1_OD", "a", "1_OD", "b", "left"],
                    ["2_OD", "a", "2_OD", "b", "right"],
                    ["3_OD", "a", "3_OD", "b", "similar"],
                ],
                columns=PAIRWISE_COLUMNS,
            )
        )
        x = np.linspace(0.0, 200.0, 120)
        geometries = {
            ("1_OD", "a"): np.column_stack([x, 9.0 * np.sin(x / 8.0)]),
            ("1_OD", "b"): np.column_stack([x, 2.0 * np.sin(x / 20.0)]),
            ("2_OD", "a"): np.column_stack([x, 1.5 * np.sin(x / 20.0)]),
            ("2_OD", "b"): np.column_stack([x, 8.0 * np.sin(x / 8.0)]),
            ("3_OD", "a"): np.column_stack([x, 4.0 * np.sin(x / 12.0)]),
            ("3_OD", "b"): np.column_stack([x, 4.1 * np.sin(x / 12.0)]),
        }
        grid = [V2Parameters(), V2Parameters(4.0, 7, 0.20, 0.40)]

        best, evaluation = select_best_parameters(labels, geometries, grid)
        summary, folds, details = leave_one_patient_out_calibration(
            labels,
            geometries,
            grid,
            bootstrap_samples=100,
        )

        self.assertIn(best, grid)
        self.assertEqual(len(evaluation), 2)
        self.assertEqual(summary["method"], "local_bump_v2")
        self.assertEqual(summary["parameter_grid_size"], 2)
        self.assertFalse(folds.empty)
        self.assertFalse(details.empty)

    def test_bootstrap_interval_is_deterministic(self) -> None:
        first = bootstrap_mean_interval([0.0, 0.2, 0.4], samples=200, random_seed=7)
        second = bootstrap_mean_interval([0.0, 0.2, 0.4], samples=200, random_seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])


if __name__ == "__main__":
    unittest.main()
