from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tortuosite_score.vessels_detection.local_bump_score import LocalBumpSettings
from tortuosite_score.vessels_detection.synthetic_tortuosity_benchmark import (
    _spearman_or_nan,
    amplitude_family,
    frequency_family,
    global_bend_family,
    noise_robustness_report,
    preprocessing_robustness_report,
    run_benchmark,
    score_curve,
    sensitivity_and_concordance_report,
)


class CurveFamilyTests(unittest.TestCase):
    def test_families_produce_geometrically_valid_polylines(self) -> None:
        for family in (amplitude_family(), frequency_family(), global_bend_family()):
            for sample in family.samples:
                self.assertGreaterEqual(len(sample.points), 4)
                segment_lengths = np.linalg.norm(np.diff(sample.points, axis=0), axis=1)
                self.assertTrue(np.all(np.isfinite(segment_lengths)))
                self.assertGreater(float(segment_lengths.sum()), 0.0)

    def test_amplitude_family_is_zero_at_baseline_and_increases_for_grisan(self) -> None:
        settings = LocalBumpSettings()
        family = amplitude_family()

        baseline_score = score_curve(family.samples[0].points, "grisan", settings)
        final_score = score_curve(family.samples[-1].points, "grisan", settings)

        self.assertEqual(family.samples[0].control_value, 0.0)
        self.assertAlmostEqual(baseline_score, 0.0, places=6)
        self.assertGreater(final_score, baseline_score)

    def test_frequency_family_is_zero_at_baseline_and_increases_for_tortuosity_density(self) -> None:
        from tortuosite_score.vessels_detection.local_bump_score import tortuosity_density_metrics

        settings = LocalBumpSettings()
        family = frequency_family()

        baseline_score = tortuosity_density_metrics(family.samples[0].points, settings)["tortuosity_density_score"]
        final_score = tortuosity_density_metrics(family.samples[-1].points, settings)["tortuosity_density_score"]

        self.assertEqual(family.samples[0].control_value, 0.0)
        self.assertAlmostEqual(float(baseline_score), 0.0, places=6)
        self.assertGreater(float(final_score), float(baseline_score))


class SensitivityConcordanceTests(unittest.TestCase):
    def test_spearman_or_nan_matches_expected_values_on_toy_sequences(self) -> None:
        self.assertAlmostEqual(
            _spearman_or_nan(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])), 1.0
        )
        self.assertAlmostEqual(
            _spearman_or_nan(np.array([4.0, 3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])), -1.0
        )
        self.assertTrue(math.isnan(_spearman_or_nan(np.array([5.0, 5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0]))))
        self.assertTrue(math.isnan(_spearman_or_nan(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 5.0, 5.0, 5.0]))))

    def test_global_bend_family_flags_arc_chord_but_not_curvature_based_methods(self) -> None:
        report = sensitivity_and_concordance_report(
            [global_bend_family()],
            method_ids=("arc_chord", "grisan", "tortuosity_density"),
        )
        by_method = report.set_index("method_id")

        # arc/chord conflates a smooth, large-scale bend with tortuosity (the paper's own
        # critique, Sec. I/VI): its score rises monotonically with bend angle alone.
        self.assertGreater(float(by_method.loc["arc_chord", "spearman_vs_control_value"]), 0.9)
        # grisan and tortuosity_density do not show that same spurious positive correlation.
        self.assertLessEqual(float(by_method.loc["grisan", "spearman_vs_control_value"]), 0.0)
        self.assertLessEqual(float(by_method.loc["tortuosity_density", "spearman_vs_control_value"]), 0.0)

    def test_amplitude_and_frequency_families_report_strong_concordance_for_all_methods(self) -> None:
        report = sensitivity_and_concordance_report(
            [amplitude_family(), frequency_family()],
            method_ids=("local_bump", "arc_chord", "curvature_squared", "tortuosity_density", "grisan"),
        )

        self.assertTrue((report["spearman_vs_control_value"] > 0.9).all())
        self.assertTrue((report["valid_fraction"] == 1.0).all())


class NoiseRobustnessTests(unittest.TestCase):
    def test_zero_noise_is_deterministic_and_matches_the_clean_ranking(self) -> None:
        report = noise_robustness_report(
            amplitude_family(),
            method_ids=("arc_chord",),
            sigmas=(0.0,),
            repetitions=5,
        )

        row = report.iloc[0]
        self.assertEqual(row["repetition_count"], 5)
        self.assertAlmostEqual(float(row["median_spearman_vs_clean_ranking"]), 1.0)
        self.assertAlmostEqual(float(row["std_spearman_vs_clean_ranking"]), 0.0)

    def test_noise_report_is_reproducible_for_a_fixed_seed(self) -> None:
        first = noise_robustness_report(
            amplitude_family(), method_ids=("grisan",), sigmas=(1.0,), repetitions=8, seed=3
        )
        second = noise_robustness_report(
            amplitude_family(), method_ids=("grisan",), sigmas=(1.0,), repetitions=8, seed=3
        )

        pd.testing.assert_frame_equal(first, second)


class PreprocessingRobustnessTests(unittest.TestCase):
    def test_arc_chord_has_a_single_variant_with_undefined_coefficient_of_variation(self) -> None:
        report = preprocessing_robustness_report(amplitude_family(), method_ids=("arc_chord",))

        self.assertEqual(len(report), 1)
        self.assertTrue(math.isnan(float(report.iloc[0]["representative_curve_coefficient_of_variation"])))

    def test_grisan_reports_one_row_per_smoothing_variant(self) -> None:
        report = preprocessing_robustness_report(amplitude_family(), method_ids=("grisan",))

        self.assertEqual(len(report), 5)
        self.assertTrue((report["spearman_vs_clean_ranking"] > 0.9).all())


class RunBenchmarkTests(unittest.TestCase):
    def test_run_benchmark_writes_the_expected_workbook_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = run_benchmark(tmp, noise_repetitions=3)
            workbook_path = Path(tmp) / "synthetic_tortuosity_benchmark.xlsx"

            self.assertTrue(workbook_path.exists())
            with pd.ExcelFile(workbook_path) as workbook:
                sheet_names = set(workbook.sheet_names)

        self.assertEqual(
            sheet_names,
            {"raw_scores", "sensitivity_concordance", "noise_robustness", "preprocessing_robustness"},
        )
        self.assertEqual(set(reports.keys()), sheet_names)
        self.assertFalse(reports["raw_scores"].empty)


if __name__ == "__main__":
    unittest.main()
