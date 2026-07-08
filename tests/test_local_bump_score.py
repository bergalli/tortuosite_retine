from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.draw import line

from tortuosite_score.vessels_detection.local_bump_score import (
    LocalBumpSettings,
    local_bump_metrics,
    score_run,
    score_saved_vessels,
    score_root_to_leaf_systems,
    summarize_eye_score,
)


class LocalBumpMetricTests(unittest.TestCase):
    def test_straight_line_scores_near_zero(self) -> None:
        points = [[x, 0.0] for x in np.linspace(0, 200, 80)]

        score = local_bump_metrics(points)

        self.assertAlmostEqual(score["branch_bump_score"], 0.0)
        self.assertEqual(score["oscillation_count"], 0.0)

    def test_repeated_bumps_score_higher_than_smooth_curve(self) -> None:
        x = np.linspace(0, 240, 160)
        bumpy = np.column_stack([x, 8.0 * np.sin(x / 7.0)])
        theta = np.linspace(0, math.pi / 2, 160)
        smooth_curve = np.column_stack([150.0 * np.cos(theta), 150.0 * np.sin(theta)])

        bumpy_score = local_bump_metrics(bumpy)
        smooth_score = local_bump_metrics(smooth_curve)

        self.assertGreater(bumpy_score["oscillation_count"], smooth_score["oscillation_count"])
        self.assertGreater(bumpy_score["branch_bump_score"], smooth_score["branch_bump_score"] * 5.0)

    def test_one_pixel_jitter_does_not_dominate_bumps(self) -> None:
        x = np.linspace(0, 240, 160)
        jitter = np.column_stack([x, ((np.arange(len(x)) % 2) * 2 - 1) * 0.45])
        bumpy = np.column_stack([x, 8.0 * np.sin(x / 7.0)])

        settings = LocalBumpSettings(curvature_threshold=0.035)
        jitter_score = local_bump_metrics(jitter, settings)
        bumpy_score = local_bump_metrics(bumpy, settings)

        self.assertLess(jitter_score["branch_bump_score"], bumpy_score["branch_bump_score"] * 0.25)

    def test_eye_summary_uses_tail_component(self) -> None:
        systems = pd.DataFrame(
            {
                "eligible": [True, True, True],
                "system_bump_score": [0.01, 0.01, 0.12],
                "system_length": [300.0, 300.0, 300.0],
                "arc_chord_tortuosity": [1.0, 1.0, 1.0],
                "category": ["artere", "veine", "artere"],
                "branch_count": [1, 1, 1],
                "system_id": [1, 2, 3],
                "oscillation_count": [1.0, 1.0, 5.0],
            }
        )

        summary = summarize_eye_score(systems)

        self.assertGreater(summary["all_vessels_tail_score"], summary["all_vessels_score"])
        self.assertGreater(summary["eye_tortuosity_score"], summary["all_vessels_score"])

    def test_bifurcation_scores_complete_root_to_leaf_systems(self) -> None:
        branches = pd.DataFrame(
            {
                "branch_id": [1, 2, 3],
                "path_points": [
                    [[0, 0], [120, 0]],
                    [[120, 0], [240, -80]],
                    [[120, 0], [240, 80]],
                ],
                "vascx_category": ["artere", "artere", "artere"],
            }
        )
        branches.attrs["root_hint"] = (0.0, 0.0)

        systems = score_root_to_leaf_systems(branches, LocalBumpSettings(min_system_length=100.0))

        self.assertEqual(len(systems), 2)
        self.assertTrue((systems["branch_count"] >= 2).all())
        self.assertTrue((systems["system_length"] > 200).all())

    def test_tiny_high_score_fragment_is_not_eligible_system(self) -> None:
        x = np.linspace(0, 80, 80)
        branches = pd.DataFrame(
            {
                "branch_id": [1],
                "path_points": [np.column_stack([x, 8.0 * np.sin(x / 4.0)]).tolist()],
                "vascx_category": ["artere"],
            }
        )

        systems = score_root_to_leaf_systems(branches, LocalBumpSettings(min_system_length=250.0))

        self.assertEqual(len(systems), 1)
        self.assertFalse(bool(systems.loc[0, "eligible"]))

    def test_small_compatible_gap_is_bridged(self) -> None:
        branches = pd.DataFrame(
            {
                "branch_id": [1, 2],
                "path_points": [
                    [[0, 0], [100, 0]],
                    [[110, 0], [220, 0]],
                ],
                "vascx_category": ["artere", "artere"],
            }
        )

        systems = score_root_to_leaf_systems(
            branches,
            LocalBumpSettings(min_system_length=100.0, bridge_tolerance=15.0),
        )

        self.assertGreaterEqual(int(systems["bridge_count"].max()), 1)


class SavedRunSmokeTests(unittest.TestCase):
    def test_score_run_reads_existing_outputs_without_vascx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "1_OD"
            output_dir = run_dir / "output"
            output_dir.mkdir(parents=True)
            mask = np.zeros((80, 120), dtype=np.uint8)
            rr, cc = line(40, 10, 40, 110)
            mask[rr, cc] = 255
            cv2.imwrite(str(output_dir / "06_cleaned_mask.png"), mask)
            cv2.imwrite(str(output_dir / "06_cleaned_artery_mask.png"), mask)
            cv2.imwrite(str(output_dir / "06_cleaned_vein_mask.png"), np.zeros_like(mask))
            pd.DataFrame(
                {
                    "image-coord-src-0": [40],
                    "image-coord-src-1": [10],
                    "image-coord-dst-0": [40],
                    "image-coord-dst-1": [110],
                    "branch-distance": [100.0],
                    "euclidean-distance": [100.0],
                    "tortuosity": [1.0],
                }
            ).to_csv(output_dir / "08_full_skeleton_summary.csv", index=False)
            (run_dir / "manual_review_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "selected_segment_refs": [],
                        "manual_segments": {},
                        "vessels": {
                            "auto_vascx_1": {
                                "category": "artere",
                                "segment_refs": ["model:0"],
                                "synthetic_links": [],
                                "start_endpoint": {
                                    "kind": "geometry_point",
                                    "point": [10, 40],
                                    "segment_ref": "model:0",
                                    "distance_from_start": 0.0,
                                },
                                "end_endpoint": {
                                    "kind": "geometry_point",
                                    "point": [110, 40],
                                    "segment_ref": "model:0",
                                    "distance_from_start": 100.0,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary, systems = score_run(run_dir, LocalBumpSettings(min_system_length=50.0))

        self.assertEqual(summary["run"], "1_OD")
        self.assertEqual(summary["eye_number"], 1)
        self.assertEqual(summary["eligible_vessel_count"], 1)
        self.assertEqual(systems["vessel_name"].tolist(), ["auto_vascx_1"])
        self.assertTrue(math.isfinite(summary["eye_tortuosity_score"]))

    def test_saved_vessel_scoring_excludes_short_vessels(self) -> None:
        branches = pd.DataFrame(
            {
                "branch_id": [0],
                "path_points": [[[0, 0], [20, 0]]],
                "branch-distance": [20.0],
                "euclidean-distance": [20.0],
                "tortuosity": [1.0],
                "vascx_category": ["artere"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "manual_review_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "selected_segment_refs": [],
                        "manual_segments": {},
                        "vessels": {
                            "short": {
                                "category": "artere",
                                "segment_refs": ["model:0"],
                                "synthetic_links": [],
                                "start_endpoint": {"kind": "geometry_point", "point": [0, 0]},
                                "end_endpoint": {"kind": "geometry_point", "point": [20, 0]},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            scores = score_saved_vessels(run_dir, branches, LocalBumpSettings(min_saved_vessel_length=100.0))

        self.assertEqual(len(scores), 1)
        self.assertFalse(bool(scores.loc[0, "eligible"]))


if __name__ == "__main__":
    unittest.main()
