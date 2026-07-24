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
    curvature_squared_metrics,
    external_angle_sum_metrics,
    local_bump_metrics,
    local_bump_v2_metrics,
    rdp_simplify,
    score_run,
    score_saved_vessels,
    score_root_to_leaf_systems,
    summarize_eye_score,
    tortuosity_density_metrics,
)
from tortuosite_score.vessels_detection.scoring import (
    available_scoring_methods,
    build_manual_vessels_export,
    build_review_scores_table,
    scoring_config,
    scoring_method_fixed_parameters,
    score_run as score_run_with_method,
    summarize_eye_score as summarize_saved_eye_score,
    score_saved_vessel,
)
from tortuosite_score.vessels_detection.segments import VesselSegment, build_segment_map


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

    def test_local_bump_v2_scores_persistent_waves_above_smooth_arc(self) -> None:
        x = np.linspace(0, 240, 180)
        waves = np.column_stack([x, 10.0 * np.sin(x / 9.0)])
        theta = np.linspace(0, math.pi / 2, 180)
        smooth_arc = np.column_stack([150.0 * np.cos(theta), 150.0 * np.sin(theta)])

        wave_score = local_bump_v2_metrics(waves)
        arc_score = local_bump_v2_metrics(smooth_arc)

        self.assertGreater(wave_score["persistent_lobe_count"], arc_score["persistent_lobe_count"])
        self.assertGreater(wave_score["local_bump_v2_score"], arc_score["local_bump_v2_score"])

    def test_local_bump_v2_excludes_endpoint_filter_artifact(self) -> None:
        points = [[0, 12], *[[x, 0] for x in range(4, 101, 4)]]

        metrics = local_bump_v2_metrics(points)

        self.assertGreater(metrics["endpoint_max_turn"], 0.0)
        self.assertEqual(metrics["persistent_oscillation_count"], 0.0)
        self.assertAlmostEqual(metrics["local_bump_v2_oscillation_component"], 0.0)

    def test_local_bump_v2_removes_small_alternating_lobes(self) -> None:
        x = np.linspace(0, 200, 160)
        small_jitter = np.column_stack([x, 0.10 * np.sin(x / 2.0)])

        metrics = local_bump_v2_metrics(small_jitter)

        self.assertEqual(metrics["persistent_lobe_count"], 0.0)
        self.assertGreaterEqual(metrics["local_bump_v2_score"], 0.0)

    def test_local_bump_v2_short_and_duplicate_paths_are_safe(self) -> None:
        short = local_bump_v2_metrics([[0, 0]])
        duplicate = local_bump_v2_metrics([[0, 0], [0, 0], [0, 0]])

        self.assertEqual(short["local_bump_v2_score"], 0.0)
        self.assertEqual(duplicate["local_bump_v2_score"], 0.0)
        self.assertTrue(math.isfinite(short["endpoint_max_turn"]))

    def test_curvature_squared_straight_line_scores_near_zero(self) -> None:
        points = [[0, 0], [10, 0], [20, 0], [30, 0]]

        score = curvature_squared_metrics(points, LocalBumpSettings(resample_step=1.0, smoothing_window=1))

        self.assertAlmostEqual(score["curvature_squared_score"], 0.0)
        self.assertAlmostEqual(score["curvature_squared_integral"], 0.0)
        self.assertAlmostEqual(score["mean_curvature"], 0.0)
        self.assertAlmostEqual(score["max_curvature"], 0.0)

    def test_curvature_squared_circular_arc_matches_inverse_radius_squared(self) -> None:
        radius = 50.0
        theta = np.linspace(0, math.pi / 2.0, 120)
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

        score = curvature_squared_metrics(points, LocalBumpSettings(resample_step=1.0, smoothing_window=1))

        self.assertAlmostEqual(score["curvature_squared_score"], 1.0 / (radius * radius), delta=8e-5)
        self.assertAlmostEqual(score["mean_curvature"], 1.0 / radius, delta=2e-3)

    def test_curvature_squared_short_or_degenerate_paths_return_safe_defaults(self) -> None:
        short_score = curvature_squared_metrics([[0, 0]], LocalBumpSettings(resample_step=1.0, smoothing_window=1))
        duplicate_score = curvature_squared_metrics([[0, 0], [0, 0]], LocalBumpSettings(resample_step=1.0, smoothing_window=1))

        self.assertEqual(short_score["curvature_squared_score"], 0.0)
        self.assertEqual(duplicate_score["curvature_squared_score"], 0.0)
        self.assertTrue(math.isfinite(short_score["curvature_squared_integral"]))
        self.assertTrue(math.isfinite(duplicate_score["curvature_squared_integral"]))

    def test_curvature_squared_resampling_can_be_disabled(self) -> None:
        points = [[0, 0], [20, 6], [40, -6], [60, 6], [80, 0]]

        raw_score = curvature_squared_metrics(points, LocalBumpSettings(resample_curvature_squared=False))
        explicit_raw_score = curvature_squared_metrics(
            points,
            LocalBumpSettings(resample_step=25.0, resample_curvature_squared=False),
        )
        resampled_score = curvature_squared_metrics(points, LocalBumpSettings(resample_step=4.0))

        self.assertEqual(raw_score, explicit_raw_score)
        self.assertNotEqual(raw_score["curvature_squared_score"], resampled_score["curvature_squared_score"])

    def test_tortuosity_density_is_zero_without_an_inflection(self) -> None:
        straight = tortuosity_density_metrics([[0, 0], [10, 0], [20, 0]])
        theta = np.linspace(0.0, math.pi / 2.0, 100)
        arc = tortuosity_density_metrics(np.column_stack([50.0 * np.cos(theta), 50.0 * np.sin(theta)]))

        for metrics in (straight, arc):
            self.assertTrue(bool(metrics["tortuosity_density_valid"]))
            self.assertEqual(metrics["constant_curvature_segment_count"], 1.0)
            self.assertEqual(metrics["inflection_count"], 0.0)
            self.assertAlmostEqual(float(metrics["tortuosity_density_score"]), 0.0)

    def test_tortuosity_density_matches_hand_calculated_two_bend_curve(self) -> None:
        points = [[0, 0], [1, 1], [2, 0], [3, -1], [4, 0]]
        settings = LocalBumpSettings(resample_step=math.sqrt(2.0), smoothing_window=1, curvature_threshold=0.01)

        metrics = tortuosity_density_metrics(points, settings)
        expected = (math.sqrt(2.0) - 1.0) / (2.0 * math.sqrt(2.0))

        self.assertTrue(bool(metrics["tortuosity_density_valid"]))
        self.assertEqual(metrics["constant_curvature_segment_count"], 2.0)
        self.assertEqual(metrics["inflection_count"], 1.0)
        self.assertAlmostEqual(float(metrics["tortuosity_density_excess"]), 2.0 * (math.sqrt(2.0) - 1.0))
        self.assertAlmostEqual(float(metrics["tortuosity_density_score"]), expected)

    def test_tortuosity_density_ignores_subthreshold_sign_flips_and_duplicates(self) -> None:
        points = [[0, 0], [1, 0.005], [1, 0.005], [2, -0.005], [3, 0.005], [4, 0]]
        settings = LocalBumpSettings(resample_step=1.0, smoothing_window=1, curvature_threshold=0.035)

        metrics = tortuosity_density_metrics(points, settings)

        self.assertTrue(bool(metrics["tortuosity_density_valid"]))
        self.assertEqual(metrics["constant_curvature_segment_count"], 1.0)
        self.assertAlmostEqual(float(metrics["tortuosity_density_score"]), 0.0)

    def test_tortuosity_density_rejects_zero_chord_geometry(self) -> None:
        closed_curve = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]

        metrics = tortuosity_density_metrics(
            closed_curve,
            LocalBumpSettings(resample_step=1.0, smoothing_window=1, curvature_threshold=0.01),
        )

        self.assertFalse(bool(metrics["tortuosity_density_valid"]))
        self.assertTrue(math.isnan(float(metrics["tortuosity_density_score"])))

    def test_rdp_simplify_collapses_straight_line_to_endpoints(self) -> None:
        points = np.column_stack([np.linspace(0, 100, 40), np.zeros(40)])

        simplified = rdp_simplify(points, epsilon=1.0)

        self.assertEqual(len(simplified), 2)
        np.testing.assert_allclose(simplified[0], points[0])
        np.testing.assert_allclose(simplified[-1], points[-1])

    def test_rdp_simplify_keeps_sharp_corners(self) -> None:
        points = np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0]])

        simplified = rdp_simplify(points, epsilon=0.5)

        self.assertEqual(len(simplified), 4)

    def test_external_angle_sum_straight_line_scores_zero(self) -> None:
        points = [[x, 0.0] for x in np.linspace(0, 200, 80)]

        score = external_angle_sum_metrics(points)

        self.assertAlmostEqual(score["external_angle_sum_score"], 0.0)
        self.assertEqual(score["external_angle_sum_bend_point_count"], 0.0)

    def test_external_angle_sum_matches_hand_calculated_staircase(self) -> None:
        points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0]]
        settings = LocalBumpSettings(rdp_epsilon=0.5)

        metrics = external_angle_sum_metrics(points, settings)

        self.assertEqual(metrics["external_angle_sum_bend_point_count"], 2.0)
        self.assertAlmostEqual(metrics["external_angle_sum_score"], 180.0)
        self.assertAlmostEqual(metrics["external_angle_sum_mean_angle_deg"], 90.0)

    def test_external_angle_sum_short_or_degenerate_paths_return_safe_defaults(self) -> None:
        short_score = external_angle_sum_metrics([[0, 0]])
        duplicate_score = external_angle_sum_metrics([[0, 0], [0, 0], [0, 0]])

        self.assertEqual(short_score["external_angle_sum_score"], 0.0)
        self.assertEqual(duplicate_score["external_angle_sum_score"], 0.0)
        self.assertTrue(math.isfinite(short_score["external_angle_sum_score"]))

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


class CentralScoringServiceTests(unittest.TestCase):
    def test_method_registry_contains_arc_chord_and_local_bump(self) -> None:
        methods = {method.method_id for method in available_scoring_methods()}

        self.assertEqual(
            methods,
            {
                "local_bump",
                "local_bump_v2",
                "arc_chord",
                "curvature_squared",
                "tortuosity_density",
                "external_angle_sum",
            },
        )

    def test_same_saved_vessel_can_be_scored_by_both_methods(self) -> None:
        points = [[0, 0], [20, 6], [40, -6], [60, 6], [80, 0]]
        segment = VesselSegment.from_manual_points(1, points)
        segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
        vessel = {
            "category": "artere",
            "segment_refs": [segment.ref],
            "synthetic_links": [],
            "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref, "distance_from_start": 0.0},
            "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref, "distance_from_start": segment.path_length},
        }

        arc_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("arc_chord"))
        bump_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("local_bump"))
        bump_v2_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("local_bump_v2"))
        curvature_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("curvature_squared"))
        density_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("tortuosity_density"))
        angle_sum_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("external_angle_sum"))

        self.assertEqual(arc_score["scoring_method"], "arc_chord")
        self.assertEqual(bump_score["scoring_method"], "local_bump")
        self.assertEqual(bump_v2_score["scoring_method"], "local_bump_v2")
        self.assertEqual(curvature_score["scoring_method"], "curvature_squared")
        self.assertEqual(density_score["scoring_method"], "tortuosity_density")
        self.assertEqual(angle_sum_score["scoring_method"], "external_angle_sum")
        self.assertAlmostEqual(float(arc_score["arc_chord_diagnostic"]), float(arc_score["primary_score"]))
        self.assertGreater(float(bump_score["local_bump_score"]), 0.0)
        self.assertAlmostEqual(float(bump_v2_score["local_bump_v2_score"]), float(bump_v2_score["primary_score"]))
        self.assertAlmostEqual(
            float(curvature_score["curvature_squared_score"]),
            float(curvature_score["primary_score"]),
        )
        self.assertAlmostEqual(
            float(density_score["tortuosity_density_score"]),
            float(density_score["primary_score"]),
        )
        self.assertAlmostEqual(
            float(angle_sum_score["external_angle_sum_score"]),
            float(angle_sum_score["primary_score"]),
        )
        self.assertGreater(float(angle_sum_score["primary_score"]), 0.0)

    def test_coordinate_normalization_makes_scaled_geometry_comparable(self) -> None:
        def score_for(points: list[list[float]], scale: float) -> dict[str, object]:
            segment = VesselSegment.from_manual_points(1, points)
            segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
            vessel = {
                "category": "artere",
                "segment_refs": [segment.ref],
                "synthetic_links": [],
                "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref},
                "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref},
            }
            return score_saved_vessel(
                segment_map,
                "v1",
                vessel,
                config=scoring_config("local_bump"),
                coordinate_scale=scale,
            )

        base = [[0, 0], [40, 12], [80, -12], [120, 12], [160, 0]]
        enlarged = [[4 * x, 4 * y] for x, y in base]
        base_score = score_for(base, 1.0)
        normalized_score = score_for(enlarged, 0.25)

        self.assertAlmostEqual(float(base_score["primary_score"]), float(normalized_score["primary_score"]), places=10)
        self.assertAlmostEqual(float(base_score["vessel_length"]), float(normalized_score["vessel_length"]), places=10)
        self.assertGreater(float(normalized_score["raw_vessel_length"]), float(normalized_score["vessel_length"]))
        self.assertIn("raw_primary_score", normalized_score)
        self.assertNotAlmostEqual(
            float(normalized_score["raw_primary_score"]),
            float(normalized_score["primary_score"]),
            places=6,
        )

    def test_external_angle_sum_uses_normalized_geometry(self) -> None:
        def score_for(points: list[list[float]], scale: float) -> dict[str, object]:
            segment = VesselSegment.from_manual_points(1, points)
            segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
            vessel = {
                "category": "artere",
                "segment_refs": [segment.ref],
                "synthetic_links": [],
                "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref},
                "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref},
            }
            return score_saved_vessel(
                segment_map,
                "v1",
                vessel,
                config=scoring_config("external_angle_sum"),
                coordinate_scale=scale,
            )

        base = [[0, 0], [40, 12], [80, -12], [120, 12], [160, 0]]
        enlarged = [[4 * x, 4 * y] for x, y in base]

        self.assertAlmostEqual(
            float(score_for(base, 1.0)["primary_score"]),
            float(score_for(enlarged, 0.25)["primary_score"]),
            places=10,
        )

    def test_tortuosity_density_uses_normalized_geometry_and_inverse_length_units(self) -> None:
        def score_for(points: list[list[float]], scale: float) -> dict[str, object]:
            segment = VesselSegment.from_manual_points(1, points)
            segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
            vessel = {
                "category": "artere",
                "segment_refs": [segment.ref],
                "synthetic_links": [],
                "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref},
                "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref},
            }
            return score_saved_vessel(
                segment_map,
                "v1",
                vessel,
                config=scoring_config("tortuosity_density"),
                coordinate_scale=scale,
            )

        base = [[0, 0], [40, 40], [80, 0], [120, -40], [160, 0]]
        enlarged = [[4 * x, 4 * y] for x, y in base]
        base_score = score_for(base, 1.0)
        normalized_score = score_for(enlarged, 0.25)

        self.assertAlmostEqual(float(base_score["primary_score"]), float(normalized_score["primary_score"]), places=10)
        self.assertAlmostEqual(
            float(normalized_score["raw_primary_score"]),
            float(normalized_score["primary_score"]) / 4.0,
            places=6,
        )

    def test_local_bump_v2_uses_normalized_geometry(self) -> None:
        def score_for(points: list[list[float]], scale: float) -> dict[str, object]:
            segment = VesselSegment.from_manual_points(1, points)
            segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
            vessel = {
                "category": "artere",
                "segment_refs": [segment.ref],
                "synthetic_links": [],
                "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref},
                "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref},
            }
            return score_saved_vessel(
                segment_map,
                "v1",
                vessel,
                config=scoring_config("local_bump_v2"),
                coordinate_scale=scale,
            )

        base = [[0, 0], [40, 12], [80, -12], [120, 12], [160, 0]]
        enlarged = [[4 * x, 4 * y] for x, y in base]

        self.assertAlmostEqual(
            float(score_for(base, 1.0)["primary_score"]),
            float(score_for(enlarged, 0.25)["primary_score"]),
            places=10,
        )

    def test_tortuosity_density_eye_summary_is_length_weighted_without_tail_or_scale(self) -> None:
        vessel_scores = pd.DataFrame(
            {
                "eligible": [True, True],
                "primary_score": [1.0, 3.0],
                "vessel_length": [10.0, 30.0],
                "arc_chord_diagnostic": [1.0, 1.2],
                "category": ["artere", "veine"],
            }
        )

        summary = summarize_saved_eye_score(vessel_scores, scoring_config("tortuosity_density"))

        self.assertAlmostEqual(float(summary["eye_tortuosity_score"]), 2.5)
        self.assertAlmostEqual(float(summary["all_vessels_score"]), 2.5)
        self.assertTrue(math.isnan(float(summary["all_vessels_tail_score"])))

    def test_short_vessel_filter_applies_to_all_scoring_methods_by_default(self) -> None:
        points = [[0, 0], [20, 6], [40, -6], [60, 6], [80, 0]]
        segment = VesselSegment.from_manual_points(1, points)
        segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
        vessel = {
            "category": "artere",
            "segment_refs": [segment.ref],
            "synthetic_links": [],
            "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref, "distance_from_start": 0.0},
            "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref, "distance_from_start": segment.path_length},
        }
        settings = LocalBumpSettings(min_saved_vessel_length=10_000.0)

        bump_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("local_bump", settings))
        arc_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("arc_chord", settings))
        curvature_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("curvature_squared", settings))

        self.assertFalse(bool(bump_score["eligible"]))
        self.assertFalse(bool(arc_score["eligible"]))
        self.assertFalse(bool(curvature_score["eligible"]))

    def test_short_vessel_filter_can_be_disabled_for_all_scoring_methods(self) -> None:
        points = [[0, 0], [20, 6], [40, -6], [60, 6], [80, 0]]
        segment = VesselSegment.from_manual_points(1, points)
        segment_map = build_segment_map(pd.DataFrame(), {"1": segment.to_manual_payload()})
        vessel = {
            "category": "artere",
            "segment_refs": [segment.ref],
            "synthetic_links": [],
            "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref, "distance_from_start": 0.0},
            "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref, "distance_from_start": segment.path_length},
        }
        settings = LocalBumpSettings(min_saved_vessel_length=10_000.0, filter_short_vessels=False)

        bump_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("local_bump", settings))
        arc_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("arc_chord", settings))
        curvature_score = score_saved_vessel(segment_map, "v1", vessel, config=scoring_config("curvature_squared", settings))

        self.assertTrue(bool(bump_score["eligible"]))
        self.assertTrue(bool(arc_score["eligible"]))
        self.assertTrue(bool(curvature_score["eligible"]))

    def test_parameter_free_eye_summaries_ignore_local_bump_aggregation_settings(self) -> None:
        vessel_scores = pd.DataFrame(
            {
                "eligible": [False, True],
                "primary_score": [1.0, 3.0],
                "vessel_length": [10.0, 30.0],
                "arc_chord_diagnostic": [1.0, 1.2],
                "category": ["artere", "veine"],
            }
        )
        default_summary = summarize_saved_eye_score(
            vessel_scores,
            scoring_config("curvature_squared", LocalBumpSettings(filter_short_vessels=False)),
        )
        extreme_summary = summarize_saved_eye_score(
            vessel_scores,
            scoring_config(
                "curvature_squared",
                LocalBumpSettings(
                    min_saved_vessel_length=10_000.0,
                    filter_short_vessels=False,
                    tail_length_fraction=1.0,
                    global_weight=0.0,
                    tail_weight=1.0,
                ),
            ),
        )

        self.assertEqual(default_summary["eligible_vessel_count"], 2)
        self.assertEqual(extreme_summary["eligible_vessel_count"], 2)
        self.assertEqual(default_summary["eye_tortuosity_score"], extreme_summary["eye_tortuosity_score"])

    def test_review_and_export_follow_active_scoring_method(self) -> None:
        points = [[0, 0], [20, 6], [40, -6], [60, 6], [80, 0]]
        branches = pd.DataFrame()
        segment = VesselSegment.from_manual_points(1, points)
        review_state = {
            "manual_segments": {"1": segment.to_manual_payload()},
            "vessels": {
                "manual_1": {
                    "category": "artere",
                    "segment_refs": [segment.ref],
                    "synthetic_links": [],
                    "start_endpoint": {"kind": "geometry_point", "point": points[0], "segment_ref": segment.ref, "distance_from_start": 0.0},
                    "end_endpoint": {"kind": "geometry_point", "point": points[-1], "segment_ref": segment.ref, "distance_from_start": segment.path_length},
                }
            },
        }

        review_table = build_review_scores_table(branches, review_state, config=scoring_config("curvature_squared"))
        export_df = build_manual_vessels_export(branches, review_state, config=scoring_config("curvature_squared"))

        self.assertEqual(review_table.loc[0, "Methode"], "Courbure quadratique")
        self.assertEqual(export_df.loc[0, "scoring_method"], "curvature_squared")
        self.assertEqual(export_df.loc[0, "primary_score_label"], "Score courbure^2")
        self.assertIn("curvature_squared_score", export_df.columns)

    def test_fixed_parameters_are_exposed_per_scoring_method(self) -> None:
        local_bump_parameters = dict(scoring_method_fixed_parameters(scoring_config("local_bump")))
        curvature_parameters = dict(scoring_method_fixed_parameters(scoring_config("curvature_squared")))
        arc_chord_parameters = dict(scoring_method_fixed_parameters(scoring_config("arc_chord")))
        density_parameters = dict(scoring_method_fixed_parameters(scoring_config("tortuosity_density")))
        v2_parameters = dict(scoring_method_fixed_parameters(scoring_config("local_bump_v2")))
        angle_sum_parameters = dict(scoring_method_fixed_parameters(scoring_config("external_angle_sum")))

        self.assertIn("Seuil de courbure locale", local_bump_parameters)
        self.assertIn("Pas de re-echantillonnage", local_bump_parameters)
        self.assertEqual(
            curvature_parameters,
            {
                "Normalisation geometrie": "diametre du fond d'oeil ramene a 1024 px",
                "Filtre petits vaisseaux": "actif, longueur minimale 100 px normalises",
                "Pretraitement re-echantillonnage": "actif, pas 4.0 px",
            },
        )
        self.assertEqual(
            arc_chord_parameters,
            {
                "Normalisation geometrie": "diametre du fond d'oeil ramene a 1024 px",
                "Filtre petits vaisseaux": "actif, longueur minimale 100 px normalises",
            },
        )
        self.assertEqual(
            density_parameters,
            {
                "Normalisation geometrie": "diametre du fond d'oeil ramene a 1024 px",
                "Filtre petits vaisseaux": "actif, longueur minimale 100 px normalises",
                "Pas de re-echantillonnage": "4.0 px",
                "Fenetre de lissage": "5 points",
                "Seuil d'inflexion": "0.035 rad",
            },
        )
        self.assertEqual(v2_parameters["Angle minimal d'un lobe persistant"], "0.150 rad")
        self.assertEqual(v2_parameters["Poids de l'angularite"], "0.25")
        self.assertEqual(
            angle_sum_parameters,
            {
                "Normalisation geometrie": "diametre du fond d'oeil ramene a 1024 px",
                "Filtre petits vaisseaux": "actif, longueur minimale 100 px normalises",
                "Tolerance de simplification RDP": "3.0 px normalises",
            },
        )


class LocalBumpV2DiagnosticExamplesTests(unittest.TestCase):
    def test_reviewed_examples_show_expected_robustness(self) -> None:
        runs_root = Path(__file__).parents[1] / "demo" / "streamlit_runs"
        if not runs_root.exists():
            self.skipTest("Saved diagnostic runs are not available.")
        targets = {
            "1_OD": ["1°A-inf"],
            "5_OG": ["a destinée maculaire"],
            "11_OG": ["1°A-sup"],
            "9_OD": ["3°A-inf2"],
            "7_OD": ["2°A-sup"],
        }
        old_scores: dict[tuple[str, str], float] = {}
        new_scores: dict[tuple[str, str], float] = {}
        for run_name, vessel_names in targets.items():
            _summary, old = score_run_with_method(runs_root / run_name, scoring_config("local_bump"))
            _summary, new = score_run_with_method(runs_root / run_name, scoring_config("local_bump_v2"))
            for vessel_name in vessel_names:
                old_scores[(run_name, vessel_name)] = float(
                    old.loc[old["vessel_name"] == vessel_name, "primary_score"].iloc[0] * 1000.0
                )
                new_scores[(run_name, vessel_name)] = float(
                    new.loc[new["vessel_name"] == vessel_name, "primary_score"].iloc[0]
                )

        aberrant = ("1_OD", "1°A-inf")
        self.assertLess(new_scores[aberrant], old_scores[aberrant] * 0.60)
        similar = [
            new_scores[("11_OG", "1°A-sup")],
            new_scores[("9_OD", "3°A-inf2")],
            new_scores[("7_OD", "2°A-sup")],
        ]
        similar_mean = float(np.mean(similar))
        self.assertTrue(all(abs(value - similar_mean) / similar_mean <= 0.25 for value in similar))
        macular = new_scores[("5_OG", "a destinée maculaire")]
        self.assertGreater(macular, max(similar))
        self.assertGreater(macular, new_scores[aberrant])


if __name__ == "__main__":
    unittest.main()
