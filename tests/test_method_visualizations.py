from __future__ import annotations

import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tortuosite_score.vessels_detection.clinical_excel import parse_vessel_name
from tortuosite_score.vessels_detection.local_bump_score import (
    LocalBumpSettings,
    local_bump_diagnostics,
    local_bump_metrics,
)
from tortuosite_score.visualization.method_figures import (
    AtlasVein,
    generate_local_bump_atlas,
    is_ranked_publication_vein,
)


class LocalBumpDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_reproduce_public_metrics(self) -> None:
        x = np.arange(0.0, 161.0)
        points = np.column_stack([x, 12.0 * np.sin(x / 11.0)])
        diagnostics = local_bump_diagnostics(points)

        self.assertEqual(diagnostics.metrics, local_bump_metrics(points))
        self.assertEqual(len(diagnostics.arc_distance), len(diagnostics.angle_change))
        self.assertEqual(
            len(diagnostics.angle_change), len(diagnostics.filtered_angle_change)
        )
        self.assertEqual(
            int(diagnostics.metrics["oscillation_count"]),
            len(diagnostics.sign_change_indices),
        )

    def test_straight_and_subthreshold_paths_have_no_oscillation(self) -> None:
        settings = LocalBumpSettings(curvature_threshold=0.20)
        straight = np.column_stack([np.arange(0.0, 121.0), np.zeros(121)])
        shallow = np.column_stack(
            [np.arange(0.0, 121.0), 0.02 * np.sin(np.arange(0.0, 121.0) / 25.0)]
        )

        for points in [straight, shallow]:
            diagnostics = local_bump_diagnostics(points, settings)
            self.assertEqual(diagnostics.metrics["oscillation_count"], 0.0)
            self.assertEqual(diagnostics.metrics["branch_bump_score"], 0.0)
            self.assertTrue(np.all(diagnostics.filtered_angle_change == 0.0))

    def test_alternating_curve_exposes_sign_change_locations(self) -> None:
        x = np.arange(0.0, 241.0)
        points = np.column_stack([x, 18.0 * np.sin(x / 12.0)])
        diagnostics = local_bump_diagnostics(points)

        self.assertGreater(len(diagnostics.sign_change_indices), 0)
        self.assertTrue(
            np.all(diagnostics.sign_change_indices < len(diagnostics.angle_change))
        )
        self.assertTrue(math.isfinite(diagnostics.metrics["branch_bump_score"]))


class PublicationVeinFilterTests(unittest.TestCase):
    def test_filter_requires_export_eligibility_and_ranked_vein_name(self) -> None:
        base = {
            "vessel_name": "2°V-inf",
            "category": "veine",
            "vessel_length": 150.0,
            "eligible": True,
            "primary_score": 0.08,
        }
        self.assertTrue(is_ranked_publication_vein(base))
        self.assertFalse(
            is_ranked_publication_vein({**base, "vessel_name": "VEINE TEMPORALE INF"})
        )
        self.assertFalse(
            is_ranked_publication_vein(
                {**base, "vessel_length": 80.0, "eligible": False}
            )
        )
        self.assertFalse(
            is_ranked_publication_vein({**base, "primary_score": math.nan})
        )
        self.assertFalse(
            is_ranked_publication_vein(
                {**base, "vessel_name": "2°A-inf", "category": "artere"}
            )
        )


class AtlasSmokeTests(unittest.TestCase):
    def test_one_record_atlas_has_cover_index_and_detail_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.png"
            Image.new("RGB", (320, 240), (75, 42, 35)).save(source_path)
            x = np.linspace(35.0, 285.0, 90)
            y = 120.0 + 25.0 * np.sin(x / 28.0)
            points = np.column_stack([x, y])
            metrics = local_bump_metrics(points)
            vessel_scores = pd.DataFrame(
                [
                    {
                        "vessel_id": 1,
                        "vessel_name": "2°V-inf",
                        "category": "veine",
                        "eligible": True,
                        "primary_score": metrics["branch_bump_score"],
                        "vessel_length": metrics["branch_length"],
                    },
                    {
                        "vessel_id": 2,
                        "vessel_name": "1°A-sup",
                        "category": "artere",
                        "eligible": True,
                        "primary_score": 0.04,
                        "vessel_length": 140.0,
                    },
                ]
            )
            weighted = float(
                np.average(
                    vessel_scores["primary_score"],
                    weights=vessel_scores["vessel_length"],
                )
            )
            summary = {
                "all_vessels_score": weighted * 1000.0,
                "all_vessels_tail_score": metrics["branch_bump_score"] * 1000.0,
                "eye_tortuosity_score": (
                    0.7 * weighted + 0.3 * metrics["branch_bump_score"]
                )
                * 1000.0,
            }
            row = {
                **vessel_scores.iloc[0].to_dict(),
                "path_points": points.tolist(),
                "coordinate_scale": 1.0,
            }
            record = AtlasVein(
                run_dir=root / "1_OD",
                source_path=source_path,
                row=row,
                name_info=parse_vessel_name("2°V-inf", "veine"),
                summary=summary,
                vessel_scores=vessel_scores,
            )
            output = root / "atlas.pdf"

            generate_local_bump_atlas([record], output)

            payload = output.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF"))
            self.assertGreater(len(payload), 10_000)
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", payload)), 3)


class ImageAnnotationTests(unittest.TestCase):
    def test_normalized_angle_vertices_and_image_coordinates(self):
        from tortuosite_score.visualization.anatomy_panels import (
            angle_vertex_indices,
            image_points,
            angle_triplet,
        )

        x = np.arange(121.0)
        d = local_bump_diagnostics(np.column_stack([x, 8 * np.sin(x / 13)]))
        np.testing.assert_array_equal(
            angle_vertex_indices(d), np.arange(1, len(d.smoothed) - 1)
        )
        np.testing.assert_allclose(image_points(d, 2.5) * 2.5, d.smoothed)
        for i in range(len(d.angle_change)):
            vectors = np.diff(angle_triplet(d, i), axis=0)
            expected = np.diff(np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0])))[0]
            self.assertAlmostEqual(expected, d.angle_change[i], places=12)
        with self.assertRaises(ValueError):
            image_points(d, 0)

    def test_duplicate_smoothed_points_follow_valid_segment_mapping(self):
        from dataclasses import replace
        from tortuosite_score.visualization.anatomy_panels import (
            angle_vertex_indices,
            angle_triplet,
        )
        from tortuosite_score.vessels_detection.local_bump_score import (
            local_angle_change,
        )

        p = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [2, 1]], dtype=float)
        d = replace(
            local_bump_diagnostics(p), smoothed=p, angle_change=local_angle_change(p)
        )
        np.testing.assert_array_equal(angle_vertex_indices(d), [2, 3])
        np.testing.assert_array_equal(angle_triplet(d, 0), [[0, 0], [1, 0], [1, 1]])

    def test_threshold_equality_and_transitions_across_zero_samples(self):
        from unittest.mock import patch

        tau = 0.035
        values = np.array([tau, 0, -tau, tau / 2, tau])
        with patch(
            "tortuosite_score.vessels_detection.local_bump_score.local_angle_change",
            return_value=values,
        ):
            d = local_bump_diagnostics([[0, 0], [100, 0]])
        np.testing.assert_array_equal(d.filtered_angle_change, [tau, 0, -tau, 0, tau])
        np.testing.assert_array_equal(d.sign_change_indices, [2, 4])
        self.assertAlmostEqual(d.metrics["local_bump_energy"], 3 * tau / 5)
        self.assertEqual(d.metrics["oscillation_count"], 2)
        self.assertEqual(d.metrics["oscillation_density"], 2)
        self.assertAlmostEqual(
            d.metrics["branch_bump_score"], 3 * tau / 5 * math.sqrt(2)
        )

    def test_single_sign_can_have_energy_but_zero_score(self):
        from unittest.mock import patch

        with patch(
            "tortuosite_score.vessels_detection.local_bump_score.local_angle_change",
            return_value=np.array([0.07, 0, 0.09]),
        ):
            d = local_bump_diagnostics([[0, 0], [100, 0]])
        self.assertGreater(d.metrics["local_bump_energy"], 0)
        self.assertEqual(d.metrics["branch_bump_score"], 0)
        self.assertEqual(len(d.sign_change_indices), 0)

    def test_dense_transition_groups_have_exact_coverage(self):
        from dataclasses import replace
        from tortuosite_score.visualization.anatomy_panels import transition_groups

        d = replace(
            local_bump_diagnostics([[0, 0], [100, 0]]),
            sign_change_indices=np.arange(29),
        )
        groups = transition_groups(d)
        np.testing.assert_array_equal(np.concatenate(groups), np.arange(29))
        self.assertTrue(all(len(g) <= 6 for g in groups))

    def test_index_pages_point_to_physical_detail_pages(self):
        from types import SimpleNamespace
        from tortuosite_score.visualization.method_figures import (
            _atlas_index_page,
            select_comparison_records,
        )

        records = [
            SimpleNamespace(
                run_dir=Path(f"{i}_OD"),
                row={"vessel_name": "1°V", "primary_score": i, "vessel_length": 100},
                name_info=SimpleNamespace(rank=1),
            )
            for i in range(31)
        ]
        f = _atlas_index_page(records[30:], 30, 2, 2)
        table = f.axes[0].tables[0]
        self.assertEqual(table[1, 0].get_text().get_text(), "31")
        self.assertEqual(table[1, 1].get_text().get_text(), "34")
        self.assertEqual(
            select_comparison_records(records[::-1]),
            [records[0], records[15], records[-1]],
        )

    def test_partial_eye_tail_allocation_uses_length(self):
        from tortuosite_score.visualization.method_figures import _tail_allocations

        vessels = pd.DataFrame(
            {
                "eligible": [True, True, False],
                "vessel_length": [10.0, 90.0, 500.0],
                "primary_score": [0.8, 0.2, 100.0],
            }
        )
        self.assertEqual(_tail_allocations(vessels, 0.2), {0: 10.0, 1: 10.0})


class EyeLengthAnnotationTests(unittest.TestCase):
    def test_distinct_app_length_is_reported_without_changing_formula_length(self):
        from types import SimpleNamespace
        from tortuosite_score.visualization.method_figures import _eye_note

        record = SimpleNamespace(
            row={"vessel_id": 1, "vessel_length": 100.0},
            vessel_scores=pd.DataFrame(
                [
                    {
                        "vessel_id": 1,
                        "eligible": True,
                        "primary_score": 0.1,
                        "vessel_length": 100.0,
                    }
                ]
            ),
            summary={
                "all_vessels_score": 100.0,
                "all_vessels_tail_score": 100.0,
                "eye_tortuosity_score": 100.0,
            },
        )
        note = _eye_note(record, LocalBumpSettings(), 100.05)
        self.assertIn("100,000 px", note)
        self.assertIn("100,050 px", note)
        self.assertIn("20,0 %", note)
        self.assertNotIn(
            "L utilisée dans B", _eye_note(record, LocalBumpSettings(), 100.0)
        )


class ArteryVisualizationTests(unittest.TestCase):
    def _record(self, root):
        from tortuosite_score.visualization.method_figures import AtlasVessel
        from tortuosite_score.vessels_detection.local_bump_score import (
            tail_weighted_mean,
        )

        source = root / "source.png"
        Image.new("RGB", (260, 200), (90, 45, 30)).save(source)
        x = np.linspace(20, 240, 120)
        points = np.column_stack([x, 90 + 15 * np.sin(x / 17)])
        metrics = local_bump_metrics(points)
        row = dict(
            vessel_id=1,
            vessel_name="2°A-inf",
            category="artere",
            eligible=True,
            primary_score=metrics["branch_bump_score"],
            vessel_length=metrics["branch_length"],
            coordinate_scale=1.0,
            path_points=points.tolist(),
        )
        scores = pd.DataFrame(
            [
                row,
                dict(
                    vessel_id=2,
                    vessel_name="1°V-sup",
                    category="veine",
                    eligible=True,
                    primary_score=0.02,
                    vessel_length=120.0,
                ),
            ]
        )
        g = float(np.average(scores.primary_score, weights=scores.vessel_length))
        t = tail_weighted_mean(scores, "primary_score", "vessel_length", 0.2)
        return AtlasVessel(
            root / "1_OD",
            source,
            row,
            parse_vessel_name("2°A-inf", "artere"),
            dict(
                all_vessels_score=g * 1000,
                all_vessels_tail_score=t * 1000,
                eye_tortuosity_score=1000 * (0.7 * g + 0.3 * t),
            ),
            scores,
        )

    def test_arteries_require_both_export_and_scoring_eligibility(self):
        from tortuosite_score.visualization.method_figures import (
            is_ranked_publication_vessel,
        )

        base = dict(
            vessel_name="2°A-inf",
            category="artere",
            vessel_length=150,
            eligible=True,
            primary_score=0.04,
        )
        self.assertTrue(is_ranked_publication_vessel(base, "artere"))
        self.assertFalse(is_ranked_publication_vessel(base, "veine"))
        for change in [
            dict(vessel_name="artère maculaire"),
            dict(vessel_name="2°V-inf"),
            dict(vessel_name="2°A-inf-sup"),
            dict(eligible=False, vessel_length=80),
            dict(primary_score=math.nan),
            dict(primary_score=math.inf),
            dict(primary_score=None),
            dict(primary_score="invalid"),
        ]:
            with self.subTest(change=change):
                self.assertFalse(
                    is_ranked_publication_vessel({**base, **change}, "artere")
                )
        with self.assertRaises(ValueError):
            is_ranked_publication_vessel(base, "unknown")

    def test_changing_vessel_type_preserves_geometry_and_score(self):
        from dataclasses import replace
        from tortuosite_score.visualization.method_figures import (
            _record_diagnostics,
            _atlas_detail_page,
            _atlas_cover_page,
            _atlas_index_page,
        )

        with tempfile.TemporaryDirectory() as tmp:
            artery = self._record(Path(tmp))
            vein = replace(
                artery,
                row={**artery.row, "vessel_name": "2°V-inf", "category": "veine"},
                name_info=parse_vessel_name("2°V-inf", "veine"),
            )
            a = _record_diagnostics(artery, LocalBumpSettings())[2]
            v = _record_diagnostics(vein, LocalBumpSettings())[2]
            self.assertEqual(a.metrics, v.metrics)
            np.testing.assert_array_equal(a.smoothed, v.smoothed)
            figure = _atlas_detail_page(artery, 1, 1, 3, LocalBumpSettings())
            text = "\n".join(item.get_text() for item in figure.texts)
            self.assertIn("2°A-inf · 2°A", text)
            self.assertIn("Calculer le score de l’artère", text)
            self.assertIn("artères et veines", text)
            self.assertNotIn("cette veine", text)
            cover = _atlas_cover_page(1, LocalBumpSettings(), vessel_type="artere")
            self.assertIn("1 artères", "\n".join(t.get_text() for t in cover.texts))
            index = _atlas_index_page([artery], 0, 1, 1, vessel_type="artere")
            self.assertEqual(index.axes[0].tables[0][1, 4].get_text().get_text(), "2°A")
            output = Path(tmp) / "arteres.pdf"
            generate_local_bump_atlas([artery], output)
            payload = output.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF"))
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", payload)), 3)
            with self.assertRaises(ValueError):
                generate_local_bump_atlas([artery, vein], Path(tmp) / "mixed.pdf")

    def test_all_eye_artery_outputs_have_separate_names(self):
        from unittest.mock import patch
        from tortuosite_score.visualization.method_figures import (
            generate_method_visualizations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artery = self._record(root)
            for name in ["1_OD", "1_OG", "legacy"]:
                (root / name).mkdir()
            with (
                patch(
                    "tortuosite_score.visualization.method_figures.generate_pipeline_figure"
                ) as pipeline,
                patch(
                    "tortuosite_score.visualization.method_figures.collect_ranked_publication_vessels",
                    return_value=[artery],
                ),
                patch(
                    "tortuosite_score.visualization.method_figures.generate_local_bump_atlas"
                ),
                patch(
                    "tortuosite_score.visualization.method_figures.generate_local_bump_overview"
                ),
            ):
                result = generate_method_visualizations(
                    root, root / "out", "all", vessel_type="artere"
                )
            self.assertEqual(pipeline.call_count, 2)
            self.assertEqual(
                [p.name for p in result["pipeline_pdfs"]],
                [
                    "pipeline_extraction_arteres_1_OD.pdf",
                    "pipeline_extraction_arteres_1_OG.pdf",
                ],
            )
            self.assertEqual(result["atlas_pdf"].name, "atlas_local_bump_arteres.pdf")
            self.assertEqual(
                result["overview_pdf"].name, "explication_local_bump_arteres.pdf"
            )
            self.assertEqual(result["artery_count"], 1)
            self.assertNotIn("vein_count", result)


if __name__ == "__main__":
    unittest.main()
