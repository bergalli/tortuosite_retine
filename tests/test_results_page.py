from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops

from tortuosite_score.app.results_page import (
    _build_atlas_systems_table,
    _build_all_systems_table,
    _build_local_bump_summary_table,
    _build_local_bump_pvalue_matrices,
    _kept_vessel_count_label,
    _local_bump_scores_pdf_table,
    _pvalue_cell_color,
    _single_line_cell_text,
    _stats_table_font_size,
    _top_system_rows,
    _top_system_table,
    _render_ranked_system_crop,
    _render_saved_vessel_overlay,
    _normalized_vessel_label,
    _path_midpoint_and_angle,
    build_adjusted_pvalue_matrix,
    build_pvalue_matrix,
    build_results_viewer_segments,
    build_result_rows,
    generate_results_pdf,
    generate_vessel_type_segmentation_zip,
    render_vessel_type_segmentation_image,
    tortuosity_values,
    visible_result_table,
    _wrapped_cell_text,
)
from tortuosite_score.vessels_detection.segments import create_geometry_endpoint


class ResultsPageTests(unittest.TestCase):
    def test_build_result_rows_from_saved_vessels(self) -> None:
        review_state = {
            "manual_segments": {
                "1": {"points": [[0, 0], [10, 0]]},
            },
            "vessels": {
                "temporal": {
                    "category": "artere",
                    "segment_refs": ["manual:1"],
                    "synthetic_links": [],
                    "start_endpoint": create_geometry_endpoint([0, 0], "manual:1", 0.0),
                    "end_endpoint": create_geometry_endpoint([10, 0], "manual:1", 10.0),
                }
            },
        }

        rows = build_result_rows(review_state, pd.DataFrame())

        self.assertEqual(rows["Label"].tolist(), ["V1"])
        self.assertEqual(rows["Vaisseau"].tolist(), ["temporal"])
        self.assertEqual(rows["Categorie"].tolist(), ["artere"])
        self.assertEqual(rows["Segments manuels"].tolist(), [1])
        self.assertAlmostEqual(float(rows.loc[0, "Tortuosite"]), 1.0)

    def test_tortuosity_values_ignores_missing_and_nan(self) -> None:
        rows = pd.DataFrame({"Tortuosite": [1.1, float("nan"), None, "bad", 1.4]})

        self.assertEqual(tortuosity_values(rows), [1.1, 1.4])

    def test_pvalue_matrix_is_directional_and_handles_na(self) -> None:
        matrix = build_pvalue_matrix(
            {
                "image_a": [1.0, 1.1, 1.2],
                "image_b": [1.8, 1.9, 2.0],
                "image_empty": [],
            }
        )

        self.assertEqual(matrix.index.tolist(), ["image_a", "image_b", "image_empty"])
        self.assertEqual(matrix.columns.tolist(), ["image_a", "image_b", "image_empty"])
        self.assertEqual(matrix.loc["image_a", "image_a"], "-")
        self.assertEqual(matrix.loc["image_a", "image_empty"], "NA")
        self.assertEqual(matrix.loc["image_empty", "image_b"], "NA")
        self.assertGreater(float(matrix.loc["image_a", "image_b"]), 0.9)
        self.assertLess(float(matrix.loc["image_b", "image_a"]), 0.1)

    def test_adjusted_pvalue_matrix_preserves_shape_and_na(self) -> None:
        raw = build_pvalue_matrix(
            {
                "image_a": [1.0, 1.1, 1.2],
                "image_b": [1.8, 1.9, 2.0],
                "image_empty": [],
            }
        )

        adjusted = build_adjusted_pvalue_matrix(raw)

        self.assertEqual(adjusted.index.tolist(), raw.index.tolist())
        self.assertEqual(adjusted.columns.tolist(), raw.columns.tolist())
        self.assertEqual(adjusted.loc["image_a", "image_a"], "-")
        self.assertEqual(adjusted.loc["image_a", "image_empty"], "NA")
        self.assertLessEqual(float(raw.loc["image_b", "image_a"]), float(adjusted.loc["image_b", "image_a"]))

    def test_visible_result_table_hides_detail_columns(self) -> None:
        rows = pd.DataFrame(
            {
                "Label": ["V1"],
                "Vaisseau": ["temporal"],
                "Categorie": ["artere"],
                "Segments modele": [2],
                "Segments manuels": [1],
                "Composantes": [1],
                "Statut du pont": ["connecte"],
                "Longueur du trajet": [12.0],
                "Corde": [10.0],
                "Tortuosite": [1.2],
                "Debut": ["(0, 0)"],
                "Fin": ["(10, 0)"],
            }
        )

        visible = visible_result_table(rows)

        self.assertEqual(
            visible.columns.tolist(),
            ["Label", "Vaisseau", "Categorie", "Longueur du trajet", "Corde", "Tortuosite"],
        )

    def test_results_overlay_does_not_draw_synthetic_links(self) -> None:
        review_state = {
            "manual_segments": {
                "1": {"points": [[0, 0], [20, 0]]},
            },
            "vessels": {
                "temporal": {
                    "category": "artere",
                    "segment_refs": ["manual:1"],
                    "synthetic_links": [{"points": [[20, 0], [200, 0]], "length": 180.0}],
                }
            },
        }

        segments, labels = build_results_viewer_segments(review_state, pd.DataFrame())

        self.assertEqual([segment["source"] for segment in segments], ["manual"])
        self.assertEqual(len(labels), 1)

    def test_pdf_generation_returns_pdf_bytes(self) -> None:
        result_table = pd.DataFrame(
            {
                "Label": ["V1"],
                "Vaisseau": ["temporal"],
                "Categorie": ["artere"],
                "Longueur du trajet": [10.0],
                "Corde": [10.0],
                "Tortuosite": [1.0],
            }
        )
        raw = build_pvalue_matrix({"image_a": [1.0]})
        adjusted = build_adjusted_pvalue_matrix(raw)
        tiny_image_path = _write_tiny_png()

        pdf_bytes = generate_results_pdf(
            selected_run_names=["image_a"],
            counts_by_run={"image_a": 1},
            loaded_results={
                "image_a": (
                    {"image_path": str(tiny_image_path)},
                    pd.DataFrame(),
                    {"manual_segments": {}, "vessels": {}},
                )
            },
            result_tables={"image_a": result_table},
            raw_pvalue_matrix=raw,
            adjusted_pvalue_matrix=adjusted,
        )

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_local_bump_pvalue_matrices_include_clean_arteries_and_eligible_veins(self) -> None:
        scored_runs = [
            (
                _fake_run_dir("eye_a"),
                {},
                pd.DataFrame(
                    {
                        "vessel_name": [
                            "1°A-sup",
                            "VEINE TEMPORALE INF",
                            "veine courte",
                            "veine score invalide",
                            "unknown",
                        ],
                        "category": ["artere", "veine", "veine", "veine", "artere"],
                        "vessel_length": [50.0, 150.0, 40.0, 150.0, 150.0],
                        "eligible": [False, True, False, False, True],
                        "primary_score": [1.0, 9.9, 7.7, float("nan"), 8.8],
                    }
                ),
            ),
            (
                _fake_run_dir("eye_b"),
                {},
                pd.DataFrame(
                    {
                        "vessel_name": ["2°A-inf", "3°A-sup"],
                        "category": ["artere", "artere"],
                        "vessel_length": [120.0, 130.0],
                        "eligible": [True, True],
                        "primary_score": [2.0, 2.1],
                    }
                ),
            ),
        ]

        raw, adjusted = _build_local_bump_pvalue_matrices(scored_runs)
        expected_raw = build_pvalue_matrix(
            {
                "eye_a": [1.0, 9.9],
                "eye_b": [2.0, 2.1],
            }
        )
        expected_adjusted = build_adjusted_pvalue_matrix(expected_raw)

        pd.testing.assert_frame_equal(raw, expected_raw)
        pd.testing.assert_frame_equal(adjusted, expected_adjusted)

    def test_local_bump_scores_pdf_table_uses_descriptive_columns(self) -> None:
        summary_table = pd.DataFrame(
            {
                "Image": ["eye_a"],
                "Oeil": ["OD"],
                "Methode": ["Score local-bump"],
                "Score median": [1.3],
                "Score moyen": [1.2],
                "Score moyen pondere": [1.4],
                "Vaisseaux retenus": [8],
                "Vaisseaux sauvegardes": [10],
                "Vaisseaux retenus/sauvegardes": ["8/10"],
                "Longueur totale vaisseaux": [300.0],
                "Longueur totale vaisseaux retenus": [250.0],
            }
        )

        display = _local_bump_scores_pdf_table(summary_table)

        self.assertEqual(
            display.columns.tolist(),
            [
                "Image",
                "Oeil",
                "Methode",
                "Score median",
                "Score moyen",
                "Score moyen pondere",
                "Vaisseaux retenus/sauvegardes",
                "Longueur totale vaisseaux",
                "Longueur totale vaisseaux retenus",
            ],
        )

    def test_top_system_table_keeps_columns_through_segments_only(self) -> None:
        top_systems = pd.DataFrame(
            {
                "highlight_label": ["V1"],
                "vessel_name": ["1°A-sup"],
                "category": ["artere"],
                "primary_score": [2.3],
                "vessel_length": [120.0],
                "segment_count": [4],
                "bridge_count": [1],
                "arc_chord_diagnostic": [1.1],
            }
        )

        display = _top_system_table(top_systems)

        self.assertEqual(
            display.columns.tolist(),
            ["Label", "Vaisseau", "Categorie", "Score vaisseau", "Longueur", "Segments"],
        )

    def test_top_system_rows_use_clean_vessels_naming_rules(self) -> None:
        system_scores = pd.DataFrame(
            {
                "vessel_name": ["1°A-sup", "VEINE TEMPORALE INF", "unknown"],
                "category": ["artere", "artere", "artere"],
                "vessel_length": [50.0, 150.0, 150.0],
                "eligible": [False, True, True],
                "primary_score": [1.0, 9.9, 8.8],
            }
        )

        top = _top_system_rows(system_scores)

        self.assertEqual(top["vessel_name"].tolist(), ["1°A-sup"])
        self.assertEqual(top["highlight_label"].tolist(), ["V1"])

    def test_all_systems_table_hides_arc_chord_diagnostic(self) -> None:
        scored_runs = [
            (
                _fake_run_dir("eye_a"),
                {"eye_number": "OD"},
                pd.DataFrame(
                    {
                        "eligible": [True],
                        "vessel_name": ["1°A-sup"],
                        "category": ["artere"],
                        "scoring_method_label": ["Local-bump"],
                        "primary_score": [2.3],
                        "vessel_length": [120.0],
                        "segment_count": [4],
                        "bridge_count": [1],
                        "arc_chord_diagnostic": [1.1],
                        "curvature_squared_score": [0.2],
                        "oscillation_count": [3],
                        "path_points": [[[0.0, 0.0], [1.0, 1.0]]],
                    }
                ),
            )
        ]

        display = _build_all_systems_table(scored_runs)

        self.assertNotIn("Arc/chord diagnostic", display.columns.tolist())

    def test_all_systems_table_uses_clean_vessels_naming_rules(self) -> None:
        scored_runs = [
            (
                _fake_run_dir("eye_a"),
                {"eye_number": "OD"},
                pd.DataFrame(
                    {
                        "vessel_name": ["1°A-sup", "VEINE TEMPORALE INF", "unknown"],
                        "category": ["artere", "artere", "artere"],
                        "vessel_length": [50.0, 150.0, 150.0],
                        "eligible": [False, True, True],
                        "scoring_method_label": ["Local-bump", "Local-bump", "Local-bump"],
                        "primary_score": [1.0, 9.9, 8.8],
                        "segment_count": [2, 3, 4],
                        "bridge_count": [0, 1, 1],
                        "curvature_squared_score": [0.1, 0.2, 0.3],
                        "oscillation_count": [1, 2, 3],
                        "path_points": [
                            [[0.0, 0.0], [1.0, 1.0]],
                            [[0.0, 0.0], [2.0, 2.0]],
                            [[0.0, 0.0], [3.0, 3.0]],
                        ],
                    }
                ),
            )
        ]

        display = _build_all_systems_table(scored_runs)

        self.assertEqual(display["Vaisseau"].tolist(), ["1°A-sup"])
        self.assertEqual(display["Rang"].tolist(), [1])

    def test_atlas_table_uses_active_scoring_eligibility_for_arteries_and_veins(self) -> None:
        scored_runs = [
            (
                _fake_run_dir("eye_a"),
                {"eye_number": "OD"},
                pd.DataFrame(
                    {
                        "vessel_name": ["artery", "vein", "short"],
                        "category": ["artere", "veine", "veine"],
                        "eligible": [True, True, False],
                        "primary_score": [1.0, 3.0, 9.0],
                        "vessel_length": [120.0, 140.0, 20.0],
                        "path_points": [
                            [[0.0, 0.0], [1.0, 1.0]],
                            [[0.0, 0.0], [2.0, 2.0]],
                            [[0.0, 0.0], [3.0, 3.0]],
                        ],
                    }
                ),
            )
        ]

        atlas = _build_atlas_systems_table(scored_runs)

        self.assertEqual(atlas["Vaisseau"].tolist(), ["vein", "artery"])
        self.assertEqual(atlas["Categorie"].tolist(), ["veine", "artere"])

    def test_saved_vessel_overlay_uses_source_image_and_type_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            Image.new("RGB", (40, 40), "black").save(run_dir / "source.png")
            (run_dir / "metadata.json").write_text(
                json.dumps({"image_name": "source.png"}),
                encoding="utf-8",
            )
            output_dir = run_dir / "output"
            output_dir.mkdir()
            Image.new("RGB", (40, 40), "white").save(output_dir / "07b_skeleton_overlay.png")
            saved = pd.DataFrame(
                {
                    "category": ["artere", "veine"],
                    "eligible": [True, False],
                    "path_points": [
                        [[5.0, 10.0], [35.0, 10.0]],
                        [[5.0, 20.0], [35.0, 20.0]],
                    ],
                }
            )

            overlay = _render_saved_vessel_overlay(run_dir, saved)

        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.getpixel((20, 10)), (255, 69, 58))
        self.assertEqual(overlay.getpixel((20, 20)), (76, 141, 255))
        self.assertEqual(overlay.getpixel((20, 30)), (0, 0, 0))

    def test_vessel_type_segmentation_uses_ranked_clean_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "1_OD"
            run_dir.mkdir()
            Image.new("RGB", (120, 120), "black").save(run_dir / "source.png")
            (run_dir / "metadata.json").write_text(json.dumps({"image_name": "source.png"}), encoding="utf-8")
            vessels = pd.DataFrame(
                {
                    "vessel_name": [
                        "1°A-sup",
                        "2°A-inf",
                        "1°V-sup",
                        "2°V-inf",
                        "veine temp sup",
                        "1°A veine ambigu",
                    ],
                    "category": ["artere", "artere", "veine", "veine", "veine", "artere"],
                    "vessel_length": [120.0, 0.0, 120.0, 120.0, 120.0, 120.0],
                    "primary_score": [float("nan"), 0.8, 0.7, 0.6, 0.5, 0.4],
                    "eligible": [False, True, True, False, True, True],
                    "path_points": [
                        [[10.0, 20.0], [110.0, 20.0]],
                        [[10.0, 40.0], [110.0, 40.0]],
                        [[10.0, 60.0], [110.0, 60.0]],
                        [[10.0, 80.0], [110.0, 80.0]],
                        [[10.0, 100.0], [110.0, 100.0]],
                        [[10.0, 110.0], [110.0, 110.0]],
                    ],
                }
            )

            arteries = render_vessel_type_segmentation_image(run_dir, vessels, "artere", False)
            veins = render_vessel_type_segmentation_image(run_dir, vessels, "veine", False)

        self.assertIsNotNone(arteries)
        self.assertIsNotNone(veins)
        assert arteries is not None and veins is not None
        header_height = arteries.height - 120
        self.assertEqual(arteries.getpixel((30, header_height + 20)), (255, 69, 58))
        self.assertEqual(arteries.getpixel((30, header_height + 40)), (0, 0, 0))
        self.assertEqual(arteries.getpixel((30, header_height + 60)), (0, 0, 0))
        self.assertEqual(arteries.getpixel((30, header_height + 110)), (0, 0, 0))
        self.assertEqual(veins.getpixel((30, header_height + 20)), (0, 0, 0))
        self.assertEqual(veins.getpixel((30, header_height + 60)), (76, 141, 255))
        self.assertEqual(veins.getpixel((30, header_height + 80)), (0, 0, 0))
        self.assertEqual(veins.getpixel((30, header_height + 100)), (0, 0, 0))

    def test_labeled_segmentation_adds_normalized_rank_type_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "1_OD"
            run_dir.mkdir()
            Image.new("RGB", (160, 120), "black").save(run_dir / "source.png")
            (run_dir / "metadata.json").write_text(json.dumps({"image_name": "source.png"}), encoding="utf-8")
            vessels = pd.DataFrame(
                {
                    "vessel_name": ["2°A-sup"],
                    "category": ["artere"],
                    "vessel_length": [120.0],
                    "primary_score": [0.8],
                    "eligible": [True],
                    "path_points": [[[20.0, 60.0], [140.0, 60.0]]],
                }
            )

            plain = render_vessel_type_segmentation_image(run_dir, vessels, "artere", False)
            labeled = render_vessel_type_segmentation_image(run_dir, vessels, "artere", True)

        self.assertEqual(_normalized_vessel_label(2, "artere"), "2°A")
        self.assertEqual(_normalized_vessel_label(3, "veine"), "3°V")
        assert plain is not None and labeled is not None
        self.assertIsNotNone(ImageChops.difference(plain, labeled).getbbox())

    def test_vessel_label_rotation_follows_path_and_stays_upright(self) -> None:
        forward = _path_midpoint_and_angle([(0.0, 0.0), (100.0, 100.0)])
        reverse = _path_midpoint_and_angle([(100.0, 100.0), (0.0, 0.0)])

        self.assertEqual(forward, (50.0, 50.0, 45.0))
        self.assertEqual(reverse, (50.0, 50.0, 45.0))

    def test_vessel_type_segmentation_zip_has_four_images_per_eye(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "1_OD"
            run_dir.mkdir()
            Image.new("RGB", (40, 40), "black").save(run_dir / "source.png")
            (run_dir / "metadata.json").write_text(json.dumps({"image_name": "source.png"}), encoding="utf-8")
            archive_bytes = generate_vessel_type_segmentation_zip([(run_dir, {}, pd.DataFrame())])

        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            self.assertEqual(
                sorted(archive.namelist()),
                [
                    "1_OD/arteres.png",
                    "1_OD/arteres_annotees.png",
                    "1_OD/veines.png",
                    "1_OD/veines_annotees.png",
                    "README.txt",
                ],
            )

    def test_vessel_type_segmentation_zip_notes_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "missing_OD"
            run_dir.mkdir()
            archive_bytes = generate_vessel_type_segmentation_zip([(run_dir, {}, pd.DataFrame())])

        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            self.assertEqual(archive.namelist(), ["README.txt"])
            self.assertIn("missing_OD", archive.read("README.txt").decode("utf-8"))

    def test_atlas_crop_keeps_yellow_path_and_uses_type_border(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            Image.new("RGB", (40, 40), "black").save(run_dir / "source.png")
            (run_dir / "metadata.json").write_text(
                json.dumps({"image_name": "source.png"}),
                encoding="utf-8",
            )
            path_points = [[5.0, 20.0], [35.0, 20.0]]
            saved = pd.DataFrame({"category": ["veine"], "path_points": [path_points]})
            row = pd.Series(
                {
                    "_run_dir": run_dir,
                    "_path_points": path_points,
                    "Categorie": "veine",
                }
            )

            crop = _render_ranked_system_crop(row, saved)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.getpixel((0, 0)), (76, 141, 255))
        self.assertEqual(crop.getpixel((260, 150)), (255, 225, 86))

    def test_local_bump_summary_table_sorts_by_weighted_mean(self) -> None:
        scored_runs = [
            (
                _fake_run_dir("eye_a"),
                {
                    "eye_number": "OD",
                    "scoring_method_label": "Local-bump",
                    "scoring_method": "local_bump",
                    "eligible_vessel_count": 2,
                    "saved_vessel_count": 3,
                },
                pd.DataFrame(
                    {
                        "eligible": [True, True, False],
                        "primary_score": [1.0, 3.0, 10.0],
                        "vessel_length": [10.0, 30.0, 50.0],
                    }
                ),
            ),
            (
                _fake_run_dir("eye_b"),
                {
                    "eye_number": "OG",
                    "scoring_method_label": "Local-bump",
                    "scoring_method": "local_bump",
                    "eligible_vessel_count": 2,
                    "saved_vessel_count": 2,
                },
                pd.DataFrame(
                    {
                        "eligible": [True, True],
                        "primary_score": [2.0, 2.0],
                        "vessel_length": [10.0, 10.0],
                    }
                ),
            ),
        ]

        summary_table = _build_local_bump_summary_table(scored_runs)

        self.assertEqual(summary_table["Image"].tolist(), ["eye_a", "eye_b"])
        self.assertEqual(
            summary_table.columns.tolist(),
            [
                "Image",
                "Oeil",
                "Methode",
                "Score median",
                "Score moyen",
                "Score moyen pondere",
                "Vaisseaux retenus",
                "Vaisseaux sauvegardes",
                "Vaisseaux retenus/sauvegardes",
                "Longueur totale vaisseaux",
                "Longueur totale vaisseaux retenus",
            ],
        )
        self.assertEqual(summary_table.loc[0, "Vaisseaux retenus/sauvegardes"], "2/3")

    def test_kept_vessel_count_label_formats_retained_over_saved(self) -> None:
        self.assertEqual(_kept_vessel_count_label(8, 10), "8/10")
        self.assertEqual(_kept_vessel_count_label("0", "0"), "0/0")
        self.assertEqual(_kept_vessel_count_label(None, 10), "NA")

    def test_stats_table_font_size_shrinks_for_large_matrices(self) -> None:
        matrix = pd.DataFrame(
            [[f"{row}-{col}" for col in range(10)] for row in range(3)],
            index=["r1", "r2", "r3"],
            columns=[f"c{col}" for col in range(10)],
        )

        self.assertEqual(_stats_table_font_size(matrix), 9.0)
        self.assertEqual(_stats_table_font_size(pd.DataFrame(columns=[f"c{col}" for col in range(18)])), 8.0)
        self.assertEqual(_stats_table_font_size(pd.DataFrame(columns=[f"c{col}" for col in range(24)])), 7.0)

    def test_pvalue_cell_color_marks_low_values_more_strongly(self) -> None:
        self.assertEqual(_pvalue_cell_color("-"), "#eceff1")
        self.assertEqual(_pvalue_cell_color("NA"), "#eceff1")
        self.assertEqual(_pvalue_cell_color(0.001), "#8b0000")
        self.assertEqual(_pvalue_cell_color(0.03), "#fc8d59")
        self.assertEqual(_pvalue_cell_color(0.8), "#91cf60")

    def test_single_line_cell_text_removes_line_breaks_and_truncates(self) -> None:
        self.assertEqual(_single_line_cell_text("Longueur\ntotale"), "Longueur totale")
        self.assertEqual(_single_line_cell_text("a" * 30, max_chars=10), "aaaaaaaaa…")

    def test_wrapped_cell_text_breaks_long_column_names(self) -> None:
        self.assertEqual(
            _wrapped_cell_text("Longueur totale vaisseaux retenus", line_width=14),
            "Longueur\ntotale\nvaisseaux\nretenus",
        )
        self.assertEqual(
            _wrapped_cell_text("Vaisseaux retenus/sauvegardes", line_width=14),
            "Vaisseaux\nretenus/\nsauvegardes",
        )


def _write_tiny_png():
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    from PIL import Image

    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    path = Path(tmp.name)
    Image.new("RGB", (20, 20), "black").save(path)
    return path


def _fake_run_dir(name: str):
    from pathlib import Path

    return Path(name)


if __name__ == "__main__":
    unittest.main()
