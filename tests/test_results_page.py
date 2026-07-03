from __future__ import annotations

import unittest

import pandas as pd

from tortuosite_score.app.results_page import (
    build_adjusted_pvalue_matrix,
    build_pvalue_matrix,
    build_result_rows,
    generate_results_pdf,
    tortuosity_values,
    visible_result_table,
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


def _write_tiny_png():
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    from PIL import Image

    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    path = Path(tmp.name)
    Image.new("RGB", (20, 20), "black").save(path)
    return path


if __name__ == "__main__":
    unittest.main()
