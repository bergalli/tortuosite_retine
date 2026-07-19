from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tortuosite_score.vessels_detection.clinical_excel import (
    SHEET_NAMES,
    build_clinical_analysis_sheets,
    build_structured_vessel_data,
    generate_clinical_excel,
    parse_run_name,
    parse_vessel_name,
)
from tortuosite_score.vessels_detection.scoring import scoring_config


class ClinicalExcelTests(unittest.TestCase):
    def test_parse_ranked_arteries(self) -> None:
        cases = {
            "1°A-inf": (1, "artere", "inf"),
            "1°A-supbis": (1, "artere", "sup"),
            "2°A-sup1": (2, "artere", "sup"),
            "3°A-inf3": (3, "artere", "inf"),
            "artere_1": (1, "artere", None),
            "artere_2": (2, "artere", None),
            "artere_3": (3, "artere", None),
        }

        for name, expected in cases.items():
            with self.subTest(name=name):
                parsed = parse_vessel_name(name)
                self.assertEqual((parsed.rank, parsed.vessel_type, parsed.territory), expected)
                self.assertEqual(parsed.issue, "")

    def test_parse_vein_variants(self) -> None:
        cases = {
            "VEINE TEMPORALE INF": "inf",
            "veine temp sup": "sup",
            "VINE NAS": None,
        }

        for name, territory in cases.items():
            with self.subTest(name=name):
                parsed = parse_vessel_name(name)
                self.assertEqual(parsed.vessel_type, "veine")
                self.assertIsNone(parsed.rank)
                self.assertEqual(parsed.territory, territory)

    def test_manual_name_takes_precedence_over_saved_category(self) -> None:
        parsed = parse_vessel_name("VEINE TEMPORALE INF", category="artere")

        self.assertEqual(parsed.vessel_type, "veine")
        self.assertIsNone(parsed.rank)
        self.assertEqual(parsed.issue, "")

    def test_parse_run_names(self) -> None:
        self.assertEqual(parse_run_name("1_OD").group, "patient")
        self.assertEqual(parse_run_name("1_OG").eye, "OG")
        control = parse_run_name("OD_de_1")
        self.assertEqual(control.group, "temoin")
        self.assertEqual(control.patient_id, 1)
        self.assertEqual(control.eye, "OD")

    def test_structured_data_and_quality_keep_unclassified_names_and_pairings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = _make_run_dirs(root, ["1_OD", "1_OG", "OD_de_1"])
            clinical_path = root / "clinical_metadata.csv"
            clinical_path.write_text("patient_id,age,severite\n1,12,3\n", encoding="utf-8")

            with patch(
                "tortuosite_score.vessels_detection.clinical_excel.score_run",
                side_effect=lambda run_dir, config: ({}, _fake_scores(Path(run_dir).name)),
            ):
                data, quality = build_structured_vessel_data(run_dirs, scoring_config(), clinical_path)
                sheets = build_clinical_analysis_sheets(run_dirs, scoring_config(), clinical_path)

        self.assertIn("branche nasale", data["vessel_name"].tolist())
        self.assertIn("type_non_classe", quality["probleme"].tolist())
        patient_vs_control = sheets["Patient_vs_temoin"]
        self.assertIn("delta_patient_temoin", patient_vs_control.columns.tolist())
        self.assertTrue((patient_vs_control["run_patient"] == "1_OD").any())
        self.assertIn("temoin_manquant", quality["probleme"].tolist())
        od_vs_og = sheets["OD_vs_OG"]
        self.assertIn("delta_OD_OG", od_vs_og.columns.tolist())
        same_eye = sheets["Rangs_meme_oeil"]
        self.assertEqual(same_eye.iloc[0]["section"], "resume")
        self.assertIn("taille_effet_biserielle", same_eye.columns.tolist())
        self.assertIn("detail_patient", same_eye["section"].tolist())

    def test_excel_export_contains_expected_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = _make_run_dirs(root, ["1_OD", "1_OG", "OD_de_1"])
            clinical_path = root / "clinical_metadata.csv"
            clinical_path.write_text("patient_id,age,severite\n1,12,3\n", encoding="utf-8")

            with patch(
                "tortuosite_score.vessels_detection.clinical_excel.score_run",
                side_effect=lambda run_dir, config: ({}, _fake_scores(Path(run_dir).name)),
            ):
                excel_bytes = generate_clinical_excel(run_dirs, scoring_config(), clinical_path)

        workbook = pd.ExcelFile(io.BytesIO(excel_bytes))
        self.assertEqual(workbook.sheet_names, SHEET_NAMES)
        structured = pd.read_excel(workbook, "Data_structuree")
        self.assertIn("patient_id", structured.columns.tolist())
        self.assertIn("score", structured.columns.tolist())


def _make_run_dirs(root: Path, names: list[str]) -> list[Path]:
    run_dirs = []
    for name in names:
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "manual_review_state.json").write_text('{"schema_version": 2, "vessels": {}}', encoding="utf-8")
        run_dirs.append(run_dir)
    return run_dirs


def _fake_scores(run_name: str) -> pd.DataFrame:
    base_scores = {
        "1_OD": [1.0, 1.4, 2.0, 0.8],
        "1_OG": [1.1, 1.5, 1.9, 0.9],
        "OD_de_1": [0.7, 1.0, 1.2, 0.6],
    }[run_name]
    return pd.DataFrame(
        {
            "eligible": [True, True, True, True],
            "vessel_name": ["1°A-inf", "2°A-sup1", "3°A-inf3", "branche nasale"],
            "category": ["artere", "artere", "artere", "unknown"],
            "primary_score": base_scores,
            "vessel_length": [100.0, 120.0, 140.0, 80.0],
            "scoring_method_label": ["Local-bump"] * 4,
        }
    )


if __name__ == "__main__":
    unittest.main()
