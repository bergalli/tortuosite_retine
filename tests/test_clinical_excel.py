from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tortuosite_score.vessels_detection.clinical_excel import (
    CLASSIFIED_VESSEL_SHEET_NAMES,
    SHEET_NAMES,
    build_clinical_analysis_sheets,
    build_structured_vessel_data,
    generate_classified_vessels_excel,
    generate_clinical_excel,
    generate_clinical_excel_outputs,
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
        self.assertIn("temoin_manquant", quality["probleme"].tolist())
        summary = sheets["Resume"]
        self.assertEqual(summary["Comparaison"].tolist(), ["R3 vs R1", "R3 vs R2"])
        self.assertIn("Conclusion", summary.columns.tolist())
        per_eye = sheets["Par_oeil"]
        self.assertEqual(per_eye["Run"].tolist(), ["1_OD", "1_OG"])
        self.assertIn("R3_superieur_R1", per_eye.columns.tolist())
        self.assertIn("R3_superieur_R2", per_eye.columns.tolist())

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
        summary = pd.read_excel(workbook, "Resume")
        per_eye = pd.read_excel(workbook, "Par_oeil")
        self.assertEqual(summary["Comparaison"].tolist(), ["R3 vs R1", "R3 vs R2"])
        self.assertIn("Score_R3", per_eye.columns.tolist())
        self.assertIn("Delta_R3_R1", per_eye.columns.tolist())

    def test_only_ranked_arteries_with_positive_length_are_analysis_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = _make_run_dirs(root, ["1_OD"])
            scores = pd.DataFrame(
                {
                    "eligible": [False, False, False, False],
                    "vessel_name": ["1°A-inf", "VEINE TEMPORALE SUP", "branche nasale", "3°A-inf1"],
                    "category": ["artere", "veine", "artere", "artere"],
                    "primary_score": [1.0, 0.8, 0.7, float("nan")],
                    "vessel_length": [40.0, 45.0, 50.0, 0.0],
                    "scoring_method_label": ["Local-bump"] * 4,
                }
            )
            with patch(
                "tortuosite_score.vessels_detection.clinical_excel.score_run",
                return_value=({}, scores),
            ):
                data, quality = build_structured_vessel_data(run_dirs, scoring_config(), root / "missing.csv")

        eligibility = data.set_index("vessel_name")["eligible_analyse"].to_dict()
        self.assertTrue(eligibility["1°A-inf"])
        self.assertFalse(eligibility["VEINE TEMPORALE SUP"])
        self.assertFalse(eligibility["branche nasale"])
        self.assertFalse(eligibility["3°A-inf1"])
        self.assertEqual(
            data.set_index("vessel_name").loc["1°A-inf", "raison_exclusion_analyse"],
            "",
        )
        self.assertEqual(
            data.set_index("vessel_name").loc["branche nasale", "raison_exclusion_analyse"],
            "unclassified_name",
        )
        self.assertEqual(
            data.set_index("vessel_name").loc["VEINE TEMPORALE SUP", "raison_exclusion_analyse"],
            "unclassified_name",
        )
        self.assertEqual(
            data.set_index("vessel_name").loc["3°A-inf1", "raison_exclusion_analyse"],
            "zero_length",
        )
        self.assertIn("vaisseau_court_conserve_nom_valide", quality["probleme"].tolist())

    def test_classified_vessel_excel_separates_clean_and_rejected_vessels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = _make_run_dirs(root, ["1_OD", "1_OG", "OD_de_1"])
            with patch(
                "tortuosite_score.vessels_detection.clinical_excel.score_run",
                side_effect=lambda run_dir, config: ({}, _fake_scores(Path(run_dir).name)),
            ):
                excel_bytes = generate_classified_vessels_excel(run_dirs, scoring_config(), root / "missing.csv")

        workbook = pd.ExcelFile(io.BytesIO(excel_bytes))
        self.assertEqual(workbook.sheet_names, CLASSIFIED_VESSEL_SHEET_NAMES)
        clean = pd.read_excel(workbook, "Clean_vessels")
        rejected = pd.read_excel(workbook, "Rejected_vessels")
        expected_columns = [
            "patient_id",
            "group",
            "eye",
            "run",
            "vessel_name",
            "class_number",
            "vessel_type",
            "rank",
            "territory",
            "length_raw_px",
            "length_normalized_px",
            "fundus_diameter_px",
            "reference_diameter_px",
            "normalization_factor",
            "analysis_eligible",
            "analysis_exclusion_reason",
            "score_raw",
            "score_normalized",
        ]
        self.assertEqual(
            clean.columns.tolist(),
            expected_columns,
        )
        self.assertEqual(rejected.columns.tolist(), expected_columns)
        self.assertEqual(len(clean), 9)
        self.assertEqual(len(rejected), 3)
        self.assertTrue(clean["analysis_eligible"].all())
        self.assertFalse(rejected["analysis_eligible"].any())
        self.assertTrue(rejected["analysis_exclusion_reason"].notna().all())
        self.assertEqual(set(clean["group"]), {"patient", "control"})
        self.assertEqual(set(clean["eye"]), {"right", "left"})
        self.assertIn("artery", set(clean["vessel_type"]))
        self.assertIn("inferior", set(clean["territory"].dropna()))
        self.assertNotIn("global_weight", clean.columns)
        self.assertNotIn("high_tail_weight", clean.columns)

    def test_combined_outputs_score_each_run_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = _make_run_dirs(root, ["1_OD", "1_OG", "OD_de_1"])
            with patch(
                "tortuosite_score.vessels_detection.clinical_excel.score_run",
                side_effect=lambda run_dir, config: ({}, _fake_scores(Path(run_dir).name)),
            ) as mocked_score_run:
                analysis_bytes, classified_bytes = generate_clinical_excel_outputs(
                    run_dirs,
                    scoring_config(),
                    root / "missing.csv",
                )

        self.assertEqual(mocked_score_run.call_count, 3)
        self.assertEqual(pd.ExcelFile(io.BytesIO(analysis_bytes)).sheet_names, SHEET_NAMES)
        self.assertEqual(
            pd.ExcelFile(io.BytesIO(classified_bytes)).sheet_names,
            CLASSIFIED_VESSEL_SHEET_NAMES,
        )


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
            "eligible": [True, True, True, False],
            "vessel_name": ["1°A-inf", "2°A-sup1", "3°A-inf3", "branche nasale"],
            "category": ["artere", "artere", "artere", "unknown"],
            "primary_score": base_scores,
            "vessel_length": [100.0, 120.0, 140.0, 80.0],
            "scoring_method_label": ["Local-bump"] * 4,
        }
    )


if __name__ == "__main__":
    unittest.main()
