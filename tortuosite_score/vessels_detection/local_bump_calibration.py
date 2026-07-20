from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tortuosite_score.vessels_detection.local_bump_score import (
    DISPLAY_SCALE,
    LocalBumpSettings,
    local_bump_metrics,
    local_bump_v2_metrics,
)
from tortuosite_score.vessels_detection.scoring import score_run, scoring_config

PAIRWISE_COLUMNS = ["left_run", "left_vessel", "right_run", "right_vessel", "judgment"]
VALID_JUDGMENTS = {"left", "right", "similar"}


@dataclass(frozen=True)
class V2Parameters:
    resample_step: float = 4.0
    smoothing_window: int = 5
    minimum_lobe_angle: float = 0.15
    angularity_weight: float = 0.25

    def settings(self) -> LocalBumpSettings:
        return LocalBumpSettings(
            resample_step=float(self.resample_step),
            smoothing_window=int(self.smoothing_window),
            min_persistent_lobe_angle=float(self.minimum_lobe_angle),
            local_bump_v2_angularity_weight=float(self.angularity_weight),
            filter_short_vessels=False,
        )


def pairwise_label_template() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIRWISE_COLUMNS)


def write_pairwise_label_template(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    template = pairwise_label_template()
    if path.suffix.lower() == ".csv":
        template.to_csv(path, index=False)
    else:
        instructions = pd.DataFrame(
            [
                {
                    "field": "judgment",
                    "instruction": "Use left, right, or similar. Collect a larger expert set before calibration.",
                },
                {
                    "field": "example",
                    "instruction": "9_OD | 1°A-sup | 5_OG | 3°A-inf1 | left",
                },
            ]
        )
        with pd.ExcelWriter(path) as writer:
            template.to_excel(writer, index=False, sheet_name="Expert_pairs")
            instructions.to_excel(writer, index=False, sheet_name="Instructions")
    return path


def read_pairwise_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    labels = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    return validate_pairwise_labels(labels)


def validate_pairwise_labels(labels: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PAIRWISE_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"Missing pairwise label columns: {', '.join(missing)}")
    clean = labels[PAIRWISE_COLUMNS].copy().dropna(subset=PAIRWISE_COLUMNS)
    for column in PAIRWISE_COLUMNS:
        clean[column] = clean[column].astype(str).str.strip()
    clean["judgment"] = clean["judgment"].str.lower()
    invalid = sorted(set(clean["judgment"]) - VALID_JUDGMENTS)
    if invalid:
        raise ValueError(f"Invalid judgments: {', '.join(invalid)}; expected left, right, or similar")
    if clean.empty:
        raise ValueError("The pairwise label file contains no complete comparisons.")
    return clean.reset_index(drop=True)


def default_parameter_grid() -> list[V2Parameters]:
    return [
        V2Parameters(step, window, lobe_angle, weight)
        for step, window, lobe_angle, weight in itertools.product(
            [3.0, 4.0, 5.0],
            [5, 7, 9],
            [0.10, 0.15, 0.20],
            [0.10, 0.25, 0.40],
        )
    ]


def collect_referenced_geometries(
    runs_root: str | Path,
    labels: pd.DataFrame,
) -> dict[tuple[str, str], np.ndarray]:
    runs_root = Path(runs_root)
    requested = {
        (str(row[f"{side}_run"]), str(row[f"{side}_vessel"]))
        for _, row in labels.iterrows()
        for side in ("left", "right")
    }
    geometries: dict[tuple[str, str], np.ndarray] = {}
    for run_name in sorted({run for run, _vessel in requested}):
        _summary, scores = score_run(
            runs_root / run_name,
            scoring_config("local_bump", LocalBumpSettings(filter_short_vessels=False)),
        )
        for _, row in scores.iterrows():
            key = (run_name, str(row["vessel_name"]))
            if key not in requested:
                continue
            geometries[key] = np.asarray(row["path_points"], dtype=float) * float(row["coordinate_scale"])
    missing = sorted(requested - set(geometries))
    if missing:
        formatted = ", ".join(f"{run}/{vessel}" for run, vessel in missing)
        raise ValueError(f"Pairwise labels reference unknown saved vessels: {formatted}")
    return geometries


def score_geometries(
    geometries: dict[tuple[str, str], np.ndarray],
    parameters: V2Parameters,
) -> dict[tuple[str, str], float]:
    settings = parameters.settings()
    return {
        key: float(local_bump_v2_metrics(points, settings)["local_bump_v2_score"])
        for key, points in geometries.items()
    }


def score_geometries_v1(
    geometries: dict[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], float]:
    settings = LocalBumpSettings()
    return {
        key: float(local_bump_metrics(points, settings)["branch_bump_score"] * DISPLAY_SCALE)
        for key, points in geometries.items()
    }


def evaluate_pairwise_scores(
    labels: pd.DataFrame,
    scores: dict[tuple[str, str], float],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for index, pair in labels.iterrows():
        left_key = (str(pair["left_run"]), str(pair["left_vessel"]))
        right_key = (str(pair["right_run"]), str(pair["right_vessel"]))
        left_score = float(scores[left_key])
        right_score = float(scores[right_key])
        judgment = str(pair["judgment"])
        strict = judgment != "similar"
        correct = (
            left_score > right_score if judgment == "left" else right_score > left_score if judgment == "right" else math.nan
        )
        relative_gap = float(2.0 * abs(left_score - right_score) / (abs(left_score) + abs(right_score) + 1e-12))
        rows.append(
            {
                "pair_index": int(index),
                **pair.to_dict(),
                "left_score": left_score,
                "right_score": right_score,
                "strict_pair": strict,
                "strict_correct": correct,
                "similar_relative_gap": relative_gap if not strict else math.nan,
            }
        )
    details = pd.DataFrame(rows)
    strict_values = details.loc[details["strict_pair"], "strict_correct"].astype(float)
    similar_values = pd.to_numeric(
        details.loc[~details["strict_pair"], "similar_relative_gap"], errors="coerce"
    ).dropna()
    metrics: dict[str, float | int] = {
        "strict_pair_count": int(len(strict_values)),
        "strict_concordance": float(strict_values.mean()) if len(strict_values) else math.nan,
        "similar_pair_count": int(len(similar_values)),
        "mean_similar_relative_gap": float(similar_values.mean()) if len(similar_values) else math.nan,
    }
    return metrics, details


def select_best_parameters(
    labels: pd.DataFrame,
    geometries: dict[tuple[str, str], np.ndarray],
    parameter_grid: list[V2Parameters] | None = None,
) -> tuple[V2Parameters, pd.DataFrame]:
    parameter_grid = parameter_grid or default_parameter_grid()
    if not parameter_grid:
        raise ValueError("The parameter grid is empty.")
    rows: list[dict[str, object]] = []
    for parameters in parameter_grid:
        metrics, _details = evaluate_pairwise_scores(labels, score_geometries(geometries, parameters))
        rows.append({**asdict(parameters), **metrics})
    evaluation = pd.DataFrame(rows)
    sortable = evaluation.assign(
        _concordance=pd.to_numeric(evaluation["strict_concordance"], errors="coerce").fillna(-1.0),
        _similar_gap=pd.to_numeric(evaluation["mean_similar_relative_gap"], errors="coerce").fillna(float("inf")),
    ).sort_values(
        ["_concordance", "_similar_gap", "angularity_weight", "smoothing_window"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    best_row = sortable.iloc[0]
    best = V2Parameters(
        float(best_row["resample_step"]),
        int(best_row["smoothing_window"]),
        float(best_row["minimum_lobe_angle"]),
        float(best_row["angularity_weight"]),
    )
    return best, evaluation.sort_values(
        ["strict_concordance", "mean_similar_relative_gap"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def leave_one_patient_out_calibration(
    labels: pd.DataFrame,
    geometries: dict[tuple[str, str], np.ndarray],
    parameter_grid: list[V2Parameters] | None = None,
    bootstrap_samples: int = 2000,
    random_seed: int = 0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels["_patients"] = labels.apply(_pair_patients, axis=1)
    patients = sorted({patient for values in labels["_patients"] for patient in values})
    if len(patients) < 2:
        raise ValueError("Leave-one-patient-out calibration needs comparisons involving at least two patients.")

    baseline_scores = score_geometries_v1(geometries)
    fold_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for patient in patients:
        involves_patient = labels["_patients"].apply(lambda values: patient in values)
        train = labels.loc[~involves_patient, PAIRWISE_COLUMNS].reset_index(drop=True)
        test = labels.loc[involves_patient, PAIRWISE_COLUMNS].reset_index(drop=True)
        if train.empty or test.empty:
            continue
        best, _grid = select_best_parameters(train, geometries, parameter_grid)
        v2_metrics, v2_details = evaluate_pairwise_scores(test, score_geometries(geometries, best))
        v1_metrics, v1_details = evaluate_pairwise_scores(test, baseline_scores)
        improvement = _finite_difference(v2_metrics["strict_concordance"], v1_metrics["strict_concordance"])
        fold_rows.append(
            {
                "held_out_patient": patient,
                **asdict(best),
                "test_pair_count": int(len(test)),
                "v1_strict_concordance": v1_metrics["strict_concordance"],
                "v2_strict_concordance": v2_metrics["strict_concordance"],
                "strict_concordance_improvement": improvement,
                "v1_similar_gap": v1_metrics["mean_similar_relative_gap"],
                "v2_similar_gap": v2_metrics["mean_similar_relative_gap"],
            }
        )
        v2_details = v2_details.add_prefix("v2_")
        v1_details = v1_details[["left_score", "right_score"]].add_prefix("v1_")
        combined = pd.concat([v2_details, v1_details], axis=1)
        combined.insert(0, "held_out_patient", patient)
        detail_frames.append(combined)

    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        raise ValueError("No patient produced both a non-empty training and test fold.")
    improvements = pd.to_numeric(folds["strict_concordance_improvement"], errors="coerce").dropna().to_numpy()
    lower, upper = bootstrap_mean_interval(improvements, bootstrap_samples, random_seed)
    overall_best, grid_evaluation = select_best_parameters(labels[PAIRWISE_COLUMNS], geometries, parameter_grid)
    summary: dict[str, object] = {
        "method": "local_bump_v2",
        "experimental": True,
        "overall_best_parameters": asdict(overall_best),
        "held_out_patient_count": int(len(folds)),
        "mean_strict_concordance_improvement": float(np.mean(improvements)) if improvements.size else math.nan,
        "bootstrap_improvement_ci95": [lower, upper],
        "promotion_recommended": bool(math.isfinite(lower) and lower >= 0.0),
        "promotion_rule": "lower bound of patient-level bootstrap CI for v2-v1 strict concordance is >= 0",
        "parameter_grid_size": int(len(grid_evaluation)),
    }
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary, folds, details


def bootstrap_mean_interval(
    values: np.ndarray | list[float],
    samples: int = 2000,
    random_seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, values.size, size=(max(int(samples), 1), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _pair_patients(pair: pd.Series) -> tuple[int, ...]:
    patients = {_patient_from_run(str(pair["left_run"])), _patient_from_run(str(pair["right_run"]))}
    patients.discard(None)
    return tuple(sorted(int(patient) for patient in patients))


def _patient_from_run(run_name: str) -> int | None:
    match = re.search(r"(?:^|_de_)(\d+)", run_name)
    return int(match.group(1)) if match else None


def _finite_difference(left: object, right: object) -> float:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return math.nan
    return left_value - right_value if math.isfinite(left_value) and math.isfinite(right_value) else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate experimental local-bump v2 from expert vessel pairs.")
    parser.add_argument("labels", nargs="?", help="CSV or XLSX with left_run/left_vessel/right_run/right_vessel/judgment")
    parser.add_argument("--runs-root", default="demo/streamlit_runs")
    parser.add_argument("--output-dir", default="demo/local_bump_v2_calibration")
    parser.add_argument("--write-template", help="Write an expert-pair CSV/XLSX template and exit")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        print(write_pairwise_label_template(args.write_template))
        return
    if not args.labels:
        raise SystemExit("Provide a pairwise label file or use --write-template.")
    labels = read_pairwise_labels(args.labels)
    geometries = collect_referenced_geometries(args.runs_root, labels)
    summary, folds, details = leave_one_patient_out_calibration(
        labels,
        geometries,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    folds.to_csv(output_dir / "patient_folds.csv", index=False)
    details.to_csv(output_dir / "held_out_pair_evaluation.csv", index=False)
    _best, grid_evaluation = select_best_parameters(labels, geometries)
    grid_evaluation.to_csv(output_dir / "full_parameter_grid.csv", index=False)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
