"""Synthetic comparative benchmark for retinal-vessel tortuosity measures.

Generates controlled synthetic curves -- amplitude, oscillation frequency, and
global bend varied independently, plus noise and preprocessing-parameter
sweeps -- and scores them with every registered tortuosity metric (the six
pre-existing methods plus Grisan). It reports:

- Sensitivity: does the score respond to the family's control variable in the
  expected direction?
- Ranking concordance: Spearman correlation between a method's score ordering
  and the family's known-by-construction ordering.
- Robustness to noise and to resampling/smoothing: how much does ranking
  concordance against a clean reference ordering degrade when the input is
  corrupted by noise (Monte Carlo, mirroring Grisan et al. 2008 Sec. IV-C /
  Table II) or when each method's own preprocessing parameters are varied?

There is no real expert-graded data in this repository (only an empty
pairwise-judgment template in ``local_bump_calibration.py``), so "expert
agreement" is intentionally out of scope here. Ranking concordance instead
uses each synthetic family's known-by-construction ordering as ground truth,
mirroring the simulated-vessel analysis in Grisan et al. (2008) Sec. IV-A / V
rather than real clinician agreement.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tortuosite_score.vessels_detection.local_bump_score import (
    LocalBumpSettings,
    arc_chord_metrics,
    curvature_squared_metrics,
    external_angle_sum_metrics,
    grisan_metrics,
    local_bump_metrics,
    local_bump_v2_metrics,
    tortuosity_density_metrics,
)

DEFAULT_METHOD_IDS: tuple[str, ...] = (
    "local_bump",
    "local_bump_v2",
    "arc_chord",
    "curvature_squared",
    "tortuosity_density",
    "external_angle_sum",
    "grisan",
)

_METHOD_METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, LocalBumpSettings], dict[str, object]]] = {
    "local_bump": local_bump_metrics,
    "local_bump_v2": local_bump_v2_metrics,
    "arc_chord": arc_chord_metrics,
    "curvature_squared": curvature_squared_metrics,
    "tortuosity_density": tortuosity_density_metrics,
    "external_angle_sum": external_angle_sum_metrics,
    "grisan": grisan_metrics,
}

_METHOD_SCORE_KEYS: dict[str, str] = {
    "local_bump": "branch_bump_score",
    "local_bump_v2": "local_bump_v2_score",
    "arc_chord": "arc_chord_score",
    "curvature_squared": "curvature_squared_score",
    "tortuosity_density": "tortuosity_density_score",
    "external_angle_sum": "external_angle_sum_score",
    "grisan": "grisan_score",
}


def score_curve(points: np.ndarray, method_id: str, settings: LocalBumpSettings) -> float:
    """Score a bare synthetic polyline with one registered method, by id."""
    metrics = _METHOD_METRIC_FUNCTIONS[method_id](points, settings)
    value = metrics[_METHOD_SCORE_KEYS[method_id]]
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


@dataclass(frozen=True)
class SweepPoint:
    control_value: float
    points: np.ndarray
    repetition: int = 0


@dataclass(frozen=True)
class CurveFamily:
    name: str
    expected_direction: str  # "increasing" or "non_increasing"
    samples: list[SweepPoint] = field(default_factory=list)


def _sinusoid(amplitude: float, frequency: float, domain: float = 6.0 * math.pi, point_count: int = 400, x_scale: float = 20.0) -> np.ndarray:
    """A vessel of fixed chord-like extent whose local wiggle has fixed amplitude and a
    tunable oscillation frequency, matching the structure of eq. (20) in Grisan et al. (2008).
    """
    l = np.linspace(0.0, domain, point_count)
    x = l * x_scale
    y = amplitude * np.sin(frequency * l)
    return np.column_stack([x, y])


def _bent_oscillation(
    bend_angle_deg: float,
    chord_length: float = 300.0,
    local_amplitude: float = 25.0,
    local_cycles: float = 5.0,
    point_count: int = 500,
) -> np.ndarray:
    """A fixed local oscillation riding on a circular-arc baseline of variable sweep angle.

    The baseline arc's chord stays constant as ``bend_angle_deg`` grows, so any score
    increase with bend reflects the metric conflating a smooth, large-scale bend with
    tortuosity -- the failure mode the paper itself calls out (Sec. I, VI): a vessel
    following the curved retina should not be scored as more tortuous just because it
    follows a bigger arc.
    """
    bend_angle = max(math.radians(bend_angle_deg), 1e-6)
    radius = (chord_length / 2.0) / math.sin(bend_angle / 2.0)
    theta = np.linspace(-bend_angle / 2.0, bend_angle / 2.0, point_count)
    base_x = radius * np.sin(theta)
    base_y = radius * (1.0 - np.cos(theta))
    arc_param = np.linspace(0.0, local_cycles * 2.0 * math.pi, point_count)
    offset = local_amplitude * np.sin(arc_param)
    normal_x, normal_y = -np.sin(theta), np.cos(theta)
    x = base_x + offset * normal_x
    y = base_y + offset * normal_y
    return np.column_stack([x, y])


def amplitude_family(
    amplitudes: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0),
    frequency: float = 1.0,
) -> CurveFamily:
    samples = [SweepPoint(amplitude, _sinusoid(amplitude, frequency)) for amplitude in amplitudes]
    return CurveFamily("amplitude", "increasing", samples)


def frequency_family(
    frequencies: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    amplitude: float = 15.0,
) -> CurveFamily:
    samples = [SweepPoint(frequency, _sinusoid(amplitude, frequency)) for frequency in frequencies]
    return CurveFamily("frequency", "increasing", samples)


def global_bend_family(
    bend_angles_deg: tuple[float, ...] = (1.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
) -> CurveFamily:
    samples = [SweepPoint(bend_angle, _bent_oscillation(bend_angle)) for bend_angle in bend_angles_deg]
    return CurveFamily("global_bend", "non_increasing", samples)


def default_families() -> list[CurveFamily]:
    return [amplitude_family(), frequency_family(), global_bend_family()]


def family_score_table(
    families: list[CurveFamily],
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS,
    settings: LocalBumpSettings | None = None,
) -> pd.DataFrame:
    """Raw long-format table: one row per (family, control value, repetition, method)."""
    settings = settings or LocalBumpSettings()
    rows: list[dict[str, object]] = []
    for family in families:
        for sample in family.samples:
            for method_id in method_ids:
                rows.append(
                    {
                        "family": family.name,
                        "control_value": sample.control_value,
                        "repetition": sample.repetition,
                        "method_id": method_id,
                        "score": score_curve(sample.points, method_id, settings),
                    }
                )
    return pd.DataFrame(rows)


def _spearman_or_nan(scores: np.ndarray, control_values: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(control_values)
    if int(valid.sum()) < 3 or np.ptp(control_values[valid]) <= 0 or np.ptp(scores[valid]) <= 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        correlation, _p_value = spearmanr(scores[valid], control_values[valid])
    return float(correlation)


def sensitivity_and_concordance_report(
    families: list[CurveFamily],
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS,
    settings: LocalBumpSettings | None = None,
) -> pd.DataFrame:
    """Per (family, method): Spearman concordance with the family's known ordering,
    plus the relative range and validity of the raw scores.
    """
    settings = settings or LocalBumpSettings()
    rows: list[dict[str, object]] = []
    for family in families:
        control_values = np.asarray([sample.control_value for sample in family.samples], dtype=float)
        for method_id in method_ids:
            scores = np.asarray(
                [score_curve(sample.points, method_id, settings) for sample in family.samples],
                dtype=float,
            )
            valid = np.isfinite(scores)
            finite_scores = scores[valid]
            if finite_scores.size:
                score_min, score_max = float(np.min(finite_scores)), float(np.max(finite_scores))
                denominator = max(abs(score_max), abs(score_min), 1e-12)
                relative_range = float((score_max - score_min) / denominator)
            else:
                score_min = score_max = relative_range = math.nan
            rows.append(
                {
                    "family": family.name,
                    "expected_direction": family.expected_direction,
                    "method_id": method_id,
                    "spearman_vs_control_value": _spearman_or_nan(scores, control_values),
                    "relative_range": relative_range,
                    "min_score": score_min,
                    "max_score": score_max,
                    "valid_fraction": float(valid.mean()) if scores.size else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _inject_noise(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian sample-position noise, matching eq. (21)-(22) of the paper."""
    if sigma <= 0:
        return points.copy()
    return points + rng.normal(0.0, sigma, size=points.shape)


def noise_robustness_report(
    reference_family: CurveFamily,
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS,
    settings: LocalBumpSettings | None = None,
    sigmas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    repetitions: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """For each noise level, repeatedly corrupt every curve in ``reference_family`` and
    measure how well the resulting score ordering still matches the family's clean,
    known-by-construction ordering. Reports median/std of that Spearman correlation per
    (method, noise level), the same style of statistic as Table II / Figs. 12-13 of the
    paper, generalized across multiple noise levels instead of the paper's single one.
    """
    settings = settings or LocalBumpSettings()
    rng = np.random.default_rng(seed)
    control_values = np.asarray([sample.control_value for sample in reference_family.samples], dtype=float)
    rows: list[dict[str, object]] = []
    for method_id in method_ids:
        for sigma in sigmas:
            correlations: list[float] = []
            for _repetition in range(repetitions):
                scores = np.asarray(
                    [
                        score_curve(_inject_noise(sample.points, sigma, rng), method_id, settings)
                        for sample in reference_family.samples
                    ],
                    dtype=float,
                )
                correlation = _spearman_or_nan(scores, control_values)
                if math.isfinite(correlation):
                    correlations.append(correlation)
            correlations_array = np.asarray(correlations, dtype=float)
            rows.append(
                {
                    "method_id": method_id,
                    "noise_sigma": sigma,
                    "repetition_count": int(correlations_array.size),
                    "median_spearman_vs_clean_ranking": float(np.median(correlations_array)) if correlations_array.size else math.nan,
                    "std_spearman_vs_clean_ranking": float(np.std(correlations_array)) if correlations_array.size else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _preprocessing_variants(method_id: str, base: LocalBumpSettings) -> list[tuple[str, LocalBumpSettings]]:
    """Each method's own resampling/smoothing-style knobs, swept from tight to loose."""
    if method_id in {"local_bump", "local_bump_v2", "tortuosity_density"}:
        return [
            (f"resample={step:.0f}px_window={window}", replace(base, resample_step=step, smoothing_window=window))
            for step, window in [(2.0, 3), (4.0, 5), (8.0, 9), (16.0, 15)]
        ]
    if method_id == "grisan":
        return [
            (f"spline_s={s:.0f}", replace(base, grisan_spline_smoothing=s))
            for s in (0.0, 10.0, 20.0, 50.0, 100.0)
        ]
    if method_id == "external_angle_sum":
        return [
            (f"rdp_epsilon={epsilon:.0f}", replace(base, rdp_epsilon=epsilon))
            for epsilon in (1.0, 3.0, 6.0, 10.0)
        ]
    if method_id == "curvature_squared":
        return [
            ("resample_off", replace(base, resample_curvature_squared=False)),
            ("resample=4px", replace(base, resample_curvature_squared=True, resample_step=4.0)),
            ("resample=8px", replace(base, resample_curvature_squared=True, resample_step=8.0)),
        ]
    return [("default", base)]


def preprocessing_robustness_report(
    reference_family: CurveFamily,
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS,
    base_settings: LocalBumpSettings | None = None,
) -> pd.DataFrame:
    """For each method's own preprocessing knob(s): does the family ranking survive
    (Spearman vs. the clean ordering), and how much does the score on one representative
    curve wobble (coefficient of variation) as the preprocessing setting changes?
    """
    base_settings = base_settings or LocalBumpSettings()
    control_values = np.asarray([sample.control_value for sample in reference_family.samples], dtype=float)
    mid_index = len(reference_family.samples) // 2
    rows: list[dict[str, object]] = []
    for method_id in method_ids:
        variants = _preprocessing_variants(method_id, base_settings)
        representative_scores: list[float] = []
        variant_rows: list[dict[str, object]] = []
        for label, variant_settings in variants:
            scores = np.asarray(
                [score_curve(sample.points, method_id, variant_settings) for sample in reference_family.samples],
                dtype=float,
            )
            representative_scores.append(float(scores[mid_index]))
            variant_rows.append(
                {
                    "method_id": method_id,
                    "preprocessing_variant": label,
                    "spearman_vs_clean_ranking": _spearman_or_nan(scores, control_values),
                }
            )
        finite_representative = np.asarray(representative_scores, dtype=float)
        finite_representative = finite_representative[np.isfinite(finite_representative)]
        if finite_representative.size >= 2 and np.mean(np.abs(finite_representative)) > 1e-12:
            coefficient_of_variation = float(np.std(finite_representative) / np.mean(np.abs(finite_representative)))
        else:
            coefficient_of_variation = math.nan
        for row in variant_rows:
            row["representative_curve_coefficient_of_variation"] = coefficient_of_variation
        rows.extend(variant_rows)
    return pd.DataFrame(rows)


def run_benchmark(
    output_dir: str | Path,
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS,
    settings: LocalBumpSettings | None = None,
    noise_sigmas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    noise_repetitions: int = 30,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    settings = settings or LocalBumpSettings()
    families = default_families()
    reference_family = families[0]  # amplitude family: widest, cleanest known ordering

    reports = {
        "raw_scores": family_score_table(families, method_ids, settings),
        "sensitivity_concordance": sensitivity_and_concordance_report(families, method_ids, settings),
        "noise_robustness": noise_robustness_report(
            reference_family,
            method_ids,
            settings,
            sigmas=noise_sigmas,
            repetitions=noise_repetitions,
            seed=seed,
        ),
        "preprocessing_robustness": preprocessing_robustness_report(reference_family, method_ids, settings),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "synthetic_tortuosity_benchmark.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        for sheet_name, frame in reports.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark tortuosity scoring methods on synthetic curves with amplitude, "
            "oscillation frequency, global bend, and noise varied independently."
        )
    )
    parser.add_argument("--output-dir", default="demo/synthetic_benchmark")
    parser.add_argument("--noise-repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = run_benchmark(
        args.output_dir,
        noise_repetitions=args.noise_repetitions,
        seed=args.seed,
    )
    print(f"Wrote benchmark report to {Path(args.output_dir) / 'synthetic_tortuosity_benchmark.xlsx'}")
    print(reports["sensitivity_concordance"].to_string(index=False))


if __name__ == "__main__":
    main()
