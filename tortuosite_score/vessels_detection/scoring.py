from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from tortuosite_score.vessels_detection.local_bump_score import (
    DISPLAY_SCALE,
    LocalBumpSettings,
    add_comparative_hybrid_score,
    curvature_squared_metrics,
    load_saved_run_branches,
    local_bump_metrics,
    tail_weighted_mean,
    weighted_mean,
)
from tortuosite_score.vessels_detection.segments import (
    build_segment_map,
    ordered_points_for_segments,
    score_segments,
    segment_ref_sort_key,
)

ScoringMethodId = Literal["local_bump", "arc_chord", "curvature_squared"]
DEFAULT_SCORING_METHOD: ScoringMethodId = "local_bump"


@dataclass(frozen=True)
class ScoringMethodSpec:
    method_id: ScoringMethodId
    label: str
    primary_score_label: str
    eye_score_label: str
    short_description: str
    report_method_title: str
    report_steps: tuple[str, ...]
    report_equations: tuple[str, ...]
    eye_score_scale: float = 1.0


@dataclass(frozen=True)
class ScoringConfig:
    method_id: ScoringMethodId = DEFAULT_SCORING_METHOD
    local_bump_settings: LocalBumpSettings = field(default_factory=LocalBumpSettings)


SCORING_METHOD_SPECS: dict[ScoringMethodId, ScoringMethodSpec] = {
    "local_bump": ScoringMethodSpec(
        method_id="local_bump",
        label="Local-bump",
        primary_score_label="Score local-bump",
        eye_score_label="Score local-bump",
        short_description="Mesure les bosses et oscillations locales d'un vaisseau sauvegarde.",
        report_method_title="score local-bump sur vaisseaux sauvegardes",
        report_steps=(
            "1. Sauvegarder les vaisseaux manuellement ou avec le bouton d'auto-completion VascX.",
            "2. Reconstituer la ligne centrale ordonnee de chaque vaisseau sauvegarde.",
            "3. Exclure les vaisseaux trop courts selon le seuil actif.",
            "4. Re-echantillonner chaque vaisseau puis le lisser legerement.",
            "5. Mesurer les changements locaux d'angle le long du vaisseau.",
            "6. Ignorer les changements minuscules et compter les alternances de courbure.",
            "7. Calculer le score local-bump du vaisseau puis agreger l'oeil entier.",
        ),
        report_equations=(
            "theta_i = atan2(y_(i+1)-y_i, x_(i+1)-x_i)",
            "Delta theta_i = theta_(i+1) - theta_i",
            "Delta theta_i est annule si |Delta theta_i| < tau",
            "E_v = moyenne(|Delta theta_i| filtres)",
            "D_v = 100 x N_v / L_v",
            "B_v = E_v x sqrt(D_v)",
            "S_oeil = 1000 x (0.70 x G + 0.30 x T)",
        ),
        eye_score_scale=DISPLAY_SCALE,
    ),
    "arc_chord": ScoringMethodSpec(
        method_id="arc_chord",
        label="Arc/chord",
        primary_score_label="Score arc/chord",
        eye_score_label="Score arc/chord",
        short_description="Mesure l'allongement global du trajet par rapport a sa corde.",
        report_method_title="score arc/chord sur vaisseaux sauvegardes",
        report_steps=(
            "1. Sauvegarder les vaisseaux manuellement ou avec le bouton d'auto-completion VascX.",
            "2. Reconstituer la ligne centrale ordonnee de chaque vaisseau sauvegarde.",
            "3. Calculer la longueur du trajet et la longueur de la corde.",
            "4. Definir le score du vaisseau comme le rapport arc/chord.",
            "5. Agreger l'oeil entier avec la meme moyenne ponderee et la meme queue superieure.",
        ),
        report_equations=(
            "arc/chord_v = L_v / ||p_n - p_0||",
            "G = somme(L_v x score_v) / somme(L_v)",
            "T = moyenne ponderee de la queue superieure",
            "S_oeil = 0.70 x G + 0.30 x T",
        ),
        eye_score_scale=1.0,
    ),
    "curvature_squared": ScoringMethodSpec(
        method_id="curvature_squared",
        label="Courbure quadratique",
        primary_score_label="Score courbure^2",
        eye_score_label="Score courbure^2",
        short_description="Mesure la moyenne de la courbure locale au carre le long du vaisseau.",
        report_method_title="score de courbure quadratique sur vaisseaux sauvegardes",
        report_steps=(
            "1. Sauvegarder les vaisseaux manuellement ou avec le bouton d'auto-completion VascX.",
            "2. Reconstituer la ligne centrale ordonnee de chaque vaisseau sauvegarde.",
            "3. Exclure les vaisseaux trop courts selon le seuil actif.",
            "4. Re-echantillonner chaque vaisseau puis le lisser legerement.",
            "5. Estimer numeriquement la courbure locale kappa(s) le long de l'abscisse curviligne.",
            "6. Integrer kappa(s)^2 sur la longueur du vaisseau.",
            "7. Diviser par la longueur du vaisseau puis agreger l'oeil entier.",
        ),
        report_equations=(
            "kappa(s) = |x'(s)y''(s) - y'(s)x''(s)| / (x'(s)^2 + y'(s)^2)^(3/2)",
            "Q_v = (1 / L_v) x integral_0^L kappa(s)^2 ds",
            "G = somme(L_v x Q_v) / somme(L_v)",
            "T = moyenne ponderee de la queue superieure",
            "S_oeil = 0.70 x G + 0.30 x T",
        ),
        eye_score_scale=1.0,
    ),
}


def available_scoring_methods() -> list[ScoringMethodSpec]:
    return [SCORING_METHOD_SPECS[method_id] for method_id in ["local_bump", "arc_chord", "curvature_squared"]]


def scoring_method_spec(method_id: ScoringMethodId) -> ScoringMethodSpec:
    return SCORING_METHOD_SPECS[method_id]


def scoring_config(method_id: str | None = None, settings: LocalBumpSettings | None = None) -> ScoringConfig:
    normalized = str(method_id or DEFAULT_SCORING_METHOD)
    if normalized not in SCORING_METHOD_SPECS:
        normalized = DEFAULT_SCORING_METHOD
    return ScoringConfig(method_id=normalized, local_bump_settings=settings or LocalBumpSettings())


def scoring_method_fixed_parameters(config: ScoringConfig | None = None) -> list[tuple[str, str]]:
    config = config or ScoringConfig()
    settings = config.local_bump_settings
    filter_label = (
        f"actif, longueur minimale {settings.min_saved_vessel_length:.0f} px"
        if settings.filter_short_vessels
        else "desactive"
    )
    shared_parameters = [("Filtre petits vaisseaux", filter_label)]
    if config.method_id == "local_bump":
        return [
            *shared_parameters,
            ("Pas de re-echantillonnage", f"{settings.resample_step:.1f} px"),
            ("Fenetre de lissage", f"{settings.smoothing_window} points"),
            ("Seuil de courbure locale", f"{settings.curvature_threshold:.3f} rad"),
        ]
    if config.method_id == "curvature_squared":
        resampling_label = (
            f"actif, pas {settings.resample_step:.1f} px"
            if settings.resample_curvature_squared
            else "desactive"
        )
        return [
            *shared_parameters,
            ("Pretraitement re-echantillonnage", resampling_label),
        ]
    return shared_parameters


def _is_saved_vessel_eligible(path_length: float, config: ScoringConfig) -> bool:
    settings = config.local_bump_settings
    if not settings.filter_short_vessels:
        return True
    return bool(path_length >= settings.min_saved_vessel_length)


def _is_branch_fragment_eligible(branch_length: float, config: ScoringConfig) -> bool:
    settings = config.local_bump_settings
    if not settings.filter_short_vessels:
        return True
    return bool(branch_length >= settings.min_branch_length)


def score_saved_vessel(
    segment_map: dict[str, object],
    vessel_name: str,
    vessel: dict[str, object],
    config: ScoringConfig | None = None,
    vessel_id: int | None = None,
) -> dict[str, object]:
    config = config or ScoringConfig()
    method = scoring_method_spec(config.method_id)
    settings = config.local_bump_settings
    segment_refs = sorted(vessel.get("segment_refs", []), key=segment_ref_sort_key)
    synthetic_links = vessel.get("synthetic_links", [])
    if not isinstance(synthetic_links, list):
        synthetic_links = []
    path_points = ordered_points_for_segments(
        segment_map,
        segment_refs,
        synthetic_links=synthetic_links,
        start_endpoint=vessel.get("start_endpoint"),
        end_endpoint=vessel.get("end_endpoint"),
    )
    diagnostic = score_segments(
        segment_map,
        segment_refs,
        synthetic_links=synthetic_links,
        start_endpoint=vessel.get("start_endpoint"),
        end_endpoint=vessel.get("end_endpoint"),
    )
    local_metrics = local_bump_metrics(path_points, settings)
    curvature_metrics = curvature_squared_metrics(path_points, settings)
    path_length = float(diagnostic.get("length", local_metrics["branch_length"]))
    chord_length = float(diagnostic.get("chord", local_metrics["chord_length"]))
    arc_chord_value = float(diagnostic.get("tortuosity", local_metrics["arc_chord_tortuosity"]))
    if config.method_id == "local_bump":
        primary_score = float(local_metrics["branch_bump_score"])
    elif config.method_id == "curvature_squared":
        primary_score = float(curvature_metrics["curvature_squared_score"])
    else:
        primary_score = arc_chord_value
    return {
        "vessel_id": int(vessel_id) if vessel_id is not None else math.nan,
        "vessel_name": vessel_name,
        "category": vessel.get("category", "unknown"),
        "path_points": [[float(x), float(y)] for x, y in path_points],
        "scoring_method": method.method_id,
        "scoring_method_label": method.label,
        "primary_score_label": method.primary_score_label,
        "primary_score": primary_score,
        "display_name": method.primary_score_label,
        "eligible": _is_saved_vessel_eligible(path_length, config),
        "vessel_length": path_length,
        "path_length": path_length,
        "chord_length": chord_length,
        "arc_chord_diagnostic": arc_chord_value,
        "segment_count": diagnostic.get("segment_count", len(segment_refs)),
        "bridge_count": diagnostic.get("bridge_branch_count", len(synthetic_links)),
        "bridge_success": diagnostic.get("bridge_success", False),
        "component_count": diagnostic.get("component_count", math.nan),
        "model_segment_count": diagnostic.get("model_segment_count", math.nan),
        "manual_segment_count": diagnostic.get("manual_segment_count", math.nan),
        "local_bump_energy": float(local_metrics["local_bump_energy"]),
        "oscillation_count": float(local_metrics["oscillation_count"]),
        "oscillation_density": float(local_metrics["oscillation_density"]),
        "local_bump_score": float(local_metrics["branch_bump_score"]),
        "curvature_squared_score": float(curvature_metrics["curvature_squared_score"]),
        "curvature_squared_integral": float(curvature_metrics["curvature_squared_integral"]),
        "mean_curvature": float(curvature_metrics["mean_curvature"]),
        "max_curvature": float(curvature_metrics["max_curvature"]),
        "start_endpoint": vessel.get("start_endpoint"),
        "end_endpoint": vessel.get("end_endpoint"),
    }


def score_saved_vessels_for_state(
    branches: pd.DataFrame,
    review_state: dict[str, object],
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    config = config or ScoringConfig()
    segment_map = build_segment_map(branches, review_state.get("manual_segments", {}) if isinstance(review_state.get("manual_segments"), dict) else {})
    rows: list[dict[str, object]] = []
    for index, (vessel_name, vessel) in enumerate(sorted(review_state.get("vessels", {}).items()), start=1):
        if not isinstance(vessel, dict):
            continue
        rows.append(score_saved_vessel(segment_map, vessel_name, vessel, config=config, vessel_id=index))
    return pd.DataFrame(rows)


def read_manual_review_state(run_dir: Path) -> dict[str, object]:
    path = run_dir / "manual_review_state.json"
    if not path.exists():
        return {"manual_segments": {}, "vessels": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"manual_segments": {}, "vessels": {}}
    if state.get("schema_version") != 2:
        return {"manual_segments": {}, "vessels": {}}
    return {
        "manual_segments": state.get("manual_segments", {}),
        "vessels": state.get("vessels", {}),
    }


def score_saved_vessels_for_run(
    run_dir: str | Path,
    branches: pd.DataFrame | None = None,
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    config = config or ScoringConfig()
    run_dir = Path(run_dir)
    branches = branches if branches is not None else load_saved_run_branches(run_dir)
    review_state = read_manual_review_state(run_dir)
    scores = score_saved_vessels_for_state(branches, review_state, config=config)
    if scores.empty:
        return scores
    scores["run"] = run_dir.name
    scores["eye_number"] = eye_number_from_run_name(run_dir.name)
    return scores


def summarize_eye_score(
    vessel_scores: pd.DataFrame,
    config: ScoringConfig | None = None,
) -> dict[str, object]:
    config = config or ScoringConfig()
    method = scoring_method_spec(config.method_id)
    settings = config.local_bump_settings
    if vessel_scores.empty:
        eligible = pd.DataFrame()
    elif config.local_bump_settings.filter_short_vessels and "eligible" in vessel_scores:
        eligible = vessel_scores[vessel_scores["eligible"]].copy()
    else:
        eligible = vessel_scores.copy()
    result: dict[str, object] = {
        "scoring_method": method.method_id,
        "scoring_method_label": method.label,
        "primary_score_label": method.eye_score_label,
        "saved_vessel_count": int(len(vessel_scores)),
        "eligible_vessel_count": int(len(eligible)),
        "eligible_total_length": float(eligible["vessel_length"].sum()) if not eligible.empty else 0.0,
        "eye_tortuosity_score": math.nan,
        "all_vessels_score": math.nan,
        "all_vessels_tail_score": math.nan,
        "artery_score": math.nan,
        "vein_score": math.nan,
        "current_arc_chord_score": math.nan,
        "top_contributing_vessels": "[]",
    }
    if eligible.empty:
        return result

    all_global = weighted_mean(eligible["primary_score"], eligible["vessel_length"])
    if config.method_id == "local_bump":
        all_tail = tail_weighted_mean(
            eligible,
            value_column="primary_score",
            length_column="vessel_length",
            tail_length_fraction=settings.tail_length_fraction,
        )
        final_score = settings.global_weight * all_global + settings.tail_weight * all_tail
    else:
        all_tail = math.nan
        final_score = all_global
    scale = method.eye_score_scale
    result.update(
        {
            "eye_tortuosity_score": float(final_score * scale),
            "all_vessels_score": float(all_global * scale),
            "all_vessels_tail_score": float(all_tail * scale),
            "artery_score": _category_score(eligible, "artere", scale),
            "vein_score": _category_score(eligible, "veine", scale),
            "current_arc_chord_score": weighted_mean(eligible["arc_chord_diagnostic"], eligible["vessel_length"]),
            "top_contributing_vessels": json.dumps(top_contributing_vessels(eligible)),
        }
    )
    return result


def score_run(
    run_dir: str | Path,
    config: ScoringConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    config = config or ScoringConfig()
    run_dir = Path(run_dir)
    branches = load_saved_run_branches(run_dir)
    vessel_scores = score_saved_vessels_for_run(run_dir, branches=branches, config=config)
    summary = summarize_eye_score(vessel_scores, config=config)
    summary["run"] = run_dir.name
    summary["eye_number"] = eye_number_from_run_name(run_dir.name)
    return summary, vessel_scores


def score_runs(
    runs_root: str | Path,
    output_csv: str | Path | None = None,
    vessel_output_csv: str | Path | None = None,
    branch_output_csv: str | Path | None = None,
    config: ScoringConfig | None = None,
    numbered_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or ScoringConfig()
    runs_root = Path(runs_root)
    summaries: list[dict[str, object]] = []
    vessels: list[pd.DataFrame] = []
    branch_fragments: list[pd.DataFrame] = []
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if numbered_only:
        numbered = [run_dir for run_dir in run_dirs if re.match(r"^\d+_(OD|OG)$", run_dir.name)]
        if numbered:
            run_dirs = numbered
    for run_dir in run_dirs:
        if not (run_dir / "output" / "08_full_skeleton_summary.csv").exists():
            continue
        branches = load_saved_run_branches(run_dir)
        vessel_scores = score_saved_vessels_for_run(run_dir, branches=branches, config=config)
        summary = summarize_eye_score(vessel_scores, config=config)
        summary["run"] = run_dir.name
        summary["eye_number"] = eye_number_from_run_name(run_dir.name)
        summaries.append(summary)
        vessels.append(vessel_scores)
        branch_fragments.append(score_branch_fragments(branches, run_dir.name, eye_number_from_run_name(run_dir.name), config=config))

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = add_comparative_hybrid_score(summary_df)
        summary_df = summary_df.sort_values("comparative_hybrid_score", ascending=False).reset_index(drop=True)
    vessel_df = pd.concat(vessels, ignore_index=True) if vessels else pd.DataFrame()
    branch_df = pd.concat(branch_fragments, ignore_index=True) if branch_fragments else pd.DataFrame()
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
    if vessel_output_csv is not None:
        Path(vessel_output_csv).parent.mkdir(parents=True, exist_ok=True)
        vessel_df.drop(columns=["path_points", "start_endpoint", "end_endpoint"], errors="ignore").to_csv(vessel_output_csv, index=False)
    if branch_output_csv is not None:
        Path(branch_output_csv).parent.mkdir(parents=True, exist_ok=True)
        branch_df.to_csv(branch_output_csv, index=False)
    return summary_df, vessel_df


def score_branch_fragments(
    branches: pd.DataFrame,
    run_name: str = "",
    eye_number: int | None = None,
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    config = config or ScoringConfig()
    settings = config.local_bump_settings
    method = scoring_method_spec(config.method_id)
    rows: list[dict[str, object]] = []
    for _, row in branches.iterrows():
        metrics = local_bump_metrics(row["path_points"], settings)
        curvature_metrics = curvature_squared_metrics(row["path_points"], settings)
        if config.method_id == "local_bump":
            primary_score = metrics["branch_bump_score"]
        elif config.method_id == "curvature_squared":
            primary_score = curvature_metrics["curvature_squared_score"]
        else:
            primary_score = metrics["arc_chord_tortuosity"]
        rows.append(
            {
                "run": run_name,
                "eye_number": eye_number,
                "branch_id": int(row["branch_id"]),
                "category": row.get("vascx_category", "unknown"),
                "scoring_method": method.method_id,
                "primary_score_label": method.primary_score_label,
                "primary_score": primary_score,
                "eligible": _is_branch_fragment_eligible(float(metrics["branch_length"]), config),
                **metrics,
                **curvature_metrics,
            }
        )
    return pd.DataFrame(rows)


def build_manual_vessels_export(
    branches: pd.DataFrame,
    review_state: dict[str, object],
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    config = config or ScoringConfig()
    scores = score_saved_vessels_for_state(branches, review_state, config=config)
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "vessel_name",
                "category",
                "scoring_method",
                "primary_score_label",
                "primary_score",
                "path_length",
                "chord_length",
                "arc_chord_diagnostic",
                "curvature_squared_score",
                "curvature_squared_integral",
                "mean_curvature",
                "max_curvature",
            ]
        )
    state_vessels = review_state.get("vessels", {})
    rows: list[dict[str, object]] = []
    for _, row in scores.iterrows():
        vessel = state_vessels.get(row["vessel_name"], {})
        rows.append(
            {
                "vessel_name": row["vessel_name"],
                "category": row["category"],
                "segment_refs": json.dumps(sorted(vessel.get("segment_refs", []), key=segment_ref_sort_key)),
                "model_segment_count": row["model_segment_count"],
                "manual_segment_count": row["manual_segment_count"],
                "segment_count": row["segment_count"],
                "component_count": row["component_count"],
                "bridge_branch_count": row["bridge_count"],
                "bridge_success": row["bridge_success"],
                "start_endpoint": json.dumps(vessel.get("start_endpoint")),
                "end_endpoint": json.dumps(vessel.get("end_endpoint")),
                "scoring_method": row["scoring_method"],
                "scoring_method_label": row["scoring_method_label"],
                "primary_score_label": row["primary_score_label"],
                "primary_score": row["primary_score"],
                "path_length": row["path_length"],
                "chord_length": row["chord_length"],
                "arc_chord_diagnostic": row["arc_chord_diagnostic"],
                "local_bump_energy": row["local_bump_energy"],
                "oscillation_count": row["oscillation_count"],
                "oscillation_density": row["oscillation_density"],
                "local_bump_score": row["local_bump_score"],
                "curvature_squared_score": row["curvature_squared_score"],
                "curvature_squared_integral": row["curvature_squared_integral"],
                "mean_curvature": row["mean_curvature"],
                "max_curvature": row["max_curvature"],
            }
        )
    export_df = pd.DataFrame(rows)
    if not export_df.empty:
        export_df = export_df.sort_values("vessel_name").reset_index(drop=True)
    return export_df


def build_review_scores_table(
    branches: pd.DataFrame,
    review_state: dict[str, object],
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    config = config or ScoringConfig()
    scores = score_saved_vessels_for_state(branches, review_state, config=config)
    rows: list[dict[str, object]] = []
    for _, row in scores.iterrows():
        rows.append(
            {
                "Vaisseau": row["vessel_name"],
                "Categorie": row["category"],
                "Methode": row["scoring_method_label"],
                "Score principal": row["primary_score"],
                "Arc/chord diagnostic": row["arc_chord_diagnostic"],
                "Courbure^2 diagnostic": row["curvature_squared_score"],
                "Longueur du trajet": row["path_length"],
                "Corde": row["chord_length"],
                "Segments modele": row["model_segment_count"],
                "Segments manuels": row["manual_segment_count"],
                "Segments": row["segment_count"],
                "Composantes": row["component_count"],
                "Ponts automatiques": row["bridge_count"],
                "Statut du pont": "connecte" if row["bridge_success"] else "partiel",
            }
        )
    return pd.DataFrame(rows)


def top_contributing_vessels(vessel_scores: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    columns = [
        "vessel_name",
        "primary_score",
        "vessel_length",
        "category",
        "bridge_count",
        "oscillation_count",
        "curvature_squared_score",
    ]
    available = [column for column in columns if column in vessel_scores.columns]
    rows = vessel_scores.sort_values("primary_score", ascending=False).head(limit)
    return [
        {
            key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value)
            for key, value in row.items()
        }
        for row in rows[available].to_dict(orient="records")
    ]


def eye_number_from_run_name(run_name: str) -> int | None:
    match = re.search(r"(?:^|_de_)(\d+)", run_name)
    return int(match.group(1)) if match else None


def _category_score(vessels: pd.DataFrame, category: str, scale: float) -> float:
    subset = vessels[vessels["category"] == category]
    if subset.empty:
        return math.nan
    return float(weighted_mean(subset["primary_score"], subset["vessel_length"]) * scale)
