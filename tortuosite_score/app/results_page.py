from __future__ import annotations

import io
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import mannwhitneyu

from tortuosite_score.app.constants import ARTERE_COLOR, VEINE_COLOR
from tortuosite_score.app.review_data import list_runs, read_json
from tortuosite_score.app.review_state import (
    get_segment_geometry,
    score_vessel,
    segment_ref_sort_key,
    segment_refs_for_vessel,
)
from tortuosite_score.vessels_detection.local_bump_score import weighted_mean
from tortuosite_score.vessels_detection.clinical_excel import (
    analysis_vessel_eligibility,
    classified_vessel_export_eligibility,
    generate_clinical_excel_outputs,
    parse_vessel_name,
)
from tortuosite_score.vessels_detection.scoring import (
    ScoringConfig,
    scoring_config as build_scoring_config,
    scoring_method_fixed_parameters,
    scoring_method_spec,
    score_run,
)

RESULTS_SCHEMA_VERSION = 2
VISIBLE_RESULT_COLUMNS = ["Label", "Vaisseau", "Categorie", "Longueur du trajet", "Corde", "Tortuosite"]
STATS_EXPLANATION = (
    "Chaque cellule compare l'image en ligne a l'image en colonne. Une petite p-value indique des preuves que "
    "l'image en ligne a des valeurs de tortuosite plus elevees que l'image en colonne. Une grande p-value ne donne "
    "pas de preuve forte dans ce sens."
)
RAW_PVALUE_EXPLANATION = (
    "Chaque cellule compare l'image en ligne a l'image en colonne avec un test de Mann-Whitney unilateral "
    "sur les arteres classees de rang 1-3 et les veines scorees valides de l'onglet Clean_vessels de l'export brut "
    "(alternative: la distribution des scores des vaisseaux valides de la ligne est plus elevee que celle de la "
    "colonne). Ces p-values brutes repondent donc a une question simple: l'oeil de la ligne a-t-il des scores "
    "globalement plus eleves que l'oeil de la colonne ?"
)
BH_PVALUE_EXPLANATION = (
    "BH signifie Benjamini-Hochberg. Dans une matrice avec N images, on ne lit pas une seule comparaison mais "
    "N x (N - 1) comparaisons directionnelles. Plus on regarde de comparaisons, plus on augmente la chance de voir "
    "une petite p-value par hasard. La correction BH transforme les p-values brutes en p-values ajustees pour une "
    "lecture globale de la matrice. Si une seule comparaison avait ete definie a l'avance, la p-value brute serait "
    "suffisante; ici, les p-values BH sont plus adaptees pour reperer les contrastes robustes dans tout le tableau."
)
SEGMENTATION_EXPLANATION = (
    "Chaque image montre l'oeil original avec les vaisseaux sauvegardes superposes. Les labels V1, V2, etc. "
    "correspondent aux lignes du tableau, qui resume la tortuosite de chaque vaisseau sauvegarde."
)
LOCAL_BUMP_METHOD_EXPLANATION = (
    "Le score actuel est calcule uniquement sur les vaisseaux sauvegardes. Ces vaisseaux peuvent avoir ete traces "
    "manuellement ou crees par le bouton d'auto-completion du squelette VascX. La reconstruction racine-feuille est "
    "donc une etape de creation de vaisseaux, pas une partie cachee du calcul final. Pour chaque vaisseau sauvegarde, "
    "la ligne centrale ordonnee est re-echantillonnee puis legerement lissee pour limiter le bruit en escalier des "
    "pixels. Le calcul mesure ensuite les changements locaux d'angle et les alternances de courbure. Ainsi, un grand "
    "virage regulier reste peu penalise, alors qu'un vaisseau presque droit mais bossue ou ondule obtient un score "
    "plus eleve."
)
LOCAL_BUMP_V2_METHOD_EXPLANATION = (
    "Cette variante experimentale conserve les vaisseaux sauvegardes et la normalisation du modele actuel. "
    "Elle lisse la ligne centrale sans restaurer les extremites brutes, exclut la marge du filtre, puis regroupe "
    "les changements d'angle en lobes de courbure persistants. Le score combine la charge d'oscillation ainsi "
    "nettoyee avec une composante d'angularite locale. Le local-bump v1 reste disponible sans modification."
)
CURVATURE_SQUARED_METHOD_EXPLANATION = (
    "Le score actuel est calcule uniquement sur les vaisseaux sauvegardes. Pour chaque vaisseau, la ligne centrale "
    "ordonnee est re-echantillonnee si ce pretraitement est actif, sans lissage. La courbure locale est estimee "
    "numeriquement le long de l'abscisse curviligne, puis le carre de cette courbure est integre et divise par la "
    "longueur du vaisseau. Le score obtenu correspond a la valeur brute de la formule "
    "T = (1 / L) integral kappa(s)^2 ds."
)
ARC_CHORD_METHOD_EXPLANATION = (
    "Le score actuel est calcule uniquement sur les vaisseaux sauvegardes. Pour chaque vaisseau, la longueur du trajet "
    "ordonne est comparee a la corde entre le point de depart et le point d'arrivee. Le rapport arc/chord mesure donc "
    "l'allongement global du vaisseau."
)
TORTUOSITY_DENSITY_METHOD_EXPLANATION = (
    "Le score actuel est calcule uniquement sur les vaisseaux sauvegardes. La ligne centrale est re-echantillonnee "
    "et lissee uniquement pour reperer les changements significatifs du signe de la courbure. Ces inflexions "
    "decoupent le vaisseau en sous-segments. Les longueurs d'arc et les cordes utilisees par la formule sont ensuite "
    "mesurees sur la geometrie normalisee non lissee. Une courbe de convexite constante a un score nul."
)
EXTERNAL_ANGLE_SUM_METHOD_EXPLANATION = (
    "Le score actuel est calcule uniquement sur les vaisseaux sauvegardes. La ligne centrale normalisee est "
    "simplifiee en segments droits avec l'algorithme de Ramer-Douglas-Peucker, ce qui produit une liste de points "
    "de flexion sans jamais estimer une courbure continue kappa(s). L'angle externe theta_i est calcule a chaque "
    "point de flexion conserve, puis tous les angles externes sont sommes pour obtenir T = somme(theta_i), en "
    "degres. La metrique compte donc le nombre de virages et leur amplitude, sans dependre directement de la "
    "longueur du vaisseau ni du rayon de courbure."
)
HYBRID_SCORE_EXPLANATION = (
    "Le score final de l'oeil combine une moyenne des vaisseaux sauvegardes ponderee par la longueur et une composante "
    "de queue superieure pour que quelques vaisseaux tres bossues ne soient pas dilues par de nombreux vaisseaux "
    "normaux. Le score comparatif est normalise dans la cohorte du rapport."
)
LENGTH_WEIGHTED_SCORE_EXPLANATION = (
    "Le score final de l'oeil est la moyenne des scores des vaisseaux retenus, ponderee par leur longueur. "
    "Aucune composante de queue superieure ni aucun multiplicateur d'affichage n'est applique."
)
LOCAL_BUMP_EQUATION_LINES = [
    "Pour un vaisseau sauvegarde v, apres re-echantillonnage et lissage leger:",
    "theta_i = atan2(y_(i+1)-y_i, x_(i+1)-x_i)",
    "Delta theta_i = theta_(i+1) - theta_i",
    "Delta theta_i est annule si |Delta theta_i| < tau",
    "E_v = moyenne(|Delta theta_i| filtres)",
    "D_v = 100 x N_v / L_v, avec N_v = nombre d'alternances de courbure et L_v = longueur du vaisseau",
    "B_v = E_v x sqrt(D_v)",
    "",
    "Score oeil:",
    "G = moyenne ponderee par longueur des B_v",
    "T = moyenne ponderee de la queue superieure (20% de longueur cumulee la plus tortueuse)",
    "S_oeil = 1000 x (0.70 x G + 0.30 x T)",
    "Score comparatif = normalisation min-max de S_oeil dans la cohorte du rapport",
]
LOCAL_BUMP_LITERATURE = [
    "Ramos et al., Scientific Reports, 2019: tortuosity globale et facteurs anatomiques.",
    "Hervella et al., Medical & Biological Engineering & Computing, 2024: assessment explicable et aggregation globale.",
    "Ramos et al., BMC Medical Research Methodology, 2018: validation multi-experts et limites des mesures simples.",
]

MODEL_DESCRIPTION_BY_METHOD = {
    "arc_chord": [
        "Le modele analyse uniquement les vaisseaux sauvegardes dans l'app.",
        "Pour chaque vaisseau, il reconstruit la ligne centrale ordonnee, mesure la longueur du trajet et la corde, puis calcule le rapport arc/chord.",
        "Le seul pretraitement transversal est le filtre de petits vaisseaux si l'option est activee.",
    ],
    "curvature_squared": [
        "Le modele analyse uniquement les vaisseaux sauvegardes dans l'app.",
        "Pour chaque vaisseau, il reconstruit la ligne centrale ordonnee puis applique la formule T = (1 / L) integral kappa(s)^2 ds.",
        "Le re-echantillonnage, quand il est actif, est un pretraitement numerique avant l'estimation des derivees; il ne change pas la formule appliquee ensuite.",
    ],
    "tortuosity_density": [
        "Le modele analyse uniquement les vaisseaux sauvegardes dans l'app.",
        "Il re-echantillonne et lisse la ligne centrale pour detecter les inflexions significatives, sans ajouter de parametre propre a cette methode.",
        "Il mesure ensuite les arcs et les cordes sur la geometrie normalisee non lissee et applique tau_TD = ((n - 1) / L) x somme(L_i / C_i - 1).",
    ],
    "external_angle_sum": [
        "Le modele analyse uniquement les vaisseaux sauvegardes dans l'app.",
        "Il simplifie la ligne centrale normalisee avec l'algorithme de Ramer-Douglas-Peucker pour obtenir des points de flexion.",
        "Il calcule ensuite l'angle externe theta_i a chaque point de flexion conserve et applique T = somme(theta_i), sans jamais estimer une courbure continue.",
    ],
    "local_bump_v2": [
        "Le modele experimental analyse les memes vaisseaux sauvegardes que le local-bump v1.",
        "Il applique un lissage sans restauration des extremites, exclut la marge du filtre et supprime les lobes de courbure non persistants.",
        "Il combine une composante d'oscillation persistante et une composante d'angularite; les deux composantes sont exportees pour audit.",
    ],
    "local_bump": [
        "Le modele analyse uniquement les vaisseaux sauvegardes dans l'app.",
        "Pour chaque vaisseau, il re-echantillonne et lisse la ligne centrale, mesure les changements locaux d'angle, ignore les variations sous le seuil actif, puis compte les alternances de courbure.",
        "Le score est concu pour valoriser les petites bosses et oscillations locales plutot qu'un grand virage regulier unique.",
    ],
}


def load_saved_review_state(run_dir: Path) -> dict:
    state = read_json(run_dir / "manual_review_state.json")
    if state.get("schema_version") != RESULTS_SCHEMA_VERSION:
        return {"manual_segments": {}, "vessels": {}}
    return {
        "manual_segments": state.get("manual_segments", {}),
        "vessels": state.get("vessels", {}),
    }


def saved_vessel_count(run_dir: Path) -> int:
    return len(load_saved_review_state(run_dir).get("vessels", {}))


def build_result_rows(review_state: dict, branches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manual_segments = review_state.get("manual_segments", {})
    for index, (vessel_name, vessel) in enumerate(sorted(review_state.get("vessels", {}).items()), start=1):
        metrics = score_vessel(branches_df, manual_segments, vessel)
        rows.append(
            {
                "Label": f"V{index}",
                "Vaisseau": vessel_name,
                "Categorie": vessel.get("category", "artere"),
                "Segments modele": metrics["model_segment_count"],
                "Segments manuels": metrics["manual_segment_count"],
                "Composantes": metrics["component_count"],
                "Statut du pont": "connecte" if metrics["bridge_success"] else "partiel",
                "Longueur du trajet": metrics["length"],
                "Corde": metrics["chord"],
                "Tortuosite": metrics["tortuosity"],
                "Debut": _endpoint_caption(metrics["start_endpoint"]),
                "Fin": _endpoint_caption(metrics["end_endpoint"]),
            }
        )
    return pd.DataFrame(rows)


def tortuosity_values(result_rows: pd.DataFrame) -> list[float]:
    if result_rows.empty or "Tortuosite" not in result_rows.columns:
        return []
    values: list[float] = []
    for value in result_rows["Tortuosite"]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def build_pvalue_matrix(scores_by_image: dict[str, list[float]]) -> pd.DataFrame:
    image_names = list(scores_by_image)
    rows: list[dict[str, str]] = []
    for row_name in image_names:
        row: dict[str, str] = {"Image": row_name}
        for col_name in image_names:
            if row_name == col_name:
                row[col_name] = "-"
                continue
            row_scores = scores_by_image.get(row_name, [])
            col_scores = scores_by_image.get(col_name, [])
            if not row_scores or not col_scores:
                row[col_name] = "NA"
                continue
            pvalue = mannwhitneyu(row_scores, col_scores, alternative="greater", method="auto").pvalue
            row[col_name] = f"{float(pvalue):.4g}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Image") if rows else pd.DataFrame()


def build_adjusted_pvalue_matrix(raw_matrix: pd.DataFrame) -> pd.DataFrame:
    if raw_matrix.empty:
        return pd.DataFrame()
    adjusted = raw_matrix.copy()
    pvalue_positions: list[tuple[str, str]] = []
    pvalues: list[float] = []
    for row_name in raw_matrix.index:
        for col_name in raw_matrix.columns:
            raw_value = raw_matrix.loc[row_name, col_name]
            try:
                pvalue = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(pvalue):
                pvalue_positions.append((row_name, col_name))
                pvalues.append(pvalue)
    adjusted_values = benjamini_hochberg(pvalues)
    for (row_name, col_name), adjusted_value in zip(pvalue_positions, adjusted_values):
        adjusted.loc[row_name, col_name] = f"{adjusted_value:.4g}"
    return adjusted


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1], reverse=True)
    adjusted = [1.0] * len(pvalues)
    running_min = 1.0
    total = len(pvalues)
    for rank_from_end, (original_index, pvalue) in enumerate(indexed, start=1):
        rank = total - rank_from_end + 1
        running_min = min(running_min, float(pvalue) * total / rank)
        adjusted[original_index] = min(1.0, running_min)
    return adjusted


def visible_result_table(result_table: pd.DataFrame) -> pd.DataFrame:
    return result_table[[column for column in VISIBLE_RESULT_COLUMNS if column in result_table.columns]].copy()


def build_results_viewer_segments(
    review_state: dict,
    branches_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    geometry_map = get_segment_geometry(branches_df, review_state.get("manual_segments", {}))
    segments: list[dict] = []
    labels: list[dict] = []
    for index, (vessel_name, vessel) in enumerate(sorted(review_state.get("vessels", {}).items()), start=1):
        del vessel_name
        color = ARTERE_COLOR if vessel.get("category") == "artere" else VEINE_COLOR
        vessel_points: list[list[float]] = []
        for segment_ref in sorted(segment_refs_for_vessel(vessel), key=segment_ref_sort_key):
            geometry = geometry_map.get(segment_ref)
            if geometry is None:
                continue
            vessel_points.extend(geometry["points"])
            segments.append(
                {
                    "segmentRef": f"result:{index}:{segment_ref}",
                    "source": geometry["source"],
                    "points": geometry["points"],
                    "label": None,
                    "locked": True,
                    "strokes": [{"color": color, "width": 3.4, "opacity": 1}],
                }
            )
        if vessel_points:
            point_df = pd.DataFrame(vessel_points, columns=["x", "y"])
            labels.append(
                {
                    "text": f"V{index}",
                    "position": [float(point_df["x"].mean()), float(point_df["y"].mean())],
                    "color": color,
                }
            )
    return segments, labels


def generate_results_pdf(
    selected_run_names: list[str],
    counts_by_run: dict[str, int],
    loaded_results: dict[str, tuple[dict, pd.DataFrame, dict]],
    result_tables: dict[str, pd.DataFrame],
    raw_pvalue_matrix: pd.DataFrame,
    adjusted_pvalue_matrix: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        _add_cover_pdf_page(pdf, selected_run_names, counts_by_run)
        _add_stats_pdf_page(pdf, raw_pvalue_matrix, adjusted_pvalue_matrix)
        for run_name in selected_run_names:
            bundle, branches_df, review_state = loaded_results[run_name]
            overlay = render_overlay_image(Path(bundle["image_path"]), review_state, branches_df, label_font_size=28)
            _add_segmentation_pdf_page(pdf, run_name, overlay, visible_result_table(result_tables[run_name]))
    return buffer.getvalue()


def generate_local_bump_results_pdf(run_dirs: list[Path], scoring_config: ScoringConfig | None = None) -> bytes:
    scoring_config = scoring_config or build_scoring_config()
    scoring_config, scored_runs, summary_table = _score_local_bump_report_runs(run_dirs, scoring_config)
    return _generate_local_bump_results_pdf_from_scores(scoring_config, scored_runs, summary_table)


def _score_local_bump_report_runs(
    run_dirs: list[Path],
    scoring_config: ScoringConfig | None = None,
) -> tuple[ScoringConfig, list[tuple[Path, dict[str, object], pd.DataFrame]], pd.DataFrame]:
    scoring_config = scoring_config or build_scoring_config()
    run_dirs = _local_bump_report_runs(run_dirs)
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]] = []
    for run_dir in run_dirs:
        try:
            summary, system_scores = score_run(run_dir, scoring_config)
        except (FileNotFoundError, ValueError, OSError, KeyError):
            continue
        scored_runs.append((run_dir, summary, system_scores))

    summary_table = _build_local_bump_summary_table(scored_runs)
    return scoring_config, scored_runs, summary_table


def _generate_local_bump_results_pdf_from_scores(
    scoring_config: ScoringConfig,
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]],
    summary_table: pd.DataFrame,
) -> bytes:
    raw_pvalue_matrix, adjusted_pvalue_matrix = _build_local_bump_pvalue_matrices(scored_runs)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        _add_local_bump_cover_page(pdf, summary_table, scoring_config)
        _add_model_description_page(pdf, scoring_config, summary_table)
        _add_stats_pdf_page(pdf, raw_pvalue_matrix, adjusted_pvalue_matrix)
        _add_local_bump_scores_page(pdf, summary_table, scoring_config)
        for run_dir, summary, system_scores in scored_runs:
            _add_local_bump_run_page(pdf, run_dir, summary, system_scores)
        _add_all_systems_pdf_pages(
            pdf,
            _build_atlas_systems_table(scored_runs),
            {run_dir.name: system_scores for run_dir, _summary, system_scores in scored_runs},
        )
    return buffer.getvalue()


def _local_bump_report_runs(run_dirs: list[Path]) -> list[Path]:
    numbered_runs = [run_dir for run_dir in run_dirs if re.match(r"^\d+_(OD|OG)$", run_dir.name)]
    return numbered_runs or run_dirs


def render_overlay_image(
    image_path: Path,
    review_state: dict,
    branches_df: pd.DataFrame,
    label_font_size: int,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    segments, labels = build_results_viewer_segments(review_state, branches_df)
    for segment in segments:
        points = [(float(point[0]), float(point[1])) for point in segment.get("points", [])]
        if len(points) < 2:
            continue
        strokes = segment.get("strokes") or [{"color": "#ffffff", "width": 3}]
        for stroke in strokes:
            color = _hex_to_rgb(str(stroke.get("color", "#ffffff")))
            width = max(1, int(round(float(stroke.get("width", 3)))))
            draw.line(points, fill=color, width=width, joint="curve")
    font = _load_font(label_font_size)
    for label in labels:
        position = label.get("position", [0, 0])
        text = str(label.get("text", ""))
        x = float(position[0])
        y = float(position[1])
        bbox = draw.textbbox((x, y), text, font=font, anchor="mm")
        padding = max(3, label_font_size // 5)
        draw.rounded_rectangle(
            (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
            radius=padding,
            fill=(0, 0, 0),
            outline=_hex_to_rgb(str(label.get("color", "#ffffff"))),
            width=2,
        )
        draw.text((x, y), text, fill=(255, 255, 255), font=font, anchor="mm")
    return image


def render_results_page(scoring_config: ScoringConfig | None = None) -> None:
    scoring_config = scoring_config or build_scoring_config()
    method = scoring_method_spec(scoring_config.method_id)
    runs = list_runs()
    if not runs:
        st.info("Importez au moins une image retinienne avant de generer des resultats.")
        return
    with st.expander("Methode utilisee dans l'app", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- Les vaisseaux sont d'abord sauvegardes manuellement ou via `Auto-complete skeleton into saved vessels`.",
                    "- Le score final est calcule uniquement sur ces vaisseaux sauvegardes.",
                    f"- La methode active est `{method.label}`.",
                    f"- {method.short_description}",
                ]
            )
        )
    with st.spinner(f"Calcul des scores {method.label}..."):
        scoring_config, scored_runs, summary_table = _score_local_bump_report_runs(runs, scoring_config)
        all_systems_table = _build_all_systems_table(scored_runs)

    if summary_table.empty:
        st.warning(f"Aucun run exploitable trouve pour calculer le score {method.label}.")
        return

    st.dataframe(summary_table, hide_index=True, width="stretch")
    st.download_button(
        "Telecharger les scores CSV",
        data=summary_table.to_csv(index=False).encode("utf-8"),
        file_name=f"scores_tortuosite_{method.method_id}.csv",
        mime="text/csv",
    )
    if not all_systems_table.empty:
        st.download_button(
            "Telecharger tous les vaisseaux CSV",
            data=_public_systems_table(all_systems_table).to_csv(index=False).encode("utf-8"),
            file_name="vaisseaux_tortuosite_tries.csv",
            mime="text/csv",
        )
    st.download_button(
        "Generer le rapport PDF",
        data=_generate_local_bump_results_pdf_from_scores(scoring_config, scored_runs, summary_table),
        file_name=f"rapport_tortuosite_{method.method_id}.pdf",
        mime="application/pdf",
    )
    clinical_excel, classified_vessels_excel = generate_clinical_excel_outputs(runs, scoring_config)
    st.download_button(
        "Generer l'Excel comparaison des rangs 1, 2 et 3",
        data=clinical_excel,
        file_name=f"comparaison_rangs_1_2_3_{method.method_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Generate classified vessel raw data Excel",
        data=classified_vessels_excel,
        file_name=f"classified_vessels_raw_{method.method_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _endpoint_caption(endpoint: dict[str, object] | None) -> str:
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("point"), list):
        return "Aucun"
    point = endpoint["point"]
    if len(point) != 2:
        return "Aucun"
    return f"({float(point[0]):.1f}, {float(point[1]):.1f})"


def _build_local_bump_summary_table(scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir, summary, _branch_scores in scored_runs:
        descriptive_metrics = _descriptive_eye_metrics(_branch_scores, summary.get("scoring_method"))
        eligible_count = summary.get("eligible_vessel_count")
        saved_count = summary.get("saved_vessel_count")
        rows.append(
            {
                "Image": run_dir.name,
                "Oeil": summary.get("eye_number"),
                "Methode": summary.get("scoring_method_label"),
                "Score median": descriptive_metrics["score_median"],
                "Score moyen": descriptive_metrics["score_mean"],
                "Score moyen pondere": descriptive_metrics["score_weighted_mean"],
                "Vaisseaux retenus": eligible_count,
                "Vaisseaux sauvegardes": saved_count,
                "Vaisseaux retenus/sauvegardes": _kept_vessel_count_label(eligible_count, saved_count),
                "Longueur totale vaisseaux": descriptive_metrics["saved_total_length"],
                "Longueur totale vaisseaux retenus": descriptive_metrics["eligible_total_length"],
            }
        )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    return table.sort_values("Score moyen pondere", ascending=False, na_position="last").reset_index(drop=True)


def _kept_vessel_count_label(eligible_count: object, saved_count: object) -> str:
    try:
        eligible = int(eligible_count)
        saved = int(saved_count)
    except (TypeError, ValueError):
        return "NA"
    if saved <= 0:
        return f"{eligible}/0"
    return f"{eligible}/{saved}"


def _build_local_bump_pvalue_matrices(
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_by_image = {
        run_dir.name: _classified_export_primary_scores(system_scores)
        for run_dir, _summary, system_scores in scored_runs
    }
    raw_matrix = build_pvalue_matrix(scores_by_image)
    return raw_matrix, build_adjusted_pvalue_matrix(raw_matrix)


def _classified_export_primary_scores(system_scores: pd.DataFrame) -> list[float]:
    """Scores from the same vessels exported in the raw Excel Clean_vessels sheet."""

    eligible = _classified_export_eligible_system_scores(system_scores)
    if eligible.empty or "primary_score" not in eligible.columns:
        return []
    scores: list[float] = []
    for value in eligible["primary_score"]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            scores.append(numeric)
    return scores


def _classified_export_eligible_system_scores(system_scores: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"vessel_name", "vessel_length", "primary_score"}
    if system_scores.empty or not required_columns.issubset(system_scores.columns):
        return pd.DataFrame()

    keep_mask: list[bool] = []
    for _, row in system_scores.iterrows():
        vessel_name = str(row.get("vessel_name", ""))
        category = row.get("category")
        vessel_length = row.get("vessel_length")
        analysis_eligible = analysis_vessel_eligibility(
            vessel_name,
            category,
            vessel_length,
        )[0]
        export_eligible = classified_vessel_export_eligibility(
            parse_vessel_name(vessel_name, category),
            analysis_eligible=analysis_eligible,
            scoring_eligible=bool(row.get("eligible", False)),
            score=row.get("primary_score"),
            vessel_length=vessel_length,
        )[0]
        keep_mask.append(export_eligible)
    return system_scores.loc[keep_mask].copy()


def _analysis_eligible_system_scores(system_scores: pd.DataFrame) -> pd.DataFrame:
    """Rows matching Clean_vessels / analysis_vessel_eligibility (ranked arteries only)."""

    required_columns = {"vessel_name", "vessel_length"}
    if system_scores.empty or not required_columns.issubset(system_scores.columns):
        return pd.DataFrame()
    keep_mask = [
        analysis_vessel_eligibility(
            str(row.get("vessel_name", "")),
            row.get("category"),
            row.get("vessel_length"),
        )[0]
        for _, row in system_scores.iterrows()
    ]
    return system_scores.loc[keep_mask].copy()


def _eligible_primary_scores(system_scores: pd.DataFrame) -> list[float]:
    eligible = _eligible_system_scores(system_scores)
    if eligible.empty or "primary_score" not in eligible.columns:
        return []
    scores: list[float] = []
    for value in eligible["primary_score"]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            scores.append(numeric)
    return scores


def _eligible_system_scores(system_scores: pd.DataFrame) -> pd.DataFrame:
    if system_scores.empty:
        return pd.DataFrame()
    if "eligible" not in system_scores.columns:
        return system_scores.copy()
    return system_scores[system_scores["eligible"]].copy()


def _descriptive_eye_metrics(system_scores: pd.DataFrame, method_id: object) -> dict[str, float]:
    method = scoring_method_spec(str(method_id or "local_bump"))
    scale = float(method.eye_score_scale)
    eligible = _eligible_system_scores(system_scores)
    primary_scores = _eligible_primary_scores(system_scores)
    saved_total_length = _column_total(system_scores, "vessel_length")
    eligible_total_length = _column_total(eligible, "vessel_length")
    if not primary_scores:
        return {
            "score_median": math.nan,
            "score_mean": math.nan,
            "score_weighted_mean": math.nan,
            "saved_total_length": saved_total_length,
            "eligible_total_length": eligible_total_length,
        }
    score_median = float(pd.Series(primary_scores).median()) * scale
    score_mean = float(sum(primary_scores) / len(primary_scores)) * scale
    score_weighted_mean = math.nan
    if not eligible.empty and "primary_score" in eligible.columns and "vessel_length" in eligible.columns:
        score_weighted_mean = float(weighted_mean(eligible["primary_score"], eligible["vessel_length"])) * scale
    return {
        "score_median": score_median,
        "score_mean": score_mean,
        "score_weighted_mean": score_weighted_mean,
        "saved_total_length": saved_total_length,
        "eligible_total_length": eligible_total_length,
    }


def _column_total(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return 0.0
    numeric = pd.to_numeric(table[column], errors="coerce")
    finite = numeric[numeric.notna()]
    return float(finite.sum()) if not finite.empty else 0.0


def _build_all_systems_table(scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]]) -> pd.DataFrame:
    return _build_systems_table(scored_runs, analysis_only=True)


def _build_atlas_systems_table(
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]],
) -> pd.DataFrame:
    return _build_systems_table(scored_runs, analysis_only=False)


def _build_systems_table(
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]],
    *,
    analysis_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir, summary, vessel_scores in scored_runs:
        eligible = (
            _analysis_eligible_system_scores(vessel_scores)
            if analysis_only
            else _eligible_system_scores(vessel_scores)
        )
        if eligible.empty:
            continue
        for _, row in eligible.iterrows():
            rows.append(
                {
                    "Image": run_dir.name,
                    "Oeil": summary.get("eye_number"),
                    "Vaisseau": row.get("vessel_name"),
                    "Categorie": row.get("category"),
                    "Methode": row.get("scoring_method_label"),
                    "Score vaisseau": row.get("primary_score"),
                    "Local-bump v1 diagnostic": (
                        float(row.get("local_bump_score")) * 1000.0
                        if pd.notna(row.get("local_bump_score"))
                        else math.nan
                    ),
                    "Local-bump v2 diagnostic": row.get("local_bump_v2_score"),
                    "Oscillation persistante": row.get("local_bump_v2_oscillation_component"),
                    "Angularite locale": row.get("local_bump_v2_angularity_component"),
                    "Lobes persistants": row.get("persistent_lobe_count"),
                    "Tour maximal aux extremites": row.get("endpoint_max_turn"),
                    "Longueur": row.get("vessel_length"),
                    "Segments": row.get("segment_count"),
                    "Ponts": row.get("bridge_count"),
                    "Courbure^2 diagnostic": row.get("curvature_squared_score"),
                    "Tortuosity Density diagnostic": row.get("tortuosity_density_score"),
                    "Sous-segments de courbure": row.get("constant_curvature_segment_count"),
                    "Inflexions": row.get("inflection_count"),
                    "Oscillations": row.get("oscillation_count"),
                    "_run_dir": run_dir,
                    "_path_points": row.get("path_points"),
                }
            )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table = table.sort_values("Score vaisseau", ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "Rang", range(1, len(table) + 1))
    return table


def _public_systems_table(all_systems_table: pd.DataFrame) -> pd.DataFrame:
    return all_systems_table.drop(columns=[column for column in all_systems_table.columns if column.startswith("_")])


def _add_local_bump_cover_page(pdf: PdfPages, summary_table: pd.DataFrame, scoring_config: ScoringConfig) -> None:
    method = scoring_method_spec(scoring_config.method_id)
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    image_count = len(summary_table)
    lines = [
        "Rapport de tortuosite retinienne",
        "",
        f"Methode actuelle: {method.report_method_title}",
        "",
        f"Images analysees: {image_count}",
        "",
        "Objectif clinique:",
        method.short_description,
        "",
        "Important:",
        "le calcul final ne score pas tout le squelette directement; il score seulement",
        "les vaisseaux sauvegardes dans chaque session.",
    ]
    ax.text(0.08, 0.92, "\n".join(lines), va="top", fontsize=13)
    pdf.savefig(fig, bbox_inches="tight")


def _add_model_description_page(
    pdf: PdfPages,
    scoring_config: ScoringConfig,
    summary_table: pd.DataFrame,
) -> None:
    method = scoring_method_spec(scoring_config.method_id)
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.08, 0.94, "Modele et pretraitements", va="top", fontsize=16, fontweight="bold")

    description_lines = MODEL_DESCRIPTION_BY_METHOD.get(method.method_id, MODEL_DESCRIPTION_BY_METHOD["local_bump"])
    ax.text(
        0.08,
        0.86,
        "\n".join(f"- {line}" for line in description_lines),
        va="top",
        fontsize=10,
        wrap=True,
    )

    parameter_lines = scoring_method_fixed_parameters(scoring_config)
    parameter_text = ["Parametres actifs:"] + [f"- {name}: {value}" for name, value in parameter_lines]
    ax.text(0.08, 0.62, "\n".join(parameter_text), va="top", fontsize=10)

    total_saved = _summary_column_total(summary_table, "Vaisseaux sauvegardes")
    total_kept = _summary_column_total(summary_table, "Vaisseaux retenus")
    if total_saved:
        kept_line = f"Vaisseaux retenus pour les scores: {total_kept:.0f} / {total_saved:.0f}"
    else:
        kept_line = "Vaisseaux retenus pour les scores: NA"
    ax.text(
        0.08,
        0.42,
        "\n".join(
            [
                "Population analysee:",
                kept_line,
                "Les vaisseaux exclus par le filtre restent sauvegardes dans l'app, mais ne contribuent pas aux statistiques du rapport.",
            ]
        ),
        va="top",
        fontsize=10,
        wrap=True,
    )

    ax.text(
        0.08,
        0.23,
        "Sortie principale du rapport: les matrices de p-values comparent les distributions de scores des vaisseaux retenus entre images.",
        va="top",
        fontsize=10,
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")


def _summary_column_total(summary_table: pd.DataFrame, column: str) -> float:
    if summary_table.empty or column not in summary_table.columns:
        return 0.0
    numeric = pd.to_numeric(summary_table[column], errors="coerce")
    return float(numeric.sum()) if numeric.notna().any() else 0.0


def _add_local_bump_method_page(pdf: PdfPages, scoring_config: ScoringConfig) -> None:
    method = scoring_method_spec(scoring_config.method_id)
    settings = scoring_config.local_bump_settings
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.08, 0.94, "Methode de calcul", va="top", fontsize=16, fontweight="bold")
    explanation = {
        "local_bump_v2": LOCAL_BUMP_V2_METHOD_EXPLANATION,
        "arc_chord": ARC_CHORD_METHOD_EXPLANATION,
        "curvature_squared": CURVATURE_SQUARED_METHOD_EXPLANATION,
        "tortuosity_density": TORTUOSITY_DENSITY_METHOD_EXPLANATION,
        "external_angle_sum": EXTERNAL_ANGLE_SUM_METHOD_EXPLANATION,
    }.get(method.method_id, LOCAL_BUMP_METHOD_EXPLANATION)
    ax.text(0.08, 0.88, explanation, va="top", fontsize=10, wrap=True)
    steps = list(method.report_steps)
    if method.method_id == "local_bump":
        steps[2] = f"3. Normaliser le diametre du fond d'oeil a 1024 px, puis exclure les vaisseaux sous {settings.min_saved_vessel_length:.0f} px normalises."
        steps[3] = f"4. Re-echantillonner chaque vaisseau tous les {settings.resample_step:.1f} px normalises."
        steps[4] = f"5. Lisser legerement la ligne centrale: fenetre {settings.smoothing_window} points."
    if method.method_id == "local_bump":
        steps[6] = f"7. Ignorer les changements minuscules: seuil {settings.curvature_threshold:.3f} rad."
    elif method.method_id == "curvature_squared":
        steps[2] = f"3. Normaliser le diametre du fond d'oeil a 1024 px, puis exclure les vaisseaux sous {settings.min_saved_vessel_length:.0f} px normalises."
        steps[3] = (
            f"4. Re-echantillonner chaque vaisseau tous les {settings.resample_step:.1f} px normalises, sans lissage."
            if settings.resample_curvature_squared
            else "4. Conserver les points normalises d'origine, sans re-echantillonnage ni lissage."
        )
    elif method.method_id == "local_bump_v2":
        steps[2] = f"3. Normaliser le fond d'oeil puis exclure les vaisseaux sous {settings.min_saved_vessel_length:.0f} px normalises."
        steps[3] = f"4. Re-echantillonner tous les {settings.resample_step:.1f} px et lisser sur {settings.smoothing_window} points sans restaurer les extremites."
        steps[4] = f"5. Exclure la marge derivee du filtre et supprimer les lobes sous {settings.min_persistent_lobe_angle:.3f} rad."
        steps[6] = f"7. Combiner avec un poids d'angularite w = {settings.local_bump_v2_angularity_weight:.2f}."
    elif method.method_id == "tortuosity_density":
        steps[2] = f"3. Normaliser le diametre du fond d'oeil a 1024 px, puis exclure les vaisseaux sous {settings.min_saved_vessel_length:.0f} px normalises."
        steps[3] = f"4. Re-echantillonner tous les {settings.resample_step:.1f} px et lisser sur {settings.smoothing_window} points pour detecter les inflexions."
        steps[4] = f"5. Ignorer les changements d'angle sous {settings.curvature_threshold:.3f} rad, puis diviser aux changements de signe restants."
    elif method.method_id == "external_angle_sum":
        steps[2] = f"3. Normaliser le diametre du fond d'oeil a 1024 px, puis exclure les vaisseaux sous {settings.min_saved_vessel_length:.0f} px normalises."
        steps[3] = f"4. Simplifier la ligne centrale avec Ramer-Douglas-Peucker (tolerance {settings.rdp_epsilon:.1f} px normalises)."
        steps[4] = "5. Calculer l'angle externe (en degres) a chaque point de flexion conserve."
    ax.text(0.08, 0.62, "\n".join(steps), va="top", fontsize=10)
    aggregation_explanation = HYBRID_SCORE_EXPLANATION if method.method_id == "local_bump" else LENGTH_WEIGHTED_SCORE_EXPLANATION
    ax.text(0.08, 0.34, aggregation_explanation, va="top", fontsize=9, wrap=True)
    ax.text(0.08, 0.24, "\n".join(method.report_equations), va="top", fontsize=8.3)
    literature_lines = ["Litterature utilisee:"] + [f"- {item}" for item in LOCAL_BUMP_LITERATURE]
    ax.text(0.08, 0.06, "\n".join(literature_lines), va="bottom", fontsize=8.2)
    pdf.savefig(fig, bbox_inches="tight")


def _local_bump_scores_pdf_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "Image",
        "Oeil",
        "Methode",
        "Score median",
        "Score moyen",
        "Score moyen pondere",
        "Vaisseaux retenus/sauvegardes",
        "Longueur totale vaisseaux",
        "Longueur totale vaisseaux retenus",
    ]
    return summary_table[[column for column in ordered_columns if column in summary_table.columns]].copy()


def _add_local_bump_scores_page(
    pdf: PdfPages,
    summary_table: pd.DataFrame,
    scoring_config: ScoringConfig,
) -> None:
    method = scoring_method_spec(scoring_config.method_id)
    fig = Figure(figsize=(11.69, 8.27))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.02, 0.98, "Resume des scores par image", va="top", fontsize=15, fontweight="bold")
    ax.text(
        0.02,
        0.92,
        f"Ces statistiques descriptives sont calculees pour la methode `{method.label}` a partir des vaisseaux "
        "retenus. Le tableau est trie par score moyen pondere; pour la comparaison principale entre yeux, les "
        "p-values restent la sortie de reference.",
        va="top",
        fontsize=9,
        wrap=True,
    )
    display = _local_bump_scores_pdf_table(summary_table)
    if not display.empty:
        display = display.head(28)
    _draw_pdf_table(ax, display, title="", bbox=[0.02, 0.06, 0.96, 0.78])
    pdf.savefig(fig, bbox_inches="tight")


def _add_local_bump_run_page(
    pdf: PdfPages,
    run_dir: Path,
    summary: dict[str, object],
    system_scores: pd.DataFrame,
) -> None:
    descriptive_metrics = _descriptive_eye_metrics(system_scores, summary.get("scoring_method"))
    fig = Figure(figsize=(11.69, 8.27))
    axes = fig.subplots(1, 2, width_ratios=[1.1, 1.0])
    image_ax, text_ax = axes
    image_ax.axis("off")
    text_ax.axis("off")
    image_ax.set_title(f"{run_dir.name} - vaisseaux sauvegardes scores", fontsize=13)
    top_systems = _top_system_rows(system_scores)
    overlay = _load_skeleton_report_image(run_dir, system_scores, top_systems)
    if overlay is not None:
        image_ax.imshow(overlay)
    else:
        image_ax.text(0.5, 0.5, "Squelette non disponible", ha="center", va="center", fontsize=11)

    score_lines = [
        str(summary.get("scoring_method_label", "Score principal")),
        "",
        f"Score median: {_format_pdf_value(descriptive_metrics.get('score_median'))}",
        f"Score moyen: {_format_pdf_value(descriptive_metrics.get('score_mean'))}",
        f"Score moyen pondere: {_format_pdf_value(descriptive_metrics.get('score_weighted_mean'))}",
        "Vaisseaux retenus/sauvegardes: "
        f"{_kept_vessel_count_label(summary.get('eligible_vessel_count'), summary.get('saved_vessel_count'))}",
        f"Longueur totale vaisseaux: {_format_pdf_value(descriptive_metrics.get('saved_total_length'))}",
        f"Longueur totale vaisseaux retenus: {_format_pdf_value(descriptive_metrics.get('eligible_total_length'))}",
    ]
    text_ax.text(0.0, 0.98, "\n".join(score_lines), va="top", fontsize=11)
    top_table = _top_system_table(top_systems)
    _draw_pdf_table(text_ax, top_table, title="Vaisseaux ayant le plus haut score", bbox=[0.0, 0.06, 1.0, 0.52])
    pdf.savefig(fig, bbox_inches="tight")


def _add_all_systems_pdf_pages(
    pdf: PdfPages,
    all_systems_table: pd.DataFrame,
    system_scores_by_run: dict[str, pd.DataFrame],
) -> None:
    rows_per_page = 12
    total_rows = len(all_systems_table)
    page_count = max(1, math.ceil(total_rows / rows_per_page))
    for page_index in range(page_count):
        page = all_systems_table.iloc[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        fig = Figure(figsize=(11.69, 8.27))
        title_ax = fig.add_axes([0.02, 0.88, 0.96, 0.10])
        title_ax.axis("off")
        title_ax.text(0.0, 1.0, "Atlas visuel des vaisseaux tries par tortuosite", va="top", fontsize=15, fontweight="bold")
        title_ax.text(
            0.0,
            0.55,
            "Chaque vignette montre un vaisseau sauvegarde retenu par la methode active, trie du plus au moins "
            "tortueux. Le jaune correspond a la geometrie mesuree; la bordure indique le type: rouge = artere, bleu = veine.",
            va="top",
            fontsize=9,
            wrap=True,
        )
        title_ax.text(0.0, 0.06, f"Page {page_index + 1}/{page_count} - {total_rows} vaisseaux", va="bottom", fontsize=8.5)

        axes = fig.subplots(4, 3)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.84, hspace=0.42, wspace=0.10)
        flat_axes = list(axes.flat)
        for ax, (_, row) in zip(flat_axes, page.iterrows(), strict=False):
            ax.axis("off")
            saved_vessels = system_scores_by_run.get(str(row.get("Image", "")), pd.DataFrame())
            crop = _render_ranked_system_crop(row, saved_vessels)
            if crop is not None:
                ax.imshow(crop)
            else:
                ax.text(0.5, 0.5, "Image non disponible", ha="center", va="center", fontsize=8)
            title = (
                f"#{int(row.get('Rang', 0))}  {row.get('Image', 'NA')}  {row.get('Vaisseau', 'NA')}\n"
                f"score {_format_pdf_value(row.get('Score vaisseau'))} | "
                f"L {_format_pdf_value(row.get('Longueur'))} px"
            )
            ax.set_title(title, fontsize=7.5, pad=2)
        for ax in flat_axes[len(page) :]:
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")


def _render_ranked_system_crop(
    system_row: pd.Series,
    saved_vessels: pd.DataFrame,
    output_size: tuple[int, int] = (520, 300),
) -> Image.Image | None:
    run_dir = system_row.get("_run_dir")
    if not isinstance(run_dir, Path):
        return None
    image = _render_saved_vessel_overlay(run_dir, saved_vessels)
    if image is None:
        return None
    points = _valid_path_points(system_row.get("_path_points"))
    if len(points) < 2:
        canvas = image.resize(output_size)
        _draw_atlas_type_border(canvas, system_row.get("Categorie"))
        return canvas

    padding = 70
    min_x = max(0, int(math.floor(min(point[0] for point in points) - padding)))
    max_x = min(image.width, int(math.ceil(max(point[0] for point in points) + padding)))
    min_y = max(0, int(math.floor(min(point[1] for point in points) - padding)))
    max_y = min(image.height, int(math.ceil(max(point[1] for point in points) + padding)))
    if max_x <= min_x or max_y <= min_y:
        canvas = image.resize(output_size)
        _draw_atlas_type_border(canvas, system_row.get("Categorie"))
        return canvas

    crop = image.crop((min_x, min_y, max_x, max_y)).convert("RGB")
    scale = min(output_size[0] / crop.width, output_size[1] / crop.height)
    resized_size = (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale))))
    crop = crop.resize(resized_size)
    canvas = Image.new("RGB", output_size, (0, 0, 0))
    offset = ((output_size[0] - crop.width) // 2, (output_size[1] - crop.height) // 2)
    canvas.paste(crop, offset)

    shifted = [
        ((point[0] - min_x) * scale + offset[0], (point[1] - min_y) * scale + offset[1])
        for point in points
    ]
    draw = ImageDraw.Draw(canvas)
    draw.line(shifted, fill=(0, 0, 0), width=9, joint="curve")
    draw.line(shifted, fill=(255, 225, 86), width=5, joint="curve")
    _draw_atlas_type_border(canvas, system_row.get("Categorie"))
    return canvas


def _load_source_image(run_dir: Path) -> Image.Image | None:
    metadata = read_json(run_dir / "metadata.json")
    image_name = metadata.get("image_name")
    if isinstance(image_name, str):
        image_path = run_dir / image_name
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
    return None


def _vessel_rgb(category: object) -> tuple[int, int, int]:
    color = ARTERE_COLOR if str(category) == "artere" else VEINE_COLOR
    return _hex_to_rgb(color)


def _draw_atlas_type_border(image: Image.Image, category: object) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (0, 0, image.width - 1, image.height - 1),
        outline=_vessel_rgb(category),
        width=6,
    )


def _render_saved_vessel_overlay(run_dir: Path, saved_vessels: pd.DataFrame) -> Image.Image | None:
    image = _load_source_image(run_dir)
    if image is None:
        return None
    if saved_vessels.empty or "path_points" not in saved_vessels.columns:
        return image

    draw = ImageDraw.Draw(image)
    stroke_width = max(3, int(round(min(image.size) / 350.0)))
    for _, row in saved_vessels.iterrows():
        points = _valid_path_points(row.get("path_points"))
        if len(points) < 2:
            continue
        draw.line(points, fill=_vessel_rgb(row.get("category")), width=stroke_width, joint="curve")
    return image


def _valid_path_points(points: object) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    valid: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            valid.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return valid


def _load_skeleton_report_image(
    run_dir: Path,
    saved_vessels: pd.DataFrame,
    top_systems: pd.DataFrame,
) -> Image.Image | None:
    image = _render_saved_vessel_overlay(run_dir, saved_vessels)
    if image is None:
        return None
    original_size = image.size
    image.thumbnail((1500, 1100))
    _draw_top_system_highlights(image, original_size, top_systems)
    return image


def _top_system_rows(system_scores: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    eligible = _analysis_eligible_system_scores(system_scores)
    if eligible.empty:
        return pd.DataFrame()
    top = eligible.sort_values("primary_score", ascending=False).head(limit).copy()
    top.insert(0, "highlight_label", [f"V{index}" for index in range(1, len(top) + 1)])
    return top


def _top_system_table(top_systems: pd.DataFrame) -> pd.DataFrame:
    if top_systems.empty:
        return pd.DataFrame()
    columns = [
        "highlight_label",
        "vessel_name",
        "category",
        "primary_score",
        "vessel_length",
        "segment_count",
    ]
    display = top_systems.sort_values("primary_score", ascending=False, na_position="last")[
        [column for column in columns if column in top_systems.columns]
    ].copy()
    return display.rename(
        columns={
            "highlight_label": "Label",
            "vessel_name": "Vaisseau",
            "category": "Categorie",
            "primary_score": "Score vaisseau",
            "vessel_length": "Longueur",
            "segment_count": "Segments",
        }
    )


def _draw_top_system_highlights(
    image: Image.Image,
    original_size: tuple[int, int],
    top_systems: pd.DataFrame,
) -> None:
    if top_systems.empty or "path_points" not in top_systems.columns:
        return
    draw = ImageDraw.Draw(image)
    font = _load_font(max(15, min(24, image.width // 55)))
    scale_x = image.width / float(original_size[0])
    scale_y = image.height / float(original_size[1])
    palette = [
        "#ffe156",
        "#00d1ff",
        "#ff4f9a",
        "#6cff8d",
        "#ff9f1c",
        "#b388ff",
        "#ff6b6b",
        "#2ec4b6",
    ]
    for index, (_, row) in enumerate(top_systems.iterrows()):
        points = _scaled_branch_points(row.get("path_points"), scale_x, scale_y)
        if len(points) < 2:
            continue
        color = _hex_to_rgb(palette[index % len(palette)])
        draw.line(points, fill=(0, 0, 0), width=10, joint="curve")
        draw.line(points, fill=color, width=6, joint="curve")
        label = str(row.get("highlight_label", f"S{index + 1}"))
        label_x, label_y = points[len(points) // 2]
        bbox = draw.textbbox((label_x, label_y), label, font=font, anchor="mm")
        padding = 4
        draw.rounded_rectangle(
            (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
            radius=5,
            fill=(0, 0, 0),
            outline=color,
            width=2,
        )
        draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font, anchor="mm")


def _scaled_branch_points(points: object, scale_x: float, scale_y: float) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    scaled: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            scaled.append((float(point[0]) * scale_x, float(point[1]) * scale_y))
        except (TypeError, ValueError):
            continue
    return scaled


def _format_pdf_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.4g}"


def _add_cover_pdf_page(pdf: PdfPages, selected_run_names: list[str], counts_by_run: dict[str, int]) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    lines = ["Resultats de tortuosite retinienne", "", "Images incluses:"]
    lines.extend(f"- {run_name}: {counts_by_run.get(run_name, 0)} vaisseaux sauvegardes" for run_name in selected_run_names)
    ax.text(0.08, 0.92, "\n".join(lines), va="top", fontsize=13)
    pdf.savefig(fig, bbox_inches="tight")


def _add_stats_pdf_page(pdf: PdfPages, raw_matrix: pd.DataFrame, adjusted_matrix: pd.DataFrame) -> None:
    _add_stats_matrix_pdf_page(
        pdf,
        raw_matrix,
        title="P-values brutes",
        explanation=f"{STATS_EXPLANATION}\n\n{RAW_PVALUE_EXPLANATION}",
    )
    _add_stats_matrix_pdf_page(
        pdf,
        adjusted_matrix,
        title="P-values ajustees (BH)",
        explanation=f"{STATS_EXPLANATION}\n\n{BH_PVALUE_EXPLANATION}",
    )


def _add_stats_matrix_pdf_page(
    pdf: PdfPages,
    matrix: pd.DataFrame,
    title: str,
    explanation: str,
) -> None:
    fig = Figure(figsize=(16.54, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.02, 0.98, title, va="top", fontsize=17, fontweight="bold")
    ax.text(0.02, 0.92, explanation, va="top", fontsize=8.2, wrap=True)
    font_size = _stats_table_font_size(matrix)
    _draw_pdf_table(
        ax,
        matrix,
        title="",
        bbox=[0.02, 0.05, 0.96, 0.68],
        font_size=font_size,
        y_scale=1.18,
        colorizer=_pvalue_cell_color,
        keep_single_line=True,
    )
    pdf.savefig(fig, bbox_inches="tight")


def _stats_table_font_size(matrix: pd.DataFrame) -> float:
    max_dimension = max(len(matrix.columns), len(matrix.index), 1)
    if max_dimension >= 24:
        return 7.0
    if max_dimension >= 18:
        return 8.0
    return 9.0


def _add_segmentation_pdf_page(pdf: PdfPages, run_name: str, overlay: Image.Image, result_table: pd.DataFrame) -> None:
    fig = Figure(figsize=(11.69, 8.27))
    axes = fig.subplots(1, 2, width_ratios=[1.1, 1.0])
    image_ax, table_ax = axes
    image_ax.axis("off")
    table_ax.axis("off")
    image_ax.set_title(run_name, fontsize=13)
    image_ax.imshow(overlay)
    table_ax.text(0.0, 0.98, "Resultats de segmentation", va="top", fontsize=13, fontweight="bold")
    table_ax.text(0.0, 0.91, SEGMENTATION_EXPLANATION, va="top", fontsize=8, wrap=True)
    _draw_pdf_table(table_ax, _format_table_for_display(result_table), title="", bbox=[0.0, 0.05, 1.0, 0.72])
    pdf.savefig(fig, bbox_inches="tight")


def _draw_pdf_table(
    ax,
    table_df: pd.DataFrame,
    title: str,
    bbox: list[float],
    font_size: float | None = None,
    y_scale: float = 1.18,
    colorizer=None,
    keep_single_line: bool = True,
) -> None:
    if title:
        ax.text(bbox[0], bbox[1] + bbox[3] + 0.035, title, fontsize=11, fontweight="bold")
    if table_df.empty:
        ax.text(bbox[0], bbox[1] + bbox[3] / 2, "Aucune donnee", fontsize=9)
        return
    display_df = _format_table_for_display(table_df)
    table = ax.table(
        cellText=display_df.values,
        colLabels=list(display_df.columns),
        rowLabels=list(display_df.index) if display_df.index.name is not None else None,
        cellLoc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    resolved_font_size = font_size if font_size is not None else (6.5 if len(display_df.columns) > 5 else 8)
    table.set_fontsize(resolved_font_size)
    table.scale(1.0, y_scale)
    _style_pdf_table(table, display_df, colorizer=colorizer, keep_single_line=keep_single_line)


def _style_pdf_table(table, display_df: pd.DataFrame, colorizer=None, keep_single_line: bool = False) -> None:
    has_row_labels = display_df.index.name is not None
    for (row, col), cell in table.get_celld().items():
        text = cell.get_text()
        if row == 0 or col == -1:
            cell.set_facecolor("#f1f3f5")
            text.set_fontweight("bold")
            text.set_wrap(True)
            text.set_text(_wrapped_cell_text(text.get_text()))
            continue
        if keep_single_line:
            text.set_wrap(False)
            text.set_text(_single_line_cell_text(text.get_text()))
        if colorizer is None:
            continue
        data_row = row - 1
        data_col = col if not has_row_labels else col
        if data_row < 0 or data_row >= len(display_df.index) or data_col < 0 or data_col >= len(display_df.columns):
            continue
        cell.set_facecolor(colorizer(display_df.iloc[data_row, data_col]))


def _single_line_cell_text(value: object, max_chars: int = 24) -> str:
    text = str(value).replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1]}…"


def _wrapped_cell_text(value: object, line_width: int = 13) -> str:
    source = str(value).replace("\n", " ").replace("/", "/ ").strip()
    words = source.split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
            continue
        if len(current) + 1 + len(word) <= line_width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _pvalue_cell_color(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if text in {"-", "NA"}:
            return "#eceff1"
        return "#ffffff"
    if not math.isfinite(numeric):
        return "#eceff1"
    clipped = min(1.0, max(0.0, numeric))
    if clipped <= 0.001:
        return "#8b0000"
    if clipped <= 0.01:
        return "#d7301f"
    if clipped <= 0.05:
        return "#fc8d59"
    if clipped <= 0.1:
        return "#fee08b"
    if clipped <= 0.5:
        return "#d9ef8b"
    return "#91cf60"


def _format_table_for_display(table_df: pd.DataFrame) -> pd.DataFrame:
    display_df = table_df.copy()
    for column in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.4g}" if pd.notna(value) else "NA")
    return display_df


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.strip()
    if not color.startswith("#") or len(color) not in {4, 7}:
        return (255, 255, 255)
    if len(color) == 4:
        color = "#" + "".join(channel * 2 for channel in color[1:])
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()
