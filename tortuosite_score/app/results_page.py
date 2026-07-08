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
from tortuosite_score.vessels_detection.local_bump_score import (
    LocalBumpSettings,
    add_comparative_hybrid_score,
    score_run,
)

RESULTS_SCHEMA_VERSION = 2
VISIBLE_RESULT_COLUMNS = ["Label", "Vaisseau", "Categorie", "Longueur du trajet", "Corde", "Tortuosite"]
STATS_EXPLANATION = (
    "Chaque cellule compare l'image en ligne a l'image en colonne. Une petite p-value indique des preuves que "
    "l'image en ligne a des valeurs de tortuosite plus elevees que l'image en colonne. Une grande p-value ne donne "
    "pas de preuve forte dans ce sens. Les p-values ajustees sont a privilegier quand plusieurs yeux sont compares."
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
HYBRID_SCORE_EXPLANATION = (
    "Le score final de l'oeil combine une moyenne des vaisseaux sauvegardes ponderee par la longueur et une composante "
    "de queue superieure pour que quelques vaisseaux tres bossues ne soient pas dilues par de nombreux vaisseaux "
    "normaux. Le score comparatif est normalise dans la cohorte du rapport."
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
    synthetic_index = 0
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
        for synthetic_link in vessel.get("synthetic_links", []):
            points = synthetic_link.get("points")
            if not isinstance(points, list) or len(points) != 2:
                continue
            vessel_points.extend(points)
            segments.append(
                {
                    "segmentRef": f"result-synthetic:{synthetic_index}",
                    "source": "synthetic",
                    "points": points,
                    "label": None,
                    "locked": True,
                    "strokes": [
                        {"color": color, "width": 5.2, "opacity": 0.45},
                        {"color": color, "width": 2.6, "opacity": 1},
                    ],
                }
            )
            synthetic_index += 1
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


def generate_local_bump_results_pdf(run_dirs: list[Path]) -> bytes:
    settings, scored_runs, summary_table = _score_local_bump_report_runs(run_dirs)
    return _generate_local_bump_results_pdf_from_scores(settings, scored_runs, summary_table)


def _score_local_bump_report_runs(
    run_dirs: list[Path],
) -> tuple[LocalBumpSettings, list[tuple[Path, dict[str, object], pd.DataFrame]], pd.DataFrame]:
    settings = LocalBumpSettings()
    run_dirs = _local_bump_report_runs(run_dirs)
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]] = []
    for run_dir in run_dirs:
        try:
            summary, system_scores = score_run(run_dir, settings)
        except (FileNotFoundError, ValueError, OSError, KeyError):
            continue
        scored_runs.append((run_dir, summary, system_scores))

    summary_table = _build_local_bump_summary_table(scored_runs)
    comparative_scores = dict(zip(summary_table.get("Image", []), summary_table.get("Score comparatif", []), strict=False))
    for run_dir, summary, _system_scores in scored_runs:
        if run_dir.name in comparative_scores:
            summary["comparative_hybrid_score"] = comparative_scores[run_dir.name]
    return settings, scored_runs, summary_table


def _generate_local_bump_results_pdf_from_scores(
    settings: LocalBumpSettings,
    scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]],
    summary_table: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        _add_local_bump_cover_page(pdf, summary_table)
        _add_local_bump_method_page(pdf, settings)
        _add_local_bump_scores_page(pdf, summary_table)
        for run_dir, summary, system_scores in scored_runs:
            _add_local_bump_run_page(pdf, run_dir, summary, system_scores)
        _add_all_systems_pdf_pages(pdf, _build_all_systems_table(scored_runs))
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


def render_results_page() -> None:
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
                    "- Le score principal mesure des bosses et oscillations locales, pas seulement un grand virage regulier.",
                    "- `Arc/chord` reste visible seulement comme diagnostic.",
                ]
            )
        )
    with st.spinner("Calcul des scores local-bump..."):
        settings, scored_runs, summary_table = _score_local_bump_report_runs(runs)
        all_systems_table = _build_all_systems_table(scored_runs)

    if summary_table.empty:
        st.warning("Aucun run exploitable trouve pour calculer le score local-bump.")
        return

    st.dataframe(summary_table, hide_index=True, width="stretch")
    st.download_button(
        "Telecharger les scores CSV",
        data=summary_table.to_csv(index=False).encode("utf-8"),
        file_name="scores_tortuosite_local_bump.csv",
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
        data=_generate_local_bump_results_pdf_from_scores(settings, scored_runs, summary_table),
        file_name="rapport_tortuosite_local_bump.pdf",
        mime="application/pdf",
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
        rows.append(
            {
                "Image": run_dir.name,
                "Oeil": summary.get("eye_number"),
                "Score local-bump": summary.get("eye_tortuosity_score"),
                "Score moyen": summary.get("all_vessels_score"),
                "Queue superieure": summary.get("all_vessels_tail_score"),
                "Arteres": summary.get("artery_score"),
                "Veines": summary.get("vein_score"),
                "Arc/chord diagnostic": summary.get("current_arc_chord_score"),
                "Vaisseaux retenus": summary.get("eligible_vessel_count"),
                "Vaisseaux sauvegardes": summary.get("saved_vessel_count"),
                "Longueur totale": summary.get("eligible_total_length"),
            }
        )
    if not rows:
        return pd.DataFrame()
    table = add_comparative_hybrid_score(pd.DataFrame(rows).rename(columns={
        "Score local-bump": "eye_tortuosity_score",
        "Score moyen": "all_vessels_score",
    }))
    table = table.rename(columns={
        "eye_tortuosity_score": "Score local-bump",
        "all_vessels_score": "Score moyen",
        "comparative_hybrid_score": "Score comparatif",
    })
    table = table.drop(columns=["system_component_norm", "saved_vessel_component_norm"], errors="ignore")
    return table.sort_values("Score comparatif", ascending=False).reset_index(drop=True)


def _build_all_systems_table(scored_runs: list[tuple[Path, dict[str, object], pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir, summary, vessel_scores in scored_runs:
        if vessel_scores.empty or "eligible" not in vessel_scores.columns:
            continue
        eligible = vessel_scores[vessel_scores["eligible"]].copy()
        for _, row in eligible.iterrows():
            rows.append(
                {
                    "Image": run_dir.name,
                    "Oeil": summary.get("eye_number"),
                    "Vaisseau": row.get("vessel_name"),
                    "Categorie": row.get("category"),
                    "Score vaisseau": row.get("vessel_bump_score"),
                    "Longueur": row.get("vessel_length"),
                    "Segments": row.get("segment_count"),
                    "Ponts": row.get("bridge_count"),
                    "Arc/chord diagnostic": row.get("arc_chord_diagnostic"),
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


def _add_local_bump_cover_page(pdf: PdfPages, summary_table: pd.DataFrame) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    image_count = len(summary_table)
    lines = [
        "Rapport de tortuosite retinienne",
        "",
        "Methode actuelle: score local-bump sur vaisseaux sauvegardes",
        "",
        f"Images analysees: {image_count}",
        "",
        "Objectif clinique:",
        "mesurer les irregularites locales, bosses et oscillations des vaisseaux sauvegardes,",
        "plutot qu'un simple allongement arc/corde.",
        "",
        "Important:",
        "le calcul final ne score pas tout le squelette directement; il score seulement",
        "les vaisseaux sauvegardes dans chaque session.",
    ]
    ax.text(0.08, 0.92, "\n".join(lines), va="top", fontsize=13)
    pdf.savefig(fig, bbox_inches="tight")


def _add_local_bump_method_page(pdf: PdfPages, settings: LocalBumpSettings) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.08, 0.94, "Methode de calcul", va="top", fontsize=16, fontweight="bold")
    ax.text(0.08, 0.88, LOCAL_BUMP_METHOD_EXPLANATION, va="top", fontsize=10, wrap=True)
    steps = [
        "1. Sauvegarder les vaisseaux manuellement ou avec le bouton d'auto-completion VascX.",
        "2. Reconstituer la ligne centrale ordonnee de chaque vaisseau sauvegarde.",
        f"3. Exclure les vaisseaux trop courts: longueur minimale {settings.min_saved_vessel_length:.0f} px.",
        f"4. Re-echantillonner chaque vaisseau tous les {settings.resample_step:.1f} px.",
        f"5. Lisser legerement la ligne centrale: fenetre {settings.smoothing_window} points.",
        "6. Calculer les changements locaux d'angle le long du vaisseau.",
        f"7. Ignorer les changements minuscules: seuil {settings.curvature_threshold:.3f} rad.",
        "8. Compter les alternances de courbure et calculer un score local-bump.",
        "9. Agreger l'oeil entier: 70% moyenne ponderee par longueur + 30% queue superieure.",
    ]
    ax.text(0.08, 0.62, "\n".join(steps), va="top", fontsize=10)
    ax.text(0.08, 0.34, HYBRID_SCORE_EXPLANATION, va="top", fontsize=9, wrap=True)
    ax.text(0.08, 0.24, "\n".join(LOCAL_BUMP_EQUATION_LINES), va="top", fontsize=8.3)
    literature_lines = ["Litterature utilisee:"] + [f"- {item}" for item in LOCAL_BUMP_LITERATURE]
    ax.text(0.08, 0.06, "\n".join(literature_lines), va="bottom", fontsize=8.2)
    pdf.savefig(fig, bbox_inches="tight")


def _add_local_bump_scores_page(pdf: PdfPages, summary_table: pd.DataFrame) -> None:
    fig = Figure(figsize=(11.69, 8.27))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.02, 0.98, "Resume des scores par image", va="top", fontsize=15, fontweight="bold")
    ax.text(
        0.02,
        0.92,
        "Les images sont triees par score comparatif. Ce score est calcule a partir des vaisseaux sauvegardes; "
        "les colonnes voisines donnent les composantes brutes.",
        va="top",
        fontsize=9,
        wrap=True,
    )
    display = summary_table.copy()
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
    fig = Figure(figsize=(11.69, 8.27))
    axes = fig.subplots(1, 2, width_ratios=[1.1, 1.0])
    image_ax, text_ax = axes
    image_ax.axis("off")
    text_ax.axis("off")
    image_ax.set_title(f"{run_dir.name} - vaisseaux sauvegardes scores", fontsize=13)
    top_systems = _top_system_rows(system_scores)
    overlay = _load_skeleton_report_image(run_dir, top_systems)
    if overlay is not None:
        image_ax.imshow(overlay)
    else:
        image_ax.text(0.5, 0.5, "Squelette non disponible", ha="center", va="center", fontsize=11)

    score_lines = [
        "Score local-bump",
        "",
        f"Score final: {_format_pdf_value(summary.get('eye_tortuosity_score'))}",
        f"Score comparatif: {_format_pdf_value(summary.get('comparative_hybrid_score'))}",
        f"Score moyen: {_format_pdf_value(summary.get('all_vessels_score'))}",
        f"Queue superieure: {_format_pdf_value(summary.get('all_vessels_tail_score'))}",
        f"Arteres: {_format_pdf_value(summary.get('artery_score'))}",
        f"Veines: {_format_pdf_value(summary.get('vein_score'))}",
        f"Arc/chord diagnostic: {_format_pdf_value(summary.get('current_arc_chord_score'))}",
        f"Vaisseaux retenus: {summary.get('eligible_vessel_count', 0)}",
        f"Vaisseaux sauvegardes: {summary.get('saved_vessel_count', 0)}",
    ]
    text_ax.text(0.0, 0.98, "\n".join(score_lines), va="top", fontsize=11)
    top_table = _top_system_table(top_systems)
    _draw_pdf_table(text_ax, top_table, title="Vaisseaux qui contribuent le plus", bbox=[0.0, 0.06, 1.0, 0.52])
    pdf.savefig(fig, bbox_inches="tight")


def _add_all_systems_pdf_pages(pdf: PdfPages, all_systems_table: pd.DataFrame) -> None:
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
            "Chaque vignette montre un vaisseau sauvegarde score, trie du plus au moins tortueux. Le surlignage "
            "jaune correspond exactement a la geometrie mesuree.",
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
            crop = _render_ranked_system_crop(row)
            if crop is not None:
                ax.imshow(crop)
            else:
                ax.text(0.5, 0.5, "Image non disponible", ha="center", va="center", fontsize=8)
            title = (
                f"#{int(row.get('Rang', 0))}  {row.get('Image', 'NA')}  {row.get('Vaisseau', 'NA')}\n"
                f"score {_format_pdf_value(row.get('Score vaisseau'))} | "
                f"L {_format_pdf_value(row.get('Longueur'))} px | "
                f"arc/chord {_format_pdf_value(row.get('Arc/chord diagnostic'))}"
            )
            ax.set_title(title, fontsize=7.5, pad=2)
        for ax in flat_axes[len(page) :]:
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")


def _render_ranked_system_crop(system_row: pd.Series, output_size: tuple[int, int] = (520, 300)) -> Image.Image | None:
    run_dir = system_row.get("_run_dir")
    if not isinstance(run_dir, Path):
        return None
    image = _load_base_skeleton_image(run_dir)
    if image is None:
        return None
    points = _valid_path_points(system_row.get("_path_points"))
    if len(points) < 2:
        return image.resize(output_size)

    padding = 70
    min_x = max(0, int(math.floor(min(point[0] for point in points) - padding)))
    max_x = min(image.width, int(math.ceil(max(point[0] for point in points) + padding)))
    min_y = max(0, int(math.floor(min(point[1] for point in points) - padding)))
    max_y = min(image.height, int(math.ceil(max(point[1] for point in points) + padding)))
    if max_x <= min_x or max_y <= min_y:
        return image.resize(output_size)

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
    draw.rectangle((0, 0, output_size[0] - 1, output_size[1] - 1), outline=(255, 255, 255), width=1)
    return canvas


def _load_base_skeleton_image(run_dir: Path) -> Image.Image | None:
    for file_name in ["07b_skeleton_overlay.png", "07_skeleton.png"]:
        image_path = run_dir / "output" / file_name
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
    return None


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


def _load_skeleton_report_image(run_dir: Path, top_systems: pd.DataFrame) -> Image.Image | None:
    for file_name in ["07b_skeleton_overlay.png", "07_skeleton.png"]:
        image_path = run_dir / "output" / file_name
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            image.thumbnail((1500, 1100))
            _draw_top_system_highlights(image, original_size, top_systems)
            return image
    return None


def _top_system_rows(system_scores: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if system_scores.empty:
        return pd.DataFrame()
    eligible = system_scores[system_scores["eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame()
    top = eligible.sort_values("vessel_bump_score", ascending=False).head(limit).copy()
    top.insert(0, "highlight_label", [f"V{index}" for index in range(1, len(top) + 1)])
    return top


def _top_system_table(top_systems: pd.DataFrame) -> pd.DataFrame:
    if top_systems.empty:
        return pd.DataFrame()
    columns = [
        "highlight_label",
        "vessel_name",
        "category",
        "vessel_bump_score",
        "vessel_length",
        "segment_count",
        "bridge_count",
        "arc_chord_diagnostic",
        "local_bump_energy",
        "oscillation_count",
    ]
    display = top_systems[[column for column in columns if column in top_systems.columns]].copy()
    return display.rename(
        columns={
            "highlight_label": "Label",
            "vessel_name": "Vaisseau",
            "category": "Categorie",
            "vessel_bump_score": "Score vaisseau",
            "vessel_length": "Longueur",
            "segment_count": "Segments",
            "bridge_count": "Ponts",
            "arc_chord_diagnostic": "Arc/chord",
            "local_bump_energy": "Energie locale",
            "oscillation_count": "Oscillations",
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
    fig = Figure(figsize=(11.69, 8.27))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.02, 0.98, "Statistiques", va="top", fontsize=15, fontweight="bold")
    ax.text(0.02, 0.91, STATS_EXPLANATION, va="top", fontsize=9, wrap=True)
    _draw_pdf_table(ax, adjusted_matrix, title="P-values ajustees (BH)", bbox=[0.02, 0.47, 0.96, 0.30])
    _draw_pdf_table(ax, raw_matrix, title="P-values brutes", bbox=[0.02, 0.08, 0.96, 0.30])
    pdf.savefig(fig, bbox_inches="tight")


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


def _draw_pdf_table(ax, table_df: pd.DataFrame, title: str, bbox: list[float]) -> None:
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
    table.set_fontsize(6.5 if len(display_df.columns) > 5 else 8)
    table.scale(1.0, 1.18)


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
