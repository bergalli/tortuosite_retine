from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import mannwhitneyu

from tortuosite_score.app.constants import ARTERE_COLOR, RUNS_ROOT, VEINE_COLOR
from tortuosite_score.app.review_data import list_runs, load_review_bundle, read_json
from tortuosite_score.app.review_state import (
    get_segment_geometry,
    score_vessel,
    segment_ref_sort_key,
    segment_refs_for_vessel,
)
from tortuosite_score.app.viewer_component import BRANCH_VIEWER

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

    counts_by_run = {run.name: saved_vessel_count(run) for run in runs}
    default_runs = [run.name for run in runs if counts_by_run[run.name] > 0]
    option_labels = {
        run.name: f"{run.name} ({counts_by_run[run.name]} vaisseaux sauvegardes)"
        for run in runs
    }
    selected_run_names = st.multiselect(
        "Images a comparer",
        options=[run.name for run in runs],
        default=default_runs,
        format_func=lambda run_name: option_labels[run_name],
    )
    if not selected_run_names:
        st.info("Selectionnez au moins une image pour afficher les resultats.")
        return
    label_font_size = st.slider("Taille des labels des vaisseaux", min_value=14, max_value=42, value=24, step=2)

    result_tables: dict[str, pd.DataFrame] = {}
    loaded_results: dict[str, tuple[dict, pd.DataFrame, dict]] = {}
    for run_name in selected_run_names:
        run_dir = RUNS_ROOT / run_name
        bundle = load_review_bundle(str(run_dir))
        branches_df = pd.DataFrame(bundle["branches_df"])
        review_state = load_saved_review_state(run_dir)
        result_table = build_result_rows(review_state, branches_df)
        result_tables[run_name] = result_table
        loaded_results[run_name] = (bundle, branches_df, review_state)

    st.header("Statistiques")
    st.write(STATS_EXPLANATION)
    raw_pvalue_matrix = build_pvalue_matrix(
        {run_name: tortuosity_values(result_table) for run_name, result_table in result_tables.items()}
    )
    adjusted_pvalue_matrix = build_adjusted_pvalue_matrix(raw_pvalue_matrix)
    st.subheader("P-values ajustees (BH)")
    st.dataframe(adjusted_pvalue_matrix, width="stretch")
    st.subheader("P-values brutes")
    st.dataframe(raw_pvalue_matrix, width="stretch")

    pdf_bytes = generate_results_pdf(
        selected_run_names,
        counts_by_run,
        loaded_results,
        result_tables,
        raw_pvalue_matrix,
        adjusted_pvalue_matrix,
    )
    st.download_button(
        "Telecharger le PDF des resultats",
        data=pdf_bytes,
        file_name="resultats_tortuosite.pdf",
        mime="application/pdf",
    )

    st.header("Resultats de segmentation")
    st.write(SEGMENTATION_EXPLANATION)
    for run_name in selected_run_names:
        bundle, branches_df, review_state = loaded_results[run_name]
        result_table = result_tables[run_name]
        st.subheader(run_name)
        if result_table.empty:
            st.info("Aucun vaisseau sauvegarde pour cette image.")
            continue
        segments, labels = build_results_viewer_segments(review_state, branches_df)
        BRANCH_VIEWER(
            data={
                "imageUrl": bundle["image_url"],
                "imageWidth": bundle["image_width"],
                "imageHeight": bundle["image_height"],
                "segments": segments,
                "selectedSegmentRefs": [],
                "interactionMode": "readonly",
                "showBaseImage": True,
                "showSkeleton": True,
                "showLabels": False,
                "showVesselLabels": True,
                "vesselLabels": labels,
                "vesselLabelFontSize": label_font_size,
                "baseOpacity": 0.95,
            },
            key=f"results_viewer::{run_name}",
            width="stretch",
            height=640,
        )
        st.dataframe(visible_result_table(result_table), width="stretch", hide_index=True)


def _endpoint_caption(endpoint: dict[str, object] | None) -> str:
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("point"), list):
        return "Aucun"
    point = endpoint["point"]
    if len(point) != 2:
        return "Aucun"
    return f"({float(point[0]):.1f}, {float(point[1]):.1f})"


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
