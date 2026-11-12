from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from tortuosite_score.visualization import anatomy_panels as anatomy

from tortuosite_score.vessels_detection.clinical_excel import (
    VesselNameInfo,
    analysis_vessel_eligibility,
    classified_vessel_export_eligibility,
    parse_vessel_name,
)
from tortuosite_score.vessels_detection.local_bump_score import (
    DISPLAY_SCALE,
    LocalBumpDiagnostics,
    LocalBumpSettings,
    local_bump_diagnostics,
)
from tortuosite_score.vessels_detection.scoring import (
    ScoringConfig,
    score_run,
    scoring_config,
)


INK = "#17202a"
MUTED = "#68727d"
GRID = "#d9dee3"
VEIN = "#2468b4"
ARTERY = "#b34a3c"
MANUAL = "#7b4ab5"
PAPER = "#fbfbfa"
INDEX_ROWS_PER_PAGE = 30


@dataclass(frozen=True)
class AtlasVessel:
    run_dir: Path
    source_path: Path
    row: dict[str, object]
    name_info: VesselNameInfo
    summary: dict[str, object]
    vessel_scores: pd.DataFrame


# Keep the original import and saved diagnostic records compatible.
AtlasVein = AtlasVessel


@dataclass(frozen=True)
class VesselLabels:
    vessel_type: str
    singular: str
    plural: str
    filename_plural: str
    symbol: str
    color: str

    @property
    def article(self) -> str:
        return "l’artère" if self.vessel_type == "artere" else "la veine"

    @property
    def of(self) -> str:
        return "de l’artère" if self.vessel_type == "artere" else "de la veine"

    def rank(self, value: int) -> str:
        return f"{value}°{self.symbol}"


VESSEL_LABELS = {
    "veine": VesselLabels("veine", "veine", "veines", "veines", "V", VEIN),
    "artere": VesselLabels("artere", "artère", "artères", "arteres", "A", ARTERY),
}


def _vessel_labels(vessel_type: str) -> VesselLabels:
    try:
        return VESSEL_LABELS[vessel_type]
    except KeyError:
        raise ValueError(f"Type de vaisseau inconnu : {vessel_type}") from None


def _record_labels(record) -> VesselLabels:
    return _vessel_labels(
        getattr(getattr(record, "name_info", None), "vessel_type", "veine")
    )


def _numbered_runs(runs_root: Path) -> list[Path]:
    return sorted(
        (
            p
            for p in runs_root.iterdir()
            if p.is_dir() and re.fullmatch(r"\d+_(OD|OG)", p.name)
        ),
        key=lambda p: (int(p.name.split("_")[0]), p.name),
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def _source_path(run_dir: Path) -> Path:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées absentes : {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_name = metadata.get("image_name")
    if not image_name:
        raise ValueError(f"Nom d'image absent des métadonnées de {run_dir.name}")
    path = run_dir / str(image_name)
    if not path.exists():
        raise FileNotFoundError(f"Image source absente : {path}")
    return path


def is_ranked_publication_vessel(
    row: pd.Series | dict[str, object], vessel_type: str = "veine"
) -> bool:
    """Mirror the publication overlay filter for a single saved vessel."""

    _vessel_labels(vessel_type)
    vessel_name = str(row.get("vessel_name", ""))
    category = row.get("category")
    vessel_length = row.get("vessel_length")
    name_info = parse_vessel_name(vessel_name, category)
    analysis_eligible = analysis_vessel_eligibility(
        vessel_name,
        category,
        vessel_length,
    )[0]
    export_eligible = classified_vessel_export_eligibility(
        name_info,
        analysis_eligible=analysis_eligible,
        scoring_eligible=bool(row.get("eligible", False)),
        score=row.get("primary_score"),
        vessel_length=vessel_length,
    )[0]
    return bool(
        export_eligible
        and name_info.vessel_type == vessel_type
        and name_info.rank in {1, 2, 3}
        and bool(row.get("eligible", False))
        and pd.notna(pd.to_numeric(row.get("primary_score"), errors="coerce"))
        and math.isfinite(float(row["primary_score"]))
    )


def collect_ranked_publication_vessels(
    runs_root: str | Path,
    config: ScoringConfig | None = None,
    *,
    vessel_type: str = "veine",
) -> list[AtlasVessel]:
    """Load the numbered cohort; select one publication vessel type."""
    _vessel_labels(vessel_type)

    runs_root = Path(runs_root)
    config = config or scoring_config("local_bump")
    records: list[AtlasVessel] = []
    for run_dir in _numbered_runs(runs_root):
        if not (run_dir / "output" / "08_full_skeleton_summary.csv").exists():
            continue
        summary, vessel_scores = score_run(run_dir, config)
        source_path = _source_path(run_dir)
        for _, row in vessel_scores.iterrows():
            if not is_ranked_publication_vessel(row, vessel_type):
                continue
            row_dict = row.to_dict()
            records.append(
                AtlasVessel(
                    run_dir=run_dir,
                    source_path=source_path,
                    row=row_dict,
                    name_info=parse_vessel_name(
                        str(row_dict.get("vessel_name", "")),
                        row_dict.get("category"),
                    ),
                    summary=summary,
                    vessel_scores=vessel_scores,
                )
            )
    return sorted(
        records,
        key=lambda record: (
            float(record.row["primary_score"]),
            record.run_dir.name,
            str(record.row["vessel_name"]),
        ),
    )


def is_ranked_publication_vein(row: pd.Series | dict[str, object]) -> bool:
    return is_ranked_publication_vessel(row, "veine")


def collect_ranked_publication_veins(runs_root, config=None) -> list[AtlasVessel]:
    return collect_ranked_publication_vessels(runs_root, config, vessel_type="veine")


def generate_method_visualizations(
    runs_root: str | Path,
    output_dir: str | Path,
    pipeline_run: str = "2_OD",
    *,
    vessel_type: str = "veine",
) -> dict[str, object]:
    _configure_style()
    labels = _vessel_labels(vessel_type)
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = scoring_config("local_bump")
    runs = (
        _numbered_runs(runs_root)
        if pipeline_run == "all"
        else [runs_root / pipeline_run]
    )
    if not runs:
        raise ValueError("Aucun œil numéroté à illustrer")
    pipeline_prefix = "pipeline_extraction" + (
        "_arteres" if vessel_type == "artere" else ""
    )
    pipelines = []
    for run in runs:
        pdf = output_dir / f"{pipeline_prefix}_{run.name}.pdf"
        png = pdf.with_suffix(".png")
        generate_pipeline_figure(run, pdf, png, config=config, vessel_type=vessel_type)
        pipelines.append((pdf, png))
        print(f"Extraction {labels.plural} : {run.name}", flush=True)

    records = collect_ranked_publication_vessels(
        runs_root, config=config, vessel_type=vessel_type
    )
    atlas_pdf = output_dir / f"atlas_local_bump_{labels.filename_plural}.pdf"
    generate_local_bump_atlas(
        records, atlas_pdf, settings=config.local_bump_settings, vessel_type=vessel_type
    )
    overview_stem = "explication_local_bump" + (
        "_arteres" if vessel_type == "artere" else ""
    )
    overview_pdf = output_dir / f"{overview_stem}.pdf"
    overview_png = overview_pdf.with_suffix(".png")
    generate_local_bump_overview(
        records, overview_pdf, overview_png, config.local_bump_settings
    )
    return {
        "overview_pdf": overview_pdf,
        "overview_png": overview_png,
        "pipeline_pdf": pipelines[0][0] if len(pipelines) == 1 else None,
        "pipeline_png": pipelines[0][1] if len(pipelines) == 1 else None,
        "pipeline_pdfs": [pair[0] for pair in pipelines],
        "pipeline_pngs": [pair[1] for pair in pipelines],
        "atlas_pdf": atlas_pdf,
        "vessel_count": len(records),
        "vessel_type": vessel_type,
        "vein_count" if vessel_type == "veine" else "artery_count": len(records),
    }


def generate_pipeline_figure(
    run_dir: str | Path,
    output_pdf: str | Path,
    output_png: str | Path,
    config: ScoringConfig | None = None,
    *,
    vessel_type: str = "veine",
) -> None:
    """A numbered image sequence: preparation, saved VascX outputs, review."""
    labels = _vessel_labels(vessel_type)
    run_dir = Path(run_dir)
    config = config or scoring_config("local_bump")
    output = run_dir / "output"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if str(metadata.get("deep_backend", "")).upper() != "VASCX":
        raise ValueError("La figure d'extraction nécessite une session VascX")
    _, vessels = score_run(run_dir, config)
    source = Image.open(_source_path(run_dir)).convert("RGB")
    empty_disc = not np.asarray(Image.open(output / "04c_vascx_disc_mask.png")).any()
    panels = [
        (
            "Image couleur",
            "01_loaded_image.png",
            "Image RGB fournie au modèle VascX.",
            None,
        ),
        (
            "Canal vert",
            "02_green_channel.png",
            "Canal utilisé pour délimiter le fond d’œil.",
            None,
        ),
        (
            "Champ rétinien",
            "02b_fundus_mask.png",
            "Zone conservée dans les prédictions\naprès l’inférence VascX.",
            None,
        ),
        (
            "Vaisseaux détectés",
            "04_vascx_vessel_mask.png",
            "Sortie VascX : ensemble des pixels\nidentifiés comme vaisseaux.",
            None,
        ),
        (
            "Artères et veines",
            "04b_vascx_artery_vein_overlay.png",
            "Classification VascX sur l’image :\nartères en rouge, veines en bleu.",
            None,
        ),
        (
            "Détection du disque optique",
            "04d_vascx_disc_overlay.png",
            (
                "Sortie VascX : aucun disque détecté ici.\nL’estimation de debug est distincte."
                if empty_disc
                else "Région du disque optique détectée\npar le second modèle VascX."
            ),
            None,
        ),
        (
            "Masque artériel nettoyé",
            "06_cleaned_artery_mask.png",
            "Petits objets retirés ; fermeture\ndes petites discontinuités.",
            None,
        ),
        (
            "Masque veineux nettoyé",
            "06_cleaned_vein_mask.png",
            "Même nettoyage, appliqué séparément\naux pixels classés comme veines.",
            None,
        ),
        (
            "Lignes centrales",
            "07b_skeleton_overlay.png",
            "Artères et veines squelettisées séparément,\npuis réunies pour la revue dans l’application.",
            None,
        ),
        (
            "Assemblage et corrections",
            "Tracés et corrections sauvegardés",
            "Résultat de la revue manuelle.\nPointillés : portions ajoutées à la main.",
            _manual_review_overlay(
                run_dir, source, vessels, label_examples=False, show_endpoints=False
            ),
        ),
        (
            "Extrémités, catégorie et nom",
            "État final de la revue sauvegardée",
            "Cercles : extrémités choisies.\nÉtiquettes : exemples de noms enregistrés.",
            _manual_review_overlay(run_dir, source, vessels),
        ),
        (
            f"{labels.plural.capitalize()} retenues pour l’export",
            f"Filtre de publication des {labels.plural}",
            f"{labels.plural.capitalize()} éligibles nommées {labels.rank(1)}, {labels.rank(2)} ou {labels.rank(3)}.\nLibellés normalisés ; tracés inchangés.",
            _ranked_vessel_overlay(source, vessels, vessel_type),
        ),
    ]
    figure = Figure(figsize=(17.5, 14.2), facecolor="white")
    figure.text(
        0.035,
        0.967,
        f"De l’image aux {labels.plural} retenues pour l’analyse",
        fontsize=23,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.035,
        0.929,
        f"{run_dir.name} · lire de gauche à droite, puis passer à la ligne suivante · 1–3 préparation ; 4–9 extraction ; 10–12 revue et export",
        fontsize=11,
        color=MUTED,
        va="top",
    )
    for i, (title, filename, caption, provided) in enumerate(panels):
        row, col = divmod(i, 4)
        x = 0.035 + col * 0.245
        y = 0.685 - row * 0.275
        axis = figure.add_axes([x, y, 0.215, 0.205])
        image = provided if provided is not None else Image.open(output / filename)
        axis.imshow(image, cmap="gray" if image.mode in {"1", "L"} else None)
        axis.set_axis_off()
        figure.text(
            x,
            y + 0.221,
            f"{i+1:02d}  {title}",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        figure.text(x, y - 0.006, caption, fontsize=9.5, va="top", linespacing=1.4)
        figure.text(x, y - 0.043, filename, fontsize=7.5, color=MUTED, va="top")
    contrast = "activé" if metadata["vascx_use_contrast_enhancement"] else "désactivé"
    figure.text(
        0.035,
        0.073,
        f"Dans VascX : redimensionnement A/V à {metadata['vascx_av_size']} px, disque à 512 px ; contraste {contrast}. Les entrées transformées ne sont pas sauvegardées.\n"
        f"Nettoyage : objets < {metadata['vascx_min_object_size']} pixels retirés, fermeture de rayon {metadata['vascx_closing_radius']} pixel. Les vues ci-dessus sont les sorties réelles sauvegardées.",
        fontsize=9,
        color=MUTED,
        va="top",
        linespacing=1.5,
    )
    figure.text(
        0.035,
        0.032,
        "03_preprocessed.png relève du prétraitement classique et n’est pas fourni à VascX. L’estimation du disque (04e/04f) sert uniquement au debug.\n"
        "Les panneaux de revue montrent l’état sauvegardé, sans reconstituer l’historique des gestes. Le calcul du score est expliqué dans l’atlas Local-bump.",
        fontsize=8.5,
        color=MUTED,
        va="top",
        linespacing=1.4,
    )
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf)
    figure.savefig(output_png, dpi=300)
    plt.close(figure)


def _manual_review_overlay(
    run_dir: Path,
    source: Image.Image,
    vessel_scores: pd.DataFrame,
    *,
    label_examples: bool = True,
    show_endpoints: bool = True,
) -> Image.Image:
    image = source.copy()
    draw = ImageDraw.Draw(image)
    width = max(3, int(round(min(image.size) / 280)))
    for _, row in vessel_scores.iterrows():
        points = _path_points(row.get("path_points"))
        if len(points) < 2:
            continue
        color = VEIN if row.get("category") == "veine" else ARTERY
        draw.line(points, fill=color, width=width, joint="curve")
        radius = max(3, width)
        for endpoint in (points[0], points[-1]) if show_endpoints else ():
            draw.ellipse(
                (
                    endpoint[0] - radius,
                    endpoint[1] - radius,
                    endpoint[0] + radius,
                    endpoint[1] + radius,
                ),
                fill="#ffffff",
                outline=color,
                width=max(1, width // 2),
            )
    state_path = run_dir / "manual_review_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for segment in state.get("manual_segments", {}).values():
            points = _path_points(segment.get("points"))
            if len(points) >= 2:
                # Dash by arc length, independent of the path's point density.
                traveled = 0.0
                for start, end in zip(points[:-1], points[1:]):
                    a, b = np.asarray(start), np.asarray(end)
                    length = float(np.linalg.norm(b - a))
                    pieces = max(1, math.ceil(length / width))
                    for j in range(pieces):
                        if int((traveled + length * j / pieces) / (width * 2)) % 2 == 0:
                            draw.line(
                                [
                                    tuple(a + (b - a) * j / pieces),
                                    tuple(a + (b - a) * (j + 1) / pieces),
                                ],
                                fill=MANUAL,
                                width=width + 2,
                            )
                    traveled += length
    # Three real names illustrate labelling without covering the saved tracings.
    candidates = [
        row
        for _, row in vessel_scores.iterrows()
        if len(_path_points(row.get("path_points"))) >= 2
    ]
    examples = candidates[:: max(1, len(candidates) // 3)][:3] if label_examples else []
    font = _pil_font(max(18, round(min(image.size) / 23)))
    for j, row in enumerate(examples):
        points = _path_points(row.get("path_points"))
        middle = points[len(points) // 2]
        position = (int(image.width * 0.07), int(image.height * (0.09 + 0.12 * j)))
        name = str(row.get("vessel_name", ""))
        bounds = draw.textbbox(position, name, font=font)
        draw.line(
            [middle, (bounds[2], (bounds[1] + bounds[3]) / 2)],
            fill="white",
            width=width,
        )
        draw.rectangle(
            (bounds[0] - 5, bounds[1] - 5, bounds[2] + 5, bounds[3] + 5), fill="white"
        )
        draw.text(position, name, font=font, fill=INK)
    return image


def _ranked_vessel_overlay(
    source: Image.Image, vessel_scores: pd.DataFrame, vessel_type: str = "veine"
) -> Image.Image:
    labels = _vessel_labels(vessel_type)
    image = source.copy()
    veil = Image.new("RGBA", image.size, (255, 255, 255, 130))
    image = Image.alpha_composite(image.convert("RGBA"), veil).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(4, int(round(min(image.size) / 220)))
    font = _pil_font(max(18, int(round(min(image.size) / 19))))
    for _, row in vessel_scores.iterrows():
        if not is_ranked_publication_vessel(row, vessel_type):
            continue
        points = _path_points(row.get("path_points"))
        if len(points) < 2:
            continue
        draw.line(points, fill="#ffffff", width=width + 4, joint="curve")
        draw.line(points, fill=labels.color, width=width, joint="curve")
        middle = points[len(points) // 2]
        info = parse_vessel_name(str(row.get("vessel_name", "")), row.get("category"))
        draw.text(
            (middle[0] + width, middle[1] - width),
            labels.rank(info.rank),
            fill=INK,
            font=font,
            stroke_width=2,
            stroke_fill="#ffffff",
        )
    return image


def _ranked_vein_overlay(source, vessel_scores):
    return _ranked_vessel_overlay(source, vessel_scores, "veine")


def generate_local_bump_atlas(
    records: Iterable[AtlasVessel],
    output_pdf: str | Path,
    settings: LocalBumpSettings | None = None,
    *,
    vessel_type: str | None = None,
) -> None:
    settings = settings or LocalBumpSettings()
    records = list(records)
    vessel_type = vessel_type or (
        _record_labels(records[0]).vessel_type if records else "veine"
    )
    _vessel_labels(vessel_type)
    if any(_record_labels(record).vessel_type != vessel_type for record in records):
        raise ValueError("Un atlas doit contenir un seul type de vaisseau")
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    index_page_count = max(1, math.ceil(len(records) / INDEX_ROWS_PER_PAGE))
    with PdfPages(output_pdf) as pdf:
        cover = _atlas_cover_page(len(records), settings, vessel_type=vessel_type)
        pdf.savefig(cover)
        plt.close(cover)
        for page_index in range(index_page_count):
            start = page_index * INDEX_ROWS_PER_PAGE
            page_records = records[start : start + INDEX_ROWS_PER_PAGE]
            index_figure = _atlas_index_page(
                page_records,
                start_index=start,
                index_page_count=index_page_count,
                index_page_number=page_index + 1,
                vessel_type=vessel_type,
            )
            pdf.savefig(index_figure)
            plt.close(index_figure)
        for record_index, record in enumerate(records, start=1):
            page_number = 1 + index_page_count + record_index
            figure = _atlas_detail_page(
                record, record_index, len(records), page_number, settings
            )
            pdf.savefig(figure)
            plt.close(figure)


def _atlas_cover_page(
    vessel_count: int, settings: LocalBumpSettings, vessel_type: str = "veine"
) -> Figure:
    """Standalone plain-language methodology, including every formula symbol."""
    labels = _vessel_labels(vessel_type)
    figure = Figure(figsize=(16.54, 11.69), facecolor="white")

    def text(x, y, value, size=12, **kw):
        figure.text(x / 16.54, y / 11.69, value, fontsize=size, va="top", **kw)

    def section(x, y, title, body):
        text(x, y, title, 13, fontweight="bold")
        text(x, y - 0.34, textwrap.fill(body, 83), 11.5, linespacing=1.45)

    text(
        0.5, 11.28, "Local-bump v1 — comprendre toute la méthode", 23, fontweight="bold"
    )
    text(
        0.5,
        10.82,
        f"{vessel_count} {labels.plural} de la cohorte *_OD / *_OG · fiches classées par score croissant · exemples sans interprétation clinique",
        11,
        color=MUTED,
    )
    section(
        0.5,
        10.30,
        f"1 · Partir du tracé {labels.of} et définir L",
        f"La ligne centrale est une suite ordonnée de points sur {labels.article}. On multiplie leurs coordonnées par un facteur propre à l’image pour ramener le diamètre rétinien à 1024 pixels. L est la longueur totale de ce tracé normalisé, mesurée avant le rééchantillonnage et le lissage. Les distances suivantes sont en pixels normalisés.",
    )
    section(
        0.5,
        8.68,
        "2 · Espacer régulièrement les points",
        f"On parcourt ce tracé et on place un point tous les {anatomy.number(settings.resample_step,0)} pixels de distance le long de la ligne. Le dernier intervalle peut être plus court. Ce rééchantillonnage permet de mesurer les changements de direction à une même échelle spatiale dans toutes les images.",
    )
    section(
        0.5,
        7.25,
        "3 · Lisser la ligne avant de mesurer les angles",
        f"Une moyenne glissante sur {settings.smoothing_window} points atténue les petites irrégularités du tracé pixelisé ; les extrémités sont conservées. On obtient une nouvelle ligne centrale, plus régulière. C’est sur cette ligne lissée, et non sur le tracé brut, que tous les angles du score sont mesurés.",
    )
    section(
        0.5,
        5.79,
        "4 · Mesurer Δθᵢ à chaque point intérieur",
        f"Trois points successifs de la ligne lissée définissent deux segments. Δθᵢ est la rotation entre le prolongement du segment entrant et le segment sortant ; i repère l’angle dans la suite. Le signe indique le sens de cette rotation dans les coordonnées de l’image. Les encarts a et b des fiches illustrent chacun un seul angle réel ; le calcul est répété sur toute {labels.article}.",
    )
    section(
        0.5,
        4.13,
        "5 · Écarter les changements trop petits : δᵢ et τ",
        f"Le seuil fixe τ vaut {anatomy.number(settings.curvature_threshold)} radian. Si la valeur absolue de Δθᵢ est inférieure à τ, on remplace cet angle par zéro. Sinon, on conserve sa valeur signée. δᵢ est l’angle ainsi filtré. Un angle exactement égal au seuil en valeur absolue est conservé.",
    )
    text(
        0.6,
        2.73,
        r"$\delta_i=\Delta\theta_i$ si $|\Delta\theta_i|\geq\tau$, sinon $0$",
        17,
    )
    section(
        8.7,
        10.30,
        "6 · Calculer l’amplitude angulaire moyenne E",
        "m est le nombre total d’angles mesurés, y compris ceux remplacés par zéro. On additionne les valeurs absolues des angles filtrés : Σ|δᵢ|. On divise cette somme par m pour obtenir E, en radians. Les signes opposés ne s’annulent donc pas dans cette moyenne.",
    )
    text(8.8, 8.96, r"$E=\frac{\sum_{i=1}^{m}|\delta_i|}{m}$", 19)
    section(
        8.7,
        8.17,
        "7 · Compter les alternances : N",
        "On lit les angles filtrés dans l’ordre du tracé et on ignore les zéros. N compte les passages d’un signe positif à un signe négatif, ou l’inverse. Par exemple, +, +, 0, −, 0, + donne N = 2. Les numéros sur les images localisent exactement les transitions retenues par le calcul.",
    )
    section(
        8.7,
        6.62,
        "8 · Rapporter ce nombre à la longueur : D",
        f"D est le nombre de transitions pour 100 pixels normalisés. On divise N par la longueur L du tracé normalisé, puis on multiplie par 100. Deux {labels.plural} de longueurs différentes peuvent ainsi être comparées sur la même distance de référence.",
    )
    text(8.8, 5.36, r"$D=100\,\frac{N}{L}$", 19)
    section(
        8.7,
        4.60,
        "9 · Combiner l’amplitude et la densité : B",
        f"B est le score Local-bump {labels.of}. On multiplie E par la racine carrée de D. Le score associe donc la taille moyenne des changements d’angle à la fréquence des alternances. Si N = 0, alors D = 0 et B = 0, même lorsque certains angles sont non nuls.",
    )
    text(8.8, 3.25, r"$B=E\sqrt{D}$", 21)
    text(0.5, 2.00, "Population incluse et score de l’œil", 13, fontweight="bold")
    text(
        0.5,
        1.67,
        f"Atlas : {labels.plural} retenues pour l’export avec score fini, nom de rang 1–3 et longueur admissible ≥ {settings.min_saved_vessel_length:.0f} px. La longueur « app » sert au filtre et à la pondération de l’œil.\n"
        "G est la moyenne des scores Bᵥ de tous les vaisseaux sauvegardés éligibles (artères et veines), pondérée par leur longueur app.\n"
        "T est la moyenne pondérée des 20 % de longueur aux scores les plus élevés ; le dernier vaisseau peut n’y contribuer qu’en partie.\n"
        "Le score affiché pour l’œil est Sœil = 1000 × (0,70 G + 0,30 T). Il utilise une population différente de celle de cet atlas.",
        11,
        linespacing=1.55,
    )
    return figure


def _atlas_index_page(
    records: list[AtlasVessel],
    start_index: int,
    index_page_count: int,
    index_page_number: int,
    vessel_type: str = "veine",
) -> Figure:
    labels = _vessel_labels(vessel_type)
    figure = Figure(figsize=(8.27, 11.69), facecolor=PAPER)
    axis = figure.subplots()
    axis.set_axis_off()
    axis.text(
        0.02,
        0.975,
        f"Index des {labels.plural}",
        fontsize=19,
        fontweight="bold",
        transform=axis.transAxes,
        va="top",
    )
    axis.text(
        0.98,
        0.975,
        f"{index_page_number}/{index_page_count} · tri croissant par B",
        fontsize=9,
        color=MUTED,
        transform=axis.transAxes,
        va="top",
        ha="right",
    )
    rows = []
    for local_index, record in enumerate(records):
        atlas_index = start_index + local_index + 1
        physical_page = 1 + index_page_count + atlas_index
        rows.append(
            [
                atlas_index,
                physical_page,
                record.run_dir.name,
                str(record.row["vessel_name"]),
                labels.rank(record.name_info.rank),
                anatomy.number(float(record.row["primary_score"]), 5),
                anatomy.number(float(record.row["vessel_length"]), 1),
            ]
        )
    table = axis.table(
        cellText=rows,
        colLabels=[
            "Fiche",
            "Page PDF",
            "Œil",
            "Nom sauvegardé",
            "Label",
            "B",
            "L app (px)",
        ],
        colLoc="left",
        cellLoc="left",
        colWidths=[0.07, 0.10, 0.12, 0.40, 0.08, 0.10, 0.10],
        bbox=[0.02, 0.035, 0.96, 0.89],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row_index == 0:
            cell.set_facecolor("#dfeaf3")
            cell.set_text_props(weight="bold", color=INK)
        elif row_index % 2 == 0:
            cell.set_facecolor("#f1f3f5")
    return figure


def _record_diagnostics(record: AtlasVessel, settings: LocalBumpSettings):
    raw = np.asarray(_path_points(record.row.get("path_points")), dtype=float)
    scale = float(record.row.get("coordinate_scale", 1.0))
    diagnostics = local_bump_diagnostics(raw * scale, settings)
    if not np.isclose(
        diagnostics.metrics["branch_bump_score"],
        float(record.row["primary_score"]),
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(
            f"Score incohérent : {record.run_dir.name} / {record.row['vessel_name']}"
        )
    return raw, scale, diagnostics


def _atlas_detail_page(
    record, atlas_index, total, page_number, settings, *, comparisons=None
):
    labels = _record_labels(record)
    raw, scale, diagnostics = _record_diagnostics(record, settings)
    source = Image.open(record.source_path).convert("RGB")
    metrics = diagnostics.metrics
    extra = 3.0 if comparisons is not None else 0.0
    width, height = 16.54, 11.69 + extra
    figure = Figure(figsize=(width, height), facecolor="white")

    def ax(x, y, w, h):
        return figure.add_axes([x / width, (y + extra) / height, w / width, h / height])

    def text(x, y, value, size=11, **kwargs):
        return figure.text(
            x / width, (y + extra) / height, value, fontsize=size, va="top", **kwargs
        )

    title = (
        f"Comprendre le score Local-bump sur une {labels.singular} réelle"
        if comparisons is not None
        else f"{record.run_dir.name} · {record.row['vessel_name']} · {labels.rank(record.name_info.rank)}"
    )
    text(0.45, 11.3, title, 21, fontweight="bold")
    text(
        0.45,
        10.78,
        f"{record.run_dir.name} · {record.row['vessel_name']}     B = {anatomy.number(metrics['branch_bump_score'],5)}     L = {anatomy.number(metrics['branch_length'],1)} px normalisés",
        12,
        color=MUTED,
    )
    if comparisons is None:
        text(
            16.05,
            11.25,
            f"Fiche {atlas_index}/{total} · page {page_number}",
            10,
            ha="right",
            color=MUTED,
        )
    text(
        0.45,
        10.15,
        f"1  Repérer {labels.article} et son sens de parcours",
        12,
        fontweight="bold",
    )
    context_axis = ax(0.45, 6.65, 4.65, 2.55)
    box = anatomy.context(context_axis, source, raw)
    anatomy.retina_locator(ax(0.45, 9.10, 1.15, 0.85), source, box)
    text(
        1.8,
        9.72,
        "Cadre blanc : zone agrandie\nFlèche : ordre des points sauvegardés",
        10,
        color=MUTED,
    )
    text(5.7, 10.15, "2  Régulariser puis lisser le tracé", 12, fontweight="bold")
    kept, rejected = anatomy.example_indices(diagnostics)
    anatomy.preparation(
        ax(5.7, 7.25, 4.65, 2.5), source, diagnostics, scale, kept, settings
    )
    text(
        5.7,
        6.98,
        f"1. Ramener le diamètre rétinien à 1024 px (× {anatomy.number(scale,4)}).\n2. Placer un point ○ tous les {anatomy.number(settings.resample_step,0)} px le long du tracé.\n3. Lisser par moyenne glissante sur {settings.smoothing_window} points.\nLa ligne bleue lissée sert à mesurer tous les angles.",
        11,
    )
    text(
        5.7,
        6.12,
        "Pointillés gris : tracé brut. Le dernier pas peut être plus court.",
        9,
        color=MUTED,
    )
    text(
        11, 10.15, "3  Exemples : un seul angle réel par encart", 12, fontweight="bold"
    )
    text(
        11, 9.84, f"a · Angle de cette {labels.singular} : retenu", 9, fontweight="bold"
    )
    text(
        13.7,
        9.84,
        f"b · Angle de cette {labels.singular} : annulé",
        9,
        fontweight="bold",
    )
    anatomy.angle_example(
        ax(11, 7.48, 2.15, 2.15), source, diagnostics, scale, kept, settings
    )
    anatomy.angle_example(
        ax(13.7, 7.48, 2.15, 2.15),
        source,
        diagnostics,
        scale,
        rejected,
        settings,
        False,
    )
    text(
        11,
        6.30,
        f"Chaque encart illustre un angle local, pas toute {labels.article}.\nLe même calcul est répété à chaque point intérieur.\nPointillés : prolongement entrant ; Δθ : rotation (rad).",
        10,
        color=MUTED,
    )
    text(
        0.45,
        5.20,
        f"4  Repérer les angles retenus sur {labels.article}",
        12,
        fontweight="bold",
    )
    anatomy.retained(
        ax(0.45, 2.35, 4.65, 2.45), source, diagnostics, scale, label_examples=True
    )
    text(0.45, 2.08, "+ positif", 11, color=anatomy.POSITIVE)
    text(1.65, 2.08, "− négatif", 11, color=anatomy.NEGATIVE)
    text(2.90, 2.08, "• annulé sous le seuil", 11, color=anatomy.MUTED)
    text(
        0.45,
        1.76,
        "Signes dans les coordonnées de l’image : y vers le bas.",
        9,
        color=MUTED,
    )
    text(5.7, 5.20, "5  Compter les alternances de signe", 12, fontweight="bold")
    groups = anatomy.transition_groups(diagnostics)
    cols = 1 if len(groups) <= 2 else 2
    rows = math.ceil(len(groups) / cols)
    for j, group in enumerate(groups):
        col, row = j % cols, j // cols
        w = 4.9 / cols
        h = 2.8 / rows
        anatomy.transitions(
            ax(5.7 + col * w, 2.12 + (rows - row - 1) * h + 0.16, w - 0.18, h - 0.36),
            source,
            diagnostics,
            scale,
            group,
            first_number=1 + j * 6,
        )
    text(
        5.7,
        1.92,
        f"N = {int(metrics['oscillation_count'])} transitions numérotées ; les δ = 0 sont ignorés.",
        10,
        fontweight="bold",
    )
    text(
        5.7,
        1.63,
        "Repères discrets du calcul, pas des inflexions interpolées.",
        9,
        color=MUTED,
    )
    text(11, 5.20, f"6  Calculer le score {labels.of}", 12, fontweight="bold")
    m = len(diagnostics.filtered_angle_change)
    absolute_sum = float(np.abs(diagnostics.filtered_angle_change).sum())
    num = anatomy.number
    text(
        11,
        4.75,
        f"L = {num(metrics['branch_length'],3)} px     m = {m} angles\n"
        f"Σ|δᵢ| = {num(absolute_sum,6)} rad\n\n"
        f"E = {num(absolute_sum,6)} / {m}\n"
        f"    = {num(metrics['local_bump_energy'],6)} rad\n"
        f"D = 100 × {int(metrics['oscillation_count'])} / {num(metrics['branch_length'],3)} = {num(metrics['oscillation_density'],4)}\n\n"
        f"B = {num(metrics['local_bump_energy'],6)} × √{num(metrics['oscillation_density'],4)}\n"
        f"    = {num(metrics['branch_bump_score'],6)}",
        12,
        linespacing=1.45,
    )
    text(
        11,
        2.05,
        "m comprend les angles annulés ; L est mesurée\nsur le tracé normalisé avant lissage.\nD : transitions pour 100 px normalisés.",
        10,
        color=MUTED,
    )
    text(
        0.45,
        1.15,
        _eye_note(record, settings, metrics["branch_length"]),
        10,
        linespacing=1.5,
    )
    text(
        0.45,
        0.42,
        f"Local-bump v1 · valeurs affichées arrondies · mêmes mesures que l’application · {labels.plural} éligibles de rang 1–3.",
        9,
        color=MUTED,
    )
    if comparisons is not None:
        _draw_comparisons(figure, comparisons, settings, width, height)
    return figure


def _eye_note(record, settings, path_length=None):
    labels = _record_labels(record)
    allocations = _tail_allocations(record.vessel_scores, settings.tail_length_fraction)
    target = next(
        (
            i
            for i, row in record.vessel_scores.iterrows()
            if row.get("vessel_id") == record.row.get("vessel_id")
        ),
        None,
    )
    take = allocations.get(target, 0)
    length = float(record.row["vessel_length"])
    participation = (
        "non"
        if take <= 0
        else f"oui ({anatomy.number(100*take/length,1)} % de sa longueur)"
    )
    summary = record.summary
    g = float(summary["all_vessels_score"]) / DISPLAY_SCALE
    t = float(summary["all_vessels_tail_score"]) / DISPLAY_SCALE
    note = (
        f"Œil : tous les vaisseaux sauvegardés éligibles, artères et veines. G = {anatomy.number(g,5)} ; T = {anatomy.number(t,5)} ; "
        f"Sœil = 1000 × (0,70 G + 0,30 T) = {anatomy.number(float(summary['eye_tortuosity_score']),2)}.\n"
        f"T porte sur les 20 % de longueur aux scores les plus élevés. Cette {labels.singular} y contribue : {participation}."
    )

    if path_length is not None and not math.isclose(path_length, length, rel_tol=1e-12):
        note += (
            f"\nLongueur de cette {labels.singular} pour le filtre et la pondération de l’œil : {anatomy.number(length,3)} px ; "
            f"L utilisée dans B : {anatomy.number(path_length,3)} px."
        )
    return note


def select_comparison_records(records):
    records = sorted(
        records,
        key=lambda r: (
            float(r.row["primary_score"]),
            r.run_dir.name,
            str(r.row["vessel_name"]),
        ),
    )
    if not records:
        raise ValueError("Aucun vaisseau éligible pour l’explication Local-bump")
    return [records[0], records[len(records) // 2], records[-1]]


def _draw_comparisons(figure, records, settings, width, height):
    figure.text(
        0.45 / width,
        2.85 / height,
        "Scores minimum, médian et maximum de la cohorte — exemples sans interprétation clinique",
        fontsize=12,
        fontweight="bold",
    )
    diagnostics = [_record_diagnostics(r, settings)[2] for r in records]
    # Equal normalized field of view, hence equal physical scale across examples.
    span = max(max(np.ptp(d.geometry, axis=0)) for d in diagnostics) * 1.22
    for j, (record, d, label) in enumerate(
        zip(records, diagnostics, ("Minimum", "Médiane", "Maximum"))
    ):
        x = 0.45 + j * 5.35
        scale = float(record.row["coordinate_scale"])
        source = Image.open(record.source_path).convert("RGB")
        axis = figure.add_axes([x / width, 0.60 / height, 4.8 / width, 1.72 / height])
        anatomy.retained(axis, source, d, scale)
        center = (d.geometry.min(axis=0) + d.geometry.max(axis=0)) / (2 * scale)
        axis.set_xlim(center[0] - span / (2 * scale), center[0] + span / (2 * scale))
        axis.set_ylim(center[1] + span / (2 * scale), center[1] - span / (2 * scale))
        figure.text(
            x / width,
            2.48 / height,
            f"{label} · {record.run_dir.name} · {record.row['vessel_name']}",
            fontsize=11,
        )
        figure.text(
            x / width,
            0.30 / height,
            f"B = {anatomy.number(d.metrics['branch_bump_score'],5)} · N = {int(d.metrics['oscillation_count'])} · même échelle normalisée",
            fontsize=10,
        )


def generate_local_bump_overview(records, output_pdf, output_png, settings=None):
    settings = settings or LocalBumpSettings()
    comparisons = select_comparison_records(records)
    figure = _atlas_detail_page(
        comparisons[1], 1, len(records), 1, settings, comparisons=comparisons
    )
    figure.savefig(output_pdf)
    figure.savefig(output_png, dpi=300)
    plt.close(figure)


def _tail_allocations(vessels: pd.DataFrame, fraction: float) -> dict[object, float]:
    eligible = (
        vessels[vessels["eligible"]].copy() if "eligible" in vessels else vessels.copy()
    )
    eligible = eligible.sort_values("primary_score", ascending=False)
    remaining = float(eligible["vessel_length"].sum()) * fraction
    allocations: dict[object, float] = {}
    for row_index, row in eligible.iterrows():
        length = max(float(row["vessel_length"]), 0.0)
        take = min(length, remaining)
        allocations[row_index] = take
        remaining -= take
        if remaining <= 1e-9:
            break
    return allocations


def _path_points(value: object) -> list[tuple[float, float]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return points


def _pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=size)
    except (OSError, ValueError):
        return ImageFont.load_default()
