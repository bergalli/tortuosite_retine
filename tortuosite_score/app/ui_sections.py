from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tortuosite_score.app.review_data import read_json
from tortuosite_score.vessels_detection.dl_extra import (
    DEEP_INSTALL_HINT,
    deep_learning_available,
)
from tortuosite_score.vessels_detection.local_bump_score import LocalBumpSettings
from tortuosite_score.vessels_detection.scoring import (
    DEFAULT_SCORING_METHOD,
    available_scoring_methods,
    scoring_config,
    scoring_method_fixed_parameters,
)


def render_sidebar_run_setup() -> dict:
    with st.sidebar:
        st.header("Configuration de la session")
        uploaded_file = st.file_uploader(
            "Image retinienne",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="Importez une image du fond d'oeil pour generer un squelette vasculaire a revoir manuellement.",
        )
        deep_available = deep_learning_available()
        method_options = ["deep", "classical"] if deep_available else ["classical"]
        scoring_methods = available_scoring_methods()
        scoring_method_ids = [method.method_id for method in scoring_methods]
        default_scoring_index = (
            scoring_method_ids.index(st.session_state.get("active_scoring_method", DEFAULT_SCORING_METHOD))
            if st.session_state.get("active_scoring_method", DEFAULT_SCORING_METHOD) in scoring_method_ids
            else 0
        )
        active_scoring_method = st.selectbox(
            "Methode de score",
            options=scoring_method_ids,
            index=default_scoring_index,
            format_func=lambda method_id: next(
                method.label for method in scoring_methods if method.method_id == method_id
            ),
            help="Cette methode est appliquee partout: revue, resultats, PDF et CSV regeneres.",
        )
        st.session_state["active_scoring_method"] = active_scoring_method
        filter_short_vessels = st.toggle(
            "Filtrer les petits vaisseaux",
            value=st.session_state.get("filter_short_vessels", True),
            help="Exclut les vaisseaux sauvegardes trop courts du tableau de resultats et des p-values.",
        )
        st.session_state["filter_short_vessels"] = filter_short_vessels
        min_saved_vessel_length = st.number_input(
            "Longueur minimale vaisseau retenu (px normalises)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get("min_saved_vessel_length", 100.0)),
            step=10.0,
            disabled=not filter_short_vessels,
            help="Seuil applique apres normalisation du diametre du fond d'oeil a 1024 px.",
        )
        st.session_state["min_saved_vessel_length"] = min_saved_vessel_length
        resample_curvature_squared = st.toggle(
            "Re-echantillonner courbure quadratique",
            value=st.session_state.get("resample_curvature_squared", True),
            disabled=active_scoring_method != "curvature_squared",
            help="Pretraitement numerique avant l'estimation des derivees de courbure. Active par defaut pour reduire l'effet escalier des pixels.",
        )
        st.session_state["resample_curvature_squared"] = resample_curvature_squared
        curvature_resample_step = st.number_input(
            "Pas de re-echantillonnage courbure",
            min_value=0.5,
            max_value=50.0,
            value=float(st.session_state.get("curvature_resample_step", 4.0)),
            step=0.5,
            disabled=active_scoring_method != "curvature_squared" or not resample_curvature_squared,
            help="Distance entre points apres re-echantillonnage pour la courbure quadratique.",
        )
        st.session_state["curvature_resample_step"] = curvature_resample_step
        scoring_settings = LocalBumpSettings(
            resample_step=float(curvature_resample_step) if active_scoring_method == "curvature_squared" else 4.0,
            min_saved_vessel_length=float(min_saved_vessel_length),
            filter_short_vessels=bool(filter_short_vessels),
            resample_curvature_squared=bool(resample_curvature_squared),
        )
        active_scoring_config = scoring_config(active_scoring_method, scoring_settings)
        parameter_lines = scoring_method_fixed_parameters(active_scoring_config)
        if parameter_lines:
            st.caption("Parametres actifs de la methode selectionnee")
            st.markdown("\n".join(f"- {label}: `{value}`" for label, value in parameter_lines))
        method = st.selectbox(
            "Methode de segmentation",
            options=method_options,
            index=0,
            format_func=lambda value: "Apprentissage profond" if value == "deep" else "Classique",
            help="Choisissez le mode de segmentation utilise pour generer le squelette a revoir.",
        )
        if not deep_available:
            st.info(
                "Les moteurs d'apprentissage profond ne sont pas installes. "
                f"Installez-les avec `{DEEP_INSTALL_HINT}`."
            )

        if method == "deep":
            deep_backend = st.selectbox(
                "Moteur profond",
                options=["VascX", "DCP"],
                index=0,
                help="VascX ajoute la segmentation artere/veine et disque optique; DCP est le repli d'origine.",
            )
            deep_threshold = st.slider(
                "Seuil profond",
                0.0,
                1.0,
                0.30,
                0.01,
                help="Seuil de probabilite applique a la segmentation neuronale.",
                disabled=deep_backend == "VascX",
            )
            deep_modality = st.selectbox(
                "Modalite profonde",
                options=["CFP", "UWF", "FFA", "SLO", "OCTA"],
                index=0,
                disabled=deep_backend == "VascX",
            )
            if deep_backend == "VascX":
                vascx_av_size = st.select_slider(
                    "Taille entree arteres/veines VascX",
                    options=[512, 768, 1024, 1280],
                    value=1024,
                    help="Des valeurs plus grandes preservent mieux les petits vaisseaux mais demandent plus de memoire et de temps.",
                )
                vascx_use_contrast_enhancement = st.toggle(
                    "Renforcement du contraste VascX",
                    value=True,
                    help="Utiliser le renforcement du contraste VascX avant l'inference.",
                )
                vascx_min_object_size = st.slider(
                    "Taille minimale de nettoyage VascX",
                    0,
                    100,
                    12,
                    1,
                    help="Supprimer les petites composantes connexes apres la segmentation artere/veine.",
                )
                vascx_closing_radius = st.slider(
                    "Rayon de fermeture VascX",
                    0,
                    4,
                    1,
                    1,
                    help="Refermer les petits trous apres la segmentation artere/veine.",
                )
                vascx_auto_create_vessels = False
                vascx_auto_min_vessel_length = 25.0
            else:
                vascx_av_size = 1024
                vascx_use_contrast_enhancement = True
                vascx_min_object_size = 12
                vascx_closing_radius = 1
                vascx_auto_create_vessels = False
                vascx_auto_min_vessel_length = 25.0
            vessel_percentile = 95.0
            vessel_low_percentile = 90.0
        else:
            vessel_percentile = st.slider("Percentile vaisseaux", 50.0, 99.9, 95.0, 0.1)
            vessel_low_percentile = st.slider("Percentile bas vaisseaux", 0.0, 99.0, 90.0, 0.1)
            deep_threshold = 0.30
            deep_modality = "CFP"
            deep_backend = "DCP"
            vascx_av_size = 1024
            vascx_use_contrast_enhancement = True
            vascx_min_object_size = 12
            vascx_closing_radius = 1
            vascx_auto_create_vessels = False
            vascx_auto_min_vessel_length = 25.0

        run_btn = st.button(
            "Lancer la segmentation",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

    return {
        "uploaded_file": uploaded_file,
        "method": method,
        "vessel_percentile": vessel_percentile,
        "vessel_low_percentile": vessel_low_percentile,
        "deep_threshold": deep_threshold,
        "deep_modality": deep_modality,
        "deep_backend": deep_backend,
        "vascx_av_size": vascx_av_size,
        "vascx_use_contrast_enhancement": vascx_use_contrast_enhancement,
        "vascx_min_object_size": vascx_min_object_size,
        "vascx_closing_radius": vascx_closing_radius,
        "vascx_auto_create_vessels": vascx_auto_create_vessels,
        "vascx_auto_min_vessel_length": vascx_auto_min_vessel_length,
        "active_scoring_method": active_scoring_method,
        "active_scoring_settings": scoring_settings,
        "run_btn": run_btn,
    }


def render_debug_tab(run_dir: Path) -> None:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        st.subheader("Metadonnees de la session")
        st.json(read_json(metadata_path), expanded=False)

    manual_csv = run_dir / "manual_vessels.csv"
    if manual_csv.exists():
        st.subheader("Vaisseaux manuels sauvegardes")
        st.dataframe(pd.read_csv(manual_csv), use_container_width=True)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        st.subheader("Sortie historique de selection automatique")
        st.dataframe(pd.read_csv(results_csv), use_container_width=True)

    logs_path = run_dir / "logs.txt"
    if logs_path.exists():
        logs = logs_path.read_text(encoding="utf-8").strip()
        if logs:
            st.subheader("Journaux de session")
            st.code(logs, language="text")

    output_dir = run_dir / "output"
    image_files = sorted(output_dir.glob("*.png"))
    if image_files:
        st.subheader("Sorties intermediaires")
        for image_file in image_files:
            st.image(str(image_file), caption=image_file.name, use_container_width=True)
