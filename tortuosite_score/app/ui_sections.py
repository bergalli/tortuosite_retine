from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tortuosite_score.app.review_data import read_json


def render_sidebar_run_setup() -> dict:
    with st.sidebar:
        st.header("Run setup")
        uploaded_file = st.file_uploader(
            "Retinal image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="Upload a fundus image to generate a vessel skeleton for manual review.",
        )
        method = st.selectbox(
            "Segmentation method",
            options=["deep", "classical"],
            index=0,
            help="Choose the segmentation mode used to generate the review skeleton.",
        )

        if method == "deep":
            deep_threshold = st.slider(
                "Deep threshold",
                0.0,
                1.0,
                0.30,
                0.01,
                help="Probability cutoff applied to the neural segmentation.",
            )
            deep_modality = st.selectbox(
                "Deep modality",
                options=["CFP", "UWF", "FFA", "SLO", "OCTA"],
                index=0,
            )
            vessel_percentile = 95.0
            vessel_low_percentile = 90.0
        else:
            vessel_percentile = st.slider("Vessel percentile", 50.0, 99.9, 95.0, 0.1)
            vessel_low_percentile = st.slider("Vessel low percentile", 0.0, 99.0, 90.0, 0.1)
            deep_threshold = 0.30
            deep_modality = "CFP"

        run_btn = st.button(
            "Run segmentation",
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
        "run_btn": run_btn,
    }


def render_debug_tab(run_dir: Path) -> None:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        st.subheader("Run metadata")
        st.json(read_json(metadata_path), expanded=False)

    manual_csv = run_dir / "manual_vessels.csv"
    if manual_csv.exists():
        st.subheader("Saved manual vessels")
        st.dataframe(pd.read_csv(manual_csv), use_container_width=True)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        st.subheader("Legacy auto-selection output")
        st.dataframe(pd.read_csv(results_csv), use_container_width=True)

    logs_path = run_dir / "logs.txt"
    if logs_path.exists():
        logs = logs_path.read_text(encoding="utf-8").strip()
        if logs:
            st.subheader("Run logs")
            st.code(logs, language="text")

    output_dir = run_dir / "output"
    image_files = sorted(output_dir.glob("*.png"))
    if image_files:
        st.subheader("Intermediate outputs")
        for image_file in image_files:
            st.image(str(image_file), caption=image_file.name, use_container_width=True)
