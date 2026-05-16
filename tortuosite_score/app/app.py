from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from tortuosite_score.vessels_detection.main import run_pipeline


st.set_page_config(page_title="Tortuosite Retine", layout="wide")
st.title("Retinal Tortuosity Scoring")
st.write("Run your vessel detection pipeline and inspect outputs from the interface.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "demo" / "streamlit_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def _slugify_name(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cleaned.strip("_") or "image"


def _render_run(run_dir: Path) -> None:
    st.subheader(f"Run: {run_dir.name}")
    st.caption(f"Folder: {run_dir}")

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.json(metadata, expanded=False)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        st.subheader("Selected branches")
        st.dataframe(pd.read_csv(results_csv), use_container_width=True)
        st.download_button(
            "Download results.csv",
            data=results_csv.read_bytes(),
            file_name=f"{run_dir.name}_results.csv",
            mime="text/csv",
            key=f"download_{run_dir.name}",
        )

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


uploaded_file = st.file_uploader(
    "Retinal image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    help="Upload a fundus image to analyze vessel tortuosity.",
)

with st.sidebar:
    st.header("Parameters")
    method = st.selectbox(
        "Method",
        options=["deep", "classical"],
        index=0,
        help="`deep`: neural network segmentation. `classical`: Frangi + thresholding.",
    )
    max_branches = st.number_input(
        "Max branches",
        min_value=1,
        max_value=30,
        value=3,
        help="Maximum number of significant branches kept in final CSV output.",
    )

    if method == "deep":
        deep_threshold = st.slider(
            "Deep threshold",
            0.0,
            1.0,
            0.30,
            0.01,
            help="Probability cutoff applied to deep model output (higher = stricter vessel detection).",
        )
        deep_modality = st.selectbox(
            "Deep modality",
            options=["CFP", "UWF", "FFA", "SLO", "OCTA"],
            index=0,
            help="Image modality hint used by the deep model.",
        )
        vessel_percentile = 95.0
        vessel_low_percentile = 90.0
    else:
        vessel_percentile = st.slider(
            "Vessel percentile",
            50.0,
            99.9,
            95.0,
            0.1,
            help="High threshold percentile for Frangi vesselness binarization.",
        )
        vessel_low_percentile = st.slider(
            "Vessel low percentile",
            0.0,
            99.0,
            90.0,
            0.1,
            help="Low threshold percentile for hysteresis (set below high percentile).",
        )
        deep_threshold = 0.30
        deep_modality = "CFP"

run_btn = st.button("Run analysis", type="primary", disabled=uploaded_file is None)

if run_btn and uploaded_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_slugify_name(uploaded_file.name)}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image_path = run_dir / uploaded_file.name
    image_path.write_bytes(uploaded_file.getvalue())

    output_csv = run_dir / "results.csv"
    output_dir = run_dir / "output"
    log_buffer = io.StringIO()

    with st.spinner("Running pipeline..."):
        with redirect_stdout(log_buffer):
            run_pipeline(
                image_path=str(image_path),
                output_csv=str(output_csv),
                max_branches=int(max_branches),
                output_dir=str(output_dir),
                method=method,
                vessel_percentile=float(vessel_percentile),
                vessel_low_percentile=float(vessel_low_percentile),
                deep_threshold=float(deep_threshold),
                deep_modality=deep_modality,
            )

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "image_name": uploaded_file.name,
        "method": method,
        "max_branches": int(max_branches),
        "vessel_percentile": float(vessel_percentile),
        "vessel_low_percentile": float(vessel_low_percentile),
        "deep_threshold": float(deep_threshold),
        "deep_modality": deep_modality,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "logs.txt").write_text(log_buffer.getvalue(), encoding="utf-8")

    st.success("Run completed.")
    _render_run(run_dir)
