from __future__ import annotations

import base64
import io
import json
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skan import Skeleton

from tortuosite_score.app.constants import RUNS_ROOT
from tortuosite_score.vessels_detection.analyse import skeletonize_mask
from tortuosite_score.vessels_detection.main import run_pipeline


def slugify_name(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cleaned.strip("_") or "image"


def list_runs() -> list[Path]:
    return sorted(
        (path for path in RUNS_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def image_to_data_url(path: Path) -> str:
    mime_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_by_suffix.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_review_bundle(run_dir_str: str) -> dict:
    run_dir = Path(run_dir_str)
    metadata = read_json(run_dir / "metadata.json")
    output_dir = run_dir / "output"
    image_name = metadata.get("image_name")
    image_path = run_dir / image_name if image_name else None

    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Source image not found for run {run_dir.name}")

    cleaned_mask = cv2.imread(str(output_dir / "06_cleaned_mask.png"), cv2.IMREAD_GRAYSCALE)
    if cleaned_mask is None:
        raise FileNotFoundError(f"Missing cleaned mask for run {run_dir.name}")

    cleaned_artery = cv2.imread(
        str(output_dir / "06_cleaned_artery_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    cleaned_vein = cv2.imread(
        str(output_dir / "06_cleaned_vein_mask.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if cleaned_artery is not None and cleaned_vein is not None:
        skeleton = skeletonize_mask(cleaned_artery > 0) | skeletonize_mask(cleaned_vein > 0)
    else:
        skeleton = skeletonize_mask(cleaned_mask > 0)
    skeleton_graph = Skeleton(skeleton)

    summary_path = output_dir / "08_full_skeleton_summary.csv"
    branches_df = pd.read_csv(summary_path).copy()
    branch_count = min(len(branches_df), skeleton_graph.n_paths)
    branches_df = branches_df.iloc[:branch_count].copy()
    branches_df["branch_id"] = np.arange(branch_count, dtype=int)

    paths_payload: list[dict] = []
    for branch_id in range(branch_count):
        coords = skeleton_graph.path_coordinates(branch_id)
        if coords.shape[0] < 2:
            continue
        centroid = coords.mean(axis=0)
        paths_payload.append(
            {
                "branchId": int(branch_id),
                "points": [[int(col), int(row)] for row, col in coords],
                "label": [int(round(centroid[1])), int(round(centroid[0]))],
            }
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read source image for run {run_dir.name}")

    image_height, image_width = image_bgr.shape[:2]

    return {
        "metadata": metadata,
        "run_dir": str(run_dir),
        "image_path": str(image_path),
        "image_url": image_to_data_url(image_path),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "branches_df": branches_df.to_dict(orient="records"),
        "paths_payload": paths_payload,
    }


def run_uploaded_analysis(
    uploaded_file,
    method: str,
    vessel_percentile: float,
    vessel_low_percentile: float,
    deep_threshold: float,
    deep_modality: str,
    deep_backend: str,
) -> str:
    run_id = slugify_name(uploaded_file.name)
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image_path = run_dir / uploaded_file.name
    image_path.write_bytes(uploaded_file.getvalue())

    output_csv = run_dir / "results.csv"
    output_dir = run_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for stale_file in [
        run_dir / "results.csv",
        run_dir / "metadata.json",
        run_dir / "logs.txt",
        run_dir / "manual_review_state.json",
        run_dir / "manual_vessels.csv",
    ]:
        if stale_file.exists():
            stale_file.unlink()

    log_buffer = io.StringIO()

    with st.spinner("Running segmentation and skeleton extraction..."):
        with redirect_stdout(log_buffer):
            run_pipeline(
                image_path=str(image_path),
                output_csv=str(output_csv),
                max_branches=30,
                output_dir=str(output_dir),
                method=method,
                vessel_percentile=float(vessel_percentile),
                vessel_low_percentile=float(vessel_low_percentile),
                deep_threshold=float(deep_threshold),
                deep_modality=deep_modality,
                deep_backend=deep_backend,
            )

    metadata = {
        "run_id": run_id,
        "image_name": uploaded_file.name,
        "method": method,
        "vessel_percentile": float(vessel_percentile),
        "vessel_low_percentile": float(vessel_low_percentile),
        "deep_threshold": float(deep_threshold),
        "deep_modality": deep_modality,
        "deep_backend": deep_backend,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "logs.txt").write_text(log_buffer.getvalue(), encoding="utf-8")
    load_review_bundle.clear()
    return run_id
