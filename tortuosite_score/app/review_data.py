from __future__ import annotations

import base64
import io
import json
import shutil
from contextlib import redirect_stdout
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tortuosite_score.app.constants import RUNS_ROOT
from tortuosite_score.vessels_detection.analyse import skeletonize_mask
from tortuosite_score.vessels_detection.main import run_pipeline
from tortuosite_score.vessels_detection.skan_extra import skan_available
from tortuosite_score.vessels_detection.skeleton_graph import skeleton_graph_from_image


def _path_endpoint_signature(points: list[list[int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(points) < 2:
        return None
    start = tuple(int(value) for value in points[0])
    end = tuple(int(value) for value in points[-1])
    return tuple(sorted((start, end)))


def _row_endpoint_signature(row: pd.Series) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(
        sorted(
            (
                (int(row["image-coord-src-1"]), int(row["image-coord-src-0"])),
                (int(row["image-coord-dst-1"]), int(row["image-coord-dst-0"])),
            )
        )
    )


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
    if skan_available():
        from skan import Skeleton

        skeleton_graph = Skeleton(skeleton)
    else:
        skeleton_graph = skeleton_graph_from_image(skeleton)

    summary_path = output_dir / "08_full_skeleton_summary.csv"
    branches_df = pd.read_csv(summary_path).copy()
    branch_count = min(len(branches_df), skeleton_graph.n_paths)
    branches_df = branches_df.iloc[:branch_count].copy().reset_index(drop=True)
    branches_df["branch_id"] = np.arange(branch_count, dtype=int)
    branches_df["vascx_category"] = "unknown"
    branches_df["vascx_artery_pixels"] = 0
    branches_df["vascx_vein_pixels"] = 0
    branches_df["path_points"] = [[] for _ in range(branch_count)]

    disc_center_xy: tuple[float, float] | None = None
    for disc_file_name in ["04c_vascx_disc_mask.png", "04e_estimated_disc_mask.png"]:
        disc_mask = cv2.imread(str(output_dir / disc_file_name), cv2.IMREAD_GRAYSCALE)
        if disc_mask is None or not np.any(disc_mask > 0):
            continue
        disc_rows, disc_cols = np.where(disc_mask > 0)
        disc_center_xy = (float(disc_cols.mean()), float(disc_rows.mean()))
        break
    if disc_center_xy is not None:
        disc_x, disc_y = disc_center_xy
        branches_df["root-distance-src"] = np.hypot(
            branches_df["image-coord-src-1"].astype(float) - disc_x,
            branches_df["image-coord-src-0"].astype(float) - disc_y,
        )
        branches_df["root-distance-dst"] = np.hypot(
            branches_df["image-coord-dst-1"].astype(float) - disc_x,
            branches_df["image-coord-dst-0"].astype(float) - disc_y,
        )

    path_points_by_signature: dict[
        tuple[tuple[int, int], tuple[int, int]],
        deque[list[list[int]]],
    ] = defaultdict(deque)
    for path_index in range(skeleton_graph.n_paths):
        coords = skeleton_graph.path_coordinates(path_index)
        if coords.shape[0] < 2:
            continue
        path_points = [[int(col), int(row)] for row, col in coords]
        signature = _path_endpoint_signature(path_points)
        if signature is None:
            continue
        path_points_by_signature[signature].append(path_points)

    paths_payload: list[dict] = []
    for branch_id in range(branch_count):
        row = branches_df.loc[branch_id]
        signature = _row_endpoint_signature(row)
        matched_points = (
            path_points_by_signature[signature].popleft()
            if path_points_by_signature.get(signature)
            else None
        )
        if matched_points is None:
            matched_points = [
                [int(row["image-coord-src-1"]), int(row["image-coord-src-0"])],
                [int(row["image-coord-dst-1"]), int(row["image-coord-dst-0"])],
            ]
        coords = np.array([[point[1], point[0]] for point in matched_points], dtype=int)
        if cleaned_artery is not None and cleaned_vein is not None:
            rows = coords[:, 0].astype(int)
            cols = coords[:, 1].astype(int)
            artery_pixels = int(np.count_nonzero(cleaned_artery[rows, cols] > 0))
            vein_pixels = int(np.count_nonzero(cleaned_vein[rows, cols] > 0))
            branches_df.loc[branch_id, "vascx_artery_pixels"] = artery_pixels
            branches_df.loc[branch_id, "vascx_vein_pixels"] = vein_pixels
            if artery_pixels > vein_pixels * 1.25:
                branches_df.loc[branch_id, "vascx_category"] = "artere"
            elif vein_pixels > artery_pixels * 1.25:
                branches_df.loc[branch_id, "vascx_category"] = "veine"
            elif artery_pixels + vein_pixels > 0:
                branches_df.loc[branch_id, "vascx_category"] = "mixed"
        centroid = coords.mean(axis=0)
        branches_df.at[branch_id, "path_points"] = matched_points
        paths_payload.append(
            {
                "branchId": int(branch_id),
                "points": matched_points,
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
    vascx_av_size: int,
    vascx_use_contrast_enhancement: bool,
    vascx_min_object_size: int,
    vascx_closing_radius: int,
    vascx_auto_create_vessels: bool,
    vascx_auto_min_vessel_length: float,
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

    with st.spinner("Segmentation et extraction du squelette en cours..."):
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
                vascx_av_size=int(vascx_av_size),
                vascx_use_contrast_enhancement=bool(vascx_use_contrast_enhancement),
                vascx_min_object_size=int(vascx_min_object_size),
                vascx_closing_radius=int(vascx_closing_radius),
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
        "vascx_av_size": int(vascx_av_size),
        "vascx_use_contrast_enhancement": bool(vascx_use_contrast_enhancement),
        "vascx_min_object_size": int(vascx_min_object_size),
        "vascx_closing_radius": int(vascx_closing_radius),
        "vascx_auto_create_vessels": bool(vascx_auto_create_vessels),
        "vascx_auto_min_vessel_length": float(vascx_auto_min_vessel_length),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "logs.txt").write_text(log_buffer.getvalue(), encoding="utf-8")
    load_review_bundle.clear()
    return run_id
