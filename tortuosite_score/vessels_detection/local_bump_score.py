from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tortuosite_score.vessels_detection.analyse import skeletonize_mask
from tortuosite_score.vessels_detection.skan_extra import skan_available
from tortuosite_score.vessels_detection.skeleton_graph import skeleton_graph_from_image
from tortuosite_score.vessels_detection.segments import (
    build_segment_map,
    ordered_points_for_segments,
    score_segments,
    segment_ref_sort_key,
)


DEFAULT_MIN_BRANCH_LENGTH = 50.0
DEFAULT_RESAMPLE_STEP = 4.0
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_CURVATURE_THRESHOLD = 0.035
DEFAULT_TAIL_LENGTH_FRACTION = 0.20
DEFAULT_GLOBAL_WEIGHT = 0.70
DEFAULT_TAIL_WEIGHT = 0.30
DEFAULT_MIN_SYSTEM_LENGTH = 250.0
DEFAULT_BRIDGE_TOLERANCE = 18.0
DEFAULT_BRIDGE_DIRECTION_COSINE = 0.35
DEFAULT_MIN_SAVED_VESSEL_LENGTH = 100.0
DISPLAY_SCALE = 1000.0
REFERENCE_FUNDUS_DIAMETER = 1024.0


@dataclass(frozen=True)
class LocalBumpSettings:
    min_branch_length: float = DEFAULT_MIN_BRANCH_LENGTH
    min_system_length: float = DEFAULT_MIN_SYSTEM_LENGTH
    resample_step: float = DEFAULT_RESAMPLE_STEP
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW
    curvature_threshold: float = DEFAULT_CURVATURE_THRESHOLD
    tail_length_fraction: float = DEFAULT_TAIL_LENGTH_FRACTION
    global_weight: float = DEFAULT_GLOBAL_WEIGHT
    tail_weight: float = DEFAULT_TAIL_WEIGHT
    bridge_tolerance: float = DEFAULT_BRIDGE_TOLERANCE
    bridge_direction_cosine: float = DEFAULT_BRIDGE_DIRECTION_COSINE
    min_saved_vessel_length: float = DEFAULT_MIN_SAVED_VESSEL_LENGTH
    filter_short_vessels: bool = True
    resample_curvature_squared: bool = True


def local_bump_metrics(
    points: list[list[float]] | np.ndarray,
    settings: LocalBumpSettings | None = None,
) -> dict[str, float]:
    settings = settings or LocalBumpSettings()
    path = _prepare_path(points)
    if len(path) < 2:
        return _empty_branch_metrics()

    branch_length = polyline_length(path)
    chord_length = float(np.linalg.norm(path[-1] - path[0]))
    if branch_length <= 0:
        return _empty_branch_metrics(branch_length=branch_length, chord_length=chord_length)

    resampled = resample_polyline(path, settings.resample_step)
    smoothed = smooth_polyline(resampled, settings.smoothing_window)
    angle_change = local_angle_change(smoothed)
    if angle_change.size == 0:
        return _empty_branch_metrics(branch_length=branch_length, chord_length=chord_length)

    meaningful = np.where(np.abs(angle_change) >= settings.curvature_threshold, angle_change, 0.0)
    local_bump_energy = float(np.mean(np.abs(meaningful)))
    oscillation_count = count_curvature_sign_changes(meaningful)
    oscillation_density = float(oscillation_count / branch_length * 100.0)
    branch_bump_score = float(local_bump_energy * math.sqrt(oscillation_density)) if oscillation_density > 0 else 0.0
    arc_chord_tortuosity = float(branch_length / chord_length) if chord_length > 0 else math.nan

    return {
        "branch_length": float(branch_length),
        "chord_length": chord_length,
        "arc_chord_tortuosity": arc_chord_tortuosity,
        "local_bump_energy": local_bump_energy,
        "oscillation_count": float(oscillation_count),
        "oscillation_density": oscillation_density,
        "branch_bump_score": branch_bump_score,
    }


def curvature_squared_metrics(
    points: list[list[float]] | np.ndarray,
    settings: LocalBumpSettings | None = None,
) -> dict[str, float]:
    settings = settings or LocalBumpSettings()
    path = _prepare_path(points)
    if len(path) < 2:
        return _empty_curvature_squared_metrics()

    branch_length = polyline_length(path)
    chord_length = float(np.linalg.norm(path[-1] - path[0]))
    if branch_length <= 0:
        return _empty_curvature_squared_metrics(branch_length=branch_length, chord_length=chord_length)

    if settings.resample_curvature_squared:
        path = resample_polyline(path, settings.resample_step)

    if len(path) < 3:
        return _empty_curvature_squared_metrics(branch_length=branch_length, chord_length=chord_length)

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    path = path[keep]
    s = s[keep]
    if len(path) < 3 or s[-1] <= 0:
        return _empty_curvature_squared_metrics(branch_length=branch_length, chord_length=chord_length)

    edge_order = 2 if len(path) >= 3 else 1
    dx = np.gradient(path[:, 0], s, edge_order=edge_order)
    dy = np.gradient(path[:, 1], s, edge_order=edge_order)
    ddx = np.gradient(dx, s, edge_order=edge_order)
    ddy = np.gradient(dy, s, edge_order=edge_order)
    speed_squared = dx * dx + dy * dy
    denominator = np.power(speed_squared, 1.5)
    curvature = np.divide(
        np.abs(dx * ddy - dy * ddx),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 1e-12,
    )
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    squared = curvature * curvature
    integral = float(np.trapezoid(squared, s))
    score = float(integral / branch_length) if branch_length > 0 else 0.0
    return {
        "curvature_squared_score": score,
        "curvature_squared_integral": integral,
        "mean_curvature": float(np.mean(curvature)),
        "max_curvature": float(np.max(curvature)),
    }


def tortuosity_density_metrics(
    points: list[list[float]] | np.ndarray,
    settings: LocalBumpSettings | None = None,
) -> dict[str, float | bool]:
    """Compute Grisan tortuosity density on one ordered vessel centerline.

    Resampling and smoothing are used only to locate stable curvature-sign
    changes. Arc lengths and chords in the formula are measured on the input
    geometry so that preprocessing does not alter the measured bends.
    """
    settings = settings or LocalBumpSettings()
    path = _prepare_path(points)
    if len(path) < 2:
        return _empty_tortuosity_density_metrics(valid=False)

    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    total_length = float(cumulative[-1])
    if total_length <= 0:
        return _empty_tortuosity_density_metrics(valid=False)

    resampled = resample_polyline(path, settings.resample_step)
    sample_distances = _resample_distances(total_length, settings.resample_step)
    if len(resampled) != len(sample_distances):
        return _empty_tortuosity_density_metrics(valid=False)
    smoothed = smooth_polyline(resampled, settings.smoothing_window)
    angle_change = local_angle_change(smoothed)
    if len(angle_change) != max(len(resampled) - 2, 0):
        return _empty_tortuosity_density_metrics(valid=False)
    meaningful = np.where(np.abs(angle_change) >= settings.curvature_threshold, angle_change, 0.0)

    nonzero_indices = np.flatnonzero(meaningful)
    boundaries: list[float] = []
    for previous, current in zip(nonzero_indices[:-1], nonzero_indices[1:]):
        if np.sign(meaningful[previous]) == np.sign(meaningful[current]):
            continue
        previous_distance = float(sample_distances[int(previous) + 1])
        current_distance = float(sample_distances[int(current) + 1])
        boundaries.append((previous_distance + current_distance) / 2.0)

    split_distances = np.asarray([0.0, *boundaries, total_length], dtype=float)
    segment_count = len(split_distances) - 1
    excess_sum = 0.0
    for start_distance, end_distance in zip(split_distances[:-1], split_distances[1:]):
        arc_length = float(end_distance - start_distance)
        start_point = _point_at_arc_distance(path, cumulative, float(start_distance))
        end_point = _point_at_arc_distance(path, cumulative, float(end_distance))
        chord_length = float(np.linalg.norm(end_point - start_point))
        if arc_length <= 0 or chord_length <= 1e-9:
            return _empty_tortuosity_density_metrics(
                segment_count=segment_count,
                inflection_count=len(boundaries),
                valid=False,
            )
        excess_sum += arc_length / chord_length - 1.0

    score = float((segment_count - 1) / total_length * excess_sum)
    return {
        "tortuosity_density_score": score,
        "constant_curvature_segment_count": float(segment_count),
        "inflection_count": float(len(boundaries)),
        "tortuosity_density_excess": float(excess_sum),
        "tortuosity_density_valid": True,
    }


def score_run(
    run_dir: str | Path,
    settings: LocalBumpSettings | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    from tortuosite_score.vessels_detection.scoring import scoring_config, score_run as score_run_with_method

    return score_run_with_method(run_dir, scoring_config("local_bump", settings or LocalBumpSettings()))


def score_branch_fragments(
    branches: pd.DataFrame,
    run_name: str = "",
    eye_number: int | None = None,
    settings: LocalBumpSettings | None = None,
) -> pd.DataFrame:
    settings = settings or LocalBumpSettings()
    scored_rows: list[dict[str, object]] = []
    for _, row in branches.iterrows():
        metrics = local_bump_metrics(row["path_points"], settings)
        scored_rows.append(
            {
                "run": run_name,
                "eye_number": eye_number,
                "branch_id": int(row["branch_id"]),
                "category": row.get("vascx_category", "unknown"),
                "eligible": bool(metrics["branch_length"] >= settings.min_branch_length),
                **metrics,
            }
        )
    return pd.DataFrame(scored_rows)


def score_runs(
    runs_root: str | Path,
    output_csv: str | Path | None = None,
    branch_output_csv: str | Path | None = None,
    system_output_csv: str | Path | None = None,
    settings: LocalBumpSettings | None = None,
    numbered_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tortuosite_score.vessels_detection.scoring import scoring_config, score_runs as score_runs_with_method

    return score_runs_with_method(
        runs_root,
        output_csv=output_csv,
        branch_output_csv=branch_output_csv,
        vessel_output_csv=system_output_csv,
        config=scoring_config("local_bump", settings or LocalBumpSettings()),
        numbered_only=numbered_only,
    )


def summarize_eye_score(
    system_scores: pd.DataFrame,
    settings: LocalBumpSettings | None = None,
) -> dict[str, object]:
    settings = settings or LocalBumpSettings()
    eligible = system_scores[system_scores["eligible"]].copy()
    result: dict[str, object] = {
        "eligible_system_count": int(len(eligible)),
        "eligible_branch_count": int(eligible["branch_count"].sum()) if "branch_count" in eligible else int(len(eligible)),
        "eligible_total_length": float(eligible["system_length"].sum()) if not eligible.empty else 0.0,
        "eye_tortuosity_score": math.nan,
        "all_vessels_score": math.nan,
        "all_vessels_tail_score": math.nan,
        "artery_score": math.nan,
        "vein_score": math.nan,
        "current_arc_chord_score": math.nan,
        "top_contributing_systems": "[]",
    }
    if eligible.empty:
        return result

    all_global = weighted_mean(eligible["system_bump_score"], eligible["system_length"])
    all_tail = tail_weighted_mean(
        eligible,
        value_column="system_bump_score",
        length_column="system_length",
        tail_length_fraction=settings.tail_length_fraction,
    )
    final_score = settings.global_weight * all_global + settings.tail_weight * all_tail

    result.update(
        {
            "eye_tortuosity_score": float(final_score * DISPLAY_SCALE),
            "all_vessels_score": float(all_global * DISPLAY_SCALE),
            "all_vessels_tail_score": float(all_tail * DISPLAY_SCALE),
            "artery_score": _category_score(eligible, "artere"),
            "vein_score": _category_score(eligible, "veine"),
            "current_arc_chord_score": weighted_mean(
                eligible["arc_chord_tortuosity"],
                eligible["system_length"],
            ),
            "top_contributing_systems": json.dumps(top_contributing_systems(eligible)),
        }
    )
    return result


def score_saved_vessels(
    run_dir: str | Path,
    branches: pd.DataFrame,
    settings: LocalBumpSettings | None = None,
) -> pd.DataFrame:
    from tortuosite_score.vessels_detection.scoring import scoring_config, score_saved_vessels_for_run

    return score_saved_vessels_for_run(run_dir, branches=branches, config=scoring_config("local_bump", settings or LocalBumpSettings()))


def summarize_saved_vessel_eye_score(
    vessel_scores: pd.DataFrame,
    settings: LocalBumpSettings | None = None,
) -> dict[str, object]:
    from tortuosite_score.vessels_detection.scoring import scoring_config, summarize_eye_score

    return summarize_eye_score(vessel_scores, config=scoring_config("local_bump", settings or LocalBumpSettings()))


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


def saved_vessel_metrics(
    run_dir: str | Path,
    settings: LocalBumpSettings | None = None,
) -> dict[str, object]:
    settings = settings or LocalBumpSettings()
    path = Path(run_dir) / "manual_vessels.csv"
    result: dict[str, object] = {
        "saved_vessel_count": 0,
        "saved_long_vessel_count": 0,
        "saved_long_top2_tortuosity": math.nan,
        "saved_long_weighted_tortuosity": math.nan,
        "saved_long_tail_fraction": math.nan,
        "saved_long_total_length": 0.0,
    }
    if not path.exists():
        return result
    try:
        vessels = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return result
    if vessels.empty or "tortuosity" not in vessels or "path_length" not in vessels:
        return result
    vessels = vessels.replace([np.inf, -np.inf], np.nan).dropna(subset=["tortuosity", "path_length"])
    vessels = vessels[vessels["path_length"].astype(float) > 0].copy()
    result["saved_vessel_count"] = int(len(vessels))
    if vessels.empty:
        return result
    long_vessels = vessels[vessels["path_length"].astype(float) >= settings.min_saved_vessel_length].copy()
    result["saved_long_vessel_count"] = int(len(long_vessels))
    if long_vessels.empty:
        return result
    tortuosity = long_vessels["tortuosity"].astype(float)
    length = long_vessels["path_length"].astype(float)
    result.update(
        {
            "saved_long_top2_tortuosity": float(tortuosity.nlargest(min(2, len(tortuosity))).mean()),
            "saved_long_weighted_tortuosity": float(np.average(tortuosity, weights=length)),
            "saved_long_tail_fraction": float(length[tortuosity >= 1.18].sum() / length.sum()),
            "saved_long_total_length": float(length.sum()),
        }
    )
    return result


def add_comparative_hybrid_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    score_raw = pd.to_numeric(summary_df.get("eye_tortuosity_score", pd.Series(dtype=float)), errors="coerce")
    score_component = minmax_series(score_raw)
    summary_df["saved_vessel_component_norm"] = score_component
    summary_df["comparative_hybrid_score"] = 100.0 * score_component
    return summary_df


def minmax_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=float)
    fill_value = float(numeric.min()) if pd.notna(numeric.min()) else 0.0
    numeric = numeric.fillna(fill_value)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum <= minimum:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)


def load_saved_run_branches(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    output_dir = run_dir / "output"
    summary_path = output_dir / "08_full_skeleton_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing skeleton summary for {run_dir}")

    summary = pd.read_csv(summary_path).copy()
    cleaned_mask = _read_mask(output_dir / "06_cleaned_mask.png")
    cleaned_artery = _read_optional_mask(output_dir / "06_cleaned_artery_mask.png")
    cleaned_vein = _read_optional_mask(output_dir / "06_cleaned_vein_mask.png")
    if cleaned_artery is not None and cleaned_vein is not None:
        skeleton = skeletonize_mask(cleaned_artery) | skeletonize_mask(cleaned_vein)
    else:
        skeleton = skeletonize_mask(cleaned_mask)

    skeleton_graph = _skeleton_graph(skeleton)
    branch_count = min(len(summary), skeleton_graph.n_paths)
    branches = summary.iloc[:branch_count].copy().reset_index(drop=True)
    branches["branch_id"] = np.arange(branch_count, dtype=int)
    branches["path_points"] = [[] for _ in range(branch_count)]
    branches["vascx_category"] = "unknown"
    branches["vascx_artery_pixels"] = 0
    branches["vascx_vein_pixels"] = 0
    branches.attrs["root_hint"] = disc_center_from_output_dir(output_dir)
    coordinate_scale, fundus_diameter = fundus_coordinate_scale(output_dir)
    branches.attrs["coordinate_scale"] = coordinate_scale
    branches.attrs["fundus_diameter_px"] = fundus_diameter
    branches.attrs["reference_fundus_diameter_px"] = REFERENCE_FUNDUS_DIAMETER

    path_points_by_signature: dict[
        tuple[tuple[int, int], tuple[int, int]],
        deque[list[list[int]]],
    ] = defaultdict(deque)
    for path_index in range(skeleton_graph.n_paths):
        coords = skeleton_graph.path_coordinates(path_index)
        if coords.shape[0] < 2:
            continue
        points = [[int(col), int(row)] for row, col in coords]
        signature = _path_endpoint_signature(points)
        if signature is not None:
            path_points_by_signature[signature].append(points)

    for branch_id in range(branch_count):
        row = branches.loc[branch_id]
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
        branches.at[branch_id, "path_points"] = matched_points
        if cleaned_artery is not None and cleaned_vein is not None:
            coords = np.array([[point[1], point[0]] for point in matched_points], dtype=int)
            rows = np.clip(coords[:, 0], 0, cleaned_artery.shape[0] - 1)
            cols = np.clip(coords[:, 1], 0, cleaned_artery.shape[1] - 1)
            artery_pixels = int(np.count_nonzero(cleaned_artery[rows, cols]))
            vein_pixels = int(np.count_nonzero(cleaned_vein[rows, cols]))
            branches.loc[branch_id, "vascx_artery_pixels"] = artery_pixels
            branches.loc[branch_id, "vascx_vein_pixels"] = vein_pixels
            branches.loc[branch_id, "vascx_category"] = _category_from_pixels(artery_pixels, vein_pixels)
    return branches


def fundus_coordinate_scale(output_dir: str | Path) -> tuple[float, float]:
    """Return a scale that expresses geometry in a common retinal image space.

    The equivalent diameter of the segmented fundus is used instead of the raw
    image width so rectangular crops and black borders do not change the unit.
    A scale of 1 is a safe fallback for legacy runs without a fundus mask.
    """

    fundus_mask = _read_optional_mask(Path(output_dir) / "02b_fundus_mask.png")
    if fundus_mask is None:
        return 1.0, math.nan
    area = int(np.count_nonzero(fundus_mask))
    if area <= 0:
        return 1.0, math.nan
    equivalent_diameter = float(2.0 * math.sqrt(area / math.pi))
    if not math.isfinite(equivalent_diameter) or equivalent_diameter <= 0:
        return 1.0, math.nan
    return float(REFERENCE_FUNDUS_DIAMETER / equivalent_diameter), equivalent_diameter


def disc_center_from_output_dir(output_dir: Path) -> tuple[float, float] | None:
    for file_name in ["04c_vascx_disc_mask.png", "04e_estimated_disc_mask.png"]:
        disc_mask = _read_optional_mask(output_dir / file_name)
        if disc_mask is None or not np.any(disc_mask):
            continue
        rows, cols = np.where(disc_mask)
        return float(cols.mean()), float(rows.mean())
    return None


def score_root_to_leaf_systems(
    branches: pd.DataFrame,
    settings: LocalBumpSettings | None = None,
) -> pd.DataFrame:
    settings = settings or LocalBumpSettings()
    graph = build_system_graph(branches, settings)
    system_rows: list[dict[str, object]] = []
    for system_id, path in enumerate(root_to_leaf_paths(graph)):
        path_points = points_for_node_edge_path(graph, path["node_path"], path["edge_ids"])
        metrics = local_bump_metrics(path_points, settings)
        system_length = metrics["branch_length"]
        category_mix = category_mix_for_edges(graph, path["edge_ids"])
        branch_ids = [
            graph["edges"][edge_id]["branch_id"]
            for edge_id in path["edge_ids"]
            if graph["edges"][edge_id].get("branch_id") is not None
        ]
        bridge_count = sum(1 for edge_id in path["edge_ids"] if graph["edges"][edge_id].get("is_bridge"))
        system_rows.append(
            {
                "system_id": int(system_id),
                "path_points": [[float(x), float(y)] for x, y in path_points],
                "branch_ids": json.dumps([int(branch_id) for branch_id in branch_ids]),
                "branch_count": int(len(set(branch_ids))),
                "bridge_count": int(bridge_count),
                "category": dominant_category(category_mix),
                "artere_pct": category_mix.get("artere", 0.0),
                "veine_pct": category_mix.get("veine", 0.0),
                "mixed_pct": category_mix.get("mixed", 0.0),
                "unknown_pct": category_mix.get("unknown", 0.0),
                "eligible": bool(system_length >= settings.min_system_length),
                "system_length": float(system_length),
                "system_bump_score": metrics["branch_bump_score"],
                **metrics,
            }
        )
    columns = [
        "system_id",
        "path_points",
        "branch_ids",
        "branch_count",
        "bridge_count",
        "category",
        "artere_pct",
        "veine_pct",
        "mixed_pct",
        "unknown_pct",
        "eligible",
        "system_length",
        "system_bump_score",
        "branch_length",
        "chord_length",
        "arc_chord_tortuosity",
        "local_bump_energy",
        "oscillation_count",
        "oscillation_density",
        "branch_bump_score",
    ]
    return pd.DataFrame(system_rows, columns=columns)


def build_system_graph(branches: pd.DataFrame, settings: LocalBumpSettings | None = None) -> dict[str, object]:
    settings = settings or LocalBumpSettings()
    nodes: list[dict[str, object]] = []
    node_lookup: dict[tuple[int, int], int] = {}
    edges: list[dict[str, object]] = []
    adjacency: dict[int, list[tuple[int, float, int]]] = defaultdict(list)

    def node_for_point(point: list[float] | tuple[float, float]) -> int:
        key = (int(round(float(point[0]))), int(round(float(point[1]))))
        if key not in node_lookup:
            node_lookup[key] = len(nodes)
            nodes.append({"point": [float(key[0]), float(key[1])]})
        return node_lookup[key]

    for _, row in branches.iterrows():
        points = normalize_path_points(row.get("path_points"))
        if len(points) < 2:
            continue
        start_node = node_for_point(points[0])
        end_node = node_for_point(points[-1])
        edge_id = len(edges)
        length = polyline_length(np.asarray(points, dtype=float))
        edge = {
            "edge_id": edge_id,
            "start_node": start_node,
            "end_node": end_node,
            "points": points,
            "length": float(length),
            "branch_id": int(row.get("branch_id", edge_id)),
            "category": str(row.get("vascx_category", "unknown")),
            "is_bridge": False,
        }
        edges.append(edge)
        adjacency[start_node].append((end_node, float(length), edge_id))
        adjacency[end_node].append((start_node, float(length), edge_id))

    add_endpoint_bridges(nodes, edges, adjacency, settings)
    return {"nodes": nodes, "edges": edges, "adjacency": adjacency, "root_hint": branches.attrs.get("root_hint")}


def add_endpoint_bridges(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    adjacency: dict[int, list[tuple[int, float, int]]],
    settings: LocalBumpSettings,
) -> None:
    endpoints = [node_id for node_id in range(len(nodes)) if len(adjacency.get(node_id, [])) == 1]
    used: set[int] = set()
    for idx, node_a in enumerate(endpoints):
        if node_a in used:
            continue
        best_node = None
        best_distance = float("inf")
        for node_b in endpoints[idx + 1 :]:
            if node_b in used:
                continue
            point_a = np.asarray(nodes[node_a]["point"], dtype=float)
            point_b = np.asarray(nodes[node_b]["point"], dtype=float)
            gap = float(np.linalg.norm(point_b - point_a))
            if gap <= 0 or gap > settings.bridge_tolerance or gap >= best_distance:
                continue
            if not endpoints_direction_compatible(node_a, node_b, nodes, edges, adjacency, settings):
                continue
            best_node = node_b
            best_distance = gap
        if best_node is None:
            continue
        used.update({node_a, best_node})
        point_a = nodes[node_a]["point"]
        point_b = nodes[best_node]["point"]
        edge_id = len(edges)
        edge = {
            "edge_id": edge_id,
            "start_node": node_a,
            "end_node": best_node,
            "points": [[float(point_a[0]), float(point_a[1])], [float(point_b[0]), float(point_b[1])]],
            "length": best_distance,
            "branch_id": None,
            "category": "bridge",
            "is_bridge": True,
        }
        edges.append(edge)
        adjacency[node_a].append((best_node, best_distance, edge_id))
        adjacency[best_node].append((node_a, best_distance, edge_id))


def endpoints_direction_compatible(
    node_a: int,
    node_b: int,
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    adjacency: dict[int, list[tuple[int, float, int]]],
    settings: LocalBumpSettings,
) -> bool:
    tangent_a = endpoint_tangent(node_a, edges[adjacency[node_a][0][2]])
    tangent_b = endpoint_tangent(node_b, edges[adjacency[node_b][0][2]])
    if tangent_a is None or tangent_b is None:
        return False
    point_a = np.asarray(nodes[node_a]["point"], dtype=float)
    point_b = np.asarray(nodes[node_b]["point"], dtype=float)
    direction_ab = normalized_vector(point_b - point_a)
    if direction_ab is None:
        return False
    direction_ba = -direction_ab
    return (
        float(np.dot(tangent_a, direction_ab)) >= settings.bridge_direction_cosine
        and float(np.dot(tangent_b, direction_ba)) >= settings.bridge_direction_cosine
    )


def endpoint_tangent(node_id: int, edge: dict[str, object]) -> np.ndarray | None:
    points = np.asarray(edge["points"], dtype=float)
    if len(points) < 2:
        return None
    if node_id == edge["start_node"]:
        vector = points[0] - points[min(4, len(points) - 1)]
    else:
        vector = points[-1] - points[max(0, len(points) - 5)]
    return normalized_vector(vector)


def root_to_leaf_paths(graph: dict[str, object]) -> list[dict[str, object]]:
    adjacency: dict[int, list[tuple[int, float, int]]] = graph["adjacency"]
    node_ids = set(adjacency)
    visited: set[int] = set()
    paths: list[dict[str, object]] = []
    for start in sorted(node_ids):
        if start in visited:
            continue
        component = connected_graph_component(start, adjacency)
        visited.update(component)
        if len(component) < 2:
            continue
        root = root_node_for_component(component, graph)
        leaves = [node_id for node_id in component if node_id != root and len(adjacency.get(node_id, [])) <= 1]
        if not leaves:
            leaves = [node_id for node_id in component if node_id != root]
        distances, previous = shortest_path_tree(root, adjacency)
        for leaf in sorted(leaves, key=lambda node_id: distances.get(node_id, 0.0), reverse=True):
            if leaf not in previous and leaf != root:
                continue
            node_path, edge_ids = path_to_node(root, leaf, previous)
            if edge_ids:
                paths.append(
                    {
                        "root": root,
                        "leaf": leaf,
                        "node_path": node_path,
                        "edge_ids": edge_ids,
                        "distance": distances.get(leaf, 0.0),
                    }
                )
    return paths


def connected_graph_component(start: int, adjacency: dict[int, list[tuple[int, float, int]]]) -> set[int]:
    stack = [start]
    component = {start}
    while stack:
        node = stack.pop()
        for neighbor, _, _ in adjacency.get(node, []):
            if neighbor not in component:
                component.add(neighbor)
                stack.append(neighbor)
    return component


def root_node_for_component(component: set[int], graph: dict[str, object]) -> int:
    nodes = graph["nodes"]
    adjacency = graph["adjacency"]
    root_hint = graph.get("root_hint")
    if isinstance(root_hint, tuple) and len(root_hint) == 2:
        hint = np.asarray(root_hint, dtype=float)
        return min(
            component,
            key=lambda node_id: float(np.linalg.norm(np.asarray(nodes[node_id]["point"], dtype=float) - hint)),
        )
    points = np.asarray([nodes[node_id]["point"] for node_id in component], dtype=float)
    centroid = points.mean(axis=0)
    high_degree = [node_id for node_id in component if len(adjacency.get(node_id, [])) > 1]
    candidates = high_degree or list(component)
    return min(candidates, key=lambda node_id: float(np.linalg.norm(np.asarray(nodes[node_id]["point"], dtype=float) - centroid)))


def shortest_path_tree(
    root: int,
    adjacency: dict[int, list[tuple[int, float, int]]],
) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
    from heapq import heappop, heappush

    distances = {root: 0.0}
    previous: dict[int, tuple[int, int]] = {}
    heap = [(0.0, root)]
    while heap:
        distance, node = heappop(heap)
        if distance > distances.get(node, float("inf")):
            continue
        for neighbor, length, edge_id in adjacency.get(node, []):
            next_distance = distance + float(length)
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                previous[neighbor] = (node, edge_id)
                heappush(heap, (next_distance, neighbor))
    return distances, previous


def path_to_node(root: int, leaf: int, previous: dict[int, tuple[int, int]]) -> tuple[list[int], list[int]]:
    edge_ids: list[int] = []
    node_ids: list[int] = [leaf]
    current = leaf
    while current != root:
        if current not in previous:
            return [], []
        parent, edge_id = previous[current]
        edge_ids.append(edge_id)
        node_ids.append(parent)
        current = parent
    return list(reversed(node_ids)), list(reversed(edge_ids))


def points_for_graph_path(graph: dict[str, object], edge_ids: list[int]) -> list[list[float]]:
    if not edge_ids:
        return []
    edges = graph["edges"]
    points: list[list[float]] = []
    current_end = None
    for edge_id in edge_ids:
        edge = edges[edge_id]
        edge_points = [list(point) for point in edge["points"]]
        if not points:
            points.extend(edge_points)
            current_end = edge["end_node"]
            continue
        if edge["start_node"] == current_end:
            oriented = edge_points
            current_end = edge["end_node"]
        elif edge["end_node"] == current_end:
            oriented = list(reversed(edge_points))
            current_end = edge["start_node"]
        else:
            previous = np.asarray(points[-1], dtype=float)
            start_distance = float(np.linalg.norm(np.asarray(edge_points[0], dtype=float) - previous))
            end_distance = float(np.linalg.norm(np.asarray(edge_points[-1], dtype=float) - previous))
            oriented = edge_points if start_distance <= end_distance else list(reversed(edge_points))
            current_end = edge["end_node"] if oriented is edge_points else edge["start_node"]
        points.extend(oriented[1:] if points and oriented else oriented)
    return points


def points_for_node_edge_path(graph: dict[str, object], node_path: list[int], edge_ids: list[int]) -> list[list[float]]:
    edges = graph["edges"]
    points: list[list[float]] = []
    for index, edge_id in enumerate(edge_ids):
        if index + 1 >= len(node_path):
            break
        start_node = node_path[index]
        end_node = node_path[index + 1]
        edge = edges[edge_id]
        edge_points = [list(point) for point in edge["points"]]
        if edge["start_node"] == start_node and edge["end_node"] == end_node:
            oriented = edge_points
        elif edge["end_node"] == start_node and edge["start_node"] == end_node:
            oriented = list(reversed(edge_points))
        else:
            oriented = edge_points
        points.extend(oriented if not points else oriented[1:])
    return points


def category_mix_for_edges(graph: dict[str, object], edge_ids: list[int]) -> dict[str, float]:
    totals = {"artere": 0.0, "veine": 0.0, "mixed": 0.0, "unknown": 0.0}
    total_length = 0.0
    for edge_id in edge_ids:
        edge = graph["edges"][edge_id]
        if edge.get("is_bridge"):
            continue
        category = str(edge.get("category", "unknown"))
        if category not in totals:
            category = "unknown"
        length = float(edge.get("length", 0.0))
        totals[category] += length
        total_length += length
    if total_length <= 0:
        return totals
    return {category: 100.0 * length / total_length for category, length in totals.items()}


def dominant_category(category_mix: dict[str, float]) -> str:
    if not category_mix:
        return "unknown"
    category, share = max(category_mix.items(), key=lambda item: item[1])
    return category if share > 0 else "unknown"


def normalize_path_points(points: object) -> list[list[float]]:
    if not isinstance(points, list):
        return []
    normalized: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        normalized.append([float(point[0]), float(point[1])])
    return normalized


def normalized_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return vector / norm


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    step = max(float(step), 1.0)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return points[[0]].copy()
    sample_distances = np.arange(0.0, total, step)
    if sample_distances.size == 0 or sample_distances[-1] < total:
        sample_distances = np.append(sample_distances, total)
    x = np.interp(sample_distances, cumulative, points[:, 0])
    y = np.interp(sample_distances, cumulative, points[:, 1])
    return np.column_stack([x, y])


def _resample_distances(total_length: float, step: float) -> np.ndarray:
    step = max(float(step), 1.0)
    sample_distances = np.arange(0.0, float(total_length), step)
    if sample_distances.size == 0 or sample_distances[-1] < total_length:
        sample_distances = np.append(sample_distances, float(total_length))
    return sample_distances


def _point_at_arc_distance(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    if index >= len(points) - 1:
        return points[-1].copy()
    segment_length = float(cumulative[index + 1] - cumulative[index])
    if segment_length <= 0:
        return points[index].copy()
    fraction = (distance - float(cumulative[index])) / segment_length
    return points[index] + fraction * (points[index + 1] - points[index])


def smooth_polyline(points: np.ndarray, window: int) -> np.ndarray:
    if len(points) < 3:
        return points.copy()
    window = int(window)
    if window < 3:
        return points.copy()
    if window % 2 == 0:
        window += 1
    if len(points) <= window:
        window = max(3, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window < 3:
        return points.copy()
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def local_angle_change(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([], dtype=float)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-6
    vectors = vectors[valid]
    if len(vectors) < 2:
        return np.array([], dtype=float)
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return np.diff(angles)


def count_curvature_sign_changes(angle_change: np.ndarray) -> int:
    signs = np.sign(angle_change)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = pd.DataFrame({"value": values, "weight": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid["weight"] > 0]
    if valid.empty:
        return math.nan
    return float(np.average(valid["value"].astype(float), weights=valid["weight"].astype(float)))


def tail_weighted_mean(
    data: pd.DataFrame,
    value_column: str,
    length_column: str,
    tail_length_fraction: float,
) -> float:
    if data.empty:
        return math.nan
    sorted_data = data.sort_values(value_column, ascending=False)
    total_length = float(sorted_data[length_column].sum())
    target_length = total_length * float(np.clip(tail_length_fraction, 0.0, 1.0))
    if target_length <= 0:
        return math.nan
    remaining = target_length
    weighted_sum = 0.0
    used_length = 0.0
    for _, row in sorted_data.iterrows():
        branch_length = float(row[length_column])
        if branch_length <= 0:
            continue
        take = min(branch_length, remaining)
        weighted_sum += float(row[value_column]) * take
        used_length += take
        remaining -= take
        if remaining <= 1e-9:
            break
    return float(weighted_sum / used_length) if used_length > 0 else math.nan


def top_contributing_branches(branch_scores: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    columns = ["branch_id", "category", "branch_bump_score", "branch_length", "oscillation_count"]
    available = [column for column in columns if column in branch_scores.columns]
    rows = branch_scores.sort_values("branch_bump_score", ascending=False).head(limit)
    return [
        {
            key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value)
            for key, value in row.items()
        }
        for row in rows[available].to_dict(orient="records")
    ]


def top_contributing_systems(system_scores: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    columns = ["system_id", "system_bump_score", "system_length", "branch_count", "bridge_count"]
    available = [column for column in columns if column in system_scores.columns]
    rows = system_scores.sort_values("system_bump_score", ascending=False).head(limit)
    return [
        {
            key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value)
            for key, value in row.items()
        }
        for row in rows[available].to_dict(orient="records")
    ]


def top_contributing_vessels(vessel_scores: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    columns = ["vessel_name", "vessel_bump_score", "vessel_length", "category", "bridge_count", "oscillation_count"]
    available = [column for column in columns if column in vessel_scores.columns]
    rows = vessel_scores.sort_values("vessel_bump_score", ascending=False).head(limit)
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


def _prepare_path(points: list[list[float]] | np.ndarray) -> np.ndarray:
    path = np.asarray(points, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    if len(path) <= 1:
        return path
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9])
    return path[keep]


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _empty_branch_metrics(branch_length: float = 0.0, chord_length: float = 0.0) -> dict[str, float]:
    return {
        "branch_length": float(branch_length),
        "chord_length": float(chord_length),
        "arc_chord_tortuosity": math.nan,
        "local_bump_energy": 0.0,
        "oscillation_count": 0.0,
        "oscillation_density": 0.0,
        "branch_bump_score": 0.0,
    }


def _empty_curvature_squared_metrics(branch_length: float = 0.0, chord_length: float = 0.0) -> dict[str, float]:
    del branch_length, chord_length
    return {
        "curvature_squared_score": 0.0,
        "curvature_squared_integral": 0.0,
        "mean_curvature": 0.0,
        "max_curvature": 0.0,
    }


def _empty_tortuosity_density_metrics(
    segment_count: int = 0,
    inflection_count: int = 0,
    valid: bool = False,
) -> dict[str, float | bool]:
    return {
        "tortuosity_density_score": math.nan,
        "constant_curvature_segment_count": float(segment_count),
        "inflection_count": float(inflection_count),
        "tortuosity_density_excess": math.nan,
        "tortuosity_density_valid": bool(valid),
    }


def _category_score(branches: pd.DataFrame, category: str) -> float:
    subset = branches[branches["category"] == category]
    if subset.empty:
        return math.nan
    value_column = "system_bump_score" if "system_bump_score" in subset else "branch_bump_score"
    length_column = "system_length" if "system_length" in subset else "branch_length"
    return float(weighted_mean(subset[value_column], subset[length_column]) * DISPLAY_SCALE)


def _saved_vessel_category_score(vessels: pd.DataFrame, category: str) -> float:
    subset = vessels[vessels["category"] == category]
    if subset.empty:
        return math.nan
    return float(weighted_mean(subset["vessel_bump_score"], subset["vessel_length"]) * DISPLAY_SCALE)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {path}")
    return mask > 0


def _read_optional_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 0


def _skeleton_graph(skeleton: np.ndarray):
    if skan_available():
        from skan import Skeleton

        return Skeleton(skeleton)
    return skeleton_graph_from_image(skeleton)


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


def _category_from_pixels(artery_pixels: int, vein_pixels: int) -> str:
    if artery_pixels > vein_pixels * 1.25:
        return "artere"
    if vein_pixels > artery_pixels * 1.25:
        return "veine"
    if artery_pixels + vein_pixels > 0:
        return "mixed"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute saved-vessel eye-level tortuosity from saved Streamlit runs.")
    parser.add_argument("runs_root", nargs="?", default="demo/streamlit_runs")
    parser.add_argument("--output", default="demo/local_bump_eye_scores.csv")
    parser.add_argument("--systems-output", "--vessels-output", dest="systems_output", default="demo/local_bump_vessel_scores.csv")
    parser.add_argument("--branches-output", default="demo/local_bump_branch_scores.csv")
    parser.add_argument("--include-all-runs", action="store_true")
    parser.add_argument(
        "--method",
        choices=["local_bump", "arc_chord", "curvature_squared", "tortuosity_density"],
        default="local_bump",
    )
    parser.add_argument("--min-branch-length", type=float, default=DEFAULT_MIN_BRANCH_LENGTH)
    parser.add_argument("--min-system-length", type=float, default=DEFAULT_MIN_SYSTEM_LENGTH)
    parser.add_argument("--resample-step", type=float, default=DEFAULT_RESAMPLE_STEP)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--curvature-threshold", type=float, default=DEFAULT_CURVATURE_THRESHOLD)
    parser.add_argument("--bridge-tolerance", type=float, default=DEFAULT_BRIDGE_TOLERANCE)
    parser.add_argument("--min-saved-vessel-length", type=float, default=DEFAULT_MIN_SAVED_VESSEL_LENGTH)
    parser.add_argument("--no-filter-short-vessels", action="store_true")
    parser.add_argument("--no-resample-curvature-squared", action="store_true")
    return parser.parse_args()


def main() -> None:
    from tortuosite_score.vessels_detection.scoring import scoring_config, score_runs as score_runs_with_method

    args = parse_args()
    settings = LocalBumpSettings(
        min_branch_length=args.min_branch_length,
        min_system_length=args.min_system_length,
        resample_step=args.resample_step,
        smoothing_window=args.smoothing_window,
        curvature_threshold=args.curvature_threshold,
        bridge_tolerance=args.bridge_tolerance,
        min_saved_vessel_length=args.min_saved_vessel_length,
        filter_short_vessels=not args.no_filter_short_vessels,
        resample_curvature_squared=not args.no_resample_curvature_squared,
    )
    config = scoring_config(args.method, settings)
    summary_df, _ = score_runs_with_method(
        args.runs_root,
        output_csv=args.output,
        branch_output_csv=args.branches_output,
        vessel_output_csv=args.systems_output,
        config=config,
        numbered_only=not args.include_all_runs,
    )
    if summary_df.empty:
        print("No saved Streamlit runs found.")
        return
    visible_columns = [
        "run",
        "eye_number",
        "eye_tortuosity_score",
        "comparative_hybrid_score",
        "all_vessels_score",
        "all_vessels_tail_score",
        "artery_score",
        "vein_score",
        "current_arc_chord_score",
        "eligible_vessel_count",
        "saved_vessel_count",
    ]
    print(summary_df[[column for column in visible_columns if column in summary_df.columns]].to_string(index=False))
    print(f"\nSaved eye scores to {args.output}")
    print(f"Saved saved-vessel scores to {args.systems_output}")
    print(f"Saved diagnostic branch scores to {args.branches_output}")


if __name__ == "__main__":
    main()
