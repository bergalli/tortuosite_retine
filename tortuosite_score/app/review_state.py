from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tortuosite_score.app.review_data import read_json
from tortuosite_score.vessels_detection.segments import (
    VesselSegment,
    build_segment_map,
    create_geometry_endpoint,
    manual_segment_ref,
    model_segment_ref,
    normalize_endpoint,
    parse_segment_ref,
    score_segments,
    segment_ref_sort_key,
    sorted_unique_refs,
    split_segment_refs,
    synthesize_segment_links,
)
from tortuosite_score.vessels_detection.local_bump_score import (
    LocalBumpSettings,
    build_system_graph,
    category_mix_for_edges,
    dominant_category,
    local_bump_metrics,
    points_for_node_edge_path,
    root_to_leaf_paths,
)

SCHEMA_VERSION = 2


def get_or_create_review_state(
    run_dir: Path,
    branches_df: pd.DataFrame | None = None,
    auto_create_vessels: bool = False,
    auto_min_vessel_length: float = 25.0,
) -> dict:
    del auto_create_vessels, auto_min_vessel_length
    state_key = f"review_state::{run_dir.name}"
    if state_key not in st.session_state:
        saved_path = run_dir / "manual_review_state.json"
        state = read_json(saved_path) if saved_path.exists() else {}
        legacy_ignored = bool(state) and state.get("schema_version") != SCHEMA_VERSION
        if legacy_ignored:
            state = {}
        review_state = {
            "schema_version": SCHEMA_VERSION,
            "legacy_state_ignored": legacy_ignored,
            "selected_segment_refs": sorted_unique_refs(state.get("selected_segment_refs", [])),
            "manual_segments": _normalize_manual_segments(state.get("manual_segments", {})),
            "vessels": _normalize_vessels(state.get("vessels", {})),
        }
        st.session_state[state_key] = review_state
        if saved_path.exists() and not legacy_ignored:
            persist_manual_review(run_dir, review_state, branches_df)
    return st.session_state[state_key]


def next_default_vessel_name(vessels: dict[str, dict], category: str) -> str:
    prefix = "artere" if category == "artere" else "veine"
    index = 1
    existing_names = set(vessels)
    while f"{prefix}_{index}" in existing_names:
        index += 1
    return f"{prefix}_{index}"


def segment_refs_for_vessel(vessel: dict) -> list[str]:
    return sorted_unique_refs(vessel.get("segment_refs", []))


def segment_refs_for_review_state(review_state: dict) -> list[str]:
    return sorted_unique_refs(review_state.get("selected_segment_refs", []))


def normalize_selection_refs(segment_refs: list[str]) -> list[str]:
    return sorted_unique_refs(segment_refs)


def push_selection_history(
    undo_stack: list[list[str]],
    redo_stack: list[list[str]],
    previous_refs: list[str],
    next_refs: list[str],
    limit: int,
) -> bool:
    previous_normalized = normalize_selection_refs(previous_refs)
    next_normalized = normalize_selection_refs(next_refs)
    if previous_normalized == next_normalized:
        return False
    undo_stack.append(previous_normalized)
    del undo_stack[:-limit]
    redo_stack.clear()
    return True


def undo_selection(
    undo_stack: list[list[str]],
    redo_stack: list[list[str]],
    current_refs: list[str],
) -> list[str]:
    current_normalized = normalize_selection_refs(current_refs)
    if not undo_stack:
        return current_normalized
    previous_refs = normalize_selection_refs(undo_stack.pop())
    redo_stack.append(current_normalized)
    return previous_refs


def redo_selection(
    undo_stack: list[list[str]],
    redo_stack: list[list[str]],
    current_refs: list[str],
) -> list[str]:
    current_normalized = normalize_selection_refs(current_refs)
    if not redo_stack:
        return current_normalized
    next_refs = normalize_selection_refs(redo_stack.pop())
    undo_stack.append(current_normalized)
    return next_refs


def next_manual_segment_id(review_state: dict) -> int:
    existing = [int(raw_id) for raw_id in review_state.get("manual_segments", {})]
    return max(existing, default=0) + 1


def upsert_manual_segment(
    review_state: dict,
    points: list[list[float]],
    manual_segment_id: int | None = None,
) -> str | None:
    if manual_segment_id is None:
        manual_segment_id = next_manual_segment_id(review_state)
    try:
        segment = VesselSegment.from_manual_points(manual_segment_id, points)
    except (TypeError, ValueError, IndexError):
        return None
    review_state.setdefault("manual_segments", {})[str(int(manual_segment_id))] = segment.to_manual_payload()
    return segment.ref


def remove_manual_segment(review_state: dict, manual_segment_id: int) -> None:
    segment_ref = manual_segment_ref(manual_segment_id)
    review_state.setdefault("manual_segments", {}).pop(str(int(manual_segment_id)), None)
    review_state["selected_segment_refs"] = [
        ref for ref in review_state.get("selected_segment_refs", []) if ref != segment_ref
    ]
    for vessel in review_state.get("vessels", {}).values():
        vessel["segment_refs"] = [
            ref for ref in vessel.get("segment_refs", []) if ref != segment_ref
        ]


def get_segment_geometry(
    branches_df: pd.DataFrame,
    manual_segments: dict[str, dict],
) -> dict[str, dict[str, object]]:
    return {
        segment_ref: segment.to_viewer_geometry()
        for segment_ref, segment in build_segment_map(branches_df, manual_segments).items()
    }


def synthesize_selection_links(
    branches_df: pd.DataFrame,
    manual_segments: dict[str, dict],
    selected_segment_refs: list[str],
) -> dict[str, object]:
    return synthesize_segment_links(
        build_segment_map(branches_df, manual_segments),
        selected_segment_refs,
    )


def build_vessel_payload(
    branches_df: pd.DataFrame,
    manual_segments: dict[str, dict],
    selected_segment_refs: list[str],
    vessel_category: str,
    start_endpoint: dict[str, object] | None,
    end_endpoint: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    segment_refs = sorted_unique_refs(selected_segment_refs)
    segment_map = build_segment_map(branches_df, manual_segments)
    resolution = synthesize_segment_links(segment_map, segment_refs)
    payload = {
        "category": vessel_category,
        "segment_refs": segment_refs,
        "synthetic_links": resolution["synthetic_links"],
        "start_endpoint": normalize_endpoint(start_endpoint),
        "end_endpoint": normalize_endpoint(end_endpoint),
    }
    return payload, resolution


def score_vessel(
    branches_df: pd.DataFrame,
    manual_segments: dict[str, dict],
    vessel: dict,
) -> dict[str, object]:
    return score_segments(
        build_segment_map(branches_df, manual_segments),
        segment_refs_for_vessel(vessel),
        synthetic_links=vessel.get("synthetic_links", []),
        start_endpoint=vessel.get("start_endpoint"),
        end_endpoint=vessel.get("end_endpoint"),
    )


def persist_manual_review(run_dir: Path, state: dict, branches_df: pd.DataFrame | None) -> None:
    persisted_state = {
        "schema_version": SCHEMA_VERSION,
        "selected_segment_refs": [],
        "manual_segments": state.get("manual_segments", {}),
        "vessels": state.get("vessels", {}),
    }
    (run_dir / "manual_review_state.json").write_text(
        json.dumps(persisted_state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    if branches_df is None:
        return

    rows: list[dict[str, object]] = []
    manual_segments = state.get("manual_segments", {})
    for vessel_name, vessel in state.get("vessels", {}).items():
        metrics = score_vessel(branches_df, manual_segments, vessel)
        model_ids, manual_ids = split_segment_refs(segment_refs_for_vessel(vessel))
        rows.append(
            {
                "vessel_name": vessel_name,
                "category": vessel.get("category", "artere"),
                "segment_refs": json.dumps(segment_refs_for_vessel(vessel)),
                "model_segment_ids": json.dumps(model_ids),
                "manual_segment_ids": json.dumps(manual_ids),
                "model_segment_count": metrics["model_segment_count"],
                "manual_segment_count": metrics["manual_segment_count"],
                "segment_count": metrics["segment_count"],
                "component_count": metrics["component_count"],
                "bridge_branch_count": metrics["bridge_branch_count"],
                "resolved_component_count": metrics["resolved_component_count"],
                "bridge_success": metrics["bridge_success"],
                "start_endpoint": json.dumps(metrics["start_endpoint"]),
                "end_endpoint": json.dumps(metrics["end_endpoint"]),
                "path_length": metrics["length"],
                "chord_length": metrics["chord"],
                "tortuosity": metrics["tortuosity"],
            }
        )

    columns = [
        "vessel_name",
        "category",
        "segment_refs",
        "model_segment_ids",
        "manual_segment_ids",
        "model_segment_count",
        "manual_segment_count",
        "segment_count",
        "component_count",
        "bridge_branch_count",
        "resolved_component_count",
        "bridge_success",
        "start_endpoint",
        "end_endpoint",
        "path_length",
        "chord_length",
        "tortuosity",
    ]
    manual_df = pd.DataFrame(rows, columns=columns)
    if not manual_df.empty:
        manual_df = manual_df.sort_values("vessel_name")
    manual_df.to_csv(run_dir / "manual_vessels.csv", index=False)


def build_selection_table(
    branches_df: pd.DataFrame,
    manual_segments: dict[str, dict],
    selected_segment_refs: list[str],
) -> pd.DataFrame:
    geometry = get_segment_geometry(branches_df, manual_segments)
    rows: list[dict[str, object]] = []
    for segment_ref in sorted_unique_refs(selected_segment_refs):
        segment = geometry.get(segment_ref)
        if segment is None:
            continue
        rows.append(
            {
                "Segment": segment_ref,
                "Source": segment["source"],
                "Longueur": segment["path_length"],
                "Corde": segment["chord_length"],
                "Tortuosite": segment["tortuosity"],
                "Type": segment["branch_type"],
                "Categorie": segment["vascx_category"],
            }
        )
    return pd.DataFrame(rows)


def build_vessel_scores_table(review_state: dict, branches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manual_segments = review_state.get("manual_segments", {})
    for vessel_name, vessel in sorted(review_state.get("vessels", {}).items()):
        metrics = score_vessel(branches_df, manual_segments, vessel)
        rows.append(
            {
                "Vaisseau": vessel_name,
                "Categorie": vessel.get("category", "artere"),
                "Debut": _endpoint_caption(metrics["start_endpoint"]),
                "Fin": _endpoint_caption(metrics["end_endpoint"]),
                "Segments modele": metrics["model_segment_count"],
                "Segments manuels": metrics["manual_segment_count"],
                "Segments": metrics["segment_count"],
                "Composantes": metrics["component_count"],
                "Ponts automatiques": metrics["bridge_branch_count"],
                "Statut du pont": "connecte" if metrics["bridge_success"] else "partiel",
                "Longueur du trajet": metrics["length"],
                "Corde": metrics["chord"],
                "Tortuosite": metrics["tortuosity"],
            }
        )
    return pd.DataFrame(rows)


def build_auto_vascx_vessels(
    branches_df: pd.DataFrame,
    min_total_length: float = 25.0,
) -> dict[str, dict]:
    if "vascx_category" not in branches_df.columns:
        return {}
    vessels: dict[str, dict] = {}
    for category in ["artere", "veine"]:
        rows = branches_df[branches_df["vascx_category"] == category]
        for index, (_, row) in enumerate(rows.iterrows(), start=1):
            if float(row.get("branch-distance", 0.0)) < min_total_length:
                continue
            segment_ref = model_segment_ref(int(row["branch_id"]))
            points = _points_from_row(row)
            vessels[f"{category}_vascx_{index}"] = {
                "category": category,
                "segment_refs": [segment_ref],
                "synthetic_links": [],
                "start_endpoint": create_geometry_endpoint(points[0], segment_ref, 0.0),
                "end_endpoint": create_geometry_endpoint(points[-1], segment_ref, float(row["branch-distance"])),
            }
    return vessels


def replace_auto_completed_vessels(
    review_state: dict,
    branches_df: pd.DataFrame,
    settings: LocalBumpSettings | None = None,
    prefix: str = "auto_vascx",
) -> int:
    settings = settings or LocalBumpSettings()
    review_state.setdefault("vessels", {})
    review_state["vessels"] = {
        vessel_name: vessel
        for vessel_name, vessel in review_state["vessels"].items()
        if not vessel_name.startswith(f"{prefix}_")
    }
    generated = build_auto_completed_vessels(branches_df, settings=settings, prefix=prefix)
    review_state["vessels"].update(generated)
    return len(generated)


def build_auto_completed_vessels(
    branches_df: pd.DataFrame,
    settings: LocalBumpSettings | None = None,
    prefix: str = "auto_vascx",
) -> dict[str, dict]:
    settings = settings or LocalBumpSettings()
    graph = build_system_graph(branches_df, settings)
    vessels: dict[str, dict] = {}
    index = 1
    for path in root_to_leaf_paths(graph):
        path_points = points_for_node_edge_path(graph, path["node_path"], path["edge_ids"])
        metrics = local_bump_metrics(path_points, settings)
        if metrics["branch_length"] < settings.min_saved_vessel_length:
            continue
        branch_ids = [
            graph["edges"][edge_id]["branch_id"]
            for edge_id in path["edge_ids"]
            if graph["edges"][edge_id].get("branch_id") is not None
        ]
        segment_refs = sorted_unique_refs([model_segment_ref(int(branch_id)) for branch_id in branch_ids])
        if not segment_refs:
            continue
        synthetic_links = [
            {
                "points": graph["edges"][edge_id]["points"],
                "length": graph["edges"][edge_id]["length"],
                "display_length": graph["edges"][edge_id]["length"],
            }
            for edge_id in path["edge_ids"]
            if graph["edges"][edge_id].get("is_bridge")
        ]
        category = dominant_category(category_mix_for_edges(graph, path["edge_ids"]))
        vessels[f"{prefix}_{index}"] = {
            "category": "veine" if category == "veine" else "artere",
            "segment_refs": segment_refs,
            "synthetic_links": synthetic_links,
            "start_endpoint": create_geometry_endpoint(path_points[0]) if path_points else None,
            "end_endpoint": create_geometry_endpoint(path_points[-1]) if path_points else None,
            "source": "auto_vascx",
        }
        index += 1
    return vessels


def _normalize_vessels(vessels: dict[str, dict]) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for vessel_name, vessel in vessels.items():
        if not isinstance(vessel, dict):
            continue
        segment_refs = sorted_unique_refs(vessel.get("segment_refs", []))
        normalized[vessel_name] = {
            "category": vessel.get("category", "artere"),
            "segment_refs": segment_refs,
            "synthetic_links": list(vessel.get("synthetic_links", [])),
            "start_endpoint": normalize_endpoint(vessel.get("start_endpoint")),
            "end_endpoint": normalize_endpoint(vessel.get("end_endpoint")),
        }
    return normalized


def _normalize_manual_segments(manual_segments: dict | None) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    if not isinstance(manual_segments, dict):
        return normalized
    for raw_id, payload in manual_segments.items():
        if not isinstance(payload, dict):
            continue
        try:
            segment = VesselSegment.from_manual_payload(int(raw_id), payload)
        except (TypeError, ValueError, IndexError):
            continue
        normalized[str(segment.segment_id)] = segment.to_manual_payload()
    return normalized


def _endpoint_caption(endpoint: dict[str, object] | None) -> str:
    endpoint = normalize_endpoint(endpoint)
    if endpoint is None:
        return "Aucun"
    x, y = endpoint["point"]
    return f"({float(x):.1f}, {float(y):.1f})"


def _points_from_row(row: pd.Series) -> list[list[float]]:
    points = row.get("path_points")
    if isinstance(points, list) and len(points) >= 2:
        return [[float(point[0]), float(point[1])] for point in points]
    return [
        [float(row["image-coord-src-1"]), float(row["image-coord-src-0"])],
        [float(row["image-coord-dst-1"]), float(row["image-coord-dst-0"])],
    ]


# Backward-compatible aliases used by older import paths in the app.
branch_ref_sort_key = segment_ref_sort_key
branch_refs_for_vessel = segment_refs_for_vessel
branch_refs_for_review_state = segment_refs_for_review_state
manual_branch_ref = manual_segment_ref
model_branch_ref = model_segment_ref
parse_branch_ref = parse_segment_ref
remove_manual_branch = remove_manual_segment
split_branch_refs = split_segment_refs
upsert_manual_branch = upsert_manual_segment
