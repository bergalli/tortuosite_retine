from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tortuosite_score.app.review_data import read_json


def get_or_create_review_state(
    run_dir: Path,
    branches_df: pd.DataFrame | None = None,
    auto_create_vessels: bool = False,
    auto_min_vessel_length: float = 25.0,
) -> dict:
    state_key = f"review_state::{run_dir.name}"
    if state_key not in st.session_state:
        saved_path = run_dir / "manual_review_state.json"
        state = read_json(saved_path) if saved_path.exists() else {}
        vessels = {
            vessel_name: {
                "category": vessel.get("category", "artere"),
                "branch_ids": vessel.get("branch_ids", []),
                "synthetic_links": vessel.get("synthetic_links", []),
                "start_node_id": vessel.get("start_node_id"),
                "end_node_id": vessel.get("end_node_id"),
            }
            for vessel_name, vessel in state.get("vessels", {}).items()
        }
        st.session_state[state_key] = {
            "selected_branch_ids": [],
            "vessels": vessels,
        }
        if vessels and not saved_path.exists():
            persist_manual_review(run_dir, st.session_state[state_key], branches_df)
    return st.session_state[state_key]


def next_default_vessel_name(vessels: dict[str, dict], category: str) -> str:
    prefix = "artere" if category == "artere" else "veine"
    index = 1
    existing_names = set(vessels)
    while f"{prefix}_{index}" in existing_names:
        index += 1
    return f"{prefix}_{index}"


def _build_graph(
    branches_df: pd.DataFrame,
    branch_ids: list[int] | None = None,
) -> tuple[dict[int, list[tuple[int, float, int]]], dict[int, tuple[float, float]]]:
    if branch_ids is None:
        graph_df = branches_df
    else:
        graph_df = branches_df[branches_df["branch_id"].isin(branch_ids)].copy()

    adjacency: dict[int, list[tuple[int, float, int]]] = {}
    node_positions: dict[int, tuple[float, float]] = {}
    for _, row in graph_df.iterrows():
        src = int(row["node-id-src"])
        dst = int(row["node-id-dst"])
        length = float(row["branch-distance"])
        branch_id = int(row["branch_id"])
        adjacency.setdefault(src, []).append((dst, length, branch_id))
        adjacency.setdefault(dst, []).append((src, length, branch_id))
        node_positions[src] = (
            float(row["image-coord-src-1"]),
            float(row["image-coord-src-0"]),
        )
        node_positions[dst] = (
            float(row["image-coord-dst-1"]),
            float(row["image-coord-dst-0"]),
        )
    return adjacency, node_positions


def _connected_components(adjacency: dict[int, list[tuple[int, float, int]]]) -> list[set[int]]:
    unvisited = set(adjacency)
    components: list[set[int]] = []
    while unvisited:
        start = unvisited.pop()
        queue = [start]
        component = {start}
        while queue:
            current = queue.pop()
            for neighbor, _, _ in adjacency[current]:
                if neighbor not in component:
                    component.add(neighbor)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _branch_ids_for_component(
    component: set[int],
    adjacency: dict[int, list[tuple[int, float, int]]],
) -> list[int]:
    branch_ids = set()
    for node_id in component:
        for neighbor, _, branch_id in adjacency[node_id]:
            if neighbor in component and branch_id >= 0:
                branch_ids.add(int(branch_id))
    return sorted(branch_ids)


def _node_root_distances(branches_df: pd.DataFrame) -> dict[int, float]:
    if "root-distance-src" not in branches_df.columns or "root-distance-dst" not in branches_df.columns:
        return {}

    distances: dict[int, float] = {}
    for _, row in branches_df.iterrows():
        src = int(row["node-id-src"])
        dst = int(row["node-id-dst"])
        distances[src] = min(
            distances.get(src, float("inf")),
            float(row["root-distance-src"]),
        )
        distances[dst] = min(
            distances.get(dst, float("inf")),
            float(row["root-distance-dst"]),
        )
    return distances


def _root_node_for_component(
    component: set[int],
    adjacency: dict[int, list[tuple[int, float, int]]],
    root_distances: dict[int, float],
) -> int:
    if root_distances:
        return min(component, key=lambda node_id: root_distances.get(node_id, float("inf")))
    return max(component, key=lambda node_id: (len(adjacency[node_id]), node_id))


def _shortest_branch_paths_from_root(
    root_node_id: int,
    component: set[int],
    adjacency: dict[int, list[tuple[int, float, int]]],
) -> dict[int, tuple[float, list[int]]]:
    distances = {root_node_id: 0.0}
    branch_paths: dict[int, list[int]] = {root_node_id: []}
    visited: set[int] = set()

    while len(visited) < len(component):
        available = [node_id for node_id in component if node_id not in visited]
        if not available:
            break
        current = min(available, key=lambda node_id: distances.get(node_id, float("inf")))
        if distances.get(current, float("inf")) == float("inf"):
            break
        visited.add(current)
        for neighbor, length, branch_id in adjacency[current]:
            if neighbor not in component:
                continue
            candidate = distances[current] + float(length)
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                branch_paths[neighbor] = branch_paths[current] + [int(branch_id)]

    return {
        node_id: (distances[node_id], branch_paths[node_id])
        for node_id in distances
        if node_id in branch_paths
    }


def build_auto_vascx_vessels(
    branches_df: pd.DataFrame,
    min_total_length: float = 25.0,
) -> dict[str, dict]:
    if "vascx_category" not in branches_df.columns:
        return {}

    vessels: dict[str, dict] = {}
    for category in ["artere", "veine"]:
        category_branch_ids = (
            branches_df.loc[branches_df["vascx_category"] == category, "branch_id"]
            .astype(int)
            .tolist()
        )
        if not category_branch_ids:
            continue

        adjacency, _ = _build_graph(branches_df, category_branch_ids)
        root_distances = _node_root_distances(branches_df)
        components = _connected_components(adjacency)
        path_rows: list[tuple[float, list[int]]] = []
        for component in components:
            if not component:
                continue
            root_node_id = _root_node_for_component(component, adjacency, root_distances)
            paths = _shortest_branch_paths_from_root(root_node_id, component, adjacency)
            leaves = [
                node_id
                for node_id in component
                if node_id != root_node_id and len(adjacency[node_id]) <= 1
            ]
            if not leaves:
                leaves = [
                    node_id
                    for node_id, (_, branch_ids) in paths.items()
                    if node_id != root_node_id and branch_ids
                ]
            for leaf_node_id in leaves:
                if leaf_node_id not in paths:
                    continue
                total_length, branch_ids = paths[leaf_node_id]
                branch_ids = [branch_id for branch_id in branch_ids if branch_id >= 0]
                if branch_ids and total_length >= float(min_total_length):
                    path_rows.append((float(total_length), sorted(dict.fromkeys(branch_ids))))

        deduplicated_path_rows: dict[tuple[int, ...], float] = {}
        for total_length, branch_ids in path_rows:
            path_key = tuple(branch_ids)
            deduplicated_path_rows[path_key] = max(
                total_length,
                deduplicated_path_rows.get(path_key, 0.0),
            )

        sorted_paths = sorted(
            ((length, list(branch_ids)) for branch_ids, length in deduplicated_path_rows.items()),
            reverse=True,
            key=lambda item: item[0],
        )
        for index, (_, branch_ids) in enumerate(sorted_paths, start=1):
            vessels[f"{category}_vascx_{index}"] = {
                "category": category,
                "branch_ids": branch_ids,
                "synthetic_links": [],
            }

    return vessels


def resolve_vessel_branch_ids(
    branches_df: pd.DataFrame,
    branch_ids: list[int],
) -> dict[str, object]:
    resolved_branch_ids = sorted({int(branch_id) for branch_id in branch_ids})
    return {
        "branch_ids": resolved_branch_ids,
        "bridge_branch_count": 0,
        "resolved_component_count": 0,
        "bridge_success": True,
        "synthetic_links": [],
    }


def has_branching_nodes(branches_df: pd.DataFrame, branch_ids: list[int]) -> bool:
    adjacency, _ = _build_graph(branches_df, branch_ids)
    return any(len(edges) > 2 for edges in adjacency.values())


def build_node_options(branches_df: pd.DataFrame, branch_ids: list[int]) -> list[dict[str, object]]:
    adjacency, node_positions = _build_graph(branches_df, branch_ids)
    root_distances = _node_root_distances(branches_df)
    options: list[dict[str, object]] = []
    for node_id, position in node_positions.items():
        x, y = position
        degree = len(adjacency.get(node_id, []))
        options.append(
            {
                "node_id": int(node_id),
                "x": float(x),
                "y": float(y),
                "degree": int(degree),
                "root_distance": float(root_distances.get(node_id, np.nan)),
            }
        )
    return sorted(
        options,
        key=lambda option: (
            np.inf
            if np.isnan(float(option["root_distance"]))
            else float(option["root_distance"]),
            int(option["degree"]) > 1,
            int(option["node_id"]),
        ),
    )


def _candidate_nodes_for_component(
    component: set[int],
    adjacency: dict[int, list[tuple[int, float, int]]],
) -> list[int]:
    endpoints = [node_id for node_id in component if len(adjacency[node_id]) <= 1]
    return endpoints if endpoints else list(component)


def _branch_points(row: pd.Series) -> np.ndarray:
    points = row.get("path_points")
    if isinstance(points, list) and len(points) >= 2:
        return np.array(points, dtype=float)
    return np.array(
        [
            [float(row["image-coord-src-1"]), float(row["image-coord-src-0"])],
            [float(row["image-coord-dst-1"]), float(row["image-coord-dst-0"])],
        ],
        dtype=float,
    )


def _nearest_point_on_polyline(
    point_xy: tuple[float, float],
    polyline: np.ndarray,
) -> dict[str, object]:
    point = np.array(point_xy, dtype=float)
    best_distance = float("inf")
    best_point = polyline[0]
    best_distance_from_start = 0.0
    traversed_distance = 0.0

    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length == 0:
            continue
        fraction = float(np.clip(np.dot(point - start, segment) / (segment_length**2), 0.0, 1.0))
        projected = start + fraction * segment
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_point = projected
            best_distance_from_start = traversed_distance + fraction * segment_length
        traversed_distance += segment_length

    return {
        "point": [float(best_point[0]), float(best_point[1])],
        "distance": best_distance,
        "distance_from_start": best_distance_from_start,
        "total_length": traversed_distance,
    }


def _nearest_component_target(
    branches_df: pd.DataFrame,
    adjacency: dict[int, list[tuple[int, float, int]]],
    component: set[int],
    point_xy: tuple[float, float],
) -> dict[str, object] | None:
    component_branch_ids = _branch_ids_for_component(component, adjacency)
    branch_rows = branches_df.set_index("branch_id", drop=False)
    best_target: dict[str, object] | None = None

    for branch_id in component_branch_ids:
        if branch_id not in branch_rows.index:
            continue
        row = branch_rows.loc[branch_id]
        polyline = _branch_points(row)
        if len(polyline) < 2:
            continue
        nearest = _nearest_point_on_polyline(point_xy, polyline)
        distance_to_src = float(nearest["distance_from_start"])
        distance_to_dst = float(nearest["total_length"]) - distance_to_src
        if distance_to_src <= distance_to_dst:
            graph_node_id = int(row["node-id-src"])
            graph_extra_distance = distance_to_src
        else:
            graph_node_id = int(row["node-id-dst"])
            graph_extra_distance = distance_to_dst
        visual_distance = float(nearest["distance"])
        graph_distance = visual_distance + graph_extra_distance
        candidate = {
            "branch_id": int(branch_id),
            "graph_node_id": graph_node_id,
            "point": nearest["point"],
            "visual_distance": visual_distance,
            "graph_distance": graph_distance,
        }
        if best_target is None or (
            visual_distance,
            graph_distance,
        ) < (
            float(best_target["visual_distance"]),
            float(best_target["graph_distance"]),
        ):
            best_target = candidate

    return best_target


def synthesize_missing_links(
    branches_df: pd.DataFrame,
    branch_ids: list[int],
) -> dict[str, object]:
    selected_branch_ids = sorted({int(branch_id) for branch_id in branch_ids})
    if not selected_branch_ids:
        return {
            "branch_ids": [],
            "synthetic_links": [],
            "component_count": 0,
            "resolved_component_count": 0,
            "bridge_success": True,
        }

    adjacency, node_positions = _build_graph(branches_df, selected_branch_ids)
    if not adjacency:
        return {
            "branch_ids": selected_branch_ids,
            "synthetic_links": [],
            "component_count": 0,
            "resolved_component_count": 0,
            "bridge_success": True,
        }

    components = _connected_components(adjacency)
    if len(components) <= 1:
        return {
            "branch_ids": selected_branch_ids,
            "synthetic_links": [],
            "component_count": 1,
            "resolved_component_count": 1,
            "bridge_success": True,
        }

    working_components = [set(component) for component in components]
    synthetic_links: list[dict[str, object]] = []

    while len(working_components) > 1:
        best_pair: tuple[int, int] | None = None
        best_link: dict[str, object] | None = None

        for idx, component_a in enumerate(working_components):
            candidates_a = _candidate_nodes_for_component(component_a, adjacency)
            for jdx, component_b in enumerate(working_components[idx + 1 :], start=idx + 1):
                candidates_b = _candidate_nodes_for_component(component_b, adjacency)
                for node_a in candidates_a:
                    ax, ay = node_positions[node_a]
                    target = _nearest_component_target(
                        branches_df,
                        adjacency,
                        component_b,
                        (ax, ay),
                    )
                    if target is not None:
                        candidate = {
                            "src_node_id": int(node_a),
                            "dst_node_id": int(target["graph_node_id"]),
                            "points": [[float(ax), float(ay)], target["point"]],
                            "length": float(target["graph_distance"]),
                            "display_length": float(target["visual_distance"]),
                            "target_branch_id": int(target["branch_id"]),
                        }
                        if best_link is None or (
                            float(candidate["display_length"]),
                            float(candidate["length"]),
                        ) < (
                            float(best_link["display_length"]),
                            float(best_link["length"]),
                        ):
                            best_pair = (idx, jdx)
                            best_link = candidate
                for node_b in candidates_b:
                    bx, by = node_positions[node_b]
                    target = _nearest_component_target(
                        branches_df,
                        adjacency,
                        component_a,
                        (bx, by),
                    )
                    if target is not None:
                        candidate = {
                            "src_node_id": int(node_b),
                            "dst_node_id": int(target["graph_node_id"]),
                            "points": [[float(bx), float(by)], target["point"]],
                            "length": float(target["graph_distance"]),
                            "display_length": float(target["visual_distance"]),
                            "target_branch_id": int(target["branch_id"]),
                        }
                        if best_link is None or (
                            float(candidate["display_length"]),
                            float(candidate["length"]),
                        ) < (
                            float(best_link["display_length"]),
                            float(best_link["length"]),
                        ):
                            best_pair = (idx, jdx)
                            best_link = candidate

        if best_pair is None or best_link is None:
            return {
                "branch_ids": selected_branch_ids,
                "synthetic_links": synthetic_links,
                "component_count": len(components),
                "resolved_component_count": len(working_components),
                "bridge_success": False,
            }

        synthetic_links.append(best_link)
        src_node_id = int(best_link["src_node_id"])
        dst_node_id = int(best_link["dst_node_id"])
        link_length = float(best_link["length"])
        adjacency.setdefault(src_node_id, []).append((dst_node_id, link_length, -1))
        adjacency.setdefault(dst_node_id, []).append((src_node_id, link_length, -1))

        idx, jdx = best_pair
        merged = working_components[idx] | working_components[jdx]
        working_components[idx] = merged
        del working_components[jdx]

    return {
        "branch_ids": selected_branch_ids,
        "synthetic_links": synthetic_links,
        "component_count": len(components),
        "resolved_component_count": 1,
        "bridge_success": True,
    }


def score_vessel(
    branches_df: pd.DataFrame,
    branch_ids: list[int],
    synthetic_links: list[dict[str, object]] | None = None,
    start_node_id: int | None = None,
    end_node_id: int | None = None,
) -> dict[str, object]:
    selected = branches_df[branches_df["branch_id"].isin(branch_ids)].copy()
    if selected.empty:
        return {
            "branch_count": 0,
            "component_count": 0,
            "bridge_branch_count": 0,
            "resolved_component_count": 0,
            "bridge_success": True,
            "length": np.nan,
            "chord": np.nan,
            "tortuosity": np.nan,
            "start_node_id": np.nan,
            "end_node_id": np.nan,
        }

    initial_adjacency, _ = _build_graph(branches_df, branch_ids)
    initial_components = _connected_components(initial_adjacency)
    if synthetic_links is None:
        resolution = synthesize_missing_links(branches_df, branch_ids)
        synthetic_links = resolution["synthetic_links"]
    else:
        synthetic_links = list(synthetic_links)
        resolution = {
            "branch_ids": sorted({int(branch_id) for branch_id in branch_ids}),
            "synthetic_links": synthetic_links,
            "component_count": len(initial_components),
            "resolved_component_count": 1 if initial_components else 0,
            "bridge_success": True,
        }

    adjacency, node_positions = _build_graph(branches_df, resolution["branch_ids"])
    for link_index, link in enumerate(synthetic_links):
        src_node_id = int(link["src_node_id"])
        dst_node_id = int(link["dst_node_id"])
        length = float(link["length"])
        synthetic_edge_id = -(link_index + 1)
        adjacency.setdefault(src_node_id, []).append((dst_node_id, length, synthetic_edge_id))
        adjacency.setdefault(dst_node_id, []).append((src_node_id, length, synthetic_edge_id))
        src_x, src_y = link["points"][0]
        dst_x, dst_y = link["points"][1]
        if src_node_id not in node_positions:
            node_positions[src_node_id] = (float(src_x), float(src_y))
        if dst_node_id not in node_positions:
            node_positions[dst_node_id] = (float(dst_x), float(dst_y))

    components = _connected_components(adjacency)
    component_branch_lengths: list[tuple[set[int], float]] = []
    for component in components:
        # Sum each unique segment once in this undirected component.
        branch_ids_in_component: set[int] = set()
        total_length = 0.0
        for node_id in component:
            for neighbor, length, branch_id in adjacency[node_id]:
                if neighbor in component and branch_id not in branch_ids_in_component:
                    branch_ids_in_component.add(branch_id)
                    total_length += length
        component_branch_lengths.append((component, total_length))

    active_component, active_component_total_length = max(
        component_branch_lengths, key=lambda item: item[1]
    )

    def shortest_paths(start: int) -> dict[int, float]:
        distances = {start: 0.0}
        visited: set[int] = set()
        while len(visited) < len(active_component):
            available = [node for node in active_component if node not in visited]
            if not available:
                break
            current = min(available, key=lambda node_id: distances.get(node_id, float("inf")))
            if distances.get(current, float("inf")) == float("inf"):
                break
            visited.add(current)
            for neighbor, length, _ in adjacency[current]:
                if neighbor not in active_component:
                    continue
                candidate = distances[current] + length
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
        return distances

    manual_start = int(start_node_id) if start_node_id is not None else None
    manual_end = int(end_node_id) if end_node_id is not None else None
    if (
        manual_start in active_component
        and manual_end in active_component
        and manual_start != manual_end
    ):
        best_start = int(manual_start)
        best_end = int(manual_end)
        best_length = shortest_paths(best_start).get(best_end, 0.0)
    else:
        leaves = [node_id for node_id in active_component if len(adjacency[node_id]) <= 1]
        if len(leaves) < 2:
            leaves = list(active_component)

        best_start = leaves[0]
        best_end = leaves[0]
        best_length = 0.0
        for start in leaves:
            distances = shortest_paths(start)
            for end in leaves:
                candidate = distances.get(end, 0.0)
                if candidate > best_length:
                    best_start = start
                    best_end = end
                    best_length = candidate

    root_xy = np.array(node_positions[best_start], dtype=float)
    end_xy = np.array(node_positions[best_end], dtype=float)
    chord = float(np.linalg.norm(end_xy - root_xy))
    path_length = float(active_component_total_length)

    return {
        "branch_count": int(len(selected)),
        "component_count": int(len(initial_components)),
        "bridge_branch_count": int(len(synthetic_links)),
        "resolved_component_count": int(resolution["resolved_component_count"]),
        "bridge_success": bool(resolution["bridge_success"]),
        "length": path_length,
        "chord": chord,
        "tortuosity": (path_length / chord) if chord > 0 else np.nan,
        "start_node_id": int(best_start),
        "end_node_id": int(best_end),
    }


def persist_manual_review(run_dir: Path, state: dict, branches_df: pd.DataFrame) -> None:
    state_path = run_dir / "manual_review_state.json"
    persisted_state = {
        "selected_branch_ids": [],
        "vessels": state["vessels"],
    }
    state_path.write_text(
        json.dumps(persisted_state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for vessel_name, vessel in state["vessels"].items():
        metrics = score_vessel(
            branches_df,
            vessel["branch_ids"],
            vessel.get("synthetic_links", []),
            vessel.get("start_node_id"),
            vessel.get("end_node_id"),
        )
        rows.append(
            {
                "vessel_name": vessel_name,
                "category": vessel["category"],
                "branch_ids": json.dumps(vessel["branch_ids"]),
                "start_node_id": metrics["start_node_id"],
                "end_node_id": metrics["end_node_id"],
                "branch_count": metrics["branch_count"],
                "component_count": metrics["component_count"],
                "bridge_branch_count": metrics["bridge_branch_count"],
                "resolved_component_count": metrics["resolved_component_count"],
                "bridge_success": metrics["bridge_success"],
                "path_length": metrics["length"],
                "chord_length": metrics["chord"],
                "tortuosity": metrics["tortuosity"],
            }
        )

    manual_csv = run_dir / "manual_vessels.csv"
    manual_df = pd.DataFrame(
        rows,
        columns=[
            "vessel_name",
            "category",
            "branch_ids",
            "start_node_id",
            "end_node_id",
            "branch_count",
            "component_count",
            "bridge_branch_count",
            "resolved_component_count",
            "bridge_success",
            "path_length",
            "chord_length",
            "tortuosity",
        ],
    )
    if not manual_df.empty:
        manual_df = manual_df.sort_values("vessel_name")
    manual_df.to_csv(manual_csv, index=False)


def build_selection_table(branches_df: pd.DataFrame, selected_branch_ids: list[int]) -> pd.DataFrame:
    selection_df = branches_df[branches_df["branch_id"].isin(selected_branch_ids)].copy()
    if selection_df.empty:
        return selection_df
    columns = [
        "branch_id",
        "branch-distance",
        "euclidean-distance",
        "tortuosity",
        "branch-type",
    ]
    if "vascx_category" in selection_df.columns:
        columns.append("vascx_category")
    return selection_df[columns].rename(
        columns={
            "branch_id": "Branch ID",
            "branch-distance": "Length",
            "euclidean-distance": "Chord",
            "tortuosity": "Tortuosity",
            "branch-type": "Type",
            "vascx_category": "VascX label",
        }
    )


def build_vessel_scores_table(review_state: dict, branches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for vessel_name, vessel in sorted(review_state["vessels"].items()):
        metrics = score_vessel(
            branches_df,
            vessel["branch_ids"],
            vessel.get("synthetic_links", []),
            vessel.get("start_node_id"),
            vessel.get("end_node_id"),
        )
        rows.append(
            {
                "Vessel": vessel_name,
                "Category": vessel["category"],
                "Start node": metrics["start_node_id"],
                "End node": metrics["end_node_id"],
                "Branches": len(vessel["branch_ids"]),
                "Components": metrics["component_count"],
                "Auto bridges": metrics["bridge_branch_count"],
                "Bridge status": "connected" if metrics["bridge_success"] else "partial",
                "Path length": metrics["length"],
                "Chord": metrics["chord"],
                "Tortuosity": metrics["tortuosity"],
            }
        )
    return pd.DataFrame(rows)
