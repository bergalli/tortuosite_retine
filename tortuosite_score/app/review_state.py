from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tortuosite_score.app.review_data import read_json


def get_or_create_review_state(run_dir: Path) -> dict:
    state_key = f"review_state::{run_dir.name}"
    if state_key not in st.session_state:
        saved_path = run_dir / "manual_review_state.json"
        state = read_json(saved_path) if saved_path.exists() else {}
        st.session_state[state_key] = {
            "selected_branch_ids": state.get("selected_branch_ids", []),
            "vessels": state.get("vessels", {}),
        }
    return st.session_state[state_key]


def next_default_vessel_name(vessels: dict[str, dict], category: str) -> str:
    prefix = "artere" if category == "artere" else "veine"
    index = 1
    existing_names = set(vessels)
    while f"{prefix}_{index}" in existing_names:
        index += 1
    return f"{prefix}_{index}"


def score_vessel(branches_df: pd.DataFrame, branch_ids: list[int]) -> dict[str, object]:
    selected = branches_df[branches_df["branch_id"].isin(branch_ids)].copy()
    if selected.empty:
        return {
            "branch_count": 0,
            "component_count": 0,
            "length": np.nan,
            "chord": np.nan,
            "tortuosity": np.nan,
        }

    adjacency: dict[int, list[tuple[int, float, int]]] = {}
    node_positions: dict[int, tuple[float, float]] = {}
    for _, row in selected.iterrows():
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

    component_branch_lengths: list[tuple[set[int], float]] = []
    for component in components:
        branch_ids_in_component: set[int] = set()
        total_length = 0.0
        for node_id in component:
            for neighbor, length, branch_id in adjacency[node_id]:
                if neighbor in component and branch_id not in branch_ids_in_component:
                    branch_ids_in_component.add(branch_id)
                    total_length += length
        component_branch_lengths.append((component, total_length))

    active_component = max(component_branch_lengths, key=lambda item: item[1])[0]
    leaves = [node_id for node_id in active_component if len(adjacency[node_id]) <= 1]
    if len(leaves) < 2:
        leaves = list(active_component)

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
    path_length = float(best_length)

    return {
        "branch_count": int(len(selected)),
        "component_count": int(len(components)),
        "length": path_length,
        "chord": chord,
        "tortuosity": (path_length / chord) if chord > 0 else np.nan,
    }


def persist_manual_review(run_dir: Path, state: dict, branches_df: pd.DataFrame) -> None:
    state_path = run_dir / "manual_review_state.json"
    state_path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for vessel_name, vessel in state["vessels"].items():
        metrics = score_vessel(branches_df, vessel["branch_ids"])
        rows.append(
            {
                "vessel_name": vessel_name,
                "category": vessel["category"],
                "branch_ids": json.dumps(vessel["branch_ids"]),
                "branch_count": metrics["branch_count"],
                "component_count": metrics["component_count"],
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
            "branch_count",
            "component_count",
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
    return selection_df[
        [
            "branch_id",
            "branch-distance",
            "euclidean-distance",
            "tortuosity",
            "branch-type",
        ]
    ].rename(
        columns={
            "branch_id": "Branch ID",
            "branch-distance": "Length",
            "euclidean-distance": "Chord",
            "tortuosity": "Tortuosity",
            "branch-type": "Type",
        }
    )


def build_vessel_scores_table(review_state: dict, branches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for vessel_name, vessel in sorted(review_state["vessels"].items()):
        metrics = score_vessel(branches_df, vessel["branch_ids"])
        rows.append(
            {
                "Vessel": vessel_name,
                "Category": vessel["category"],
                "Branches": len(vessel["branch_ids"]),
                "Components": metrics["component_count"],
                "Path length": metrics["length"],
                "Chord": metrics["chord"],
                "Tortuosity": metrics["tortuosity"],
            }
        )
    return pd.DataFrame(rows)
