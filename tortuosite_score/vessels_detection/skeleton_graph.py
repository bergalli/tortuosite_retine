"""Skeleton branch analysis without skan (scikit-image graph fallback)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from skimage.graph import pixel_graph


_SUMMARY_COLUMNS = [
    "image-coord-src-0",
    "image-coord-src-1",
    "image-coord-dst-0",
    "image-coord-dst-1",
    "branch-distance",
    "euclidean-distance",
]


@dataclass
class SkeletonGraph:
    """Minimal skan.Skeleton-like API for path coordinate lookup."""

    paths: list[np.ndarray]

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def path_coordinates(self, path_index: int) -> np.ndarray:
        return self.paths[path_index]


def _edge_length(row_a: int, col_a: int, row_b: int, col_b: int) -> float:
    if row_a == row_b or col_a == col_b:
        return 1.0
    return float(np.sqrt(2.0))


def _build_adjacency(
    graph: sparse.csr_matrix,
) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    degrees = np.asarray(graph.sum(axis=1)).flatten()
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(graph.shape[0])]
    coo = graph.tocoo()
    for row, col, weight in zip(coo.row, coo.col, coo.data, strict=True):
        if row >= col:
            continue
        length = float(weight)
        adjacency[row].append((col, length))
        adjacency[col].append((row, length))
    return adjacency, degrees


def _trace_branches(
    adjacency: list[list[tuple[int, float]]],
    degrees: np.ndarray,
    index_to_coord: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[float, float, int, int, int, int]]]:
    special_nodes = {index for index, degree in enumerate(degrees) if degree != 2}
    if not special_nodes:
        return [], []

    visited_edges: set[tuple[int, int]] = set()
    paths: list[np.ndarray] = []
    branch_records: list[tuple[float, float, int, int, int, int]] = []

    def mark_edge(a: int, b: int) -> bool:
        key = (a, b) if a < b else (b, a)
        if key in visited_edges:
            return False
        visited_edges.add(key)
        return True

    for start in sorted(special_nodes):
        for neighbor, step_length in adjacency[start]:
            if not mark_edge(start, neighbor):
                continue

            path_indices = [start, neighbor]
            path_length = step_length
            previous, current = start, neighbor

            while current not in special_nodes:
                next_candidates = [
                    (node, length)
                    for node, length in adjacency[current]
                    if node != previous
                ]
                if len(next_candidates) != 1:
                    break
                next_node, step = next_candidates[0]
                if not mark_edge(current, next_node):
                    break
                path_length += step
                path_indices.append(next_node)
                previous, current = current, next_node

            if current in special_nodes and current != start:
                start_row, start_col = index_to_coord[start]
                end_row, end_col = index_to_coord[current]
                euclidean = float(
                    np.hypot(start_row - end_row, start_col - end_col)
                )
                coords = index_to_coord[path_indices]
                paths.append(coords)
                branch_records.append(
                    (
                        path_length,
                        euclidean,
                        int(start_row),
                        int(start_col),
                        int(end_row),
                        int(end_col),
                    )
                )

    if not branch_records and degrees.size > 0 and np.all(degrees == 2):
        # Single closed loop: pick an arbitrary start node.
        start = 0
        neighbor, step_length = adjacency[start][0]
        mark_edge(start, neighbor)
        path_indices = [start, neighbor]
        path_length = step_length
        previous, current = start, neighbor
        while current != start:
            next_candidates = [
                (node, length)
                for node, length in adjacency[current]
                if node != previous
            ]
            if len(next_candidates) != 1:
                break
            next_node, step = next_candidates[0]
            if not mark_edge(current, next_node):
                break
            path_length += step
            path_indices.append(next_node)
            previous, current = current, next_node
        start_row, start_col = index_to_coord[start]
        end_row, end_col = index_to_coord[current]
        euclidean = float(np.hypot(start_row - end_row, start_col - end_col))
        paths.append(index_to_coord[path_indices])
        branch_records.append(
            (
                path_length,
                euclidean,
                int(start_row),
                int(start_col),
                int(end_row),
                int(end_col),
            )
        )

    return paths, branch_records


def skeleton_graph_from_image(skeleton: np.ndarray) -> SkeletonGraph:
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return SkeletonGraph(paths=[])

    graph, nodes = pixel_graph(skeleton, connectivity=2)
    index_to_coord = np.column_stack(np.unravel_index(nodes, skeleton.shape))
    adjacency, degrees = _build_adjacency(graph)
    paths, _ = _trace_branches(adjacency, degrees, index_to_coord)
    return SkeletonGraph(paths=paths)


def summarize_skeleton_branches(skeleton: np.ndarray) -> pd.DataFrame:
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    graph, nodes = pixel_graph(skeleton, connectivity=2)
    index_to_coord = np.column_stack(np.unravel_index(nodes, skeleton.shape))
    adjacency, degrees = _build_adjacency(graph)
    _, branch_records = _trace_branches(adjacency, degrees, index_to_coord)

    if not branch_records:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    summary = pd.DataFrame(
        branch_records,
        columns=[
            "branch-distance",
            "euclidean-distance",
            "image-coord-src-0",
            "image-coord-src-1",
            "image-coord-dst-0",
            "image-coord-dst-1",
        ],
    )
    return summary[_SUMMARY_COLUMNS]
