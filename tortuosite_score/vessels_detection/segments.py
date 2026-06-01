from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

import numpy as np
import pandas as pd

SegmentSource = Literal["model", "manual"]

GEOMETRY_JOIN_TOLERANCE = 10.0
MIN_MANUAL_SEGMENT_POINTS = 2
MIN_MANUAL_SEGMENT_LENGTH = 8.0


@dataclass(frozen=True)
class VesselSegment:
    ref: str
    source: SegmentSource
    segment_id: int
    points: tuple[tuple[float, float], ...]
    path_length: float
    chord_length: float
    tortuosity: float
    label_position: tuple[float, float]
    category: str = "unknown"
    branch_type: str = ""

    @classmethod
    def from_model_row(cls, row: pd.Series) -> "VesselSegment":
        segment_id = int(row["branch_id"])
        points = _row_points(row)
        path_length = float(row.get("branch-distance", polyline_length(points)))
        chord_length = float(row.get("euclidean-distance", polyline_chord(points)))
        return cls(
            ref=model_segment_ref(segment_id),
            source="model",
            segment_id=segment_id,
            points=tuple(tuple(point) for point in points),
            path_length=path_length,
            chord_length=chord_length,
            tortuosity=float(row.get("tortuosity", path_length / chord_length if chord_length > 0 else np.nan)),
            label_position=tuple(label_position(points)),
            category=str(row.get("vascx_category", "unknown")),
            branch_type=str(row.get("branch-type", "")),
        )

    @classmethod
    def from_manual_points(cls, segment_id: int, points: list[list[float]]) -> "VesselSegment":
        normalized = normalize_points(points)
        if len(normalized) < MIN_MANUAL_SEGMENT_POINTS:
            raise ValueError("A manual segment needs at least two points.")
        path_length = polyline_length(normalized)
        if path_length < MIN_MANUAL_SEGMENT_LENGTH:
            raise ValueError("A manual segment is too short.")
        chord_length = polyline_chord(normalized)
        return cls(
            ref=manual_segment_ref(segment_id),
            source="manual",
            segment_id=int(segment_id),
            points=tuple(tuple(point) for point in normalized),
            path_length=path_length,
            chord_length=chord_length,
            tortuosity=path_length / chord_length if chord_length > 0 else np.nan,
            label_position=tuple(label_position(normalized)),
            category="manual",
            branch_type="manual",
        )

    @classmethod
    def from_manual_payload(cls, segment_id: int, payload: dict) -> "VesselSegment":
        points = payload.get("points")
        if not isinstance(points, list):
            raise ValueError("Manual segment payload must contain points.")
        return cls.from_manual_points(segment_id, points)

    def to_manual_payload(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "points": [[float(x), float(y)] for x, y in self.points],
            "path_length": float(self.path_length),
            "chord_length": float(self.chord_length),
            "label_position": [float(self.label_position[0]), float(self.label_position[1])],
        }

    def to_viewer_geometry(self) -> dict[str, object]:
        return {
            "segment_ref": self.ref,
            "source": self.source,
            "id": self.segment_id,
            "points": [[float(x), float(y)] for x, y in self.points],
            "path_length": float(self.path_length),
            "chord_length": float(self.chord_length),
            "label_position": [float(self.label_position[0]), float(self.label_position[1])],
            "vascx_category": self.category,
            "branch_type": self.branch_type,
            "tortuosity": float(self.tortuosity),
        }


def model_segment_ref(segment_id: int) -> str:
    return f"model:{int(segment_id)}"


def manual_segment_ref(segment_id: int) -> str:
    return f"manual:{int(segment_id)}"


def parse_segment_ref(segment_ref: str) -> tuple[str, int]:
    source, raw_id = str(segment_ref).split(":", maxsplit=1)
    return source, int(raw_id)


def segment_ref_sort_key(segment_ref: str) -> tuple[int, int]:
    source, segment_id = parse_segment_ref(segment_ref)
    return (0 if source == "model" else 1, segment_id)


def sorted_unique_refs(segment_refs: list[str]) -> list[str]:
    return sorted({str(segment_ref) for segment_ref in segment_refs}, key=segment_ref_sort_key)


def split_segment_refs(segment_refs: list[str]) -> tuple[list[int], list[int]]:
    model_ids: list[int] = []
    manual_ids: list[int] = []
    for segment_ref in sorted_unique_refs(segment_refs):
        source, segment_id = parse_segment_ref(segment_ref)
        if source == "model":
            model_ids.append(segment_id)
        elif source == "manual":
            manual_ids.append(segment_id)
    return model_ids, manual_ids


def normalize_points(points: list[list[float]] | tuple[tuple[float, float], ...]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Segment points must be [x, y] pairs.")
        normalized.append([float(point[0]), float(point[1])])
    return normalized


def polyline_length(points: list[list[float]] | tuple[tuple[float, float], ...]) -> float:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise ValueError("A polyline must contain at least two [x, y] points.")
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def polyline_chord(points: list[list[float]] | tuple[tuple[float, float], ...]) -> float:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise ValueError("A polyline must contain at least two [x, y] points.")
    return float(np.linalg.norm(array[-1] - array[0]))


def label_position(points: list[list[float]] | tuple[tuple[float, float], ...]) -> list[float]:
    array = np.asarray(points, dtype=float)
    midpoint = array[len(array) // 2]
    return [float(midpoint[0]), float(midpoint[1])]


def distance(point_a: list[float] | tuple[float, float], point_b: list[float] | tuple[float, float]) -> float:
    return float(math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1])))


def nearest_point_on_polyline(point_xy: tuple[float, float], points: list[list[float]]) -> dict[str, object]:
    point = np.asarray(point_xy, dtype=float)
    polyline = np.asarray(points, dtype=float)
    best_distance = float("inf")
    best_point = polyline[0]
    best_distance_from_start = 0.0
    traversed = 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length == 0:
            continue
        fraction = float(np.clip(np.dot(point - start, segment) / (segment_length**2), 0.0, 1.0))
        projected = start + fraction * segment
        point_distance = float(np.linalg.norm(point - projected))
        if point_distance < best_distance:
            best_distance = point_distance
            best_point = projected
            best_distance_from_start = traversed + fraction * segment_length
        traversed += segment_length
    return {
        "point": [float(best_point[0]), float(best_point[1])],
        "distance": best_distance,
        "distance_from_start": best_distance_from_start,
        "total_length": traversed,
    }


def create_geometry_endpoint(
    point: list[float],
    segment_ref: str | None = None,
    distance_from_start: float | None = None,
) -> dict[str, object]:
    endpoint: dict[str, object] = {
        "kind": "geometry_point",
        "point": [float(point[0]), float(point[1])],
    }
    if segment_ref is not None:
        endpoint["segment_ref"] = str(segment_ref)
    if distance_from_start is not None:
        endpoint["distance_from_start"] = float(distance_from_start)
    return endpoint


def normalize_endpoint(endpoint: dict | None) -> dict[str, object] | None:
    if not isinstance(endpoint, dict):
        return None
    point = endpoint.get("point")
    if not isinstance(point, list) or len(point) != 2:
        return None
    normalized = create_geometry_endpoint(point)
    if endpoint.get("segment_ref") is not None:
        normalized["segment_ref"] = str(endpoint["segment_ref"])
    if endpoint.get("branch_ref") is not None:
        normalized["segment_ref"] = str(endpoint["branch_ref"])
    if endpoint.get("distance_from_start") is not None:
        normalized["distance_from_start"] = float(endpoint["distance_from_start"])
    return normalized


def segments_from_branches(branches_df: pd.DataFrame) -> dict[str, VesselSegment]:
    return {
        segment.ref: segment
        for _, row in branches_df.iterrows()
        for segment in [VesselSegment.from_model_row(row)]
    }


def manual_segments_from_payload(payloads: dict[str, dict]) -> dict[str, VesselSegment]:
    segments: dict[str, VesselSegment] = {}
    for raw_id, payload in payloads.items():
        try:
            segment = VesselSegment.from_manual_payload(int(raw_id), payload)
        except (TypeError, ValueError, IndexError):
            continue
        segments[segment.ref] = segment
    return segments


def build_segment_map(branches_df: pd.DataFrame, manual_payloads: dict[str, dict]) -> dict[str, VesselSegment]:
    segments = segments_from_branches(branches_df)
    segments.update(manual_segments_from_payload(manual_payloads))
    return segments


def score_segments(
    segments: dict[str, VesselSegment],
    selected_refs: list[str],
    synthetic_links: list[dict[str, object]] | None = None,
    start_endpoint: dict[str, object] | None = None,
    end_endpoint: dict[str, object] | None = None,
    join_tolerance: float = GEOMETRY_JOIN_TOLERANCE,
) -> dict[str, object]:
    selected = {
        segment_ref: segments[segment_ref]
        for segment_ref in sorted_unique_refs(selected_refs)
        if segment_ref in segments
    }
    model_count = sum(1 for segment in selected.values() if segment.source == "model")
    manual_count = sum(1 for segment in selected.values() if segment.source == "manual")
    if not selected:
        return _empty_score(model_count, manual_count)

    graph = build_segment_graph(selected, synthetic_links or [], join_tolerance)
    graph_nodes = graph["graph_nodes"]
    adjacency = graph["adjacency"]
    segment_nodes = graph["segment_nodes"]
    components = graph["components"]
    if not components:
        return _empty_score(model_count, manual_count, branch_count=len(selected), bridge_success=False)

    start = normalize_endpoint(start_endpoint)
    end = normalize_endpoint(end_endpoint)
    start_node = add_endpoint_node(start, selected, graph_nodes, adjacency, segment_nodes) if start else None
    end_node = add_endpoint_node(end, selected, graph_nodes, adjacency, segment_nodes) if end else None
    if start_node is not None and end_node is not None and start and end:
        start_ref = start.get("segment_ref")
        end_ref = end.get("segment_ref")
        if start_ref == end_ref and start_ref in selected:
            start_distance = endpoint_distance_from_start(start, selected[str(start_ref)])
            end_distance = endpoint_distance_from_start(end, selected[str(end_ref)])
            add_edge(adjacency, start_node, end_node, abs(end_distance - start_distance))
    components = connected_components(adjacency)

    active_component = None
    if start_node is not None and end_node is not None:
        start_component = component_for_node(components, start_node)
        end_component = component_for_node(components, end_node)
        if start_component is not None and start_component == end_component and start_node != end_node:
            active_component = start_component

    if active_component is None:
        active_component = largest_component(components, adjacency)
        start_node, end_node = endpoints_from_longest_path(active_component, adjacency)
        start = create_geometry_endpoint(graph_nodes[start_node]["point"]) if start_node is not None else None
        end = create_geometry_endpoint(graph_nodes[end_node]["point"]) if end_node is not None else None

    distances = shortest_paths(adjacency, start_node) if start_node is not None else {}
    path_length = float(distances.get(end_node, np.nan)) if end_node is not None else np.nan
    chord = float(distance(start["point"], end["point"])) if start and end else np.nan
    bridge_count = len(synthetic_links or [])
    return {
        "model_segment_count": model_count,
        "manual_segment_count": manual_count,
        "branch_count": len(selected),
        "segment_count": len(selected),
        "component_count": len(components),
        "bridge_branch_count": bridge_count,
        "resolved_component_count": 1 if active_component else 0,
        "bridge_success": bool(active_component is not None and not np.isnan(path_length) and (len(components) <= 1 or bridge_count > 0)),
        "length": path_length,
        "chord": chord,
        "tortuosity": path_length / chord if chord and chord > 0 else np.nan,
        "start_endpoint": start,
        "end_endpoint": end,
    }


def synthesize_segment_links(
    segments: dict[str, VesselSegment],
    selected_refs: list[str],
    join_tolerance: float = GEOMETRY_JOIN_TOLERANCE,
) -> dict[str, object]:
    selected = {
        segment_ref: segments[segment_ref]
        for segment_ref in sorted_unique_refs(selected_refs)
        if segment_ref in segments
    }
    if not selected:
        return {"synthetic_links": [], "component_count": 0, "resolved_component_count": 0, "bridge_success": True}
    graph = build_segment_graph(selected, [], join_tolerance)
    graph_nodes = graph["graph_nodes"]
    adjacency = graph["adjacency"]
    segment_nodes = graph["segment_nodes"]
    components = [set(component) for component in graph["components"]]
    if len(components) <= 1:
        return {"synthetic_links": [], "component_count": len(components), "resolved_component_count": len(components), "bridge_success": True}

    synthetic_links: list[dict[str, object]] = []
    working = [set(component) for component in components]
    while len(working) > 1:
        best_pair: tuple[int, int] | None = None
        best_link: dict[str, object] | None = None
        best_length = float("inf")
        component_refs = [segment_refs_for_component(component, segment_nodes) for component in working]

        for idx, component_a in enumerate(working):
            nodes_a = component_candidate_nodes(component_a, adjacency)
            for jdx in range(idx + 1, len(working)):
                refs_b = component_refs[jdx]
                for node_a in nodes_a:
                    point_a = graph_nodes[node_a]["point"]
                    target_b = nearest_point_on_component(refs_b, selected, point_a)
                    if target_b is None:
                        continue
                    point_b = target_b["point"]
                    link_length = distance(point_a, point_b)
                    if link_length < best_length:
                        best_length = link_length
                        best_pair = (idx, jdx)
                        best_link = {
                            "points": [[float(point_a[0]), float(point_a[1])], [float(point_b[0]), float(point_b[1])]],
                            "length": float(link_length),
                            "display_length": float(link_length),
                            "target_segment_ref": target_b["segment_ref"],
                            "target_distance_from_start": target_b["distance_from_start"],
                        }

        if best_pair is None or best_link is None:
            return {
                "synthetic_links": synthetic_links,
                "component_count": len(components),
                "resolved_component_count": len(working),
                "bridge_success": False,
            }

        synthetic_links.append(best_link)
        start_node = register_graph_node(graph_nodes, best_link["points"][0], join_tolerance)
        end_node = register_graph_node(graph_nodes, best_link["points"][1], join_tolerance)
        add_edge(adjacency, start_node, end_node, float(best_link["length"]))
        idx, jdx = best_pair
        working[idx] = working[idx] | working[jdx]
        del working[jdx]

    return {
        "synthetic_links": synthetic_links,
        "component_count": len(components),
        "resolved_component_count": 1,
        "bridge_success": True,
    }


def build_segment_graph(
    segments: dict[str, VesselSegment],
    synthetic_links: list[dict[str, object]],
    join_tolerance: float,
) -> dict[str, object]:
    graph_nodes: list[dict[str, object]] = []
    adjacency: dict[int, list[tuple[int, float]]] = {}
    segment_nodes: dict[str, list[int]] = {}
    for segment_ref, segment in segments.items():
        node_ids = [register_graph_node(graph_nodes, [x, y], 0.1) for x, y in segment.points]
        segment_nodes[segment_ref] = node_ids
        for start_id, end_id, start_point, end_point in zip(node_ids[:-1], node_ids[1:], segment.points[:-1], segment.points[1:]):
            edge_length = distance(start_point, end_point)
            if edge_length > 0:
                add_edge(adjacency, start_id, end_id, edge_length)
    for link in synthetic_links:
        points = link.get("points", [])
        if not isinstance(points, list) or len(points) != 2:
            continue
        start_node = register_graph_node(graph_nodes, [float(points[0][0]), float(points[0][1])], join_tolerance)
        end_node = register_graph_node(graph_nodes, [float(points[1][0]), float(points[1][1])], join_tolerance)
        add_edge(adjacency, start_node, end_node, float(link.get("length", distance(points[0], points[1]))))
    return {
        "graph_nodes": graph_nodes,
        "adjacency": adjacency,
        "segment_nodes": segment_nodes,
        "components": connected_components(adjacency),
    }


def register_graph_node(graph_nodes: list[dict[str, object]], point: list[float], tolerance: float) -> int:
    closest_index = None
    closest_distance = float("inf")
    for index, existing in enumerate(graph_nodes):
        candidate_distance = distance(point, existing["point"])
        if candidate_distance < closest_distance:
            closest_index = index
            closest_distance = candidate_distance
    if closest_index is not None and closest_distance <= tolerance:
        existing = graph_nodes[closest_index]
        if closest_distance > 1e-6:
            existing["point"] = [
                float((existing["point"][0] + point[0]) / 2.0),
                float((existing["point"][1] + point[1]) / 2.0),
            ]
        return closest_index
    graph_nodes.append({"point": [float(point[0]), float(point[1])]})
    return len(graph_nodes) - 1


def add_edge(adjacency: dict[int, list[tuple[int, float]]], start_node: int, end_node: int, length: float) -> None:
    adjacency.setdefault(start_node, []).append((end_node, float(length)))
    adjacency.setdefault(end_node, []).append((start_node, float(length)))


def connected_components(adjacency: dict[int, list[tuple[int, float]]]) -> list[set[int]]:
    unvisited = set(adjacency)
    components: list[set[int]] = []
    while unvisited:
        start = unvisited.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            for neighbor, _ in adjacency.get(current, []):
                if neighbor not in component:
                    component.add(neighbor)
                    unvisited.discard(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def nearest_graph_node(point: list[float], graph_nodes: list[dict[str, object]]) -> int | None:
    if not graph_nodes:
        return None
    return min(range(len(graph_nodes)), key=lambda node_id: distance(point, graph_nodes[node_id]["point"]))


def add_endpoint_node(
    endpoint: dict[str, object],
    segments: dict[str, VesselSegment],
    graph_nodes: list[dict[str, object]],
    adjacency: dict[int, list[tuple[int, float]]],
    segment_nodes: dict[str, list[int]],
) -> int | None:
    segment_ref = endpoint.get("segment_ref")
    if segment_ref not in segments:
        return nearest_graph_node(endpoint["point"], graph_nodes)
    segment = segments[str(segment_ref)]
    distance_from_start = endpoint.get("distance_from_start")
    if distance_from_start is None:
        nearest = nearest_point_on_polyline(
            (float(endpoint["point"][0]), float(endpoint["point"][1])),
            [[x, y] for x, y in segment.points],
        )
        distance_from_start = nearest["distance_from_start"]
    endpoint_point = [float(endpoint["point"][0]), float(endpoint["point"][1])]
    endpoint_node = register_graph_node(graph_nodes, endpoint_point, 0.1)
    node_ids = segment_nodes.get(str(segment_ref), [])
    if len(node_ids) < 2:
        return endpoint_node

    traversed = 0.0
    target_distance = float(distance_from_start)
    for index, (start_point, end_point) in enumerate(zip(segment.points[:-1], segment.points[1:])):
        segment_length = distance(start_point, end_point)
        if segment_length == 0:
            continue
        if target_distance <= traversed + segment_length + 1e-6:
            start_node = node_ids[index]
            end_node = node_ids[index + 1]
            distance_to_start = max(0.0, min(segment_length, target_distance - traversed))
            distance_to_end = max(0.0, segment_length - distance_to_start)
            if endpoint_node != start_node and distance_to_start > 0:
                add_edge(adjacency, endpoint_node, start_node, distance_to_start)
            if endpoint_node != end_node and distance_to_end > 0:
                add_edge(adjacency, endpoint_node, end_node, distance_to_end)
            return endpoint_node
        traversed += segment_length

    end_node = node_ids[-1]
    if endpoint_node != end_node:
        add_edge(adjacency, endpoint_node, end_node, distance(endpoint_point, graph_nodes[end_node]["point"]))
    return endpoint_node


def endpoint_distance_from_start(endpoint: dict[str, object], segment: VesselSegment) -> float:
    if endpoint.get("distance_from_start") is not None:
        return float(endpoint["distance_from_start"])
    nearest = nearest_point_on_polyline(
        (float(endpoint["point"][0]), float(endpoint["point"][1])),
        [[x, y] for x, y in segment.points],
    )
    return float(nearest["distance_from_start"])


def component_for_node(components: list[set[int]], node_id: int | None) -> set[int] | None:
    if node_id is None:
        return None
    for component in components:
        if node_id in component:
            return component
    return None


def shortest_paths(adjacency: dict[int, list[tuple[int, float]]], start_node: int) -> dict[int, float]:
    distances = {start_node: 0.0}
    heap: list[tuple[float, int]] = [(0.0, start_node)]
    while heap:
        distance_so_far, current = heappop(heap)
        if distance_so_far > distances.get(current, float("inf")):
            continue
        for neighbor, weight in adjacency.get(current, []):
            candidate = distance_so_far + float(weight)
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heappush(heap, (candidate, neighbor))
    return distances


def largest_component(components: list[set[int]], adjacency: dict[int, list[tuple[int, float]]]) -> set[int]:
    def component_weight(component: set[int]) -> float:
        total = 0.0
        seen: set[tuple[int, int]] = set()
        for node_id in component:
            for neighbor, weight in adjacency.get(node_id, []):
                edge_key = tuple(sorted((node_id, neighbor)))
                if neighbor in component and edge_key not in seen:
                    total += float(weight)
                    seen.add(edge_key)
        return total
    return max(components, key=component_weight)


def endpoints_from_longest_path(component: set[int], adjacency: dict[int, list[tuple[int, float]]]) -> tuple[int | None, int | None]:
    candidate_nodes = [node_id for node_id in component if len(adjacency.get(node_id, [])) <= 1] or list(component)
    best_start = candidate_nodes[0] if candidate_nodes else None
    best_end = candidate_nodes[0] if candidate_nodes else None
    best_length = 0.0
    for start_node in candidate_nodes:
        distances = shortest_paths(adjacency, start_node)
        for end_node in candidate_nodes:
            candidate_length = distances.get(end_node, 0.0)
            if candidate_length > best_length:
                best_start = start_node
                best_end = end_node
                best_length = candidate_length
    return best_start, best_end


def component_candidate_nodes(component: set[int], adjacency: dict[int, list[tuple[int, float]]]) -> list[int]:
    leaves = [node_id for node_id in component if len(adjacency.get(node_id, [])) <= 1]
    return leaves or list(component)


def segment_refs_for_component(component: set[int], segment_nodes: dict[str, list[int]]) -> set[str]:
    component_node_ids = set(component)
    return {
        segment_ref
        for segment_ref, node_ids in segment_nodes.items()
        if any(node_id in component_node_ids for node_id in node_ids)
    }


def nearest_point_on_component(
    component_refs: set[str],
    segments: dict[str, VesselSegment],
    point: list[float],
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for segment_ref in component_refs:
        segment = segments.get(segment_ref)
        if segment is None:
            continue
        candidate = nearest_point_on_polyline((float(point[0]), float(point[1])), [[x, y] for x, y in segment.points])
        candidate["segment_ref"] = segment_ref
        if best is None or float(candidate["distance"]) < float(best["distance"]):
            best = candidate
    return best


def _row_points(row: pd.Series) -> list[list[float]]:
    points = row.get("path_points")
    if isinstance(points, list) and len(points) >= 2:
        return normalize_points(points)
    return [
        [float(row["image-coord-src-1"]), float(row["image-coord-src-0"])],
        [float(row["image-coord-dst-1"]), float(row["image-coord-dst-0"])],
    ]


def _empty_score(
    model_count: int,
    manual_count: int,
    branch_count: int = 0,
    bridge_success: bool = True,
) -> dict[str, object]:
    return {
        "model_segment_count": model_count,
        "manual_segment_count": manual_count,
        "branch_count": branch_count,
        "segment_count": branch_count,
        "component_count": 0,
        "bridge_branch_count": 0,
        "resolved_component_count": 0,
        "bridge_success": bridge_success,
        "length": np.nan,
        "chord": np.nan,
        "tortuosity": np.nan,
        "start_endpoint": None,
        "end_endpoint": None,
    }
