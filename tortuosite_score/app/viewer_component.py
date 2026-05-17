from __future__ import annotations

import pandas as pd
import streamlit as st

from tortuosite_score.app.constants import (
    ARTERE_COLOR,
    DEFAULT_BRANCH_COLOR,
    SELECTED_COLOR,
    VEINE_COLOR,
)


def _vascx_branch_color(branch_row: pd.Series) -> str:
    category = branch_row.get("vascx_category", "unknown")
    if category == "artere":
        return ARTERE_COLOR
    if category == "veine":
        return VEINE_COLOR
    return DEFAULT_BRANCH_COLOR


BRANCH_VIEWER = st.components.v2.component(
    name="retina_branch_viewer",
    html="""
    <div class="viewer-shell">
      <svg id="branch-viewer" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
    """,
    css="""
    .viewer-shell {
      width: 100%;
      height: 100%;
      min-height: 640px;
      border: 1px solid rgba(120, 120, 120, 0.35);
      border-radius: 0.85rem;
      overflow: hidden;
      background: #050505;
    }

    #branch-viewer {
      width: 100%;
      height: 100%;
      display: block;
      background: #050505;
      cursor: crosshair;
    }
    """,
    js="""
    export default function(component) {
      const { parentElement, data, setStateValue } = component;
      const svg = parentElement.querySelector("#branch-viewer");
      const svgNs = "http://www.w3.org/2000/svg";
      let currentSelection = Array.isArray(data.selectedBranchIds)
        ? [...data.selectedBranchIds]
        : [];

      function make(tag, attrs = {}) {
        const node = document.createElementNS(svgNs, tag);
        Object.entries(attrs).forEach(([key, value]) => {
          node.setAttribute(key, String(value));
        });
        return node;
      }

      function clear(node) {
        while (node.firstChild) {
          node.removeChild(node.firstChild);
        }
      }

      function toggleBranch(branchId) {
        const next = new Set(currentSelection);
        if (next.has(branchId)) {
          next.delete(branchId);
        } else {
          next.add(branchId);
        }
        currentSelection = Array.from(next).sort((a, b) => a - b);
        setStateValue("selected_branch_ids", currentSelection);
      }

      function toggleVessel(branch) {
        const targetBranchIds = Array.isArray(branch.vesselBranchIds) && branch.vesselBranchIds.length > 0
          ? branch.vesselBranchIds
          : [branch.branchId];
        const next = new Set(currentSelection);
        const isSelected = targetBranchIds.every((branchId) => next.has(branchId));
        targetBranchIds.forEach((branchId) => {
          if (isSelected) {
            next.delete(branchId);
          } else {
            next.add(branchId);
          }
        });
        currentSelection = Array.from(next).sort((a, b) => a - b);
        setStateValue("selected_branch_ids", currentSelection);
      }

      function toggleSelection(branch) {
        if (data.selectionMode === "vessel") {
          toggleVessel(branch);
        } else {
          toggleBranch(branch.branchId);
        }
      }

      clear(svg);

      const width = data.imageWidth ?? 1000;
      const height = data.imageHeight ?? 1000;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const background = make("rect", {
        x: 0,
        y: 0,
        width,
        height,
        fill: "#050505",
      });
      svg.appendChild(background);

      if (data.showBaseImage && data.imageUrl) {
        const image = make("image", {
          x: 0,
          y: 0,
          width,
          height,
          href: data.imageUrl,
          opacity: data.baseOpacity ?? 1,
          preserveAspectRatio: "none",
        });
        svg.appendChild(image);
      }

      const branchesGroup = make("g");
      svg.appendChild(branchesGroup);

      if (data.showSkeleton) {
        (data.branches ?? []).forEach((branch) => {
          const branchGroup = make("g");
          const points = (branch.points ?? [])
            .map((point) => `${point[0]},${point[1]}`)
            .join(" ");
          (branch.strokes ?? []).forEach((strokeLayer) => {
            const polyline = make("polyline", {
              points,
              fill: "none",
              stroke: strokeLayer.color ?? "#ff3b30",
              "stroke-width": strokeLayer.width ?? 2.2,
              "stroke-dasharray": strokeLayer.dasharray ?? "",
              "stroke-opacity": strokeLayer.opacity ?? 1,
              "stroke-linecap": "round",
              "stroke-linejoin": "round",
              "vector-effect": "non-scaling-stroke",
            });
            polyline.style.pointerEvents = "none";
            branchGroup.appendChild(polyline);
          });

          const hitArea = make("polyline", {
            points,
            fill: "none",
            stroke: "rgba(0, 0, 0, 0)",
            "stroke-width": 12,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke",
          });
          hitArea.style.cursor = branch.locked ? "default" : "pointer";
          if (!branch.locked) {
            hitArea.addEventListener("click", (event) => {
              event.stopPropagation();
              toggleSelection(branch);
            });
          }
          branchGroup.appendChild(hitArea);
          branchesGroup.appendChild(branchGroup);

          if (data.showLabels && branch.label) {
            const text = make("text", {
              x: branch.label[0],
              y: branch.label[1],
              fill: branch.labelColor ?? "#ffffff",
              "font-size": 12,
              "font-weight": 700,
              "text-anchor": "middle",
              "paint-order": "stroke",
              stroke: "rgba(0, 0, 0, 0.7)",
              "stroke-width": 2,
            });
            text.textContent = String(branch.branchId);
            text.style.pointerEvents = "none";
            branchesGroup.appendChild(text);
          }
        });
      }

      if (data.showVesselLabels) {
        (data.vesselLabels ?? []).forEach((label) => {
          const text = make("text", {
            x: label.position[0],
            y: label.position[1],
            fill: label.color ?? "#ffffff",
            "font-size": 16,
            "font-weight": 800,
            "text-anchor": "middle",
            "paint-order": "stroke",
            stroke: "rgba(0, 0, 0, 0.78)",
            "stroke-width": 3,
          });
          text.textContent = String(label.text);
          text.style.pointerEvents = "none";
          svg.appendChild(text);
        });
      }

      return () => {};
    }
    """,
)


NODE_ENDPOINT_VIEWER = st.components.v2.component(
    name="retina_node_endpoint_viewer",
    html="""
    <div class="endpoint-shell">
      <svg id="endpoint-viewer" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
    """,
    css="""
    .endpoint-shell {
      width: 100%;
      height: 100%;
      min-height: 260px;
      border: 1px solid rgba(120, 120, 120, 0.35);
      border-radius: 0.85rem;
      overflow: hidden;
      background: #050505;
    }

    #endpoint-viewer {
      width: 100%;
      height: 100%;
      display: block;
      background: #050505;
      cursor: pointer;
    }
    """,
    js="""
    export default function(component) {
      const { parentElement, data, setStateValue } = component;
      const svg = parentElement.querySelector("#endpoint-viewer");
      const svgNs = "http://www.w3.org/2000/svg";
      let startNodeId = data.startNodeId ?? null;
      let endNodeId = data.endNodeId ?? null;
      let nextTarget = data.nextEndpointTarget ?? (startNodeId == null ? "start" : "end");

      function make(tag, attrs = {}) {
        const node = document.createElementNS(svgNs, tag);
        Object.entries(attrs).forEach(([key, value]) => {
          node.setAttribute(key, String(value));
        });
        return node;
      }

      function clear(node) {
        while (node.firstChild) {
          node.removeChild(node.firstChild);
        }
      }

      function setEndpoint(nodeId) {
        if (nextTarget === "start") {
          const previousStart = startNodeId;
          startNodeId = nodeId;
          if (endNodeId === nodeId) {
            endNodeId = previousStart;
          }
          nextTarget = "end";
        } else {
          const previousEnd = endNodeId;
          endNodeId = nodeId;
          if (startNodeId === nodeId) {
            startNodeId = previousEnd;
          }
          nextTarget = "start";
        }
        setStateValue("start_node_id", startNodeId);
        setStateValue("end_node_id", endNodeId);
        setStateValue("next_endpoint_target", nextTarget);
      }

      clear(svg);

      const points = [];
      (data.branches ?? []).forEach((branch) => {
        (branch.points ?? []).forEach((point) => points.push(point));
      });
      (data.nodes ?? []).forEach((node) => points.push([node.x, node.y]));

      const imageWidth = data.imageWidth ?? 1000;
      const imageHeight = data.imageHeight ?? 1000;
      let minX = 0;
      let minY = 0;
      let maxX = imageWidth;
      let maxY = imageHeight;
      if (points.length > 0) {
        minX = Math.min(...points.map((point) => point[0]));
        minY = Math.min(...points.map((point) => point[1]));
        maxX = Math.max(...points.map((point) => point[0]));
        maxY = Math.max(...points.map((point) => point[1]));
        const spanX = Math.max(1, maxX - minX);
        const spanY = Math.max(1, maxY - minY);
        const padding = Math.max(20, Math.max(spanX, spanY) * 0.12);
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(imageWidth, maxX + padding);
        maxY = Math.min(imageHeight, maxY + padding);
      }

      const width = Math.max(1, maxX - minX);
      const height = Math.max(1, maxY - minY);
      svg.setAttribute("viewBox", `${minX} ${minY} ${width} ${height}`);

      const background = make("rect", {
        x: minX,
        y: minY,
        width,
        height,
        fill: "#050505",
      });
      svg.appendChild(background);

      if (data.imageUrl) {
        const image = make("image", {
          x: 0,
          y: 0,
          width: imageWidth,
          height: imageHeight,
          href: data.imageUrl,
          opacity: data.baseOpacity ?? 0.55,
          preserveAspectRatio: "none",
        });
        svg.appendChild(image);
      }

      (data.branches ?? []).forEach((branch) => {
        const pointsAttr = (branch.points ?? [])
          .map((point) => `${point[0]},${point[1]}`)
          .join(" ");
        (branch.strokes ?? [{ color: "#00c2a8", width: 3 }]).forEach((strokeLayer) => {
          const polyline = make("polyline", {
            points: pointsAttr,
            fill: "none",
            stroke: strokeLayer.color ?? "#00c2a8",
            "stroke-width": strokeLayer.width ?? 3,
            "stroke-opacity": strokeLayer.opacity ?? 1,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke",
          });
          polyline.style.pointerEvents = "none";
          svg.appendChild(polyline);
        });
      });

      const markerRadius = Math.max(3, Math.min(width, height) * 0.01);
      const labelFontSize = Math.max(4.5, Math.min(7.5, Math.min(width, height) * 0.012));
      (data.nodes ?? []).forEach((node) => {
        const isStart = node.node_id === startNodeId;
        const isEnd = node.node_id === endNodeId;
        const circle = make("circle", {
          cx: node.x,
          cy: node.y,
          r: markerRadius,
          fill: isStart ? "#00c2a8" : isEnd ? "#ffd166" : "rgba(5, 5, 5, 0.85)",
          stroke: isStart ? "#ffffff" : isEnd ? "#ffffff" : "rgba(255, 255, 255, 0.9)",
          "stroke-width": isStart || isEnd ? 3 : 2,
          "vector-effect": "non-scaling-stroke",
        });
        circle.addEventListener("click", (event) => {
          event.stopPropagation();
          setEndpoint(node.node_id);
        });
        svg.appendChild(circle);

        const label = make("text", {
          x: node.x,
          y: node.y - markerRadius * 1.2,
          fill: "#ffffff",
          "font-size": labelFontSize,
          "font-weight": 800,
          "text-anchor": "middle",
          "paint-order": "stroke",
          stroke: "rgba(0, 0, 0, 0.8)",
          "stroke-width": 1.1,
          "vector-effect": "non-scaling-stroke",
        });
        label.textContent = String(node.node_id);
        label.style.pointerEvents = "none";
        svg.appendChild(label);
      });

      return () => {};
    }
    """,
)


def build_viewer_branches(
    branches_df: pd.DataFrame,
    paths_payload: list[dict],
    review_state: dict,
    allow_reuse_assigned: bool,
    provisional_synthetic_links: list[dict[str, object]] | None = None,
) -> list[dict]:
    branch_memberships: dict[int, list[str]] = {
        int(branch_id): [] for branch_id in branches_df["branch_id"].astype(int).tolist()
    }
    branch_vessels: dict[int, list[str]] = {
        int(branch_id): [] for branch_id in branches_df["branch_id"].astype(int).tolist()
    }
    vessel_branch_ids = {
        vessel_name: sorted(int(branch_id) for branch_id in vessel["branch_ids"])
        for vessel_name, vessel in review_state["vessels"].items()
    }
    assigned_branch_ids: set[int] = set()
    for vessel_name, vessel in review_state["vessels"].items():
        color = ARTERE_COLOR if vessel["category"] == "artere" else VEINE_COLOR
        for branch_id in vessel["branch_ids"]:
            branch_id = int(branch_id)
            branch_memberships.setdefault(branch_id, []).append(color)
            branch_vessels.setdefault(branch_id, []).append(vessel_name)
            assigned_branch_ids.add(branch_id)

    selected_branch_set = set(int(branch_id) for branch_id in review_state["selected_branch_ids"])
    path_map = {item["branchId"]: item for item in paths_payload}
    branch_rows = branches_df.set_index("branch_id", drop=False)
    viewer_branches: list[dict] = []
    for branch_id, memberships in branch_memberships.items():
        path = path_map.get(branch_id)
        if path is None:
            continue
        is_selected = branch_id in selected_branch_set
        strokes: list[dict[str, float | str]] = []
        if memberships:
            unique_memberships = list(dict.fromkeys(memberships))
            base_width = 5.6 if len(unique_memberships) > 1 else 3.0
            width_step = 1.4 if len(unique_memberships) > 1 else 0.0
            for index, color in enumerate(unique_memberships):
                strokes.append(
                    {
                        "color": color,
                        "width": max(2.4, base_width - index * width_step),
                    }
                )
        else:
            strokes.append(
                {
                    "color": _vascx_branch_color(branch_rows.loc[branch_id]),
                    "width": 2.4,
                }
            )

        if is_selected:
            strokes.append(
                {
                    "color": SELECTED_COLOR,
                    "width": 2.8,
                }
            )

        viewer_branches.append(
            {
                "branchId": branch_id,
                "vesselBranchIds": sorted(
                    {
                        linked_branch_id
                        for vessel_name in branch_vessels.get(branch_id, [])
                        for linked_branch_id in vessel_branch_ids.get(vessel_name, [])
                    }
                ),
                "vesselNames": branch_vessels.get(branch_id, []),
                "points": path["points"],
                "label": path["label"],
                "labelColor": SELECTED_COLOR if is_selected else "#ffffff",
                "locked": (branch_id in assigned_branch_ids) and not is_selected and not allow_reuse_assigned,
                "strokes": strokes,
            }
        )

    synthetic_index = 0
    for vessel in review_state["vessels"].values():
        color = ARTERE_COLOR if vessel["category"] == "artere" else VEINE_COLOR
        for synthetic_link in vessel.get("synthetic_links", []):
            viewer_branches.append(
                {
                    "branchId": f"saved-synthetic-{synthetic_index}",
                    "points": synthetic_link["points"],
                    "label": None,
                    "labelColor": "#ffffff",
                    "locked": True,
                    "strokes": [
                        {
                            "color": color,
                            "width": 5.2,
                            "opacity": 0.45,
                        },
                        {
                            "color": color,
                            "width": 2.6,
                            "opacity": 1,
                        },
                    ],
                }
            )
            synthetic_index += 1

    for synthetic_link in provisional_synthetic_links or []:
        viewer_branches.append(
            {
                "branchId": f"provisional-synthetic-{synthetic_index}",
                "points": synthetic_link["points"],
                "label": None,
                "labelColor": "#ffffff",
                "locked": True,
                "strokes": [
                    {
                        "color": SELECTED_COLOR,
                        "width": 5.6,
                        "opacity": 0.45,
                    },
                    {
                        "color": SELECTED_COLOR,
                        "width": 2.8,
                        "opacity": 1,
                    }
                ],
            }
        )
        synthetic_index += 1
    return viewer_branches


def build_vessel_labels(
    branches_df: pd.DataFrame,
    paths_payload: list[dict],
    review_state: dict,
) -> list[dict]:
    path_map = {item["branchId"]: item for item in paths_payload}
    vessel_labels: list[dict] = []
    for vessel_name, vessel in sorted(review_state["vessels"].items()):
        points: list[list[float]] = []
        for branch_id in vessel.get("branch_ids", []):
            path = path_map.get(int(branch_id))
            if path is not None:
                points.extend(path["points"])
        for synthetic_link in vessel.get("synthetic_links", []):
            points.extend(synthetic_link.get("points", []))

        if not points:
            continue

        point_array = pd.DataFrame(points, columns=["x", "y"])
        centroid_x = float(point_array["x"].mean())
        centroid_y = float(point_array["y"].mean())
        vessel_labels.append(
            {
                "text": vessel_name,
                "position": [centroid_x, centroid_y],
                "color": ARTERE_COLOR if vessel["category"] == "artere" else VEINE_COLOR,
            }
        )
    return vessel_labels
