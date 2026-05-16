from __future__ import annotations

import pandas as pd
import streamlit as st

from tortuosite_score.app.constants import (
    ARTERE_COLOR,
    DEFAULT_BRANCH_COLOR,
    SELECTED_COLOR,
    VEINE_COLOR,
)


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
          const polyline = make("polyline", {
            points: (branch.points ?? [])
              .map((point) => `${point[0]},${point[1]}`)
              .join(" "),
            fill: "none",
            stroke: branch.stroke ?? "#ff3b30",
            "stroke-width": branch.strokeWidth ?? 2.2,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke",
          });
          polyline.style.cursor = branch.locked ? "default" : "pointer";
          if (!branch.locked) {
            polyline.addEventListener("click", (event) => {
              event.stopPropagation();
              toggleBranch(branch.branchId);
            });
          }
          branchesGroup.appendChild(polyline);

          if (data.showLabels && branch.label) {
            const text = make("text", {
              x: branch.label[0],
              y: branch.label[1],
              fill: branch.labelColor ?? "#ffffff",
              "font-size": 18,
              "font-weight": 700,
              "text-anchor": "middle",
              "paint-order": "stroke",
              stroke: "rgba(0, 0, 0, 0.7)",
              "stroke-width": 3,
            });
            text.textContent = String(branch.branchId);
            text.style.pointerEvents = "none";
            branchesGroup.appendChild(text);
          }
        });
      }

      return () => {};
    }
    """,
)


def build_viewer_branches(
    branches_df: pd.DataFrame,
    paths_payload: list[dict],
    review_state: dict,
) -> list[dict]:
    branch_color_lookup = {
        branch_id: DEFAULT_BRANCH_COLOR for branch_id in branches_df["branch_id"].astype(int).tolist()
    }
    assigned_branch_ids: set[int] = set()
    for vessel in review_state["vessels"].values():
        color = ARTERE_COLOR if vessel["category"] == "artere" else VEINE_COLOR
        for branch_id in vessel["branch_ids"]:
            branch_color_lookup[int(branch_id)] = color
            assigned_branch_ids.add(int(branch_id))

    selected_branch_set = set(int(branch_id) for branch_id in review_state["selected_branch_ids"])
    path_map = {item["branchId"]: item for item in paths_payload}
    viewer_branches: list[dict] = []
    for branch_id, color in branch_color_lookup.items():
        path = path_map.get(branch_id)
        if path is None:
            continue
        is_selected = branch_id in selected_branch_set
        viewer_branches.append(
            {
                "branchId": branch_id,
                "points": path["points"],
                "label": path["label"],
                "stroke": SELECTED_COLOR if is_selected else color,
                "strokeWidth": 4.8 if is_selected else 2.4,
                "labelColor": SELECTED_COLOR if is_selected else "#ffffff",
                "locked": (branch_id in assigned_branch_ids) and not is_selected,
            }
        )
    return viewer_branches
