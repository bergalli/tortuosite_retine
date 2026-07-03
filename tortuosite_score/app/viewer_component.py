from __future__ import annotations

import pandas as pd
import streamlit as st

from tortuosite_score.app.constants import (
    ARTERE_COLOR,
    DEFAULT_BRANCH_COLOR,
    SELECTED_COLOR,
    VEINE_COLOR,
)
from tortuosite_score.app.review_state import (
    get_segment_geometry,
    segment_ref_sort_key,
    segment_refs_for_vessel,
)


def _segment_color(label: str) -> str:
    if label == "artere":
        return ARTERE_COLOR
    if label == "veine":
        return VEINE_COLOR
    if label == "manual":
        return "#ffd166"
    return DEFAULT_BRANCH_COLOR


BRANCH_VIEWER = st.components.v2.component(
    name="retina_branch_viewer",
    html="""
    <div class="viewer-shell">
      <svg id="branch-viewer" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
    """,
    css="""
    html,
    body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #050505;
    }

    * {
      box-sizing: border-box;
    }

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
      touch-action: none;
    }
    """,
    js="""
    export default function(component) {
      const { parentElement, data, setStateValue, setTriggerValue } = component;
      const svg = parentElement.querySelector("#branch-viewer");
      const svgNs = "http://www.w3.org/2000/svg";
      let currentSelection = Array.isArray(data.selectedSegmentRefs)
        ? [...data.selectedSegmentRefs]
        : [];
      let currentStroke = [];
      let isDrawing = false;
      let activePointerId = null;
      let pendingClickSegment = null;
      const interactionMode = data.interactionMode ?? "both";
      const readOnly = interactionMode === "readonly" || data.readOnly === true;
      const selectedColor = "#00c2a8";
      const minDrawLength = 3;

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

      function pointsAttr(points) {
        return (points ?? []).map((point) => `${point[0]},${point[1]}`).join(" ");
      }

      function toggleSegment(segmentRef) {
        if (readOnly) {
          return;
        }
        const next = new Set(currentSelection);
        if (next.has(segmentRef)) {
          next.delete(segmentRef);
        } else {
          next.add(segmentRef);
        }
        currentSelection = Array.from(next).sort();
        render();
        setStateValue("selected_segment_refs", currentSelection);
      }

      function toggleVessel(segment) {
        if (readOnly) {
          return;
        }
        const targetRefs = Array.isArray(segment.vesselSegmentRefs) && segment.vesselSegmentRefs.length > 0
          ? segment.vesselSegmentRefs
          : [segment.segmentRef];
        const next = new Set(currentSelection);
        const fullySelected = targetRefs.every((segmentRef) => next.has(segmentRef));
        targetRefs.forEach((segmentRef) => {
          if (fullySelected) {
            next.delete(segmentRef);
          } else {
            next.add(segmentRef);
          }
        });
        currentSelection = Array.from(next).sort();
        render();
        setStateValue("selected_segment_refs", currentSelection);
      }

      function toggleSelection(segment) {
        if (data.selectionMode === "vessel") {
          toggleVessel(segment);
        } else {
          toggleSegment(segment.segmentRef);
        }
      }

      function clientPointToSvg(event) {
        const rect = svg.getBoundingClientRect();
        const viewBox = svg.viewBox.baseVal;
        const scaleX = viewBox.width / rect.width;
        const scaleY = viewBox.height / rect.height;
        return [
          viewBox.x + (event.clientX - rect.left) * scaleX,
          viewBox.y + (event.clientY - rect.top) * scaleY,
        ];
      }

      function strokeLength(points) {
        let length = 0;
        for (let index = 1; index < points.length; index += 1) {
          const previous = points[index - 1];
          const point = points[index];
          length += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
        }
        return length;
      }

      function beginDraw(event) {
        if (readOnly) {
          return;
        }
        if (event.button !== undefined && event.button !== 0) {
          return;
        }
        event.preventDefault();
        if (event.pointerId !== undefined && svg.setPointerCapture) {
          svg.setPointerCapture(event.pointerId);
        }
        activePointerId = event.pointerId ?? null;
        pendingClickSegment = event.target.__segmentData ?? null;
        isDrawing = true;
        currentStroke = [clientPointToSvg(event)];
        render();
      }

      function extendDraw(event) {
        if (!isDrawing) {
          return;
        }
        if (activePointerId !== null && event.pointerId !== activePointerId) {
          return;
        }
        event.preventDefault();
        const point = clientPointToSvg(event);
        const previous = currentStroke[currentStroke.length - 1];
        if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 1.5) {
          currentStroke.push(point);
          render();
        }
      }

      function finishDraw(event) {
        if (!isDrawing) {
          return;
        }
        if (activePointerId !== null && event?.pointerId !== undefined && event.pointerId !== activePointerId) {
          return;
        }
        if (event?.clientX !== undefined && event?.clientY !== undefined) {
          extendDraw(event);
        }
        if (event?.pointerId !== undefined && svg.hasPointerCapture?.(event.pointerId)) {
          svg.releasePointerCapture(event.pointerId);
        }
        isDrawing = false;
        const shouldDraw = currentStroke.length >= 2 && strokeLength(currentStroke) >= minDrawLength;
        if (shouldDraw) {
          setTriggerValue("draw_segment", {
            action: interactionMode === "redraw" ? "redraw" : "create",
            points: currentStroke,
          });
        } else if (pendingClickSegment !== null) {
          toggleSelection(pendingClickSegment);
        }
        activePointerId = null;
        pendingClickSegment = null;
        currentStroke = [];
        render();
      }

      function render() {
        clear(svg);
        const width = data.imageWidth ?? 1000;
        const height = data.imageHeight ?? 1000;
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.style.cursor = readOnly ? "default" : "crosshair";

        svg.appendChild(make("rect", { x: 0, y: 0, width, height, fill: "#050505" }));

        if (data.showBaseImage && data.imageUrl) {
          svg.appendChild(make("image", {
            x: 0,
            y: 0,
            width,
            height,
            href: data.imageUrl,
            opacity: data.baseOpacity ?? 1,
            preserveAspectRatio: "none",
          }));
        }

        const segmentsGroup = make("g");
        svg.appendChild(segmentsGroup);
        if (data.showSkeleton) {
          (data.segments ?? []).forEach((segment) => {
            const segmentGroup = make("g");
            const segmentPoints = pointsAttr(segment.points ?? []);
            (segment.strokes ?? []).forEach((strokeLayer) => {
              const polyline = make("polyline", {
                points: segmentPoints,
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
              segmentGroup.appendChild(polyline);
            });
            if (currentSelection.includes(segment.segmentRef)) {
              const selectedPolyline = make("polyline", {
                points: segmentPoints,
                fill: "none",
                stroke: selectedColor,
                "stroke-width": 3.0,
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "vector-effect": "non-scaling-stroke",
              });
              selectedPolyline.style.pointerEvents = "none";
              segmentGroup.appendChild(selectedPolyline);
            }

            const hitArea = make("polyline", {
              points: segmentPoints,
              fill: "none",
              stroke: "rgba(0, 0, 0, 0)",
              "stroke-width": 14,
              "stroke-linecap": "round",
              "stroke-linejoin": "round",
              "vector-effect": "non-scaling-stroke",
            });
            hitArea.style.cursor = (readOnly || segment.locked) ? "default" : "pointer";
            if (!readOnly && !segment.locked) {
              hitArea.__segmentData = segment;
            }
            segmentGroup.appendChild(hitArea);
            segmentsGroup.appendChild(segmentGroup);

            if (data.showLabels && segment.label) {
              const text = make("text", {
                x: segment.label[0],
                y: segment.label[1],
                fill: segment.labelColor ?? "#ffffff",
                "font-size": 12,
                "font-weight": 700,
                "text-anchor": "middle",
                "paint-order": "stroke",
                stroke: "rgba(0, 0, 0, 0.7)",
                "stroke-width": 2,
              });
              text.textContent = String(segment.labelText ?? segment.segmentRef);
              text.style.pointerEvents = "none";
              segmentsGroup.appendChild(text);
            }
          });
        }

        if (currentStroke.length >= 2) {
          const draft = make("polyline", {
            points: pointsAttr(currentStroke),
            fill: "none",
            stroke: "#ffd166",
            "stroke-width": 3.2,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke",
          });
          draft.style.pointerEvents = "none";
          svg.appendChild(draft);
        }

        if (data.showVesselLabels) {
          const vesselLabelFontSize = data.vesselLabelFontSize ?? 16;
          (data.vesselLabels ?? []).forEach((label) => {
            const text = make("text", {
              x: label.position[0],
              y: label.position[1],
              fill: label.color ?? "#ffffff",
              "font-size": vesselLabelFontSize,
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
      }

      if (!readOnly) {
        svg.onpointerdown = beginDraw;
        svg.onpointermove = extendDraw;
        svg.onpointerup = finishDraw;
        svg.onpointercancel = finishDraw;
      }

      render();
      return () => {};
    }
    """,
)


NODE_ENDPOINT_VIEWER = st.components.v2.component(
    name="retina_geometry_endpoint_viewer",
    html="""
    <div class="endpoint-shell">
      <svg id="endpoint-viewer" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
    """,
    css="""
    html,
    body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #050505;
    }

    * {
      box-sizing: border-box;
    }

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
      touch-action: none;
    }
    """,
    js="""
    export default function(component) {
      const { parentElement, data, setStateValue } = component;
      const svg = parentElement.querySelector("#endpoint-viewer");
      const svgNs = "http://www.w3.org/2000/svg";
      let startEndpoint = data.startEndpoint ?? null;
      let endEndpoint = data.endEndpoint ?? null;
      let nextTarget = data.nextEndpointTarget ?? (startEndpoint == null ? "start" : "end");

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

      function pointsAttr(points) {
        return (points ?? []).map((point) => `${point[0]},${point[1]}`).join(" ");
      }

      function nearestPointOnPolyline(point, polyline, segmentRef) {
        let best = null;
        let traversed = 0;
        for (let index = 0; index < polyline.length - 1; index += 1) {
          const start = polyline[index];
          const end = polyline[index + 1];
          const dx = end[0] - start[0];
          const dy = end[1] - start[1];
          const length = Math.hypot(dx, dy);
          if (length === 0) {
            continue;
          }
          const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (length * length)));
          const projected = [start[0] + dx * t, start[1] + dy * t];
          const distance = Math.hypot(point[0] - projected[0], point[1] - projected[1]);
          const candidate = {
            kind: "geometry_point",
            point: projected,
            segment_ref: segmentRef,
            distance_from_start: traversed + t * length,
            distance,
          };
          if (best == null || candidate.distance < best.distance) {
            best = candidate;
          }
          traversed += length;
        }
        return best;
      }

      function setEndpoint(endpoint) {
        if (nextTarget === "start") {
          startEndpoint = endpoint;
          if (endEndpoint && Math.hypot(endEndpoint.point[0] - endpoint.point[0], endEndpoint.point[1] - endpoint.point[1]) < 1) {
            endEndpoint = null;
          }
          nextTarget = "end";
        } else {
          endEndpoint = endpoint;
          if (startEndpoint && Math.hypot(startEndpoint.point[0] - endpoint.point[0], startEndpoint.point[1] - endpoint.point[1]) < 1) {
            startEndpoint = null;
          }
          nextTarget = "start";
        }
        setStateValue("start_endpoint", startEndpoint);
        setStateValue("end_endpoint", endEndpoint);
        setStateValue("next_endpoint_target", nextTarget);
        render();
      }

      function clientPointToSvg(event) {
        const rect = svg.getBoundingClientRect();
        const viewBox = svg.viewBox.baseVal;
        const scaleX = viewBox.width / rect.width;
        const scaleY = viewBox.height / rect.height;
        return [
          viewBox.x + (event.clientX - rect.left) * scaleX,
          viewBox.y + (event.clientY - rect.top) * scaleY,
        ];
      }

      function renderMarker(endpoint, fill, labelText) {
        if (!endpoint || !endpoint.point) {
          return;
        }
        svg.appendChild(make("circle", {
          cx: endpoint.point[0],
          cy: endpoint.point[1],
          r: 5.5,
          fill,
          stroke: "#ffffff",
          "stroke-width": 2.5,
          "vector-effect": "non-scaling-stroke",
        }));
        const label = make("text", {
          x: endpoint.point[0],
          y: endpoint.point[1] - 8,
          fill: "#ffffff",
          "font-size": 12,
          "font-weight": 800,
          "text-anchor": "middle",
          "paint-order": "stroke",
          stroke: "rgba(0, 0, 0, 0.8)",
          "stroke-width": 2,
          "vector-effect": "non-scaling-stroke",
        });
        label.textContent = labelText;
        label.style.pointerEvents = "none";
        svg.appendChild(label);
      }

      function render() {
        clear(svg);
        const points = [];
        (data.segments ?? []).forEach((segment) => {
          (segment.points ?? []).forEach((point) => points.push(point));
        });

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
        svg.appendChild(make("rect", { x: minX, y: minY, width, height, fill: "#050505" }));

        if (data.imageUrl) {
          svg.appendChild(make("image", {
            x: 0,
            y: 0,
            width: imageWidth,
            height: imageHeight,
            href: data.imageUrl,
            opacity: data.baseOpacity ?? 0.55,
            preserveAspectRatio: "none",
          }));
        }

        (data.segments ?? []).forEach((segment) => {
          const segmentPoints = pointsAttr(segment.points ?? []);
          (segment.strokes ?? [{ color: "#00c2a8", width: 3 }]).forEach((strokeLayer) => {
            const polyline = make("polyline", {
              points: segmentPoints,
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

          const hitArea = make("polyline", {
            points: segmentPoints,
            fill: "none",
            stroke: "rgba(0, 0, 0, 0)",
            "stroke-width": 18,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke",
          });
          hitArea.addEventListener("click", (event) => {
            event.stopPropagation();
            const clickPoint = clientPointToSvg(event);
            const endpoint = nearestPointOnPolyline(clickPoint, segment.points ?? [], segment.segmentRef);
            if (endpoint) {
              delete endpoint.distance;
              setEndpoint(endpoint);
            }
          });
          svg.appendChild(hitArea);
        });

        renderMarker(startEndpoint, "#00c2a8", "S");
        renderMarker(endEndpoint, "#ffd166", "E");
      }

      render();
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
    del paths_payload
    geometry_map = get_segment_geometry(branches_df, review_state.get("manual_segments", {}))
    memberships: dict[str, list[str]] = {segment_ref: [] for segment_ref in geometry_map}
    segment_vessels: dict[str, list[str]] = {segment_ref: [] for segment_ref in geometry_map}
    vessel_segment_refs = {
        vessel_name: segment_refs_for_vessel(vessel)
        for vessel_name, vessel in review_state.get("vessels", {}).items()
    }
    assigned_refs: set[str] = set()
    for vessel_name, vessel in review_state.get("vessels", {}).items():
        color = ARTERE_COLOR if vessel.get("category") == "artere" else VEINE_COLOR
        for segment_ref in segment_refs_for_vessel(vessel):
            memberships.setdefault(segment_ref, []).append(color)
            segment_vessels.setdefault(segment_ref, []).append(vessel_name)
            assigned_refs.add(segment_ref)

    selected_refs = set(review_state.get("selected_segment_refs", []))
    viewer_segments: list[dict] = []
    for segment_ref in sorted(geometry_map, key=segment_ref_sort_key):
        geometry = geometry_map[segment_ref]
        is_selected = segment_ref in selected_refs
        membership_colors = list(dict.fromkeys(memberships.get(segment_ref, [])))
        strokes: list[dict[str, float | str]] = []
        if membership_colors:
            base_width = 5.6 if len(membership_colors) > 1 else 3.2
            width_step = 1.4 if len(membership_colors) > 1 else 0.0
            for index, color in enumerate(membership_colors):
                strokes.append({"color": color, "width": max(2.5, base_width - index * width_step)})
        else:
            strokes.append(
                {
                    "color": _segment_color(str(geometry["vascx_category"])),
                    "width": 2.6 if geometry["source"] == "model" else 2.9,
                    "dasharray": "" if geometry["source"] == "model" else "7 5",
                    "opacity": 0.95,
                }
            )
        if is_selected:
            strokes.append({"color": SELECTED_COLOR, "width": 3.0})

        viewer_segments.append(
            {
                "segmentRef": segment_ref,
                "source": geometry["source"],
                "labelText": str(geometry["id"]),
                "vesselSegmentRefs": sorted(
                    {
                        linked_ref
                        for vessel_name in segment_vessels.get(segment_ref, [])
                        for linked_ref in vessel_segment_refs.get(vessel_name, [])
                    },
                    key=segment_ref_sort_key,
                ),
                "vesselNames": segment_vessels.get(segment_ref, []),
                "points": geometry["points"],
                "label": geometry["label_position"],
                "labelColor": SELECTED_COLOR if is_selected else "#ffffff",
                "locked": (segment_ref in assigned_refs) and not is_selected and not allow_reuse_assigned,
                "strokes": strokes,
            }
        )

    synthetic_index = 0
    for vessel in review_state.get("vessels", {}).values():
        color = ARTERE_COLOR if vessel.get("category") == "artere" else VEINE_COLOR
        for synthetic_link in vessel.get("synthetic_links", []):
            viewer_segments.append(_synthetic_viewer_segment(f"saved-synthetic:{synthetic_index}", synthetic_link, color))
            synthetic_index += 1
    for synthetic_link in provisional_synthetic_links or []:
        viewer_segments.append(_synthetic_viewer_segment(f"provisional-synthetic:{synthetic_index}", synthetic_link, SELECTED_COLOR))
        synthetic_index += 1
    return viewer_segments


def build_vessel_labels(
    branches_df: pd.DataFrame,
    paths_payload: list[dict],
    review_state: dict,
) -> list[dict]:
    del paths_payload
    geometry_map = get_segment_geometry(branches_df, review_state.get("manual_segments", {}))
    labels: list[dict] = []
    for vessel_name, vessel in sorted(review_state.get("vessels", {}).items()):
        points: list[list[float]] = []
        for segment_ref in segment_refs_for_vessel(vessel):
            geometry = geometry_map.get(segment_ref)
            if geometry is not None:
                points.extend(geometry["points"])
        for synthetic_link in vessel.get("synthetic_links", []):
            points.extend(synthetic_link.get("points", []))
        if not points:
            continue
        point_df = pd.DataFrame(points, columns=["x", "y"])
        labels.append(
            {
                "text": vessel_name,
                "position": [float(point_df["x"].mean()), float(point_df["y"].mean())],
                "color": ARTERE_COLOR if vessel.get("category") == "artere" else VEINE_COLOR,
            }
        )
    return labels


def _synthetic_viewer_segment(segment_ref: str, synthetic_link: dict[str, object], color: str) -> dict:
    return {
        "segmentRef": segment_ref,
        "source": "synthetic",
        "points": synthetic_link["points"],
        "label": None,
        "labelColor": "#ffffff",
        "locked": True,
        "strokes": [
            {"color": color, "width": 5.2, "opacity": 0.45},
            {"color": color, "width": 2.6, "opacity": 1},
        ],
    }
