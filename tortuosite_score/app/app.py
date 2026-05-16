from __future__ import annotations

import base64
import io
import json
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skan import Skeleton

from tortuosite_score.vessels_detection.analyse import skeletonize_mask
from tortuosite_score.vessels_detection.main import run_pipeline


st.set_page_config(page_title="Tortuosite Retine", layout="wide")
st.title("Retinal Vessel Review")
st.write(
    "Run the segmentation once, then review the extracted skeleton manually, "
    "define vessels, classify them, and score them from the interface."
)

ARTERE_COLOR = "#ff453a"
VEINE_COLOR = "#4c8dff"
SELECTED_COLOR = "#ffd60a"
DEFAULT_BRANCH_COLOR = "#ff7a70"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "demo" / "streamlit_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

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
        setStateValue(
          "selected_branch_ids",
          currentSelection,
        );
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
          polyline.style.cursor = "pointer";
          polyline.addEventListener("click", (event) => {
            event.stopPropagation();
            toggleBranch(branch.branchId);
          });
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


def _slugify_name(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cleaned.strip("_") or "image"


def _list_runs() -> list[Path]:
    return sorted(
        (path for path in RUNS_ROOT.iterdir() if path.is_dir()),
        reverse=True,
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _image_to_data_url(path: Path) -> str:
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
def _load_review_bundle(run_dir_str: str) -> dict:
    run_dir = Path(run_dir_str)
    metadata = _read_json(run_dir / "metadata.json")
    output_dir = run_dir / "output"
    image_name = metadata.get("image_name")
    image_path = run_dir / image_name if image_name else None

    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Source image not found for run {run_dir.name}")

    cleaned_mask = cv2.imread(str(output_dir / "06_cleaned_mask.png"), cv2.IMREAD_GRAYSCALE)
    if cleaned_mask is None:
        raise FileNotFoundError(f"Missing cleaned mask for run {run_dir.name}")

    skeleton = skeletonize_mask(cleaned_mask > 0)
    skeleton_graph = Skeleton(skeleton)

    summary_path = output_dir / "08_full_skeleton_summary.csv"
    branches_df = pd.read_csv(summary_path).copy()
    branch_count = min(len(branches_df), skeleton_graph.n_paths)
    branches_df = branches_df.iloc[:branch_count].copy()
    branches_df["branch_id"] = np.arange(branch_count, dtype=int)

    paths_payload: list[dict] = []
    for branch_id in range(branch_count):
        coords = skeleton_graph.path_coordinates(branch_id)
        if coords.shape[0] < 2:
            continue
        centroid = coords.mean(axis=0)
        paths_payload.append(
            {
                "branchId": int(branch_id),
                "points": [[int(col), int(row)] for row, col in coords],
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
        "image_url": _image_to_data_url(image_path),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "branches_df": branches_df.to_dict(orient="records"),
        "paths_payload": paths_payload,
    }


def _get_or_create_review_state(run_dir: Path, bundle: dict) -> dict:
    state_key = f"review_state::{run_dir.name}"
    if state_key not in st.session_state:
        saved_path = run_dir / "manual_review_state.json"
        if saved_path.exists():
            state = _read_json(saved_path)
        else:
            state = {}
        st.session_state[state_key] = {
            "selected_branch_ids": state.get("selected_branch_ids", []),
            "vessels": state.get("vessels", {}),
        }
    return st.session_state[state_key]


def _score_vessel(branches_df: pd.DataFrame, branch_ids: list[int]) -> dict[str, object]:
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


def _persist_manual_review(run_dir: Path, state: dict, branches_df: pd.DataFrame) -> None:
    state_path = run_dir / "manual_review_state.json"
    state_path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for vessel_name, vessel in state["vessels"].items():
        metrics = _score_vessel(branches_df, vessel["branch_ids"])
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


def _render_debug_tab(run_dir: Path) -> None:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        st.subheader("Run metadata")
        st.json(_read_json(metadata_path), expanded=False)

    manual_csv = run_dir / "manual_vessels.csv"
    if manual_csv.exists():
        st.subheader("Saved manual vessels")
        st.dataframe(pd.read_csv(manual_csv), use_container_width=True)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        st.subheader("Legacy auto-selection output")
        st.dataframe(pd.read_csv(results_csv), use_container_width=True)

    logs_path = run_dir / "logs.txt"
    if logs_path.exists():
        logs = logs_path.read_text(encoding="utf-8").strip()
        if logs:
            st.subheader("Run logs")
            st.code(logs, language="text")

    output_dir = run_dir / "output"
    image_files = sorted(output_dir.glob("*.png"))
    if image_files:
        st.subheader("Intermediate outputs")
        for image_file in image_files:
            st.image(str(image_file), caption=image_file.name, use_container_width=True)


with st.sidebar:
    st.header("Run setup")
    uploaded_file = st.file_uploader(
        "Retinal image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Upload a fundus image to generate a vessel skeleton for manual review.",
    )
    method = st.selectbox(
        "Segmentation method",
        options=["deep", "classical"],
        index=0,
        help="Choose the segmentation mode used to generate the review skeleton.",
    )

    if method == "deep":
        deep_threshold = st.slider(
            "Deep threshold",
            0.0,
            1.0,
            0.30,
            0.01,
            help="Probability cutoff applied to the neural segmentation.",
        )
        deep_modality = st.selectbox(
            "Deep modality",
            options=["CFP", "UWF", "FFA", "SLO", "OCTA"],
            index=0,
        )
        vessel_percentile = 95.0
        vessel_low_percentile = 90.0
    else:
        vessel_percentile = st.slider("Vessel percentile", 50.0, 99.9, 95.0, 0.1)
        vessel_low_percentile = st.slider("Vessel low percentile", 0.0, 99.0, 90.0, 0.1)
        deep_threshold = 0.30
        deep_modality = "CFP"

    run_btn = st.button(
        "Run segmentation",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

if run_btn and uploaded_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_slugify_name(uploaded_file.name)}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image_path = run_dir / uploaded_file.name
    image_path.write_bytes(uploaded_file.getvalue())

    output_csv = run_dir / "results.csv"
    output_dir = run_dir / "output"
    log_buffer = io.StringIO()

    with st.spinner("Running segmentation and skeleton extraction..."):
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
            )

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "image_name": uploaded_file.name,
        "method": method,
        "vessel_percentile": float(vessel_percentile),
        "vessel_low_percentile": float(vessel_low_percentile),
        "deep_threshold": float(deep_threshold),
        "deep_modality": deep_modality,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "logs.txt").write_text(log_buffer.getvalue(), encoding="utf-8")
    st.session_state["active_run_id"] = run_id
    st.rerun()

runs = _list_runs()
if not runs:
    st.info("Upload a retinal image and run the segmentation to start the manual review workflow.")
    st.stop()

run_names = [run.name for run in runs]
default_run_name = st.session_state.get("active_run_id", run_names[0])
default_index = run_names.index(default_run_name) if default_run_name in run_names else 0

selected_run_name = st.selectbox(
    "Working run",
    options=run_names,
    index=default_index,
    help="Choose which processed image you want to review.",
)
st.session_state["active_run_id"] = selected_run_name
selected_run_dir = RUNS_ROOT / selected_run_name
bundle = _load_review_bundle(str(selected_run_dir))
review_state = _get_or_create_review_state(selected_run_dir, bundle)
branches_df = pd.DataFrame(bundle["branches_df"])

selected_branch_ids = sorted(int(branch_id) for branch_id in review_state["selected_branch_ids"])

main_tab, debug_tab = st.tabs(["Manual Review", "Debug Outputs"])

with main_tab:
    top_controls_left, top_controls_mid, top_controls_right = st.columns(
        [1.1, 1.1, 1.4],
        gap="large",
    )

    with top_controls_left:
        st.subheader("Viewer")
        show_base_image = st.toggle("Show base image", value=True)
        show_skeleton = st.toggle("Show skeleton", value=True)
        show_labels = st.toggle("Show branch IDs", value=False)
        base_opacity = st.slider("Base image opacity", 0.0, 1.0, 0.95, 0.05)

    with top_controls_mid:
        st.subheader("Selection")
        st.write(f"{len(selected_branch_ids)} branch(es) selected")
        selection_actions_left, selection_actions_right = st.columns(2)
        with selection_actions_left:
            if st.button("Clear selection", use_container_width=True):
                review_state["selected_branch_ids"] = []
                st.rerun()
        with selection_actions_right:
            if st.button("Select all", use_container_width=True):
                review_state["selected_branch_ids"] = branches_df["branch_id"].astype(int).tolist()
                st.rerun()

    with top_controls_right:
        st.subheader("Current vessel")
        vessel_name = st.text_input("Vessel name", value="")
        vessel_category = st.radio(
            "Vessel category",
            options=["artere", "veine"],
            horizontal=True,
        )
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button("Save current vessel", type="primary", use_container_width=True):
                clean_name = vessel_name.strip()
                if not clean_name:
                    st.warning("Choose a vessel name before saving.")
                elif not selected_branch_ids:
                    st.warning("Select at least one branch before saving a vessel.")
                else:
                    review_state["vessels"][clean_name] = {
                        "category": vessel_category,
                        "branch_ids": selected_branch_ids,
                    }
                    _persist_manual_review(selected_run_dir, review_state, branches_df)
                    st.success(f"Saved vessel `{clean_name}`.")
        with delete_col:
            if st.button("Delete vessel", use_container_width=True):
                clean_name = vessel_name.strip()
                if clean_name and clean_name in review_state["vessels"]:
                    del review_state["vessels"][clean_name]
                    _persist_manual_review(selected_run_dir, review_state, branches_df)
                    st.rerun()

        vessel_names = sorted(review_state["vessels"])
        if vessel_names:
            vessel_to_load = st.selectbox("Saved vessels", options=vessel_names)
            if st.button("Load vessel selection", use_container_width=True):
                review_state["selected_branch_ids"] = list(
                    review_state["vessels"][vessel_to_load]["branch_ids"]
                )
                st.rerun()

    branch_color_lookup = {
        branch_id: DEFAULT_BRANCH_COLOR for branch_id in branches_df["branch_id"].astype(int).tolist()
    }
    for vessel in review_state["vessels"].values():
        color = ARTERE_COLOR if vessel["category"] == "artere" else VEINE_COLOR
        for branch_id in vessel["branch_ids"]:
            branch_color_lookup[int(branch_id)] = color

    selected_branch_set = set(selected_branch_ids)
    viewer_branches = []
    path_map = {item["branchId"]: item for item in bundle["paths_payload"]}
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
                "stroke": "#ffd60a" if is_selected else color,
                "strokeWidth": 4.8 if is_selected else 2.4,
                "labelColor": "#ffd60a" if is_selected else "#ffffff",
            }
        )

    st.subheader("Interactive skeleton viewer")
    viewer_result = BRANCH_VIEWER(
        data={
            "imageUrl": bundle["image_url"],
            "imageWidth": bundle["image_width"],
            "imageHeight": bundle["image_height"],
            "branches": viewer_branches,
            "selectedBranchIds": selected_branch_ids,
            "showBaseImage": show_base_image,
            "showSkeleton": show_skeleton,
            "showLabels": show_labels,
            "baseOpacity": base_opacity,
        },
        default={
            "selected_branch_ids": selected_branch_ids,
        },
        on_selected_branch_ids_change=lambda: None,
        key=f"branch_viewer::{selected_run_name}",
        width="stretch",
        height=860,
    )

    previous_selection = list(review_state["selected_branch_ids"])
    state_changed = False
    if getattr(viewer_result, "selected_branch_ids", None) is not None:
        updated_selection = sorted(
            int(branch_id) for branch_id in viewer_result.selected_branch_ids
        )
        if updated_selection != previous_selection:
            review_state["selected_branch_ids"] = updated_selection
            state_changed = True

    if state_changed:
        _persist_manual_review(selected_run_dir, review_state, branches_df)
        st.rerun()

    selected_branch_ids = sorted(int(branch_id) for branch_id in review_state["selected_branch_ids"])
    selection_df = branches_df[branches_df["branch_id"].isin(selected_branch_ids)].copy()
    if selection_df.empty:
        st.info("Click branches in the viewer to build a vessel selection.")
    else:
        selection_df = selection_df[
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
        st.subheader("Current selection")
        st.dataframe(selection_df, use_container_width=True, hide_index=True)

    vessel_rows: list[dict[str, object]] = []
    for vessel_name, vessel in sorted(review_state["vessels"].items()):
        metrics = _score_vessel(branches_df, vessel["branch_ids"])
        vessel_rows.append(
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

    if vessel_rows:
        vessel_df = pd.DataFrame(vessel_rows)
        st.subheader("Saved vessel scores")
        st.dataframe(vessel_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download vessel scores",
            data=vessel_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_name}_manual_vessels.csv",
            mime="text/csv",
        )
        disconnected = vessel_df[vessel_df["Components"] > 1]
        if not disconnected.empty:
            st.warning(
                "Some saved vessels are disconnected. Their score is computed on the largest connected component."
            )

with debug_tab:
    _render_debug_tab(selected_run_dir)
