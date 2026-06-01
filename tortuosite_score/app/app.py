from __future__ import annotations

import pandas as pd
import streamlit as st

from tortuosite_score.app.constants import RUNS_ROOT
from tortuosite_score.app.review_data import (
    list_runs,
    load_review_bundle,
    run_uploaded_analysis,
    slugify_name,
)
from tortuosite_score.app.review_state import (
    build_selection_table,
    build_vessel_payload,
    build_vessel_scores_table,
    get_or_create_review_state,
    next_default_vessel_name,
    normalize_selection_refs,
    parse_segment_ref,
    persist_manual_review,
    push_selection_history,
    redo_selection,
    remove_manual_segment,
    score_vessel,
    segment_ref_sort_key,
    segment_refs_for_review_state,
    segment_refs_for_vessel,
    synthesize_selection_links,
    undo_selection,
    upsert_manual_segment,
)
from tortuosite_score.app.ui_sections import render_debug_tab, render_sidebar_run_setup
from tortuosite_score.app.viewer_component import (
    BRANCH_VIEWER,
    NODE_ENDPOINT_VIEWER,
    build_vessel_labels,
    build_viewer_branches,
)

VIEWER_SELECTION_MODE_KEY = "viewer_selection_mode"
VIEWER_INTERACTION_MODE_KEY = "viewer_interaction_mode"
SELECTION_HISTORY_LIMIT = 50


st.set_page_config(page_title="Tortuosite Retine", layout="wide")
st.title("Retinal Vessel Review")
st.write(
    "Run the segmentation once, then review the extracted skeleton manually, "
    "define vessels, classify them, and score them from the interface."
)


def _handle_run_creation(sidebar_values: dict) -> None:
    if not sidebar_values["run_btn"] or sidebar_values["uploaded_file"] is None:
        return
    image_run_id = slugify_name(sidebar_values["uploaded_file"].name)
    st.session_state.pop(f"review_state::{image_run_id}", None)
    run_id = run_uploaded_analysis(
        uploaded_file=sidebar_values["uploaded_file"],
        method=sidebar_values["method"],
        vessel_percentile=sidebar_values["vessel_percentile"],
        vessel_low_percentile=sidebar_values["vessel_low_percentile"],
        deep_threshold=sidebar_values["deep_threshold"],
        deep_modality=sidebar_values["deep_modality"],
        deep_backend=sidebar_values["deep_backend"],
        vascx_av_size=sidebar_values["vascx_av_size"],
        vascx_use_contrast_enhancement=sidebar_values["vascx_use_contrast_enhancement"],
        vascx_min_object_size=sidebar_values["vascx_min_object_size"],
        vascx_closing_radius=sidebar_values["vascx_closing_radius"],
        vascx_auto_create_vessels=sidebar_values["vascx_auto_create_vessels"],
        vascx_auto_min_vessel_length=sidebar_values["vascx_auto_min_vessel_length"],
    )
    st.session_state["active_run_id"] = run_id
    st.rerun()


def _render_run_selector() -> str | None:
    runs = list_runs()
    if not runs:
        st.info("Upload a retinal image and run the segmentation to start the manual review workflow.")
        return None
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
    return selected_run_name


def _selection_history_keys(selected_run_name: str) -> tuple[str, str]:
    return f"selection_undo::{selected_run_name}", f"selection_redo::{selected_run_name}"


def _selection_history_stacks(undo_key: str, redo_key: str) -> tuple[list[list[str]], list[list[str]]]:
    st.session_state.setdefault(undo_key, [])
    st.session_state.setdefault(redo_key, [])
    return st.session_state[undo_key], st.session_state[redo_key]


def _clear_selection_history(undo_key: str, redo_key: str) -> None:
    undo_stack, redo_stack = _selection_history_stacks(undo_key, redo_key)
    undo_stack.clear()
    redo_stack.clear()


def _set_selected_segment_refs(
    review_state: dict,
    next_refs: list[str],
    undo_key: str,
    redo_key: str,
    viewer_reset_key: str | None = None,
) -> bool:
    previous_refs = segment_refs_for_review_state(review_state)
    normalized_next = normalize_selection_refs(next_refs)
    undo_stack, redo_stack = _selection_history_stacks(undo_key, redo_key)
    changed = push_selection_history(
        undo_stack,
        redo_stack,
        previous_refs,
        normalized_next,
        SELECTION_HISTORY_LIMIT,
    )
    if not changed:
        return False
    review_state["selected_segment_refs"] = normalized_next
    if viewer_reset_key is not None:
        st.session_state[viewer_reset_key] += 1
    return True


def _record_existing_selection_change(
    review_state: dict,
    previous_refs: list[str],
    undo_key: str,
    redo_key: str,
    viewer_reset_key: str | None = None,
) -> bool:
    next_refs = segment_refs_for_review_state(review_state)
    undo_stack, redo_stack = _selection_history_stacks(undo_key, redo_key)
    changed = push_selection_history(
        undo_stack,
        redo_stack,
        previous_refs,
        next_refs,
        SELECTION_HISTORY_LIMIT,
    )
    if changed and viewer_reset_key is not None:
        st.session_state[viewer_reset_key] += 1
    return changed


def _render_selection_history_controls(
    review_state: dict,
    undo_key: str,
    redo_key: str,
    viewer_reset_key: str,
) -> None:
    undo_stack, redo_stack = _selection_history_stacks(undo_key, redo_key)
    undo_col, redo_col, count_col = st.columns([1.0, 1.0, 2.5], vertical_alignment="center")
    with undo_col:
        if st.button("Undo", disabled=not undo_stack, use_container_width=True):
            review_state["selected_segment_refs"] = undo_selection(
                undo_stack,
                redo_stack,
                segment_refs_for_review_state(review_state),
            )
            st.session_state[viewer_reset_key] += 1
            st.rerun()
    with redo_col:
        if st.button("Redo", disabled=not redo_stack, use_container_width=True):
            review_state["selected_segment_refs"] = redo_selection(
                undo_stack,
                redo_stack,
                segment_refs_for_review_state(review_state),
            )
            st.session_state[viewer_reset_key] += 1
            st.rerun()
    with count_col:
        st.caption(f"{len(undo_stack)} undo step(s), {len(redo_stack)} redo step(s)")


def _clear_vessel_draft(
    review_state: dict,
    undo_key: str,
    redo_key: str,
    editing_vessel_name_key: str,
    editing_original_snapshot_key: str,
    vessel_name_reset_key: str,
    viewer_reset_key: str,
    start_endpoint_key: str,
    end_endpoint_key: str,
    next_endpoint_target_key: str,
) -> None:
    _set_selected_segment_refs(review_state, [], undo_key, redo_key)
    st.session_state.pop(editing_vessel_name_key, None)
    st.session_state.pop(editing_original_snapshot_key, None)
    st.session_state.pop(start_endpoint_key, None)
    st.session_state.pop(end_endpoint_key, None)
    st.session_state.pop(next_endpoint_target_key, None)
    st.session_state[vessel_name_reset_key] = True
    st.session_state[viewer_reset_key] += 1


def _start_vessel_edit(
    vessel_name: str,
    review_state: dict,
    undo_key: str,
    redo_key: str,
    editing_vessel_name_key: str,
    editing_original_snapshot_key: str,
    vessel_name_key: str,
    vessel_category_key: str,
    start_endpoint_key: str,
    end_endpoint_key: str,
) -> None:
    vessel = review_state["vessels"][vessel_name]
    _set_selected_segment_refs(review_state, segment_refs_for_vessel(vessel), undo_key, redo_key)
    st.session_state[editing_vessel_name_key] = vessel_name
    st.session_state[editing_original_snapshot_key] = {
        "name": vessel_name,
        "category": vessel["category"],
        "segment_refs": segment_refs_for_vessel(vessel),
        "synthetic_links": list(vessel.get("synthetic_links", [])),
        "start_endpoint": vessel.get("start_endpoint"),
        "end_endpoint": vessel.get("end_endpoint"),
    }
    st.session_state[vessel_name_key] = vessel_name
    st.session_state[vessel_category_key] = vessel["category"]
    if vessel.get("start_endpoint") is not None:
        st.session_state[start_endpoint_key] = vessel["start_endpoint"]
    if vessel.get("end_endpoint") is not None:
        st.session_state[end_endpoint_key] = vessel["end_endpoint"]


def _resolve_clean_vessel_name(vessel_name: str, vessel_category: str, vessels: dict[str, dict]) -> str:
    return vessel_name.strip() or next_default_vessel_name(vessels, vessel_category)


def _selected_complete_vessel_names(review_state: dict, selected_segment_refs: list[str]) -> list[str]:
    selected_set = set(selected_segment_refs)
    return [
        vessel_name
        for vessel_name, vessel in sorted(review_state.get("vessels", {}).items())
        if set(segment_refs_for_vessel(vessel)) and set(segment_refs_for_vessel(vessel)).issubset(selected_set)
    ]


def _dominant_category_for_vessels(
    branches_df: pd.DataFrame,
    review_state: dict,
    vessel_names: list[str],
) -> str:
    category_lengths = {"artere": 0.0, "veine": 0.0}
    for vessel_name in vessel_names:
        vessel = review_state["vessels"][vessel_name]
        metrics = score_vessel(branches_df, review_state.get("manual_segments", {}), vessel)
        category_lengths[vessel["category"]] = category_lengths.get(vessel["category"], 0.0) + float(
            metrics["length"] if pd.notna(metrics["length"]) else 0.0
        )
    return "veine" if category_lengths["veine"] > category_lengths["artere"] else "artere"


def _show_saved_vessel_status(vessel_name: str, resolution: dict) -> None:
    if len(resolution.get("synthetic_links", [])) > 0:
        st.success(f"Saved vessel `{vessel_name}` with {len(resolution['synthetic_links'])} synthetic link(s).")
    elif not resolution.get("bridge_success", True):
        st.warning(f"Saved vessel `{vessel_name}` with unresolved disconnected pieces.")
    else:
        st.success(f"Saved vessel `{vessel_name}`.")


def _single_manual_selection_ref(selected_segment_refs: list[str]) -> str | None:
    manual_refs = [segment_ref for segment_ref in selected_segment_refs if segment_ref.startswith("manual:")]
    return manual_refs[0] if len(manual_refs) == 1 else None


def _sync_viewer_state(
    viewer_result,
    review_state: dict,
    selected_run_dir,
    branches_df: pd.DataFrame,
    undo_key: str,
    redo_key: str,
    viewer_reset_key: str,
    redraw_target_ref: str | None,
) -> None:
    next_selection = getattr(viewer_result, "selected_segment_refs", None)
    if next_selection is not None:
        normalized = normalize_selection_refs([str(segment_ref) for segment_ref in next_selection])
        if _set_selected_segment_refs(review_state, normalized, undo_key, redo_key):
            st.rerun()

    draw_action = getattr(viewer_result, "draw_action", None)
    drawn_points = getattr(viewer_result, "drawn_segment_points", None)
    if draw_action not in {"create", "redraw"} or not isinstance(drawn_points, list):
        return

    if draw_action == "redraw" and redraw_target_ref and redraw_target_ref.startswith("manual:"):
        _, manual_id = parse_segment_ref(redraw_target_ref)
        updated_ref = upsert_manual_segment(review_state, drawn_points, manual_id)
        if updated_ref is None:
            st.warning("The redrawn segment was too short to keep.")
        else:
            current = [ref for ref in review_state["selected_segment_refs"] if ref != redraw_target_ref]
            current.append(updated_ref)
            _set_selected_segment_refs(review_state, current, undo_key, redo_key)
            persist_manual_review(selected_run_dir, review_state, branches_df)
        st.session_state[viewer_reset_key] += 1
        st.rerun()

    if draw_action == "create":
        created_ref = upsert_manual_segment(review_state, drawn_points)
        if created_ref is None:
            st.warning("The drawn segment was too short to keep.")
        else:
            current = list(review_state["selected_segment_refs"])
            current.append(created_ref)
            _set_selected_segment_refs(review_state, current, undo_key, redo_key)
            persist_manual_review(selected_run_dir, review_state, branches_df)
        st.session_state[viewer_reset_key] += 1
        st.rerun()


def _sync_endpoint_selection(
    endpoint_result,
    start_endpoint_key: str,
    end_endpoint_key: str,
    next_endpoint_target_key: str,
) -> bool:
    changed = False
    if getattr(endpoint_result, "start_endpoint", None) is not None:
        endpoint = endpoint_result.start_endpoint
        if st.session_state.get(start_endpoint_key) != endpoint:
            st.session_state[start_endpoint_key] = endpoint
            changed = True
    if getattr(endpoint_result, "end_endpoint", None) is not None:
        endpoint = endpoint_result.end_endpoint
        if st.session_state.get(end_endpoint_key) != endpoint:
            st.session_state[end_endpoint_key] = endpoint
            changed = True
    if getattr(endpoint_result, "next_endpoint_target", None) in {"start", "end"}:
        next_target = endpoint_result.next_endpoint_target
        if st.session_state.get(next_endpoint_target_key) != next_target:
            st.session_state[next_endpoint_target_key] = next_target
            changed = True
    if changed:
        st.rerun()
    return changed


def _render_viewer_controls() -> dict[str, object]:
    with st.sidebar:
        st.header("Viewer")
        show_base_image = st.toggle("Show base image", value=True)
        show_skeleton = st.toggle("Show skeleton", value=True)
        show_labels = st.toggle("Show segment IDs", value=False)
        show_vessel_labels = st.toggle("Show vessel IDs", value=False)
        base_opacity = st.slider("Base image opacity", 0.0, 1.0, 0.95, 0.05)

    st.session_state.setdefault(VIEWER_SELECTION_MODE_KEY, "Segment parts")
    st.session_state.setdefault(VIEWER_INTERACTION_MODE_KEY, "Select")
    if st.session_state[VIEWER_INTERACTION_MODE_KEY] not in {"Select", "Draw new"}:
        st.session_state[VIEWER_INTERACTION_MODE_KEY] = "Select"
    return {
        "show_base_image": show_base_image,
        "show_skeleton": show_skeleton,
        "show_labels": show_labels,
        "show_vessel_labels": show_vessel_labels,
        "selection_mode": st.session_state[VIEWER_SELECTION_MODE_KEY],
        "interaction_mode_label": st.session_state[VIEWER_INTERACTION_MODE_KEY],
        "base_opacity": base_opacity,
    }


def _render_image_interaction_controls() -> None:
    with st.container(border=True):
        st.radio(
            "Interaction mode",
            options=["Select", "Draw new"],
            horizontal=True,
            key=VIEWER_INTERACTION_MODE_KEY,
        )


@st.fragment
def _render_manual_review(selected_run_name: str, viewer_options: dict[str, object]) -> None:
    selected_run_dir = RUNS_ROOT / selected_run_name
    bundle = load_review_bundle(str(selected_run_dir))
    branches_df = pd.DataFrame(bundle["branches_df"])
    review_state = get_or_create_review_state(selected_run_dir, branches_df=branches_df)
    if review_state.get("legacy_state_ignored"):
        st.warning("An older manual review file exists for this run. It was ignored because this version uses the v2 segment schema.")

    vessel_name_key = f"vessel_name_input::{selected_run_name}"
    vessel_category_key = f"vessel_category_input::{selected_run_name}"
    vessel_load_key = f"saved_vessel_select::{selected_run_name}"
    vessel_name_reset_key = f"vessel_name_reset::{selected_run_name}"
    viewer_reset_key = f"viewer_reset_nonce::{selected_run_name}"
    pending_edit_key = f"pending_vessel_edit::{selected_run_name}"
    editing_vessel_name_key = f"editing_vessel_name::{selected_run_name}"
    editing_original_snapshot_key = f"editing_original_snapshot::{selected_run_name}"
    start_endpoint_key = f"vessel_start_endpoint::{selected_run_name}"
    end_endpoint_key = f"vessel_end_endpoint::{selected_run_name}"
    next_endpoint_target_key = f"next_endpoint_target::{selected_run_name}"
    selection_undo_key, selection_redo_key = _selection_history_keys(selected_run_name)
    _selection_history_stacks(selection_undo_key, selection_redo_key)

    if st.session_state.get(vessel_name_reset_key):
        st.session_state[vessel_name_key] = ""
        st.session_state[vessel_category_key] = "artere"
        st.session_state[vessel_name_reset_key] = False
    st.session_state.setdefault(vessel_name_key, "")
    st.session_state.setdefault(vessel_category_key, "artere")
    st.session_state.setdefault(viewer_reset_key, 0)
    st.session_state.setdefault(next_endpoint_target_key, "start")

    pending_vessel_edit = st.session_state.pop(pending_edit_key, None)
    if pending_vessel_edit in review_state["vessels"]:
        _start_vessel_edit(
            pending_vessel_edit,
            review_state,
            selection_undo_key,
            selection_redo_key,
            editing_vessel_name_key,
            editing_original_snapshot_key,
            vessel_name_key,
            vessel_category_key,
            start_endpoint_key,
            end_endpoint_key,
        )

    editing_vessel_name = st.session_state.get(editing_vessel_name_key)
    if editing_vessel_name and editing_vessel_name not in review_state["vessels"]:
        _clear_vessel_draft(
            review_state,
            selection_undo_key,
            selection_redo_key,
            editing_vessel_name_key,
            editing_original_snapshot_key,
            vessel_name_reset_key,
            viewer_reset_key,
            start_endpoint_key,
            end_endpoint_key,
            next_endpoint_target_key,
        )
        editing_vessel_name = None
    is_editing_vessel = editing_vessel_name is not None

    show_base_image = bool(viewer_options["show_base_image"])
    show_skeleton = bool(viewer_options["show_skeleton"])
    show_labels = bool(viewer_options["show_labels"])
    show_vessel_labels = bool(viewer_options["show_vessel_labels"])
    selection_mode = str(viewer_options["selection_mode"])
    interaction_mode_label = str(viewer_options["interaction_mode_label"])
    base_opacity = float(viewer_options["base_opacity"])

    selected_segment_refs = segment_refs_for_review_state(review_state)
    redraw_target_ref = _single_manual_selection_ref(selected_segment_refs)

    provisional_resolution = (
        synthesize_selection_links(branches_df, review_state["manual_segments"], selected_segment_refs)
        if selected_segment_refs
        else {"synthetic_links": []}
    )
    viewer_segments = build_viewer_branches(
        branches_df=branches_df,
        paths_payload=bundle["paths_payload"],
        review_state=review_state,
        allow_reuse_assigned=True,
        provisional_synthetic_links=provisional_resolution["synthetic_links"],
    )
    interaction_mode = {"Select": "select", "Draw new": "draw"}[interaction_mode_label]
    selected_start_endpoint = st.session_state.get(start_endpoint_key)
    selected_end_endpoint = st.session_state.get(end_endpoint_key)
    endpoint_segments = [
        segment
        for segment in viewer_segments
        if segment.get("segmentRef") in set(selected_segment_refs)
        or str(segment.get("segmentRef", "")).startswith("provisional-synthetic:")
    ]

    if selected_segment_refs:
        viewer_col, endpoint_col = st.columns([2.35, 1.0], gap="large", vertical_alignment="top")
    else:
        viewer_col = st.container()
        endpoint_col = None

    with viewer_col:
        _render_selection_history_controls(review_state, selection_undo_key, selection_redo_key, viewer_reset_key)
        viewer_result = BRANCH_VIEWER(
            data={
                "imageUrl": bundle["image_url"],
                "imageWidth": bundle["image_width"],
                "imageHeight": bundle["image_height"],
                "segments": viewer_segments,
                "selectedSegmentRefs": selected_segment_refs,
                "selectionMode": "vessel" if selection_mode == "Whole vessels" else "segment",
                "interactionMode": interaction_mode,
                "showBaseImage": show_base_image,
                "showSkeleton": show_skeleton,
                "showLabels": show_labels,
                "showVesselLabels": show_vessel_labels,
                "vesselLabels": build_vessel_labels(branches_df, bundle["paths_payload"], review_state),
                "baseOpacity": base_opacity,
            },
            default={"selected_segment_refs": selected_segment_refs},
            on_selected_segment_refs_change=lambda: None,
            key=f"branch_viewer::{selected_run_name}::{st.session_state[viewer_reset_key]}",
            width="stretch",
            height=720,
        )
        _render_image_interaction_controls()
    _sync_viewer_state(
        viewer_result,
        review_state,
        selected_run_dir,
        branches_df,
        selection_undo_key,
        selection_redo_key,
        viewer_reset_key,
        redraw_target_ref,
    )

    if endpoint_col is not None:
        with endpoint_col:
            st.subheader("Tortuosity endpoints")
            st.caption(
                "After selecting the vessel segments on the left, click this zoomed view to choose "
                "the start and end points for tortuosity scoring."
            )
            endpoint_result = NODE_ENDPOINT_VIEWER(
                data={
                    "imageUrl": bundle["image_url"],
                    "imageWidth": bundle["image_width"],
                    "imageHeight": bundle["image_height"],
                    "segments": endpoint_segments,
                    "startEndpoint": selected_start_endpoint,
                    "endEndpoint": selected_end_endpoint,
                    "nextEndpointTarget": st.session_state[next_endpoint_target_key],
                    "baseOpacity": 0.62,
                },
                default={
                    "start_endpoint": selected_start_endpoint,
                    "end_endpoint": selected_end_endpoint,
                    "next_endpoint_target": st.session_state[next_endpoint_target_key],
                },
                on_start_endpoint_change=lambda: None,
                on_end_endpoint_change=lambda: None,
                on_next_endpoint_target_change=lambda: None,
                key=f"geometry_endpoint_viewer::{selected_run_name}",
                width="stretch",
                height=360,
            )
            _sync_endpoint_selection(endpoint_result, start_endpoint_key, end_endpoint_key, next_endpoint_target_key)
            selected_start_endpoint = st.session_state.get(start_endpoint_key)
            selected_end_endpoint = st.session_state.get(end_endpoint_key)
            if selected_start_endpoint and selected_end_endpoint:
                start_x, start_y = selected_start_endpoint["point"]
                end_x, end_y = selected_end_endpoint["point"]
                st.caption(
                    f"Start `{start_x:.1f}, {start_y:.1f}`, end `{end_x:.1f}, {end_y:.1f}`. "
                    "Each click alternates between setting start and end."
                )

    selected_segment_refs = segment_refs_for_review_state(review_state)
    selected_complete_vessels = _selected_complete_vessel_names(review_state, selected_segment_refs)
    selection_df = build_selection_table(branches_df, review_state["manual_segments"], selected_segment_refs)
    if selection_df.empty:
        st.info("Click model segments or draw manual segments to build a vessel selection.")
    else:
        st.subheader("Current selection")
        st.dataframe(selection_df, use_container_width=True, hide_index=True)
        if selected_complete_vessels:
            st.caption(
                "Complete saved vessel(s) in selection: "
                + ", ".join(f"`{vessel_name}`" for vessel_name in selected_complete_vessels)
            )

    if selected_segment_refs and redraw_target_ref:
        delete_col, clear_endpoint_col = st.columns(2)
        with delete_col:
            if st.button("Delete selected manual segment", use_container_width=True):
                _, manual_segment_id = parse_segment_ref(redraw_target_ref)
                previous_refs = segment_refs_for_review_state(review_state)
                remove_manual_segment(review_state, manual_segment_id)
                _record_existing_selection_change(
                    review_state,
                    previous_refs,
                    selection_undo_key,
                    selection_redo_key,
                    viewer_reset_key,
                )
                persist_manual_review(selected_run_dir, review_state, branches_df)
                st.rerun()
        with clear_endpoint_col:
            if st.button("Clear picked endpoints", use_container_width=True):
                st.session_state.pop(start_endpoint_key, None)
                st.session_state.pop(end_endpoint_key, None)
                st.session_state[next_endpoint_target_key] = "start"
                st.rerun()

    bottom_left, bottom_right = st.columns([1.2, 1.0], gap="large")
    with bottom_left:
        _render_vessel_draft(
            branches_df,
            selected_run_dir,
            review_state,
            selected_segment_refs,
            selected_start_endpoint,
            selected_end_endpoint,
            selected_complete_vessels,
            is_editing_vessel,
            editing_vessel_name,
            vessel_name_key,
            vessel_category_key,
            vessel_name_reset_key,
            viewer_reset_key,
            selection_undo_key,
            selection_redo_key,
            editing_vessel_name_key,
            editing_original_snapshot_key,
            start_endpoint_key,
            end_endpoint_key,
            next_endpoint_target_key,
        )

    with bottom_right:
        _render_saved_vessels(
            selected_run_dir,
            branches_df,
            review_state,
            vessel_load_key,
            pending_edit_key,
            viewer_reset_key,
            selection_undo_key,
            selection_redo_key,
            editing_vessel_name_key,
            editing_original_snapshot_key,
            vessel_name_reset_key,
        )

    vessel_df = build_vessel_scores_table(review_state, branches_df)
    if not vessel_df.empty:
        st.subheader("Saved vessel scores")
        st.dataframe(vessel_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download vessel scores",
            data=vessel_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_name}_manual_vessels.csv",
            mime="text/csv",
        )
        if not vessel_df[vessel_df["Bridge status"] == "partial"].empty:
            st.warning("Some vessels still contain disconnected selections.")


def _render_vessel_draft(
    branches_df: pd.DataFrame,
    selected_run_dir,
    review_state: dict,
    selected_segment_refs: list[str],
    selected_start_endpoint,
    selected_end_endpoint,
    selected_complete_vessels: list[str],
    is_editing_vessel: bool,
    editing_vessel_name: str | None,
    vessel_name_key: str,
    vessel_category_key: str,
    vessel_name_reset_key: str,
    viewer_reset_key: str,
    undo_key: str,
    redo_key: str,
    editing_vessel_name_key: str,
    editing_original_snapshot_key: str,
    start_endpoint_key: str,
    end_endpoint_key: str,
    next_endpoint_target_key: str,
) -> None:
    st.subheader("Vessel draft")
    st.info(f"Editing `{editing_vessel_name}`. Changes stay temporary until you apply them." if is_editing_vessel else "New vessel draft. Select model/manual segments, then save it as a vessel.")
    vessel_category = st.radio("Vessel category", options=["artere", "veine"], horizontal=True, key=vessel_category_key)
    vessel_name = st.text_input("Vessel name", key=vessel_name_key)
    if is_editing_vessel:
        st.caption("Renaming here will replace the saved vessel name when changes are applied.")
    else:
        st.caption(f"If left empty, the next name will be `{next_default_vessel_name(review_state['vessels'], vessel_category)}`.")
    st.caption(f"{len(selected_segment_refs)} segment(s) currently selected.")

    def can_save(action: str) -> bool:
        if not selected_segment_refs:
            st.warning(f"Select at least one segment before {action}.")
            return False
        if selected_start_endpoint is None or selected_end_endpoint is None:
            st.warning(f"Pick a tortuosity start and end before {action}.")
            return False
        if selected_start_endpoint == selected_end_endpoint:
            st.warning("Start and end must be different points.")
            return False
        return True

    def save_payload(clean_name: str, replace_name: str | None = None) -> None:
        payload, resolution = build_vessel_payload(
            branches_df,
            review_state["manual_segments"],
            selected_segment_refs,
            vessel_category,
            selected_start_endpoint,
            selected_end_endpoint,
        )
        if replace_name is not None and clean_name != replace_name:
            review_state["vessels"].pop(replace_name, None)
        review_state["vessels"][clean_name] = payload
        _clear_vessel_draft(
            review_state,
            undo_key,
            redo_key,
            editing_vessel_name_key,
            editing_original_snapshot_key,
            vessel_name_reset_key,
            viewer_reset_key,
            start_endpoint_key,
            end_endpoint_key,
            next_endpoint_target_key,
        )
        persist_manual_review(selected_run_dir, review_state, branches_df)
        _show_saved_vessel_status(clean_name, resolution)
        st.rerun()

    if is_editing_vessel:
        apply_col, save_new_col, cancel_col = st.columns(3)
        with apply_col:
            if st.button("Apply changes", type="primary", use_container_width=True) and can_save("applying changes"):
                clean_name = _resolve_clean_vessel_name(vessel_name, vessel_category, review_state["vessels"])
                if clean_name != editing_vessel_name and clean_name in review_state["vessels"]:
                    st.warning(f"`{clean_name}` already exists. Choose a different name before applying.")
                else:
                    save_payload(clean_name, replace_name=editing_vessel_name)
        with save_new_col:
            if st.button("Save as new", use_container_width=True) and can_save("saving a vessel"):
                clean_name = _resolve_clean_vessel_name(
                    vessel_name if vessel_name.strip() != editing_vessel_name else "",
                    vessel_category,
                    review_state["vessels"],
                )
                if clean_name in review_state["vessels"]:
                    st.warning(f"`{clean_name}` already exists. Choose a different name for the new vessel.")
                else:
                    save_payload(clean_name)
        with cancel_col:
            if st.button("Cancel edit", use_container_width=True):
                _clear_vessel_draft(
                    review_state,
                    undo_key,
                    redo_key,
                    editing_vessel_name_key,
                    editing_original_snapshot_key,
                    vessel_name_reset_key,
                    viewer_reset_key,
                    start_endpoint_key,
                    end_endpoint_key,
                    next_endpoint_target_key,
                )
                st.rerun()
    else:
        save_col, clear_col, reset_col = st.columns(3)
        with save_col:
            if st.button("Save current vessel", type="primary", use_container_width=True) and can_save("saving a vessel"):
                clean_name = _resolve_clean_vessel_name(vessel_name, vessel_category, review_state["vessels"])
                if clean_name in review_state["vessels"]:
                    st.warning(f"`{clean_name}` already exists. Choose a different name.")
                else:
                    save_payload(clean_name)
        with clear_col:
            if st.button("Clear typed name", use_container_width=True):
                st.session_state[vessel_name_reset_key] = True
                st.rerun()
        with reset_col:
            if st.button("Clear draft selection", use_container_width=True):
                review_state["selected_segment_refs"] = []
                _clear_selection_history(undo_key, redo_key)
                st.session_state[viewer_reset_key] += 1
                st.rerun()

    _render_merge_controls(
        branches_df,
        selected_run_dir,
        review_state,
        selected_complete_vessels,
        selected_start_endpoint,
        selected_end_endpoint,
        vessel_name,
        vessel_name_reset_key,
        viewer_reset_key,
        undo_key,
        redo_key,
        editing_vessel_name_key,
        editing_original_snapshot_key,
        start_endpoint_key,
        end_endpoint_key,
        next_endpoint_target_key,
    )


def _render_merge_controls(
    branches_df: pd.DataFrame,
    selected_run_dir,
    review_state: dict,
    selected_complete_vessels: list[str],
    selected_start_endpoint,
    selected_end_endpoint,
    vessel_name: str,
    vessel_name_reset_key: str,
    viewer_reset_key: str,
    undo_key: str,
    redo_key: str,
    editing_vessel_name_key: str,
    editing_original_snapshot_key: str,
    start_endpoint_key: str,
    end_endpoint_key: str,
    next_endpoint_target_key: str,
) -> None:
    merge_ready = len(selected_complete_vessels) >= 2
    if not st.button(
        f"Merge {len(selected_complete_vessels)} selected vessels" if merge_ready else "Merge selected vessels",
        disabled=not merge_ready,
        use_container_width=True,
        help="Select at least two complete saved vessels.",
    ):
        return
    merged_refs = sorted(
        {
            segment_ref
            for vessel_name_to_merge in selected_complete_vessels
            for segment_ref in segment_refs_for_vessel(review_state["vessels"][vessel_name_to_merge])
        },
        key=segment_ref_sort_key,
    )
    merged_category = _dominant_category_for_vessels(branches_df, review_state, selected_complete_vessels)
    clean_name = _resolve_clean_vessel_name(vessel_name, merged_category, review_state["vessels"])
    remaining = set(review_state["vessels"]) - set(selected_complete_vessels)
    if clean_name in remaining:
        st.warning(f"`{clean_name}` already exists. Choose a different name for the merged vessel.")
        return
    if selected_start_endpoint is None or selected_end_endpoint is None:
        st.warning("Pick a tortuosity start and end before merging vessels.")
        return
    if selected_start_endpoint == selected_end_endpoint:
        st.warning("Start and end must be different points.")
        return
    payload, resolution = build_vessel_payload(
        branches_df,
        review_state["manual_segments"],
        merged_refs,
        merged_category,
        selected_start_endpoint,
        selected_end_endpoint,
    )
    for vessel_name_to_remove in selected_complete_vessels:
        review_state["vessels"].pop(vessel_name_to_remove, None)
    review_state["vessels"][clean_name] = payload
    _clear_vessel_draft(
        review_state,
        undo_key,
        redo_key,
        editing_vessel_name_key,
        editing_original_snapshot_key,
        vessel_name_reset_key,
        viewer_reset_key,
        start_endpoint_key,
        end_endpoint_key,
        next_endpoint_target_key,
    )
    persist_manual_review(selected_run_dir, review_state, branches_df)
    st.success(f"Merged {len(selected_complete_vessels)} vessels into `{clean_name}` as `{merged_category}`.")
    if not resolution["bridge_success"]:
        st.info("The merged vessel still has disconnected pieces.")
    st.rerun()


def _render_saved_vessels(
    selected_run_dir,
    branches_df: pd.DataFrame,
    review_state: dict,
    vessel_load_key: str,
    pending_edit_key: str,
    viewer_reset_key: str,
    undo_key: str,
    redo_key: str,
    editing_vessel_name_key: str,
    editing_original_snapshot_key: str,
    vessel_name_reset_key: str,
) -> None:
    st.subheader("Saved vessels")
    vessel_names = sorted(review_state["vessels"])
    if not vessel_names:
        st.info("Saved vessels will appear here.")
        return
    if vessel_load_key not in st.session_state or st.session_state[vessel_load_key] not in vessel_names:
        st.session_state[vessel_load_key] = vessel_names[0]
    vessel_to_load = st.selectbox("Saved vessels list", options=vessel_names, key=vessel_load_key)
    load_col, add_col, delete_col = st.columns(3)
    with load_col:
        if st.button("Open for editing", use_container_width=True):
            st.session_state[pending_edit_key] = vessel_to_load
            st.session_state[viewer_reset_key] += 1
            st.rerun()
    with add_col:
        if st.button("Add to selection", use_container_width=True):
            current = set(review_state["selected_segment_refs"])
            current.update(segment_refs_for_vessel(review_state["vessels"][vessel_to_load]))
            _set_selected_segment_refs(review_state, list(current), undo_key, redo_key, viewer_reset_key)
            st.rerun()
    with delete_col:
        if st.button("Delete saved vessel", use_container_width=True):
            del review_state["vessels"][vessel_to_load]
            if st.session_state.get(editing_vessel_name_key) == vessel_to_load:
                st.session_state.pop(editing_vessel_name_key, None)
                st.session_state.pop(editing_original_snapshot_key, None)
            _set_selected_segment_refs(review_state, [], undo_key, redo_key)
            st.session_state[vessel_name_reset_key] = True
            st.session_state[viewer_reset_key] += 1
            persist_manual_review(selected_run_dir, review_state, branches_df)
            st.rerun()


def main() -> None:
    sidebar_values = render_sidebar_run_setup()
    _handle_run_creation(sidebar_values)
    selected_run_name = _render_run_selector()
    if selected_run_name is None:
        st.stop()
    active_view = st.radio(
        "View",
        options=["Manual Review", "Debug Outputs"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if active_view == "Manual Review":
        viewer_options = _render_viewer_controls()
        _render_manual_review(selected_run_name, viewer_options)
    else:
        render_debug_tab(RUNS_ROOT / selected_run_name)


main()
