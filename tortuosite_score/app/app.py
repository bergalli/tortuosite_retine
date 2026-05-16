from __future__ import annotations

import pandas as pd
import streamlit as st

from tortuosite_score.app.constants import RUNS_ROOT
from tortuosite_score.app.review_data import list_runs, load_review_bundle, run_uploaded_analysis
from tortuosite_score.app.review_state import (
    build_selection_table,
    build_vessel_scores_table,
    get_or_create_review_state,
    next_default_vessel_name,
    persist_manual_review,
    synthesize_missing_links,
)
from tortuosite_score.app.ui_sections import render_debug_tab, render_sidebar_run_setup
from tortuosite_score.app.viewer_component import BRANCH_VIEWER, build_viewer_branches


st.set_page_config(page_title="Tortuosite Retine", layout="wide")
st.title("Retinal Vessel Review")
st.write(
    "Run the segmentation once, then review the extracted skeleton manually, "
    "define vessels, classify them, and score them from the interface."
)


def _handle_run_creation(sidebar_values: dict) -> None:
    if not sidebar_values["run_btn"] or sidebar_values["uploaded_file"] is None:
        return

    run_id = run_uploaded_analysis(
        uploaded_file=sidebar_values["uploaded_file"],
        method=sidebar_values["method"],
        vessel_percentile=sidebar_values["vessel_percentile"],
        vessel_low_percentile=sidebar_values["vessel_low_percentile"],
        deep_threshold=sidebar_values["deep_threshold"],
        deep_modality=sidebar_values["deep_modality"],
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


def _sync_viewer_selection(
    viewer_result,
    review_state: dict,
    selected_run_dir,
    branches_df: pd.DataFrame,
) -> None:
    previous_selection = list(review_state["selected_branch_ids"])
    if getattr(viewer_result, "selected_branch_ids", None) is None:
        return

    updated_selection = sorted(
        int(branch_id) for branch_id in viewer_result.selected_branch_ids
    )
    if updated_selection == previous_selection:
        return

    review_state["selected_branch_ids"] = updated_selection
    persist_manual_review(selected_run_dir, review_state, branches_df)
    st.rerun()


def _render_manual_review(selected_run_name: str) -> None:
    selected_run_dir = RUNS_ROOT / selected_run_name
    bundle = load_review_bundle(str(selected_run_dir))
    review_state = get_or_create_review_state(selected_run_dir)
    branches_df = pd.DataFrame(bundle["branches_df"])

    vessel_name_key = f"vessel_name_input::{selected_run_name}"
    vessel_load_key = f"saved_vessel_select::{selected_run_name}"
    vessel_name_reset_key = f"vessel_name_reset::{selected_run_name}"
    viewer_reset_key = f"viewer_reset_nonce::{selected_run_name}"
    if st.session_state.get(vessel_name_reset_key):
        st.session_state[vessel_name_key] = ""
        st.session_state[vessel_name_reset_key] = False
    if vessel_name_key not in st.session_state:
        st.session_state[vessel_name_key] = ""
    if viewer_reset_key not in st.session_state:
        st.session_state[viewer_reset_key] = 0

    top_controls_left, top_controls_mid = st.columns([1.2, 1.0], gap="large")

    with top_controls_left:
        st.subheader("Viewer")
        show_base_image = st.toggle("Show base image", value=True)
        show_skeleton = st.toggle("Show skeleton", value=True)
        show_labels = st.toggle("Show branch IDs", value=False)
        base_opacity = st.slider("Base image opacity", 0.0, 1.0, 0.95, 0.05)

    with top_controls_mid:
        st.subheader("Selection")
        st.write(f"{len(review_state['selected_branch_ids'])} branch(es) selected")
        allow_reuse_assigned = st.toggle(
            "Reuse assigned branches",
            value=False,
            help="Allow already saved artery/vein segments to be selected again for another vessel.",
        )
        selection_actions_left, selection_actions_right = st.columns(2)
        with selection_actions_left:
            if st.button("Clear selection", use_container_width=True):
                review_state["selected_branch_ids"] = []
                st.session_state[viewer_reset_key] += 1
                st.rerun()
        with selection_actions_right:
            if st.button("Select all", use_container_width=True):
                review_state["selected_branch_ids"] = branches_df["branch_id"].astype(int).tolist()
                st.session_state[viewer_reset_key] += 1
                st.rerun()

    st.subheader("Interactive skeleton viewer")
    provisional_resolution = synthesize_missing_links(
        branches_df,
        sorted(int(branch_id) for branch_id in review_state["selected_branch_ids"]),
    )
    viewer_result = BRANCH_VIEWER(
        data={
            "imageUrl": bundle["image_url"],
            "imageWidth": bundle["image_width"],
            "imageHeight": bundle["image_height"],
            "branches": build_viewer_branches(
                branches_df=branches_df,
                paths_payload=bundle["paths_payload"],
                review_state=review_state,
                allow_reuse_assigned=allow_reuse_assigned,
                provisional_synthetic_links=provisional_resolution["synthetic_links"],
            ),
            "selectedBranchIds": sorted(int(branch_id) for branch_id in review_state["selected_branch_ids"]),
            "showBaseImage": show_base_image,
            "showSkeleton": show_skeleton,
            "showLabels": show_labels,
            "baseOpacity": base_opacity,
        },
        default={
            "selected_branch_ids": sorted(
                int(branch_id) for branch_id in review_state["selected_branch_ids"]
            ),
        },
        on_selected_branch_ids_change=lambda: None,
        key=f"branch_viewer::{selected_run_name}::{st.session_state[viewer_reset_key]}",
        width="stretch",
        height=860,
    )
    _sync_viewer_selection(
        viewer_result=viewer_result,
        review_state=review_state,
        selected_run_dir=selected_run_dir,
        branches_df=branches_df,
    )

    selected_branch_ids = sorted(int(branch_id) for branch_id in review_state["selected_branch_ids"])
    selection_df = build_selection_table(branches_df, selected_branch_ids)
    if selection_df.empty:
        st.info("Click branches in the viewer to build a vessel selection.")
    else:
        st.subheader("Current selection")
        st.dataframe(selection_df, use_container_width=True, hide_index=True)

    bottom_left, bottom_right = st.columns([1.2, 1.0], gap="large")

    with bottom_left:
        st.subheader("Current vessel")
        vessel_category = st.radio(
            "Vessel category",
            options=["artere", "veine"],
            horizontal=True,
        )
        vessel_name = st.text_input("Vessel name", key=vessel_name_key)
        st.caption(
            f"If left empty, the next name will be `{next_default_vessel_name(review_state['vessels'], vessel_category)}`."
        )
        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("Save current vessel", type="primary", use_container_width=True):
                if not selected_branch_ids:
                    st.warning("Select at least one branch before saving a vessel.")
                else:
                    resolution = synthesize_missing_links(branches_df, selected_branch_ids)
                    clean_name = vessel_name.strip() or next_default_vessel_name(
                        review_state["vessels"],
                        vessel_category,
                    )
                    review_state["vessels"][clean_name] = {
                        "category": vessel_category,
                        "branch_ids": resolution["branch_ids"],
                        "synthetic_links": resolution["synthetic_links"],
                    }
                    review_state["selected_branch_ids"] = []
                    st.session_state[vessel_name_reset_key] = True
                    st.session_state[viewer_reset_key] += 1
                    persist_manual_review(selected_run_dir, review_state, branches_df)
                    if len(resolution["synthetic_links"]) > 0:
                        st.success(
                            f"Saved vessel `{clean_name}` with {len(resolution['synthetic_links'])} synthetic link(s)."
                        )
                    elif resolution["component_count"] > 1:
                        st.warning(
                            f"Saved vessel `{clean_name}` with unresolved disconnected pieces."
                        )
                    else:
                        st.success(f"Saved vessel `{clean_name}`.")
                    st.rerun()
        with clear_col:
            if st.button("Clear typed name", use_container_width=True):
                st.session_state[vessel_name_reset_key] = True
                st.rerun()

    with bottom_right:
        st.subheader("Saved vessels")
        vessel_names = sorted(review_state["vessels"])
        if vessel_names:
            if vessel_load_key not in st.session_state or st.session_state[vessel_load_key] not in vessel_names:
                st.session_state[vessel_load_key] = vessel_names[0]
            vessel_to_load = st.selectbox(
                "Saved vessels list",
                options=vessel_names,
                key=vessel_load_key,
            )
            load_col, delete_col = st.columns(2)
            with load_col:
                if st.button("Load vessel selection", use_container_width=True):
                    review_state["selected_branch_ids"] = list(
                        review_state["vessels"][vessel_to_load]["branch_ids"]
                    )
                    st.session_state[viewer_reset_key] += 1
                    st.rerun()
            with delete_col:
                if st.button("Delete saved vessel", use_container_width=True):
                    del review_state["vessels"][vessel_to_load]
                    review_state["selected_branch_ids"] = []
                    st.session_state[viewer_reset_key] += 1
                    persist_manual_review(selected_run_dir, review_state, branches_df)
                    st.rerun()
        else:
            st.info("Saved vessels will appear here.")

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
        partial_vessels = vessel_df[vessel_df["Bridge status"] == "partial"]
        if not partial_vessels.empty:
            st.warning(
                "Some vessels still contain disconnected selections. Synthetic links are shown as dashed segments."
            )
        elif (vessel_df["Auto bridges"] > 0).any():
            st.info("Synthetic links are shown as dashed segments between the selected vessel parts.")


def main() -> None:
    sidebar_values = render_sidebar_run_setup()
    _handle_run_creation(sidebar_values)

    selected_run_name = _render_run_selector()
    if selected_run_name is None:
        st.stop()

    main_tab, debug_tab = st.tabs(["Manual Review", "Debug Outputs"])
    with main_tab:
        _render_manual_review(selected_run_name)

    with debug_tab:
        render_debug_tab(RUNS_ROOT / selected_run_name)


main()
