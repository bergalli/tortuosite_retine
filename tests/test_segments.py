from __future__ import annotations

import math
import unittest

import pandas as pd

from tortuosite_score.vessels_detection.segments import (
    VesselSegment,
    create_geometry_endpoint,
    manual_segment_ref,
    model_segment_ref,
    ordered_points_for_segments,
    score_segments,
    segment_ref_sort_key,
    split_segment_refs,
    synthesize_segment_links,
)
from tortuosite_score.app.review_state import (
    replace_auto_completed_vessels,
    push_selection_history,
    redo_selection,
    undo_selection,
)


class VesselSegmentTests(unittest.TestCase):
    def test_model_segment_from_row(self) -> None:
        row = pd.Series(
            {
                "branch_id": 7,
                "path_points": [[0, 0], [3, 4]],
                "branch-distance": 5.0,
                "euclidean-distance": 5.0,
                "tortuosity": 1.0,
                "vascx_category": "artere",
                "branch-type": "endpoint",
            }
        )

        segment = VesselSegment.from_model_row(row)

        self.assertEqual(segment.ref, "model:7")
        self.assertEqual(segment.source, "model")
        self.assertEqual(segment.points, ((0.0, 0.0), (3.0, 4.0)))
        self.assertEqual(segment.category, "artere")

    def test_manual_segment_validation(self) -> None:
        with self.assertRaises(ValueError):
            VesselSegment.from_manual_points(1, [[0, 0]])
        with self.assertRaises(ValueError):
            VesselSegment.from_manual_points(1, [[0, 0], [2, 0]])

        segment = VesselSegment.from_manual_points(1, [[0, 0], [10, 0]])

        self.assertEqual(segment.ref, "manual:1")
        self.assertEqual(segment.path_length, 10.0)
        self.assertEqual(segment.chord_length, 10.0)

    def test_segment_ref_sorting_and_splitting(self) -> None:
        refs = ["manual:2", "model:5", "manual:1", "model:2"]

        self.assertEqual(sorted(refs, key=segment_ref_sort_key), ["model:2", "model:5", "manual:1", "manual:2"])
        self.assertEqual(split_segment_refs(refs), ([2, 5], [1, 2]))

    def test_score_single_model_segment_with_arbitrary_endpoints(self) -> None:
        segment = VesselSegment.from_manual_points(1, [[0, 0], [20, 0]])
        segments = {model_segment_ref(1): VesselSegment(
            ref=model_segment_ref(1),
            source="model",
            segment_id=1,
            points=segment.points,
            path_length=segment.path_length,
            chord_length=segment.chord_length,
            tortuosity=segment.tortuosity,
            label_position=segment.label_position,
        )}
        start = create_geometry_endpoint([5, 0], model_segment_ref(1), 5.0)
        end = create_geometry_endpoint([15, 0], model_segment_ref(1), 15.0)

        score = score_segments(segments, [model_segment_ref(1)], start_endpoint=start, end_endpoint=end)

        self.assertAlmostEqual(score["length"], 10.0)
        self.assertAlmostEqual(score["chord"], 10.0)
        self.assertAlmostEqual(score["tortuosity"], 1.0)

    def test_score_single_manual_segment(self) -> None:
        segment = VesselSegment.from_manual_points(3, [[0, 0], [0, 10]])
        start = create_geometry_endpoint([0, 0], manual_segment_ref(3), 0.0)
        end = create_geometry_endpoint([0, 10], manual_segment_ref(3), 10.0)

        score = score_segments({segment.ref: segment}, [segment.ref], start_endpoint=start, end_endpoint=end)

        self.assertEqual(score["manual_segment_count"], 1)
        self.assertAlmostEqual(score["length"], 10.0)
        self.assertAlmostEqual(score["chord"], 10.0)

    def test_score_mixed_segments_with_synthetic_link(self) -> None:
        model = VesselSegment(
            ref=model_segment_ref(1),
            source="model",
            segment_id=1,
            points=((0.0, 0.0), (10.0, 0.0)),
            path_length=10.0,
            chord_length=10.0,
            tortuosity=1.0,
            label_position=(10.0, 0.0),
        )
        manual = VesselSegment.from_manual_points(1, [[20, 0], [30, 0]])
        segments = {model.ref: model, manual.ref: manual}
        resolution = synthesize_segment_links(segments, [model.ref, manual.ref])
        start = create_geometry_endpoint([0, 0], model.ref, 0.0)
        end = create_geometry_endpoint([30, 0], manual.ref, 10.0)

        score = score_segments(
            segments,
            [model.ref, manual.ref],
            synthetic_links=resolution["synthetic_links"],
            start_endpoint=start,
            end_endpoint=end,
        )

        self.assertTrue(resolution["bridge_success"])
        self.assertEqual(len(resolution["synthetic_links"]), 1)
        self.assertTrue(math.isclose(score["length"], 30.0))
        self.assertTrue(math.isclose(score["chord"], 30.0))

    def test_ordered_points_follow_saved_vessel_endpoints(self) -> None:
        segment = VesselSegment.from_manual_points(1, [[0, 0], [10, 0], [20, 0]])
        start = create_geometry_endpoint([20, 0], manual_segment_ref(1), 20.0)
        end = create_geometry_endpoint([0, 0], manual_segment_ref(1), 0.0)

        points = ordered_points_for_segments({segment.ref: segment}, [segment.ref], start_endpoint=start, end_endpoint=end)

        self.assertEqual(points[0], [20.0, 0.0])
        self.assertEqual(points[-1], [0.0, 0.0])

    def test_ordered_points_preserve_curved_saved_vessel_geometry(self) -> None:
        source_points = [[0, 0], [5, 8], [12, -4], [20, 6], [30, 0]]
        segment = VesselSegment.from_manual_points(1, source_points)
        start = create_geometry_endpoint(source_points[0], manual_segment_ref(1), 0.0)
        end = create_geometry_endpoint(source_points[-1], manual_segment_ref(1), segment.path_length)

        points = ordered_points_for_segments({segment.ref: segment}, [segment.ref], start_endpoint=start, end_endpoint=end)

        self.assertEqual(points, [[float(x), float(y)] for x, y in source_points])
        self.assertGreater(len(points), 2)


class SelectionHistoryTests(unittest.TestCase):
    def test_push_records_previous_selection_and_clears_redo(self) -> None:
        undo_stack = [["model:9"]]
        redo_stack = [["model:3"]]

        changed = push_selection_history(undo_stack, redo_stack, ["model:2"], ["model:2", "manual:1"], 50)

        self.assertTrue(changed)
        self.assertEqual(undo_stack, [["model:9"], ["model:2"]])
        self.assertEqual(redo_stack, [])

    def test_push_ignores_duplicate_selection(self) -> None:
        undo_stack: list[list[str]] = []
        redo_stack = [["model:3"]]

        changed = push_selection_history(undo_stack, redo_stack, ["manual:1", "model:2"], ["model:2", "manual:1"], 50)

        self.assertFalse(changed)
        self.assertEqual(undo_stack, [])
        self.assertEqual(redo_stack, [["model:3"]])

    def test_undo_and_redo_move_between_stacks(self) -> None:
        undo_stack = [[], ["model:2"]]
        redo_stack: list[list[str]] = []

        restored = undo_selection(undo_stack, redo_stack, ["model:2", "manual:1"])

        self.assertEqual(restored, ["model:2"])
        self.assertEqual(undo_stack, [[]])
        self.assertEqual(redo_stack, [["model:2", "manual:1"]])

        redone = redo_selection(undo_stack, redo_stack, restored)

        self.assertEqual(redone, ["model:2", "manual:1"])
        self.assertEqual(undo_stack, [[], ["model:2"]])
        self.assertEqual(redo_stack, [])

    def test_new_selection_after_undo_clears_redo(self) -> None:
        undo_stack = [[]]
        redo_stack = [["model:2", "manual:1"]]

        push_selection_history(undo_stack, redo_stack, ["model:2"], ["model:4"], 50)

        self.assertEqual(undo_stack, [[], ["model:2"]])
        self.assertEqual(redo_stack, [])


class AutoCompleteVesselTests(unittest.TestCase):
    def test_auto_complete_preserves_manual_and_replaces_prior_auto(self) -> None:
        branches = pd.DataFrame(
            {
                "branch_id": [0, 1],
                "path_points": [
                    [[0, 0], [120, 0]],
                    [[120, 0], [240, 0]],
                ],
                "vascx_category": ["artere", "artere"],
            }
        )
        branches.attrs["root_hint"] = (0.0, 0.0)
        review_state = {
            "manual_segments": {},
            "vessels": {
                "manual_keep": {"category": "veine", "segment_refs": ["model:0"], "synthetic_links": []},
                "auto_vascx_99": {"category": "artere", "segment_refs": ["model:1"], "synthetic_links": []},
            },
        }

        created = replace_auto_completed_vessels(review_state, branches)

        self.assertGreaterEqual(created, 1)
        self.assertIn("manual_keep", review_state["vessels"])
        self.assertNotIn("auto_vascx_99", review_state["vessels"])
        self.assertTrue(any(name.startswith("auto_vascx_") for name in review_state["vessels"]))


if __name__ == "__main__":
    unittest.main()
