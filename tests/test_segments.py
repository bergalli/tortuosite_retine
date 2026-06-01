from __future__ import annotations

import math
import unittest

import pandas as pd

from tortuosite_score.vessels_detection.segments import (
    VesselSegment,
    create_geometry_endpoint,
    manual_segment_ref,
    model_segment_ref,
    score_segments,
    segment_ref_sort_key,
    split_segment_refs,
    synthesize_segment_links,
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


if __name__ == "__main__":
    unittest.main()
