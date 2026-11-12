"""Private JSON-lines worker. The parent enforces per-operation deadlines."""
from __future__ import annotations

import json
import math
import sys
import warnings

import numpy as np

from tortuosite_score.external_models.curve_toolkit.Curve import Curve
from tortuosite_score.external_models.curve_toolkit.Tortuosity import get_tortuosity_measure
from tortuosite_score.vessels_detection.toolkit_metrics import METRICS


def _snapshot(values):
    result = {}
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (float, int, str, list, tuple)):
            try:
                result[name] = json.loads(json.dumps(value, default=lambda v: v.item()))
            except (TypeError, ValueError, AttributeError):
                pass
        elif isinstance(value, np.number):
            result[name] = value.item()
    return result


def compute(payload):
    points = np.asarray(payload["points"], dtype=float)
    if payload["operation"] == "prepare":
        curve = Curve(points[:, 0], points[:, 1])
        length = curve.arclength()
        count = math.ceil(len(curve) * 0.25)
        smooth = 8 / length
        smoothed = curve.smooth("cubic_spline", smooth=smooth, num_points=count)
        processed = smoothed.parametrize("scc", num_points=count)
        return {
            "status": "ok", "reason": "", "points": processed.points.tolist(),
            "smoothed_points": smoothed.points.tolist(), "input_length": float(length),
            "input_point_count": len(curve), "output_point_count": len(processed), "smooth": smooth,
        }
    metric = payload["metric"]
    kwargs = {"smooth": 1.0} if metric in {
        "mean_curvature", "total_curvature", "tau_3", "tau_5", "asdc", "icm", "td",
    } else {}
    measure = get_tortuosity_measure(METRICS[metric], **kwargs)
    diagnostics = {"arc_chord_calls": []}

    def capture(frame, event, arg):
        if event != "return":
            return
        filename = frame.f_code.co_filename
        name = frame.f_code.co_name
        if filename.endswith("/Tortuosity.py"):
            owner = frame.f_locals.get("self")
            if owner is measure:
                diagnostics[name] = _snapshot(frame.f_locals)
                if name == "escc" and arg is not None:
                    diagnostics["escc_points"] = np.column_stack(arg).tolist()
                if name == "scc" and arg is not None:
                    diagnostics["scc_angles"] = np.asarray(arg).tolist()
            elif type(owner).__name__ == "Distance_Metric_Tortuosity" and name == "compute":
                diagnostics["arc_chord_calls"].append(_snapshot(frame.f_locals))
            if type(owner).__name__ == "Tau_3_Tortuosity" and name == "compute":
                diagnostics["tau_3"] = _snapshot(frame.f_locals)
        elif filename.endswith("/Derivatives.py") and name == "curvature" and arg is not None:
            diagnostics["curvature"] = {"t": arg[0].tolist(), "kappa": arg[1].tolist()}

    sys.setprofile(capture)
    try:
        value = float(measure.compute(points[:, 0], points[:, 1]))
    finally:
        sys.setprofile(None)
    return {
        "status": "ok" if math.isfinite(value) else "non_finite",
        "reason": "" if math.isfinite(value) else "Non-finite source result",
        "value": value if math.isfinite(value) else math.nan, "diagnostics": diagnostics,
    }


def main():
    for line in sys.stdin:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = compute(json.loads(line))
                result["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
        except Exception as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                      "value": math.nan, "diagnostics": {}}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
