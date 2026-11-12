"""Faithful Curve Toolkit batch adapter, bounded execution and geometry cache.

The subprocess is deliberately persistent: each operation has its own deadline,
but importing scipy/csaps is paid only once. Source computations stay untouched.
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache

import numpy as np

SOURCE_REVISION = "337a3f855f6900b6b1c51ecfbeaf8a291ce718c8"
TIMEOUT_SECONDS = 30.0
CACHE_VERSION = "batch-v1-diagnostics-3"
# Keys are stable public method suffixes; values are the original registry keys.
METRICS = {
    "dm": "DM", "mean_curvature": "mean_curvature",
    "total_curvature": "total_curvature", "tau_3": "tau_3", "tau_5": "tau_5",
    "asdc": "ASDC", "icm": "ICM", "soam": "SOAM", "td": "TD",
    "scc": "SCC", "escc": "ESCC",
}
FORMULAS = {
    "dm": "DM = L / C",
    "mean_curvature": "M = moyenne(|kappa_i|) (moyenne des echantillons)",
    "total_curvature": "TC = trapeze(|kappa|, s)",
    "tau_3": "tau_3 = trapeze(kappa^2, s)",
    "tau_5": "tau_5 = tau_3 / L",
    "asdc": "ASDC = trapeze(gradient(kappa)^2) / C (pas d'indice = 1)",
    "icm": "ICM = DM x (nombre d'inflexions retenues + 1)",
    "soam": "SOAM = somme(|diff(atan2(dy, dx) en degres)|) / L",
    "td": "TD = ((n - 1) / n) x somme(exces retenus) / L",
    "scc": "SCC = somme(|angles corriges / 180|)",
    "escc": "ESCC = somme(|angles corriges / 180|) sur le trace ESCC",
}
DESCRIPTION = (
    "Calcul Curve Toolkit sur les vaisseaux sauvegardes normalises. Recette batch : "
    "lissage csaps avec p=8/L, ceil(N/4) points, puis parametrisation SCC. "
    "Les formules et particularites du code source sont conservees."
)
PARAMETERS = (
    ("Source", SOURCE_REVISION),
    ("Recette batch", "p=8/L; M=ceil(N/4); lissage M points puis SCC M points"),
    ("Derivees csaps", "smooth=1.0; seuil ICM/TD=0.0001"),
    ("Echecs", "NA avec motif; limite 30 s par operation"),
    ("Aggregation app", "moyenne ponderee par longueur; aucun multiplicateur"),
)
AUDIT_COLUMNS = [
    "toolkit_status", "toolkit_failure_reason", "toolkit_raw_status",
    "toolkit_raw_failure_reason", "toolkit_source_revision", "toolkit_preprocessing",
    "toolkit_input_point_count", "toolkit_output_point_count", "toolkit_smooth",
]


def _failure(reason: str, status: str = "error") -> dict:
    return {"value": math.nan, "status": status, "reason": reason, "diagnostics": {}}


class _Worker:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.buffer = b""

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        self.buffer = b""

    def call(self, payload: dict, timeout: float) -> dict:
        with self.lock:
            try:
                if self.process is None or self.process.poll() is not None:
                    self.close()
                    self.process = subprocess.Popen(
                        [sys.executable, "-u", "-m", "tortuosite_score.vessels_detection.toolkit_worker"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        cwd=Path(__file__).resolve().parents[2],
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                             "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "tortuosite-mpl")},
                    )
                deadline = time.monotonic() + timeout
                self.process.stdin.write((json.dumps(payload) + "\n").encode())
                self.process.stdin.flush()
                while b"\n" not in self.buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                        self.close()
                        return _failure(f"Delai depasse ({timeout:g} s)", "timeout")
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        self.close()
                        return _failure("Le processus toolkit s'est termine sans resultat")
                    self.buffer += chunk
                line, self.buffer = self.buffer.split(b"\n", 1)
                return json.loads(line)
            except (OSError, ValueError) as exc:
                self.close()
                return _failure(f"{type(exc).__name__}: {exc}")


_WORKER = _Worker()
atexit.register(_WORKER.close)


@lru_cache(maxsize=1)
def _environment():
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "csaps")}


def _cached_call(payload: dict, timeout: float) -> dict:
    key = hashlib.sha256(json.dumps(
        [CACHE_VERSION, SOURCE_REVISION, _environment(), timeout, payload], sort_keys=True,
    ).encode()).hexdigest()
    root = Path(os.environ.get("TORTUOSITY_TOOLKIT_CACHE", str(Path(tempfile.gettempdir()) / "tortuosite-toolkit")))
    path = root / f"{key}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    result = _WORKER.call(payload, timeout)
    # Cache only completed calculations. Transient process failures can be retried.
    if result["status"] in {"ok", "non_finite"} or (result["status"] == "error" and result["reason"] != "Le processus toolkit s'est termine sans resultat"):
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as f:
                json.dump(result, f)
                temp_name = f.name
            os.replace(temp_name, path)
        except OSError:
            pass
    return result


@lru_cache(maxsize=1024)
def _prepare_cached(points_json: str, timeout: float) -> dict:
    return _cached_call({"operation": "prepare", "points": json.loads(points_json)}, timeout)


@lru_cache(maxsize=8192)
def _metric_cached(points_json: str, metric: str, timeout: float) -> dict:
    prepared = _prepare_cached(points_json, timeout)
    if prepared["status"] != "ok":
        return {**prepared, "value": math.nan}
    result = _cached_call({"operation": "metric", "metric": metric, "points": prepared["points"]}, timeout)
    return {**result, "preprocessing": prepared}


def toolkit_metrics(points, method: str, *, timeout: float = TIMEOUT_SECONDS) -> dict:
    """Return a scalar plus source-captured diagnostics, never alter input geometry."""
    metric = method.removeprefix("toolkit_")
    if metric not in METRICS:
        raise ValueError(f"Unknown toolkit metric: {method}")
    try:
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
            return _failure("Coordonnees attendues: tableau N x 2 fini")
        encoded = json.dumps(array.tolist())
        return copy.deepcopy(_metric_cached(encoded, metric, float(timeout)))
    except (ValueError, TypeError, importlib.metadata.PackageNotFoundError) as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def toolkit_audit(result: dict, raw_result: dict) -> dict:
    prep = result.get("preprocessing", result)
    return {
        "toolkit_status": result["status"], "toolkit_failure_reason": result["reason"],
        "toolkit_raw_status": raw_result["status"], "toolkit_raw_failure_reason": raw_result["reason"],
        "toolkit_source_revision": SOURCE_REVISION,
        "toolkit_preprocessing": "batch: cubic_spline(p=8/L,M=ceil(N/4)); SCC(M); derivatives smooth=1.0",
        "toolkit_input_point_count": prep.get("input_point_count", math.nan),
        "toolkit_output_point_count": prep.get("output_point_count", math.nan),
        "toolkit_smooth": prep.get("smooth", math.nan),
    }
