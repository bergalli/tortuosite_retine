from __future__ import annotations

from pathlib import Path


ARTERE_COLOR = "#ff453a"
VEINE_COLOR = "#4c8dff"
SELECTED_COLOR = "#ffd60a"
DEFAULT_BRANCH_COLOR = "#ff7a70"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "demo" / "streamlit_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
