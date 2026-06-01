"""Optional deep-learning dependency helpers."""

from __future__ import annotations

import importlib.util

DEEP_INSTALL_HINT = "uv sync --extra deep"


def deep_learning_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def require_deep_learning(feature: str = "Deep learning segmentation") -> None:
    if deep_learning_available():
        return
    raise RuntimeError(
        f"{feature} requires optional dependencies. Install with: {DEEP_INSTALL_HINT}"
    )
