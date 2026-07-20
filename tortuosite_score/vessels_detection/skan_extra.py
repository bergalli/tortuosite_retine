"""Optional skan dependency helpers."""

from __future__ import annotations

import importlib.util

SKAN_INSTALL_HINT = "uv sync"


def skan_available() -> bool:
    return importlib.util.find_spec("skan") is not None


def require_skan(feature: str = "Skeleton branch analysis") -> None:
    if skan_available():
        return
    raise RuntimeError(
        f"{feature} requires optional dependencies. Install with: {SKAN_INSTALL_HINT}"
    )
