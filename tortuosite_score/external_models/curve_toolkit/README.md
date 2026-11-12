# Vendored Curve Toolkit

Source: local checkout `workspace/tortuosity_metrics_python`, remote
`https://github.com/coconetlero/tortuosity_metrics_python`, commit
`337a3f855f6900b6b1c51ecfbeaf8a291ce718c8`.

The six Python modules here are copied from that revision. Only imports were
changed to the `tortuosite_score.external_models.curve_toolkit` namespace.
`__init__.py` files make the modules installable with the app. The original
checkout and example scripts are not modified or imported at runtime.

The adapter lives in `vessels_detection/toolkit_metrics.py`; its bounded worker
is `vessels_detection/toolkit_worker.py`. See `docs/toolkit-methods.md` for the
approved batch recipe, adaptations and preserved implementation quirks.
