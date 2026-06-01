# Run with

```bash
uv sync                          # classical + Streamlit (no torch, no skan)
uv sync --extra skan             # faster/more accurate skeleton branch analysis
uv sync --extra deep             # DCP / VascX backends
uv sync --extra skan --extra deep
uv run streamlit run tortuosite_score/app/app.py
```

Without `skan`, skeleton tortuosity uses a built-in scikit-image graph fallback (no LLVM/numba build required on Intel Mac).
