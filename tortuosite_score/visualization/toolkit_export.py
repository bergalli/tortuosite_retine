"""Generate all toolkit reports and atlases from saved vessels, without inference."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tortuosite_score.app.results_page import (
    _build_local_bump_summary_table, _generate_local_bump_results_pdf_from_scores,
)
from tortuosite_score.vessels_detection.clinical_excel import generate_classified_vessels_excel
from tortuosite_score.vessels_detection.scoring import scoring_config, score_run, scoring_method_fixed_parameters
from tortuosite_score.vessels_detection.toolkit_metrics import METRICS, SOURCE_REVISION
from tortuosite_score.visualization.method_figures import _numbered_runs
from tortuosite_score.visualization.toolkit_figures import write_toolkit_atlas, toolkit_diagnostics_bytes


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_output_index(output, manifest):
    lines = ["# Rapports et atlas Curve Toolkit", "",
             f"Cohorte : {len(manifest['runs'])} yeux numérotés. Source : `{SOURCE_REVISION}`.", "",
             "Calculs sur les vaisseaux sauvegardés normalisés à 1024 px, recette batch conservée.",
             "Les échecs restent NA avec leur motif; aucune correction de formule n'a été appliquée.", "",
             "| Métrique | Rapport | Atlas des artères | Artères incluses | Calculs normalisés réussis / sauvegardés |",
             "| --- | --- | --- | ---: | ---: |"]
    for method_id, stats in manifest["methods"].items():
        key = method_id.removeprefix("toolkit_")
        report = f"[PDF](rapport_tortuosite_{method_id}.pdf)"
        atlas = f"[PDF](atlas_{method_id}_arteres.pdf)" if "artery_count" in stats else "—"
        succeeded = stats["normalized_status_counts"].get("ok", 0)
        lines.append(f"| {METRICS[key]} | {report} | {atlas} | {stats.get('artery_count', '—')} | {succeeded} / {stats['saved_vessels']} |")
    lines += ["", "Chaque métrique possède également un CSV par œil, un CSV de tous les vaisseaux,",
              "un classeur Excel classé et un fichier JSON avec les calculs intermédiaires sources.",
              "`manifest.json` contient les paramètres, effectifs et motifs d'exclusion des atlas.", "",
              "Le score de l'œil agrège artères et veines éligibles par longueur app;",
              "les atlas montrent uniquement les artères admissibles de rang 1–3.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def generate_toolkit_outputs(runs_root, output_dir, methods=None, *, run_names=None, atlases=True):
    root = Path(runs_root); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    runs = _numbered_runs(root)
    if run_names:
        unknown = set(run_names) - {run.name for run in runs}
        if unknown:
            raise ValueError(f"Sessions numerotees absentes: {sorted(unknown)}")
        runs = [run for run in runs if run.name in run_names]
    if not runs:
        raise ValueError("Aucune session numerotee *_OD / *_OG trouvee")
    manifest = {"source_revision": SOURCE_REVISION, "runs_root": str(root.resolve()),
                "runs": [run.name for run in runs], "methods": {}}
    for key in methods or METRICS:
        method_id = f"toolkit_{key.removeprefix('toolkit_')}"
        if method_id.removeprefix("toolkit_") not in METRICS:
            raise ValueError(f"Metrique inconnue: {key}")
        config = scoring_config(method_id)
        scored = []
        for run in runs:
            summary, table = score_run(run, config)
            scored.append((run, summary, table))
            print(f"{method_id}: {run.name}: {len(table)} vaisseaux", flush=True)
        summary_table = _build_local_bump_summary_table(scored)
        (output / f"rapport_tortuosite_{method_id}.pdf").write_bytes(
            _generate_local_bump_results_pdf_from_scores(config, scored, summary_table))
        summary_table.to_csv(output / f"scores_{method_id}.csv", index=False)
        tables = [table.assign(run=run.name) for run, _, table in scored if not table.empty]
        vessels = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        vessels.drop(columns=["path_points", "start_endpoint", "end_endpoint"], errors="ignore").to_csv(
            output / f"vaisseaux_{method_id}.csv", index=False)
        (output / f"classified_vessels_raw_{method_id}.xlsx").write_bytes(
            generate_classified_vessels_excel(runs, config))
        (output / f"diagnostics_{method_id}.json").write_bytes(toolkit_diagnostics_bytes(scored, config))
        stats = {"saved_vessels": len(vessels),
                 "normalized_status_counts": vessels["toolkit_status"].value_counts().to_dict() if not vessels.empty else {},
                 "raw_status_counts": vessels["toolkit_raw_status"].value_counts().to_dict() if not vessels.empty else {},
                 "parameters": dict(scoring_method_fixed_parameters(config))}
        if atlases:
            with PdfPages(output / f"atlas_{method_id}_arteres.pdf") as pdf:
                stats.update(write_toolkit_atlas(pdf, scored, config))
        manifest["methods"][method_id] = stats
        (output / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
        _write_output_index(output, manifest)
        print(f"{method_id}: termine — {stats}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Rapports et atlas Curve Toolkit sur les vaisseaux sauvegardes.")
    parser.add_argument("--runs-root", type=Path, default=Path("demo/streamlit_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_visualisations/toolkit"))
    parser.add_argument("--method", choices=["all", *METRICS], default="all")
    parser.add_argument("--run", dest="run_names", action="append", help="Session numerotee; option repetable")
    parser.add_argument("--skip-atlases", action="store_true")
    args = parser.parse_args()
    generate_toolkit_outputs(args.runs_root, args.output_dir,
                             None if args.method == "all" else [args.method],
                             run_names=args.run_names, atlases=not args.skip_atlases)


if __name__ == "__main__":
    main()
