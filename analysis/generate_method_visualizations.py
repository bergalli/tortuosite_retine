from __future__ import annotations

import argparse
from pathlib import Path

from tortuosite_score.visualization.method_figures import generate_method_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère le pipeline VascX, l’explication illustrée et l’atlas Local-bump v1.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("demo/streamlit_runs"),
        help="Répertoire contenant les sessions Streamlit sauvegardées.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualisations"),
        help="Répertoire de destination des figures.",
    )
    parser.add_argument(
        "--pipeline-run",
        default="2_OD",
        help="Session réelle utilisée pour la figure, ou 'all' pour tous les yeux numérotés.",
    )
    parser.add_argument(
        "--vessel-type",
        choices=("veine", "artere"),
        default="veine",
        help="Population illustrée dans l’atlas et retenue dans le panneau final du pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_method_visualizations(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        pipeline_run=args.pipeline_run,
        vessel_type=args.vessel_type,
    )
    for pdf, png in zip(result["pipeline_pdfs"], result["pipeline_pngs"]):
        print(f"Figure du pipeline (PDF) : {pdf}")
        print(f"Figure du pipeline (PNG) : {png}")
    print(f"Atlas Local-bump : {result['atlas_pdf']}")
    print(f"Explication Local-bump (PDF) : {result['overview_pdf']}")
    print(f"Explication Local-bump (PNG) : {result['overview_png']}")
    label = "Artères" if args.vessel_type == "artere" else "Veines"
    print(f"{label} incluses : {result['vessel_count']}")


if __name__ == "__main__":
    main()
