from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "clinical_analysis_audit.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """## tl;dr

Après normalisation de la géométrie sur un diamètre de fond d'œil commun, les patients ont des scores supérieurs aux témoins pour les trois rangs (Wilcoxon apparié, p bilatérales corrigées de Holm ≤ 0,014). En revanche, les données ne montrent pas que le rang 3 est systématiquement plus tortueux que les rangs 1 et 2. L'association bilatérale est surtout visible au rang 3. La comparaison artères–veines de rang 3 est suggestive mais sous-puissante, car seules neuf paires œil/artères/veines sont disponibles. Les analyses âge et sévérité ne sont pas calculables sans `demo/clinical_metadata.csv`.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

L'unité primaire est le patient. Les comparaisons patient–témoin sont appariées sur `patient_id`, œil et rang. Les deux yeux d'un patient ne sont pas traités comme deux patients indépendants pour la comparaison des rangs. Les trajectoires sont ramenées à un diamètre de fond d'œil de 1024 px avant le rééchantillonnage, le lissage et le seuil de longueur.

### Key Assumptions

- `OD_de_<id>` ou `OG_de_<id>` désigne le témoin apparié au patient `<id>`.
- `1°A`, `2°A`, `3°A` désignent les rangs artériels; `inf` et `sup` désignent les territoires inférieur et supérieur.
- Les tests bilatéraux sont les résultats principaux; les p-values directionnelles sont secondaires et correspondent aux hypothèses cliniques préspécifiées.
- Une absence de différence OD–OG ne démontre pas une équivalence. L'association OD–OG est donc aussi évaluée par Spearman.
"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Recalculer et structurer les vaisseaux sauvegardés"),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

from tortuosite_score.vessels_detection.clinical_excel import (
    build_structured_vessel_data,
    _arteries_vs_veins_sheet,
    _od_vs_og_sheet,
    _patient_vs_control_sheet,
    _rank_eye_scores,
    _same_eye_rank_sheet,
)
from tortuosite_score.vessels_detection.scoring import scoring_config

project_root = Path.cwd()
if not (project_root / "demo" / "streamlit_runs").exists():
    project_root = Path.cwd().parent
run_dirs = sorted(path for path in (project_root / "demo" / "streamlit_runs").iterdir() if path.is_dir())
structured, quality = build_structured_vessel_data(run_dirs, scoring_config("local_bump"))
rank_scores = _rank_eye_scores(structured)
print({
    "runs_exploitables": int(structured["run"].nunique()),
    "vaisseaux_sauvegardes": int(len(structured)),
    "vaisseaux_eligibles": int(structured["eligible"].sum()),
    "patients": int(structured.loc[structured["groupe"] == "patient", "patient_id"].nunique()),
})"""
        ),
        nbf.v4.new_markdown_cell("### 2. Vérifier le naming, la complétude et la résolution"),
        nbf.v4.new_code_cell(
            """naming_profile = (
    structured.groupby(["groupe", "type_vaisseau", "rang"], dropna=False)
    .size().rename("n_vaisseaux").reset_index()
)
display(naming_profile)

resolution_profile = (
    structured[["run", "groupe", "diametre_fond_oeil_px", "facteur_normalisation"]]
    .drop_duplicates("run")
    .groupby("groupe")
    .agg(
        n_runs=("run", "count"),
        diametre_median_px=("diametre_fond_oeil_px", "median"),
        diametre_min_px=("diametre_fond_oeil_px", "min"),
        diametre_max_px=("diametre_fond_oeil_px", "max"),
    )
)
display(resolution_profile.round(1))
display(quality["probleme"].value_counts().rename_axis("probleme").reset_index(name="n"))"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 3. Comparer les rangs au niveau patient"),
        nbf.v4.new_code_cell(
            """same_eye = _same_eye_rank_sheet(rank_scores)
display(same_eye.loc[same_eye["section"] == "resume"].dropna(axis=1, how="all").round(4))"""
        ),
        nbf.v4.new_markdown_cell("### 4. Comparer chaque rang entre patient et témoin apparié"),
        nbf.v4.new_code_cell(
            """patient_control = _patient_vs_control_sheet(rank_scores)
display(patient_control.loc[patient_control["section"] == "resume"].dropna(axis=1, how="all").round(4))"""
        ),
        nbf.v4.new_markdown_cell("### 5. Évaluer la différence et l'association entre OD et OG"),
        nbf.v4.new_code_cell(
            """od_og = _od_vs_og_sheet(rank_scores)
display(od_og.loc[od_og["section"] == "resume"].dropna(axis=1, how="all").round(4))"""
        ),
        nbf.v4.new_markdown_cell("### 6. Comparer artères et veines"),
        nbf.v4.new_code_cell(
            """arteries_veins = _arteries_vs_veins_sheet(structured)
display(arteries_veins.loc[arteries_veins["section"] == "resume"].dropna(axis=1, how="all").round(4))"""
        ),
        nbf.v4.new_markdown_cell("### 7. Vérifier le biais résiduel lié au format d'acquisition"),
        nbf.v4.new_code_cell(
            """eligible_arteries = structured[
    (structured["eligible"])
    & (structured["type_vaisseau"] == "artere")
    & (structured["rang"].isin([1, 2, 3]))
].copy()
eye_scores = (
    eligible_arteries.groupby(["run", "groupe"])
    .apply(lambda group: (group["score"] * group["longueur"]).sum() / group["longueur"].sum(), include_groups=False)
    .rename("score_moyen_pondere").reset_index()
    .merge(
        structured[["run", "diametre_fond_oeil_px"]].drop_duplicates("run"),
        on="run",
        how="left",
    )
)
for group_name, group in eye_scores.groupby("groupe"):
    rho, p_value = spearmanr(group["diametre_fond_oeil_px"], group["score_moyen_pondere"])
    print(group_name, {"n": len(group), "rho_resolution_score": round(float(rho), 3), "p_value": round(float(p_value), 4)})"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Le premier export mélangeait une unité pixel non comparable entre acquisitions. Cette erreur est corrigée par une normalisation géométrique avant scoring.
2. La différence patient–témoin est robuste dans cet échantillon pour les trois rangs, mais elle reste potentiellement confondue par le protocole d'acquisition: les groupes proviennent de résolutions très différentes et la normalisation ne remplace pas un plan d'acquisition harmonisé.
3. Le rang 3 n'est pas « toujours » supérieur aux rangs 1 et 2 selon le score local-bump; environ la moitié à deux tiers des patients seulement ont un delta positif selon la comparaison.
4. La question OD–OG est mieux traitée comme une question d'association que comme un simple test de différence. Le signal le plus net concerne le rang 3.
5. Le contraste artères–veines de rang 3 va dans le sens clinique, mais neuf paires seulement sont disponibles. Il faut compléter les annotations veineuses avant de conclure.
6. Il faut fournir `demo/clinical_metadata.csv` (`patient_id,age,severite`) pour les deux corrélations cliniques.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
