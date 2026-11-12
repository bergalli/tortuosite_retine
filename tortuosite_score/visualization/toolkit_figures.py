"""French, source-backed calculation atlases for Curve Toolkit metrics."""
from __future__ import annotations

import io
import math
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image

from tortuosite_score.vessels_detection.clinical_excel import parse_vessel_name
from tortuosite_score.vessels_detection.scoring import scoring_method_spec
from tortuosite_score.vessels_detection.toolkit_metrics import toolkit_metrics, FORMULAS, SOURCE_REVISION
from tortuosite_score.visualization.method_figures import (
    AtlasVessel, is_ranked_publication_vessel, _source_path, _vessel_labels,
)
from tortuosite_score.visualization import anatomy_panels as anatomy

INDEX_ROWS = 28
INK = "#172b41"
BLUE = "#2163a0"
RED = "#b43d37"


def _num(value):
    try:
        return f"{float(value):.7g}"
    except (TypeError, ValueError):
        return "NA"


def atlas_population(scored_runs, *, vessel_type="artere"):
    _vessel_labels(vessel_type)
    records = []
    exclusions = []
    for run, summary, table in scored_runs:
        for _, row in table.iterrows():
            info = parse_vessel_name(str(row.get("vessel_name", "")), row.get("category"))
            if info.vessel_type != vessel_type or info.rank not in {1, 2, 3}:
                continue
            if is_ranked_publication_vessel(row, vessel_type):
                records.append(AtlasVessel(run, _source_path(run), row.to_dict(), info, summary, table))
            else:
                if row.get("toolkit_status") != "ok":
                    reason = f"{row.get('toolkit_status')}: {row.get('toolkit_failure_reason')}"
                elif not row.get("meets_length_threshold", False):
                    reason = "Sous le seuil de longueur app"
                else:
                    reason = "Filtre d'export classe / score non admissible"
                exclusions.append({"Image": run.name, "Vaisseau": row.get("vessel_name"), "Motif": reason})
    return sorted(records, key=lambda r: (r.row["primary_score"], r.run_dir.name, r.row["vessel_name"])), exclusions


def _text(fig, x, y, text, *, size=10, width=110, **kwargs):
    return fig.text(x, y, "\n".join(textwrap.fill(line, width) for line in str(text).split("\n")),
                    va="top", fontsize=size, color=INK, **kwargs)


def _table_page(pdf, title, table, *, widths=None):
    fig = Figure(figsize=(11.69, 8.27), facecolor="white")
    _text(fig, .04, .96, title, size=16, weight="bold")
    ax = fig.add_axes([.04, .04, .92, .85]); ax.axis("off")
    if table.empty:
        _text(fig, .05, .82, "Aucun vaisseau admissible pour cette métrique.", size=12)
        pdf.savefig(fig)
        return
    displayed = table.map(lambda v: textwrap.fill(str(v), 70 if len(str(v)) > 90 else 48))
    tab = ax.table(cellText=displayed.values, colLabels=table.columns, colWidths=widths,
                   loc="upper center", cellLoc="left")
    tab.auto_set_font_size(False); tab.set_fontsize(8)
    for (row, col), cell in tab.get_celld().items():
        cell.set_height(.94 / (len(table) + 1))
        cell.set_edgecolor("#dbe2ea")
        if row == 0:
            cell.set_facecolor(INK); cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    pdf.savefig(fig, dpi=120)


def _cover(pdf, config, records, exclusions, *, vessel_type="artere"):
    labels = _vessel_labels(vessel_type)
    method = scoring_method_spec(config.method_id)
    fig = Figure(figsize=(11.69, 8.27), facecolor="white")
    _text(fig, .05, .94, f"{method.label} — comprendre le calcul", size=22, weight="bold")
    _text(fig, .05, .86, f"Atlas de {len(records)} {labels.plural} · fiches par score croissant · {len(exclusions)} {labels.plural} classées exclues", size=12)
    left = ["1 · Reprendre le tracé sauvegardé",
            "Les mêmes segments, liens manuels et extrémités que dans l'app sont conservés. Le diamètre du fond d'œil est ramené à 1024 px.",
            "", "2 · Appliquer exactement la recette batch",
            "N est le nombre de points d'entrée; L est leur longueur normalisée. Le lissage csaps utilise p=8/L et M=ceil(N/4) points. La paramétrisation SCC du toolkit utilise ensuite M points.",
            "", "3 · Calculer la métrique du code source", FORMULAS[config.method_id.removeprefix("toolkit_")],
            "Les dérivées csaps utilisent smooth=1.0. ICM et TD gardent leur seuil d'excès 0.0001. Aucun filtre Local-bump n'est ajouté."]
    right = ["4 · Lire les fiches",
             "Le tracé brut est gris, le tracé traité bleu. Les graphiques et les substitutions numériques proviennent des valeurs intermédiaires capturées lors du calcul source du score.",
             "", "5 · Respecter les particularités du toolkit",
             "SOAM conserve les différences d'angles sans correction -180/+180°. ASDC utilise un gradient et une intégrale par indice, divisés par la corde. Les choix SCC et les partitions ICM/TD du code sont conservés.",
             "", "6 · Distinguer atlas et score de l'œil",
             f"Atlas : {labels.plural} nommées de rang 1–3, export classé autorisé, longueur admissible et score fini. Le score de l'œil utilise tous les vaisseaux sauvegardés éligibles, artères et veines, pondérés par leur longueur app.",
             "Aucun multiplicateur ni queue supérieure. Les NA sont exclus des statistiques, jamais remplacés par zéro."]
    _text(fig, .05, .77, "\n".join(left), width=65, size=10)
    _text(fig, .53, .77, "\n".join(right), width=65, size=10)
    _text(fig, .05, .09, f"Source : {SOURCE_REVISION}\nLimite de calcul : 30 s par opération; NA avec motif en cas d'échec.\nLes équations décrivent le code local; ces exemples ne constituent pas une interprétation clinique.", size=9)
    pdf.savefig(fig)


def numeric_substitution(metric, result):
    """Use captured source locals, including its indexing and accepted partitions."""
    d = result["diagnostics"]
    c = d.get("compute", {})
    value = result["value"]
    if c.get("n", 3) < 3 and metric in {"mean_curvature", "total_curvature", "tau_3", "tau_5", "asdc", "icm", "td", "soam"}:
        return f"{c['n']} points traités (< 3) : retour anticipé du code source = {_num(value)}. Aucun terme de courbure/angle n’est calculé."
    if metric == "dm":
        lines = [f"L = {_num(c.get('arc_length'))}; C = {_num(c.get('chord'))}",
                 f"DM = {_num(c.get('arc_length'))} / {_num(c.get('chord'))} = {_num(value)}"]
    elif metric == "mean_curvature":
        k = np.asarray(c["kappa"])
        lines = [f"Somme |kappa| = {_num(np.abs(k).sum())}; nombre de valeurs = {len(k)}",
                 f"M = {_num(np.abs(k).sum())} / {len(k)} = {_num(value)}",
                 "Le toolkit utilise ici une moyenne des échantillons (pas une intégrale pondérée)."]
    elif metric in {"total_curvature", "tau_3"}:
        k = np.asarray(c["kappa"]); t = np.asarray(c["t"])
        y = np.abs(k) if metric == "total_curvature" else k**2
        terms = .5 * (y[1:] + y[:-1]) * np.diff(t)
        lines = [f"{len(k)} valeurs de courbure; {len(terms)} trapèzes",
                 f"Somme des contributions trapézoïdales = {_num(terms.sum())} = {_num(value)}"]
    elif metric == "tau_5":
        lines = [f"tau_3 = {_num(c['Tau_3'])}; L = {_num(c['L_c'])}",
                 f"tau_5 = {_num(c['Tau_3'])} / {_num(c['L_c'])} = {_num(value)}"]
    elif metric == "asdc":
        lines = [f"{len(c['dkappa_dt'])} gradients par indice; intégrale = {_num(c['integral'])}",
                 f"C = {_num(c['L_x'])}; ASDC = {_num(c['integral'])} / {_num(c['L_x'])} = {_num(value)}",
                 "Le pas du gradient et de l'intégrale est 1. Le dénominateur est la corde."]
    elif metric == "icm":
        lines = [f"Inflexions retenues = {c['n_inflections']}; DM global = {_num(c['DM'])}",
                 f"ICM = {_num(c['DM'])} x ({c['n_inflections']} + 1) = {_num(value)}",
                 "Chaque portion testée doit avoir au moins 3 points et un excès DM-1 > 0.0001.",
                 "La dernière courbure n'est pas ajoutée au compteur par le code source."]
    elif metric == "td":
        terms = c["L"]
        lines = [f"n = {c['n_ic']}; L = {_num(c['L_c'])}; somme des excès = {_num(sum(terms))}",
                 f"TD = (({c['n_ic']}-1)/{c['n_ic']}) x {_num(sum(terms))} / {_num(c['L_c'])} = {_num(value)}",
                 "Excès retenus : " + "; ".join(_num(v) for v in terms),
                 "La portion finale est ajoutée sans test de seuil, comme dans la source."]
    elif metric == "soam":
        total = np.abs(c["alpha"]).sum()
        lines = [f"{len(c['alpha'])} différences d'angles; somme absolue = {_num(total)}°; L = {_num(c['arc_length'])}",
                 f"SOAM = {_num(total)} / {_num(c['arc_length'])} = {_num(value)} degrés/px normalisé",
                 "Les sauts de direction entre -180° et +180° sont conservés sans correction."]
    else:
        angles = np.asarray(c["alpha"])
        lines = [f"{len(angles)} angles corrigés, divisés par 180°",
                 f"Somme des valeurs absolues = {_num(np.abs(angles).sum())} = {_num(value)}"]
        if metric == "escc":
            lines.append(f"Construction ESCC : {len(d['escc_points'])} points; distance de référence = {_num(d['escc']['L_x'])}.")
    return "\n".join(lines)


def _detail_page(record, config, index, total, page):
    labels = _vessel_labels(record.name_info.vessel_type)
    metric = config.method_id.removeprefix("toolkit_")
    raw = np.asarray(record.row["path_points"], dtype=float)
    scale = float(record.row["coordinate_scale"])
    result = toolkit_metrics(raw * scale, config.method_id)
    if result["status"] != "ok" or not np.isclose(result["value"], record.row["primary_score"], rtol=1e-12, atol=1e-14):
        raise ValueError(f"Atlas/score incohérents : {record.run_dir.name}/{record.row['vessel_name']}")
    prep = result["preprocessing"]
    processed = np.asarray(prep["points"])
    diagnostics = result["diagnostics"]
    c = diagnostics["compute"]
    fig = Figure(figsize=(16.54, 11.69), facecolor="white")
    _text(fig, .035, .965, f"{record.run_dir.name} · {record.row['vessel_name']} · {labels.rank(record.name_info.rank)}", size=21, weight="bold")
    _text(fig, .035, .922, f"Toolkit — {metric.upper() if metric in {'dm','asdc','icm','soam','td','scc','escc'} else metric}   |   Score = {_num(result['value'])}   |   L app = {_num(record.row['vessel_length'])} px", size=12)
    fig.text(.965, .96, f"Fiche {index}/{total} · page {page}", ha="right", fontsize=10, color=INK)
    # Raw and processed coordinates share image coordinates, never rotated or stretched.
    with Image.open(record.source_path) as image:
        source = image.convert("RGB")
    left = fig.add_axes([.035, .52, .29, .33])
    box = anatomy.crop_box(np.concatenate([raw, processed / scale]), source.size)
    anatomy.backdrop(left, source, box, fade=.15)
    left.plot(*raw.T, color="#717b85", linewidth=3, label="Tracé sauvegardé")
    left.plot(*(processed / scale).T, color=BLUE, linewidth=1.7, marker=".", markersize=3, label="Tracé traité SCC")
    left.scatter(*(processed[[0, -1]] / scale).T, s=25, color=["#35a36d", RED], zorder=5)
    left.legend(loc="upper left", fontsize=8)
    left.set_title(f"1 · {labels.singular.capitalize()} réelle, points source et points traités", fontsize=11)
    right = fig.add_axes([.365, .52, .28, .33])
    right.plot(*processed.T, "o-", color=BLUE, markersize=2, linewidth=1)
    right.set_aspect("equal"); right.invert_yaxis()
    right.set_xlabel("x normalisé (px)"); right.set_ylabel("y normalisé (px)")
    right.set_title("2 · Géométrie réellement mesurée", fontsize=11)
    calls = diagnostics.get("arc_chord_calls", [])
    if metric == "icm":
        calls = calls[:-1]  # last call is the whole-curve DM, not a tested portion
    if metric in {"td", "icm"}:
        for j, call in enumerate(calls):
            x, y = call["x"], call["y"]
            excess = call["arc_length"] / call["chord"] - 1 if call["chord"] else math.inf
            accepted = excess > 1e-4 or (metric == "td" and j == len(calls)-1)
            right.plot(x, y, color=RED if accepted else "#aaaaaa", linewidth=2, alpha=.8)
            right.annotate(str(j+1), (x[len(x)//2], y[len(y)//2]), fontsize=7)
        right.set_title("2 · Portions testées : rouge retenu, gris rejeté", fontsize=11)
    elif metric == "dm":
        right.plot(*processed[[0,-1]].T, "--", color=RED, label="Corde C"); right.legend(fontsize=8)
    elif metric == "escc":
        escc = np.asarray(diagnostics["escc_points"])
        right.plot(*escc.T, "o-", color=RED, markersize=3, linewidth=1, label="Points ESCC"); right.legend(fontsize=8)
    _text(fig, .69, .84, f"3 · Paramètres de cette {labels.singular}", size=13, weight="bold")
    _text(fig, .69, .80,
          f"N = {prep['input_point_count']} points d'entrée\nL entrée = {_num(prep['input_length'])} px\nM = ceil(N/4) = {prep['output_point_count']}\np = 8/L = {_num(prep['smooth'])}\nFacteur image → normalisé = {_num(scale)}\nL app (filtre et pondération) = {_num(record.row['vessel_length'])} px\nDérivées csaps : smooth=1.0\nSeuil ICM/TD = 0.0001",
          width=42, size=10)
    ax = fig.add_axes([.055, .245, .59, .19])
    if "curvature" in diagnostics:
        curvature = diagnostics["curvature"]; k = np.asarray(curvature["kappa"]); t = np.asarray(curvature["t"])
        if metric in {"tau_3", "tau_5"}:
            y = k**2; label = "Courbure au carré kappa²"
        elif metric == "asdc":
            y = np.asarray(c["dkappa_dt"])**2; t = np.arange(len(y)); label = "gradient(kappa)², pas d'indice = 1"
        elif metric in {"mean_curvature", "total_curvature"}:
            y = np.abs(k); label = "Courbure absolue |kappa|"
        else:
            y = k; label = "Courbure signée kappa (source)"
        ax.plot(t, y, color=BLUE, linewidth=1); ax.fill_between(t, 0, y, color=BLUE, alpha=.12)
        ax.set_xlabel("Indice" if metric == "asdc" else "Paramètre source de la courbe (px)")
        ax.set_title(label, fontsize=11)
    elif metric in {"soam", "scc", "escc"} and "alpha" in c:
        y = np.asarray(c["alpha"]); ax.bar(np.arange(len(y)), y, color=np.where(y >= 0, RED, BLUE))
        ax.set_xlabel("Indice de l'angle"); ax.set_ylabel("Degrés" if metric == "soam" else "Angle / 180°")
        ax.set_title("Angles effectivement sommés (signe conservé pour la lecture)", fontsize=11)
    else:
        lengths = np.linalg.norm(np.diff(processed, axis=0), axis=1)
        ax.plot(np.cumsum(lengths), color=BLUE); ax.set_xlabel("Indice du segment"); ax.set_ylabel("Longueur cumulée (px)")
        ax.set_title("Longueur d'arc cumulée sur le tracé traité", fontsize=11)
    ax.grid(alpha=.18)
    _text(fig, .69, .435, "4 · Formule du code source", size=13, weight="bold")
    _text(fig, .69, .395, FORMULAS[metric], width=43, size=10)
    _text(fig, .035, .16, "5 · Substitution numérique\n" + numeric_substitution(metric, result), width=170, size=10)
    _text(fig, .69, .30,
          f"Œil : moyenne pondérée = {_num(record.summary.get('eye_tortuosity_score'))}\nTous les vaisseaux éligibles : {record.summary.get('eligible_vessel_count')}.\nLa pondération utilise L app, avant le lissage toolkit.\nLes artères et veines éligibles participent au score de l'œil.", width=43, size=9)
    _text(fig, .035, .025, "Même calcul que dans l'app · Source " + SOURCE_REVISION + " · Détails complets : fichier diagnostics JSON associé.", size=8, width=180)
    return fig


def write_toolkit_atlas(pdf, scored_runs, config, *, vessel_type="artere"):
    labels = _vessel_labels(vessel_type)
    records, exclusions = atlas_population(scored_runs, vessel_type=vessel_type)
    _cover(pdf, config, records, exclusions, vessel_type=vessel_type)
    pages = max(1, math.ceil(len(records) / INDEX_ROWS))
    for offset in range(0, max(1, len(records)), INDEX_ROWS):
        rows = [{"Fiche": i+1, "Page PDF": 2+pages+i, "Œil": r.run_dir.name,
                 labels.singular.capitalize(): r.row["vessel_name"], "Score": _num(r.row["primary_score"]),
                 "L app (px)": _num(r.row["vessel_length"])}
                for i, r in enumerate(records) if offset <= i < offset+INDEX_ROWS]
        _table_page(pdf, f"Index des {labels.plural} · {config.method_id} · {offset//INDEX_ROWS+1}/{pages}",
                    pd.DataFrame(rows, columns=["Fiche", "Page PDF", "Œil", labels.singular.capitalize(), "Score", "L app (px)"]))
    for i, record in enumerate(records):
        pdf.savefig(_detail_page(record, config, i+1, len(records), 2+pages+i), dpi=120)
    for offset in range(0, len(exclusions), 20):
        _table_page(pdf, f"{labels.plural.capitalize()} classées exclues : {len(exclusions)} · motifs", pd.DataFrame(exclusions[offset:offset+20]), widths=[.1,.22,.68])
    count_key = "artery_count" if vessel_type == "artere" else "vein_count"
    return {count_key: len(records), f"excluded_{count_key}": len(exclusions),
            "exclusion_reasons": dict(Counter(r["Motif"] for r in exclusions))}


def toolkit_diagnostic_records(scored_runs, config):
    """Full raw/normalized source diagnostics for all saved vessels, including NA."""
    records = []
    for run, _, table in scored_runs:
        for _, row in table.iterrows():
            raw = np.asarray(row["path_points"], dtype=float)
            records.append({"run": run.name, "vessel_name": row["vessel_name"],
                "coordinate_scale": row["coordinate_scale"], "input_points": raw.tolist(),
                "normalized": toolkit_metrics(raw * float(row["coordinate_scale"]), config.method_id),
                "raw": toolkit_metrics(raw, config.method_id)})
    return records


def toolkit_diagnostics_bytes(scored_runs, config):
    import json
    def safe(value):
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value
    return json.dumps(safe(toolkit_diagnostic_records(scored_runs, config)), ensure_ascii=False).encode("utf-8")


def toolkit_atlas_bytes(scored_runs, config, *, vessel_type="artere"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        write_toolkit_atlas(pdf, scored_runs, config, vessel_type=vessel_type)
    return buffer.getvalue()
