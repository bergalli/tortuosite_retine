# Curve Toolkit: approved integration

All eleven methods are exposed as `Toolkit — <name>`, alongside the existing
methods. The default remains `local_bump`. Stable IDs are `toolkit_dm`,
`toolkit_mean_curvature`, `toolkit_total_curvature`, `toolkit_tau_3`,
`toolkit_tau_5`, `toolkit_asdc`, `toolkit_icm`, `toolkit_soam`, `toolkit_td`,
`toolkit_scc`, and `toolkit_escc`.

## Geometry and preprocessing

The input is the existing ordered saved-vessel centerline, including the app's
manual links and endpoint choices. Segmentation and skeletonization are not
rerun. The adapter starts from those coordinates; it does not run the standalone
TXT file loader or apply additional deduplication, smoothing, endpoint repair,
point-count clamping or inflection filtering.

For the primary score, the app's existing fundus-diameter normalization brings
the image diameter to 1024 px. On these input points, the source **batch** recipe
is applied:

1. `L = Curve.arclength()` and `M = ceil(N * 0.25)`.
2. `Curve.smooth("cubic_spline", smooth=8/L, num_points=M)`.
3. `.parametrize("scc", num_points=M)`.
4. Compute the selected source metric on the resulting curve.

This intentionally follows `compute_tortuosity_batch.py`, not the single-curve
example (which evaluates the smoothing spline at N points). `Curve.smooth`
itself uses the source's uniform parameter from 0 to L, not cumulative input
arc-length spacing. The source SCC algorithm walks circle/polyline
intersections; it is not replaced by ordinary arc-length resampling.

All curvature-based metrics use derivative `smooth=1.0`. This includes mean
curvature, which is registered but absent from the source batch script; the
choice was explicitly approved. ICM/TD retain `threshold=1e-4`.

The native-image-pixel score is computed independently using the same recipe
for `raw_primary_score` / workbook `score_raw`. The primary normalized-coordinate
value maps to `primary_score` / workbook `score_normalized`. A raw calculation
failure does not invalidate an otherwise successful normalized calculation.

## Source equations and preserved quirks

| Metric | Actual implemented computation |
| --- | --- |
| DM | Processed-curve arc length / endpoint chord |
| mean curvature | Sample mean of absolute csaps curvature; its default parameter is uniform from 0 to processed length |
| total curvature | Trapezoidal integral of absolute curvature against cumulative processed arc length |
| tau_3 | Trapezoidal integral of squared curvature against cumulative processed arc length |
| tau_5 | tau_3 / processed arc length |
| ASDC | `trapezoid(gradient(kappa)**2) / chord`, with gradient and integration both using unit index spacing |
| ICM | DM times (accepted inflection count + 1) |
| SOAM | Sum of absolute differences of atan2 directions in degrees, divided by processed arc length |
| TD | `((n-1)/n) * sum(accepted_DM_minus_1) / processed_length` |
| SCC | Sum of absolute corrected discrete turns divided by 180 degrees |
| ESCC | Source heart-pattern resampling followed by the same SCC angle sum |

SOAM does **not** correct crossing the -180/+180 degree direction boundary.
ASDC's code differs from its docstring: it uses index spacing and chord length.
The source SCC sampling, index choices, and intersections are preserved.
ICM ignores the final unclosed curvature portion in its inflection counter.
TD uses the source acceptance/reset logic and adds the final portion without
applying its excess threshold. These are properties of this source revision,
not corrections or claims of equivalence to the cited publications.

DM, tau_5, TD and SOAM remain separate from existing app methods with similar
names: their preprocessing and/or formulas differ. No existing method is
replaced. Any future mathematical correction requires user approval and a
separate method/version.

## Failures, aggregation and populations

Preprocessing and each metric operation have an independent 30-second deadline
in a persistent child process. On timeout the child is killed and restarted for
the next operation. Exceptions and non-finite source results become NA with a
reason. Other successful metrics survive. No zeros or alternate algorithms are
substituted. Where a source metric explicitly returns zero for fewer than three
processed points, that is retained as a successful source result. The source
recipe can fail on sparse saved paths, including long
paths for which `ceil(N/4)` leaves too few points.

Shared preprocessing and individual calculations are cached by exact geometry,
source revision, adapter cache version, parameters, timeout and NumPy/SciPy/csaps
versions. Memory-cached results are copied on retrieval. Disk cache defaults to
`<temporary-directory>/tortuosite-toolkit`; override it using
`TORTUOSITY_TOOLKIT_CACHE`. Delete that directory to force recomputation.

The app's default 100 normalized-pixel length filter is preserved. The eye score
uses finite, eligible saved vessels (arteries and veins), weighted by their
**app length before toolkit smoothing**. It has no tail component and no display
multiplier. This aggregation is an app convention, not part of the toolkit.

Reports retain the original Grisan report's statistical populations and tests:
classified artery/vein score distributions, one-sided Mann–Whitney comparisons,
and the existing Benjamini–Hochberg correction. The workbook's existing
classification rule can keep a named artery with an NA score in `Clean_vessels`;
its NA remains visible for audit and is excluded from calculations. In
particular, report p-value populations follow that existing classification rule
rather than being silently replaced by the eye-score length-filter population.

Atlases use the existing publication filter, plus a finite eligible score:
classified arteries of rank 1–3, ascending score, one detail page per artery.
They explicitly distinguish their population from the all-saved-vessel eye
aggregation, and list excluded ranked arteries with reasons.

## Diagnostics and output

The worker captures local arrays and scalars from the unmodified source calls
using a Python profiler. It records curvature, actual angle arrays,
arc/chord calls for tested portions, and ESCC points. The figures and numerical
substitutions use these captured values and verify parity with each scored row.
The companion JSON files retain all captured terms at full precision; PDF
numbers are rounded for readability.

Generate the numbered saved cohort (default `demo/streamlit_runs`):

```bash
uv run python -m tortuosite_score.visualization.toolkit_export
```

Select a method or eye for a smaller export:

```bash
uv run python -m tortuosite_score.visualization.toolkit_export \
  --method td --run 1_OD --output-dir paper_visualisations/toolkit_preview
```

The default output directory is `paper_visualisations/toolkit/`. For each metric:

- `rapport_tortuosite_toolkit_<metric>.pdf`: Grisan-style report and failure audit.
- `atlas_toolkit_<metric>_arteres.pdf`: complete artery calculation atlas.
- `scores_toolkit_<metric>.csv`: eye summary.
- `vaisseaux_toolkit_<metric>.csv`: all saved vessels, including failure reasons.
- `classified_vessels_raw_toolkit_<metric>.xlsx`: existing classified workbook with audit columns.
- `diagnostics_toolkit_<metric>.json`: raw and normalized source diagnostics.
- `manifest.json`: source revision, settings, status counts and atlas populations.

Existing score CLI `--method` choices include all toolkit IDs. In Streamlit,
select a toolkit method in the sidebar; the Results page offers the report,
workbooks, an all-vessel audit CSV, and a single **Telecharger les 3 PDF
(rapport + arteres + veines)** button. One click generates a ZIP containing the
main report, `atlas_toolkit_<metric>_arteres.pdf`, and
`atlas_toolkit_<metric>_veines.pdf`. The vein atlas uses the same publication
filter for ranked veins, source computations, index and exclusions as the artery
atlas, with vein-specific text and V labels. Generation starts only on download;
Streamlit 1.57 or newer provides the deferred-download callback. The JSON
diagnostics remain available as a separate download.
