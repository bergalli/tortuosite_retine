# Run with

```bash
uv sync                          # classical + Streamlit + skan
uv sync --extra deep             # DCP / VascX backends; skan is still included
uv run streamlit run tortuosite_score/app/app.py
```

Without `skan`, skeleton tortuosity uses a built-in scikit-image graph fallback (no LLVM/numba build required on Intel Mac).

# Saved-vessel tortuosity scoring

Compute the saved-vessel tortuosity score from already saved Streamlit/VascX runs:

```bash
uv run python -m tortuosite_score.vessels_detection.local_bump_score demo/streamlit_runs
```

This writes eye-level and saved-vessel score CSVs without rerunning segmentation. The primary detail CSV contains saved vessels: vessels selected by hand and vessels created with the app's auto-complete button.
By default, the CLI and PDF report use the numbered `*_OD` / `*_OG` cohort when present, because those are the clinically labelled eyes. Use `--include-all-runs` to include legacy-style folders such as `OD_de_*`.
The app sidebar now exposes the active scoring method globally; review tables, Results, PDF reports, and regenerated CSV exports all follow that same method.

## Method currently used in the app

The app now separates two concerns:

1. `Auto-complete skeleton into saved vessels` is a review tool.
   It reconstructs longer VascX paths and saves them as ordinary review vessels named `auto_vascx_*`.
2. The final tortuosity score is computed only on saved vessels.
   This includes manually saved vessels and any auto-completed vessels that the user chose to keep.

So the final method does **not** score the whole skeleton directly. The whole-skeleton graph logic is only used upstream to create reviewable saved vessels.

## Where the new method replaced the old one

- The Streamlit `Results` tab uses the active saved-vessel scoring method.
- The PDF report uses the same active saved-vessel scoring method.
- The CLI `tortuosite_score.vessels_detection.local_bump_score` accepts `--method` and uses the same centralized scoring service.
- `Local-bump v2 (experimental)`, `Arc/chord`, `Courbure quadratique`, `Tortuosity Density`, and `Somme des angles externes (RDP)` can also be selected as primary methods; `Arc/chord` remains available as a diagnostic column.

## Scoring method

The score is designed for local vessel irregularity rather than broad smooth curvature.

For each saved vessel, let the ordered centerline be:

\[
\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_n \in \mathbb{R}^2
\]

After light resampling and smoothing, define local segment directions:

\[
\theta_i = \operatorname{atan2}(y_{i+1}-y_i,\; x_{i+1}-x_i)
\]

and local turning-angle changes:

\[
\Delta \theta_i = \theta_{i+1} - \theta_i
\]

Small angle changes are suppressed with a curvature threshold \(\tau\):

\[
\widetilde{\Delta \theta_i} =
\begin{cases}
\Delta \theta_i & \text{if } |\Delta \theta_i| \ge \tau \\
0 & \text{otherwise}
\end{cases}
\]

The vessel-level local-bump energy is:

\[
E_v = \frac{1}{m}\sum_{i=1}^{m} |\widetilde{\Delta \theta_i}|
\]

where \(m\) is the number of valid local angle changes.

The oscillation count is the number of sign changes in the non-zero filtered angle sequence. Its density is:

\[
D_v = 100 \cdot \frac{N_v}{L_v}
\]

where:

- \(N_v\) is the number of curvature sign changes
- \(L_v\) is the vessel path length in pixels

The saved-vessel tortuosity score is:

\[
B_v = E_v \sqrt{D_v}
\]

Only saved vessels with length at least \(L_{\min} = 100\) px contribute to the eye-level score by default.

### Eye-level score

For the eligible saved vessels \(v \in V\):

\[
G = \frac{\sum_{v \in V} L_v B_v}{\sum_{v \in V} L_v}
\]

This is the global length-weighted mean local-bump burden.

The tail component keeps the most tortuous long vessels from being diluted. Let \(\rho = 0.20\) be the target tail-length fraction. Sort vessels by \(B_v\) from high to low, then compute a length-weighted mean on the leading vessels until their cumulative length reaches \(\rho \sum_v L_v\):

\[
T = \operatorname{TailMean}_\rho(B_v, L_v)
\]

The final eye score is:

\[
S_{\text{eye}} = 1000 \cdot \left( \alpha G + (1-\alpha)T \right)
\]

with default \(\alpha = 0.70\).

### Comparative report score

The report also shows a cohort-relative comparative score:

\[
C_{\text{eye}} = 100 \cdot \frac{S_{\text{eye}} - \min(S)}{\max(S) - \min(S)}
\]

computed within the set of eyes included in the report.

### Diagnostic arc/chord value

The older geometric tortuosity is still exported for diagnosis only:

\[
\text{arc/chord} = \frac{L_v}{\|\mathbf{p}_n - \mathbf{p}_0\|}
\]

It is no longer the default primary score.

### Experimental Local-bump v2

Local-bump v2 is available alongside the unchanged default method. It uses the same normalized saved-vessel geometry, with four fixed method parameters: a 4 px resampling step, a 5-point smoothing span, a minimum persistent-lobe angle of 0.15 rad, and an angularity weight of 0.25.

The smoother does not restore raw endpoints, and the two-sample convolution boundary is excluded. Consecutive signed turns are grouped into curvature lobes; weak lobes are removed iteratively and equal-sign neighbors are merged. The retained oscillation and local angularity components are:

\[
O_v = E_v\sqrt{\frac{100N_v}{L_v}}, \qquad
A_v = \operatorname{RMS}(\Delta\theta)\sqrt{\frac{100}{L_v}}
\]

and the experimental vessel score is:

\[
S_{v2}=1000\left(0.75O_v+0.25A_v\right)
\]

The classified-vessel workbook exports both v1 and v2 scores, both v2 components, persistent-lobe counts, endpoint maximum turn, and manual/model segment counts for audit.

Expert calibration uses pairwise labels with columns `left_run`, `left_vessel`, `right_run`, `right_vessel`, and `judgment`, where judgment is `left`, `right`, or `similar`:

```bash
uv run python -m tortuosite_score.vessels_detection.local_bump_calibration \
  --write-template expert_pairs.xlsx

uv run python -m tortuosite_score.vessels_detection.local_bump_calibration \
  expert_pairs.xlsx --runs-root demo/streamlit_runs
```

The calibration report selects parameters within patient-held-out folds, compares strict expert-pair concordance with local-bump v1, and recommends promotion only when the lower bound of the patient-level bootstrap improvement interval is non-negative. Until that evidence exists, v2 remains explicitly experimental and `local_bump` remains the default.

### Optional curvature-squared score

The app can also use a direct curvature-energy score:

\[
Q_v = \frac{1}{L_v}\int_0^{L_v} \kappa(s)^2\,ds
\]

where \(s\) is arc length and the local curvature is estimated from the resampled, lightly smoothed centerline:

\[
\kappa(s) =
\frac{|x'(s)y''(s)-y'(s)x''(s)|}{(x'(s)^2+y'(s)^2)^{3/2}}
\]

This score is displayed and aggregated as the raw mathematical value. The eye-level aggregation is the length-weighted mean across eligible saved vessels.

### Optional Tortuosity Density score

The app also implements the Grisan Tortuosity Density formula supplied for this project. The normalized vessel is divided into (n) subsegments whose curvature has constant sign:

\[
\tau_{TD}=\frac{n-1}{L}\sum_{i=1}^{n}\left(\frac{L_i}{C_i}-1\right)
\]

where (L) is the total vessel length, (L_i) is the arc length of subsegment (i), and (C_i) is its chord. The centerline is resampled every 4 px and smoothed over 5 points to locate stable inflections; angle changes below 0.035 rad are ignored. Arc and chord measurements themselves use the normalized, unsmoothed geometry. A vessel with no significant change of curvature sign has (n=1) and therefore scores zero.

The eye-level Tortuosity Density score is the length-weighted mean across eligible saved vessels. It has no upper-tail component and no display multiplier.

### Optional external-angle-sum score

The app also implements a formula supplied for this project, where vessel tortuosity is defined as the sum of all recorded external angles in a single traced vessel:

> "Vessel tortuosity was defined as the sum of all recorded external angles (Θ) in a single traced vessel."

\[
T_v = \sum_{i=1}^{n} |\theta_i|
\]

The underlying algorithm never computes a continuous curvature \(\kappa(s)\). Instead, for each saved vessel it:

1. Reuses the already-extracted, already-skeletonized vessel centerline (the four earlier pipeline steps: segmentation, skeletonization, branch extraction, saved-vessel reconstruction).
2. Simplifies that ordered centerline into straight segments with the Ramer-Douglas-Peucker algorithm, using a fixed tolerance \(\epsilon\) (in normalized px) to discard points that do not bend the trace beyond that tolerance. This keeps only the true "bend points" \(p_0, p_1, \dots, p_n\).
3. Computes the external (turning) angle \(\theta_i\) at each retained interior bend point, i.e. the angle between the incoming and outgoing straight segments.
4. Sums the absolute value of every external angle, in degrees, to obtain \(T_v\).

This metric measures the number of turns and how sharp each turn is (close to the total variation of direction along the vessel). It does not directly depend on the vessel length, its radius of curvature, or the derivative of curvature. The RDP tolerance \(\epsilon\) is a fixed method parameter (default 3 normalized px), shown in the app sidebar and PDF report like the other methods' fixed parameters.

The eye-level score is the length-weighted mean across eligible saved vessels, with no upper-tail component and no display multiplier, exactly like the curvature-squared and Tortuosity Density scores above.

The current report score is project-specific and cohort-relative:

- `Score local-bump`: final saved-vessel local-bump eye score.
- `Score courbure^2`: optional raw curvature-squared eye score when that method is active.
- `Score Tortuosity Density`: optional raw Tortuosity Density eye score when that method is active.
- `Score somme des angles externes`: optional raw external-angle-sum eye score (\(T\), in degrees) when that method is active.
- `Score moyen`: length-weighted mean saved-vessel local-bump burden.
- `Queue superieure`: upper-tail saved-vessel burden.
- `Score comparatif`: cohort-normalized `Score local-bump`.
- The `100 px` saved-vessel threshold filters short reviewed vessels from the eye-level aggregation; it does not crop or shorten vessel geometry.
- The comparative score is cohort-relative: it is meant to rank the eyes included in the report, not to be interpreted as an absolute population threshold.

The literature below motivates the design choices: moving beyond simple arc/chord ratios, using eye-level aggregation, incorporating expert/clinical judgement, and favoring interpretable tortuosity features. The exact formula above is not copied from a single paper; it is a local scoring rule built from these ideas and the available reviewed-vessel annotations.

## Sources and bibliography

- Ramos L, Novo J, Rouco J, Romeo SJ, Ortega M. “Computational assessment of the retinal vascular tortuosity integrating domain-related information.” *Scientific Reports*. 2019. https://www.nature.com/articles/s41598-019-56507-7
- Hervella AS, Rouco J, Novo J, Ortega M. “Explainable artificial intelligence for the automated assessment of the retinal vascular tortuosity.” *Medical & Biological Engineering & Computing*. 2024. https://link.springer.com/article/10.1007/s11517-023-02978-w
- Ramos L, Novo J, Rouco J, Ortega M. “Retinal vascular tortuosity assessment: inter-intra expert analysis and correlation with computational measurements.” *BMC Medical Research Methodology*. 2018. https://link.springer.com/article/10.1186/s12874-018-0598-3

In this project, these papers motivate:

- preferring interpretable vessel-shape descriptors over a single raw arc/chord ratio
- aggregating vessel-level measurements at eye level
- validating against expert/clinical judgement
- treating the current formula as a project-specific scoring rule rather than a direct copy of one published metric
