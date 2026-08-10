# Mathematical definition of the tortuosity scores

## Technical summary

This document is the code-audited reference for every scoring method exposed by
the application. The implementation scores **saved vessels**, not the complete
skeleton. Geometry is normalized to a common fundus diameter, vessels shorter
than the configured threshold are excluded, and the remaining vessel scores are
aggregated at eye level.

The six implemented primary methods are Local-bump v1, Local-bump v2,
Arc/chord, curvature-squared, Tortuosity Density, and the RDP external-angle
sum. Only Local-bump v1 uses an upper-tail component and a separate display
multiplier at eye level. All other methods use a simple length-weighted mean.

## Common geometry and notation

Let a saved vessel have an ordered centreline

\[
P=(\mathbf p_0,\ldots,\mathbf p_n),\qquad \mathbf p_i=(x_i,y_i).
\]

The fundus mask area is \(A\) pixels. Its equivalent diameter and the coordinate
normalization factor are

\[
d_{\mathrm{fundus}}=2\sqrt{\frac{A}{\pi}},
\qquad
c=\frac{1024}{d_{\mathrm{fundus}}},
\qquad
\mathbf q_i=c\,\mathbf p_i.
\]

If no valid fundus mask is available, the implementation uses \(c=1\). All
primary metrics below use the normalized points \(\mathbf q_i\). The normalized
path length and chord are

\[
L=\sum_{i=0}^{n-1}\lVert\mathbf q_{i+1}-\mathbf q_i\rVert_2,
\qquad
C=\lVert\mathbf q_n-\mathbf q_0\rVert_2.
\]

By default, a vessel contributes to the eye score only if its geometry is valid
and \(L\geq100\) normalized pixels. This filter can be disabled or configured.

Unless stated otherwise, resampling places points every \(h=4\) normalized
pixels and includes the final endpoint. A moving average with a default span of
five points is used by Local-bump v1, Local-bump v2, and Tortuosity Density.

## 1. Local-bump v1 (default)

After resampling and smoothing, the segment direction is unwrapped before its
first difference is taken:

\[
\theta_i=\operatorname{unwrap}\!\left(
  \operatorname{atan2}(y_{i+1}-y_i,x_{i+1}-x_i)
\right),
\qquad
\Delta\theta_i=\theta_{i+1}-\theta_i.
\]

Here, `unwrap` removes artificial jumps at the \(-\pi/\pi\) boundary: for
example, a direction change from \(179^\circ\) to \(-179^\circ\) is treated as
\(2^\circ\), not \(-358^\circ\). It changes only the angle representation, not
the vessel geometry.

With the default threshold \(\tau=0.035\) radians,

\[
\delta_i=
\begin{cases}
\Delta\theta_i,&|\Delta\theta_i|\geq\tau,\\
0,&|\Delta\theta_i|<\tau.
\end{cases}
\]

If \(m\) is the number of angle-difference samples, including samples set to
zero, the local-bump energy is

\[
E=\frac{1}{m}\sum_{i=1}^{m}|\delta_i|.
\]

Let \(N\) be the number of sign changes after zero entries have been removed
from \((\delta_i)\). The oscillation density and vessel score are

\[
D=100\frac{N}{L},
\qquad
B=E\sqrt{D}.
\]

A vessel with no sign change has \(B=0\). The vessel-level `primary_score` is
the unscaled value \(B\).

For eligible vessels \(v\in V\), the global component is

\[
G=\frac{\sum_{v\in V}L_vB_v}{\sum_{v\in V}L_v}.
\]

For the tail component, vessels are sorted by decreasing \(B_v\). The algorithm
takes exactly \(\rho\sum_vL_v\) length from that ordering, using a fractional
length from the last vessel when necessary. With \(a_v\in[0,L_v]\) denoting the
length taken from vessel \(v\),

\[
\rho=0.20,
\qquad
\sum_v a_v=\rho\sum_vL_v,
\qquad
T=\frac{\sum_v a_vB_v}{\sum_v a_v}.
\]

The displayed eye score is

\[
S_{\mathrm{eye}}=1000\,(0.70G+0.30T).
\]

## 2. Local-bump v2 (experimental)

V2 uses endpoint-safe smoothing, removes the convolution boundary from the
angle-difference sequence, and groups consecutive same-sign turns into lobes.
Any lobe whose absolute total turn is below \(\lambda=0.15\) radians is removed
iteratively; adjacent lobes that then have the same sign are merged.

Let \(m\) be the number of angle changes after removal of the filter boundary,
let \(R\) be the multiset of angle changes belonging to retained lobes, and let
\(K\) be the number of retained lobes. The implementation defines

\[
E_p=\frac{\sum_{\delta\in R}|\delta|}{m},
\qquad
N_p=\max(K-1,0),
\qquad
D_p=100\frac{N_p}{L}.
\]

The oscillation and angularity components are

\[
O=E_p\sqrt{D_p},
\qquad
A_{\mathrm{local}}=
\sqrt{\frac{1}{m}\sum_{i=1}^{m}(\Delta\theta_i)^2}
\sqrt{\frac{100}{L}}.
\]

With angularity weight \(w=0.25\), the vessel score is already display-scaled:

\[
B_{v2}=1000\left((1-w)O+wA_{\mathrm{local}}\right).
\]

The eye score has no upper-tail term and no further multiplier:

\[
S_{\mathrm{eye}}=
\frac{\sum_{v\in V}L_vB_{v2,v}}{\sum_{v\in V}L_v}.
\]

## 3. Arc/chord

The vessel score is the length of the resolved saved-vessel route divided by
the chord between its resolved endpoints:

\[
R=\frac{L}{C}.
\]

It is invalid when \(C=0\). The eye score is the length-weighted mean only:

\[
S_{\mathrm{eye}}=
\frac{\sum_{v\in V}L_vR_v}{\sum_{v\in V}L_v}.
\]

There is no upper-tail component and no display multiplier.

## 4. Curvature-squared

The centreline is resampled by default, but it is **not smoothed** in the
implemented curvature-squared calculation. Resampling can be disabled.
Numerical first and second derivatives are evaluated with respect to cumulative
arc length \(s\), and curvature is

\[
\kappa(s)=
\frac{|x'(s)y''(s)-y'(s)x''(s)|}
{\left(x'(s)^2+y'(s)^2\right)^{3/2}}.
\]

The trapezoidal rule approximates the integral, while the denominator uses the
original normalized polyline length \(L\):

\[
Q=\frac{1}{L}\int\kappa(s)^2\,ds.
\]

The eye score is

\[
S_{\mathrm{eye}}=
\frac{\sum_{v\in V}L_vQ_v}{\sum_{v\in V}L_v}.
\]

There is no upper-tail component and no display multiplier. The unit is inverse
normalized-pixel squared.

## 5. Tortuosity Density

Resampling, smoothing, and thresholding are used only to locate significant
changes in curvature sign. Each boundary is placed halfway in arc distance
between consecutive non-zero angle changes of opposite sign. The arcs and
chords in the formula are then measured on the normalized, unsmoothed input
geometry.

If the boundaries create \(n\) constant-sign subsegments, with arc length
\(L_i\) and chord \(C_i\), the vessel score is

\[
\operatorname{TD}=
\frac{n-1}{L}\sum_{i=1}^{n}\left(\frac{L_i}{C_i}-1\right).
\]

No significant sign change gives \(n=1\) and therefore
\(\operatorname{TD}=0\). A subsegment with zero arc or chord makes the metric
invalid. The eye score is

\[
S_{\mathrm{eye}}=
\frac{\sum_{v\in V}L_v\operatorname{TD}_v}{\sum_{v\in V}L_v}.
\]

There is no upper-tail component and no display multiplier. The unit is inverse
normalized pixels.

## 6. Sum of external angles after RDP simplification

The normalized centreline is simplified by the Ramer-Douglas-Peucker algorithm
with default perpendicular-distance tolerance \(\epsilon=3\) normalized pixels.
For each retained interior bend point, the implementation computes the unwrapped
change in segment direction and converts its absolute value to degrees:

\[
\phi_i=
\frac{180}{\pi}\left|
\operatorname{unwrap}(\theta_{i+1})-\operatorname{unwrap}(\theta_i)
\right|.
\]

The vessel score is

\[
T_{\mathrm{angle}}=\sum_i\phi_i.
\]

The eye score is

\[
S_{\mathrm{eye}}=
\frac{\sum_{v\in V}L_vT_{\mathrm{angle},v}}{\sum_{v\in V}L_v}.
\]

There is no upper-tail component and no display multiplier. The result is in
degrees.

## Cohort-relative comparative score

For every primary method, a report cohort with finite eye scores \(S_j\) also
receives a min-max comparative score:

\[
C_j=
\begin{cases}
100\dfrac{S_j-S_{\min}}{S_{\max}-S_{\min}},&S_{\max}>S_{\min},\\[6pt]
0,&S_{\max}=S_{\min}.
\end{cases}
\]

Missing eye scores are filled with the cohort minimum before normalization.
This is a cohort rank aid, not an absolute clinical threshold.

## Code verification record

| Claim | Implementation checked | Automated evidence |
|---|---|---|
| Geometry normalization and eligibility | `fundus_coordinate_scale`, `score_saved_vessel` | scaled-geometry and short-vessel tests |
| Local-bump v1 and upper-tail eye score | `local_bump_metrics`, `tail_weighted_mean`, `summarize_eye_score` | straight, oscillation, jitter, and tail tests |
| Local-bump v2 | `local_bump_v2_metrics`, `persistent_curvature_lobes` | persistent-wave, endpoint, and weak-lobe tests |
| Arc/chord | `score_segments`, `score_saved_vessel` | central scoring service tests |
| Curvature-squared | `curvature_squared_metrics` | straight-line, circular-arc, and resampling tests |
| Tortuosity Density | `tortuosity_density_metrics` | hand-calculated, no-inflection, and invalid-geometry tests |
| External-angle sum | `rdp_simplify`, `external_angle_sum_metrics` | straight-line and 180-degree staircase tests |
| Eye aggregation | `scoring.summarize_eye_score` | method-specific aggregation tests |
| Comparative score | `add_comparative_hybrid_score`, `minmax_series` | formula inspection and full test suite |

During this audit, the app/PDF descriptions for Arc/chord and curvature-squared
were corrected: both previously mentioned an upper-tail aggregation that the
central scoring service does not apply. The curvature-squared description was
also corrected to remove a smoothing step that is absent from the calculation.

## Limitations and interpretation

- Pixel-derived units become comparable only to the extent that the fundus-mask
  diameter is a suitable spatial normalization.
- These are deterministic geometric scores; they do not by themselves establish
  a clinical diagnosis or calibrated risk threshold.
- The comparative score changes when the report cohort changes.
- Local-bump v2 is explicitly experimental and remains separate from the default
  Local-bump v1 method.
