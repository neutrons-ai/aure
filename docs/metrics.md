# Metrics and Deterministic Math

Every number AuRE uses to *judge* a fit, *rank* models, or *seed* a model is
computed by deterministic code, not by the LLM. This document is the reference
for what those numbers are, how they are defined, and where they are computed.

It describes the code **as implemented**. Where a formula here differs from the
convention you might expect from the literature, that difference is called out
explicitly rather than smoothed over.

- [1. χ² — goodness of fit](#1-χ--goodness-of-fit)
- [2. The χ² acceptance window](#2-the-χ-acceptance-window)
- [3. BIC — complexity penalty](#3-bic--complexity-penalty)
- [4. Regression guardrails](#4-regression-guardrails)
- [5. Feature extraction (the `analysis` node)](#5-feature-extraction-the-analysis-node)
- [6. Residual analysis](#6-residual-analysis)
- [7. SLD-profile artifact detection](#7-sld-profile-artifact-detection)
- [8. Boundary-hit detection](#8-boundary-hit-detection)
- [9. Resolution limit and thin layers](#9-resolution-limit-and-thin-layers)
- [10. Final model selection](#10-final-model-selection)
- [11. Constant and environment-variable reference](#11-constant-and-environment-variable-reference)

---

## 1. χ² — goodness of fit

### 1.1 The reported number

The χ² AuRE reports everywhere — the acceptance window, the guardrails, BIC,
the report, `--json`, both web tabs — is
[`model_builder.data_chisq(problem)`](../src/aure/nodes/model_builder.py).

```
χ²_reported  =  problem.chisq(nllf = pmodel)
```

where `pmodel` is the **data term alone** of bumps' negative log-likelihood,
taken from `FitProblem._nllf_components() → (pparameter, pconstraints, pmodel,
failing)`.

Two things follow from that definition:

1. **It excludes the penalty terms.** `FitProblem.chisq()` scales the *total*
   nllf, `pmodel + pparameter + pconstraints`. With plain box bounds and no
   constraints the extra terms are identically zero and the two agree. The
   moment a [`derived_parameters`](derived-parameters.md) `keep_physical` guard
   is declared they do not: a violated constraint pushed the total to ~10¹⁰ in
   testing. χ² is used as a statement about *how well the model describes the
   measurement*, so it must see only the data term. The penalty is not
   discarded — the optimizer still minimizes the total — it is reported
   separately by `model_builder.penalty_nllf`.

2. **Infeasible is `+inf`, not 0.** bumps short-circuits and returns
   `pmodel = 0.0` without evaluating the model when a prior or constraint is
   violated. Scaling that would report χ² = 0 for an *impossible* model — which
   does not merely look good, it lands under the acceptance floor (§2) and would
   be filed as "your error bars are wrong" rather than "this model is not
   allowed". `data_chisq` returns `float("inf")` instead. **`+inf` is the
   fit-failed sentinel** throughout the codebase.

If bumps' internals are renamed, `data_chisq` falls back to `problem.chisq()`,
which is equal for every model that declares no priors or constraints.

### 1.2 Normalization

χ² is **reduced** — normalized per degree of freedom, so a well-fitted model
with correctly estimated `dR` sits near 1. AuRE does not define the
normalization itself; it inherits `bumps.FitProblem.chisq()`, which returns

```
χ²_reduced  =  2 · nllf / dof        dof = model_points() + Σ prior.dof − n_params
```

(verified against bumps 1.0.5, `fitproblem.py` — `chisq`, `nllf_scale`, and the
`_dof` assignment). Two properties of that definition matter downstream:

- **Priors add degrees of freedom.** `dof` is not simply `N − k`; a model
  declaring priors has a larger `dof` and therefore a smaller reduced χ².
- **It degrades silently.** When `dof ≤ 0` (or is NaN/inf) `nllf_scale` returns
  a scale factor of 1.0 instead of `2/dof`, so `chisq()` returns the raw nllf —
  *not* normalized at all. Reachable only for a model with at least as many free
  parameters as data points, but it is a silent change of units, not an error.

Together these are why BIC derives its total χ² directly from `pmodel` rather
than multiplying the reduced value back up by `dof` (§3.2).

The acceptance window, the floor, and the "reduced χ² far under 1 is evidence
about the `dR` column" reasoning in §2 all rest on this normalization.

> **Caveat — two conventions coexist.** The aggregate χ² above is normalized by
> degrees of freedom. The **per-file** χ² attached to each `per_file_results`
> entry in a co-refinement is computed directly from the residuals in
> [`fitting.py`](../src/aure/nodes/fitting.py) as
>
> ```
> χ²_per_file  =  (1/n) · Σᵢ residualᵢ²        residualᵢ = (Rᵢ − R_modelᵢ) / dRᵢ
> ```
>
> — normalized by `n`, not by `n − k`, because a refl1d `Experiment` has no
> `.chisq()`. The two are therefore not the same statistic, and per-file χ² is
> systematically the smaller of the two by a factor `(n−k)/n`. Both are compared
> against the same `chi2_max` ceiling (§2), so the per-file check is slightly the
> more permissive of the two. For typical `n ≫ k` the difference is negligible;
> for a small dataset with many free parameters it is not.

### 1.3 Residual quantities

Two residual arrays are recorded per fit and per file:

```
residualᵢ       = (R_dataᵢ − R_fitᵢ) / max(|dRᵢ|, 1e-20)      # σ units
residual_ratioᵢ = R_dataᵢ / max(R_fitᵢ, 1e-20)                # dimensionless
```

`residuals` is what χ² is built from. `residual_ratio` is what the fringe
analysis of §6 operates on: dividing out the model leaves the *unmodeled*
oscillation, whose spacing names a thickness the model is missing.

---

## 2. The χ² acceptance window

The refinement loop terminates deterministically on χ², not at the LLM's
discretion. [`evaluation._clamp_acceptance_to_chi2`](../src/aure/nodes/evaluation.py)
implements:

```
chi2_min  ≤  χ²  ≤  chi2_max      and χ² finite   ⟹   acceptable = True
```

| Bound | Env | Setup key | Default |
|---|---|---|---|
| ceiling | `CHI2_MAX` | `chi2_max` | `5.0` |
| floor | `CHI2_MIN` | `chi2_min` | `0.5` (`0` disables) |

Both are **pinned into `state` by the runner on the first pass**, so a `resume`
keeps the window the run was launched with rather than inheriting the resuming
process's environment.

**The clamp is one-directional.** It only raises a verdict (`False → True`);
it never lowers one. *Above* `chi2_max` the LLM's `acceptable` is taken as-is
and none of the stand-down conditions are evaluated.

**Why there is a floor.** A reduced χ² far below 1 says the residuals are much
smaller than the quoted uncertainties — an overestimated `dR` column, or a model
free enough to absorb the noise. That is evidence about the *error bars*, not
about the structure, so it must not read as a pass. `0.5` is deliberately the
same number `_simple_evaluation` calls "possible overfitting", so the two cannot
contradict each other.

**The clamp stands down** (declines to force acceptance; the LLM's verdict
decides) on:

- a vetoed profile (§7);
- a profile that was not *verified* — no exported profile, the detector returned
  `checked=False`, or a co-refinement where any one state reported no profile;
- a per-file or per-state χ² over the ceiling, or carrying the `+inf` sentinel;
- χ² below `chi2_min`.

**Ordering invariant:** the clamp must run *after* profile-artifact detection,
whose two markers it reads. Hoisted above it, both are unset, "not checked"
means stand down, and the χ² stop silently becomes dead code.

---

## 3. BIC — complexity penalty

Adding a layer almost always lowers χ², because it gives the fit more freedom.
What we want is a model that is *statistically justified*. That is what the
Bayesian Information Criterion measures.

### 3.1 The formula

[`evaluation._compute_bic`](../src/aure/nodes/evaluation.py):

```
BIC  =  χ²_total  +  k · ln(n)
```

with `χ²_total` the **un-normalized** data-term χ², `n` the total number of data
points, and `k` the number of free parameters. Lower is better.

This is the standard Gaussian result for **known** variances, which is the case
here: the variances are the `dR` column of the data file, not something the fit
estimates. With

```
χ²_total  =  Σᵢ ((Rᵢ − R_modelᵢ) / dRᵢ)²
```

we have `−2 ln L = χ²_total` up to an additive constant (the `Σ ln(2π dRᵢ²)`
term) that is identical for every model of the same data, so it cancels in any
comparison and is omitted.

Non-finite or negative χ², or `n ≤ 0`, returns `+inf` — so a failed or
infeasible fit can never claim the BIC baseline.

> **The χ² fed to this must be the total, not the reduced value** that every
> other part of the codebase reports. `model_builder.data_chisq_total` supplies
> it; `data_chisq` (§1) does not. Passing a reduced χ² flattens the likelihood
> term by a factor of the degrees of freedom — which is most of the metric.

**Why not the RSS form.** Until this was corrected the code computed
`n·ln(χ²_red) + k·ln(n)`, the *unknown-variance* (RSS) BIC. That form is
scale-free in χ², with the consequence that the minimum **relative** χ²
improvement needed to justify a parameter did not depend on how badly the model
fitted. For `n = 2000` and four added parameters:

| χ²_red | RSS form demanded | known-variance form demands |
|---|---|---|
| 1.0 | 1.51 % | 1.33 % |
| 2.0 | 1.51 % | 0.56 % |
| 5.0 | 1.51 % | 0.10 % |

A constant threshold is the tell. On a poor fit a 1 % *relative* gain is a large
*absolute* χ² gain — real evidence that the added structure is doing work — and
the RSS form could not see it, so it favoured simplicity most strongly in
exactly the regime where the refinement loop is adding layers. The sensitivities
are `n/χ²_red` versus `n−k`, which coincide near χ²_red ≈ 1; that is why the
error was invisible on a good fit.

### 3.2 Where `n`, `k` and `χ²_total` come from

All three are read **off the bumps problem** by
[`model_builder.bic_inputs`](../src/aure/nodes/model_builder.py) and stamped onto
every `FitResult` as `_chi2_total` / `_n_data` / `_n_free_params`:

| Quantity | Source | Meaning |
|---|---|---|
| `χ²_total` | `2 · pmodel` | bumps defines `pmodel = ½·Σresiduals²` for Gaussian independent uncertainties, so this is exact. `inf` when a constraint fails. |
| `n` | `problem.model_points()` | Every point of every dataset of every state. |
| `k` | `len(problem.getp())` | Unique free parameters — tied-across-state parameters counted once, expression-derived ones not at all. |

`model_points()` and `getp()` are the same accessors bumps uses to build its own
`dof`, so there is exactly one definition of each quantity in play.

Both `fitting` and `evaluation` resolve their inputs through
[`evaluation.bic_inputs_for`](../src/aure/nodes/evaluation.py), which prefers the
recorded values. **This single source is load-bearing.** The two nodes used to
derive `n` and `k` independently and then compare the resulting BIC values:
`evaluation` used `len(state["Q"])`, which holds only the *primary* data file, so
on any co-refinement the two sides of the regression guardrail (§4.2) were
different statistics — and the guardrail either never fired or fired on every
iteration depending on which side of χ² = 1 the fit sat.

`χ²_total` is derived as `2·pmodel` rather than as `χ²_reduced × dof`
deliberately: bumps' `dof` is `model_points() + Σ prior.dof − n_params`, and
`nllf_scale` silently degrades to an un-normalized scale when that is
non-positive, so the round trip is not an identity.

### 3.3 Legacy fit results

A checkpoint written before those fields existed has only the *reduced* χ². For
those, `bic_inputs_for` reconstructs `χ²_total ≈ χ²_reduced × max(n − k, 1)` and
falls back to the structural counts below. The reconstruction ignores prior
degrees of freedom, so it is approximate — but it is in the right units, which
the stored reduced value is not.

The structural free-parameter estimate,
`evaluation._count_free_params(model)`:

```
k = 3 · n_layers                                  # thickness, SLD, roughness each
  + 1  if substrate.roughness_max is not None
  + 1  if ambient is not "air" and ambient.sld ≠ 0
  + 1  unless intensity.fixed
  + derived-parameter delta                       # §3.4
```

> **Caveat — the structural estimate undercounts.** The builder also makes free,
> when declared and not `fixed`: `probe.background` (free *by default* once a
> background block is present), `probe.sample_broadening`, `probe.theta_offset`,
> and a layer's solvation `frac`. None are counted above. This is why
> `len(problem.getp())` is preferred wherever a problem exists.

`n` for a legacy result comes from `evaluation._n_data_from_state`, which sums
every dataset of every state, falls back to the flat `data_files` list, and only
then to `state["Q"]`.

### 3.4 The derived-parameter delta

A [reparametrization](derived-parameters.md) adds one **free** parameter and
derives raw ones from it, so the raw ones leave the free set (bumps discovers
parameters by traversal, and an expression is not one). `_derived_param_delta`
adjusts the structural estimate:

```
delta = (number of declarations)  −  (assigned slots that k had actually charged for)
```

A one-for-one swap — a surface excess replacing an SLD — is BIC-neutral.
Solvation (two free, one derived: a volume fraction and a dry SLD) costs one.
Only slots the count above actually charged for are refunded, so an assignment
to something it never counted (an `air` ambient) cannot drive the total
negative. `len(problem.getp())` needs none of this, which is the other reason to
prefer it.

### 3.5 Changing the convention

`evaluation.BIC_FORMULA` marks which formula a stored `best_bic` was computed
under, and is persisted as `state["bic_formula"]`. A run resumed across a change
of convention would otherwise compare two values on entirely different scales;
`bic_baseline_is_stale` detects the mismatch (an *absent* marker counts as
stale) and the guardrails discard the baseline, logging it, so the next fit
re-establishes it. Bump `BIC_FORMULA` whenever the formula changes.

## 4. Regression guardrails

Both run in `evaluation`, **independently of the LLM's opinion**.

### 4.1 χ² regression

```
χ²  >  best_χ²  ×  1.05      ⟹   restore best_model, mark the hypothesis rejected
```

The 5 % slack is hardcoded in
[`evaluation.py`](../src/aure/nodes/evaluation.py) (no env override).

### 4.2 BIC regression

```
BIC  >  best_BIC   and not χ²-reverted   ⟹   restore best_bic_model, mark rejected
```

**No slack** — deliberately tighter than the χ² guardrail, because it also marks
the attempted hypothesis `rejected` and so does more damage when it misfires.

### 4.3 Which fits may become the baseline

`fitting._wins_baseline` decides what the guardrails revert *to*. A fit **below
the acceptance floor must not claim the baseline**: its χ² is evidence about the
`dR` column, and one noise-absorbing iteration would make every later honest fit
read as a regression, pinning the run to a model it should have moved away from.

```
in-window candidate  vs  sub-floor incumbent   →  candidate wins, whatever the scores
sub-floor candidate  vs  in-window incumbent   →  candidate loses, whatever the scores
otherwise                                      →  lower score wins
```

A sub-floor fit *is* still recorded when nothing in-window exists yet: leaving
the guardrails with no baseline at all would disable the check entirely, which
is the worse failure.

---

## 5. Feature extraction (the `analysis` node)

All deterministic, no LLM. [`tools/feature_tools.py`](../src/aure/tools/feature_tools.py).
Throughout, Q is the momentum transfer in Å⁻¹, `Q = 4π sin θ / λ`, and SLD is
reported in units of 10⁻⁶ Å⁻².

### 5.1 Critical edge → SLD contrast

Total external reflection ends at the critical edge Q_c, where

```
Q_c  =  4 √(π Δρ)          ⟺          Δρ  =  Q_c² / (16 π)
```

`extract_critical_edges` implements the inverse as `(Qc/4)² / π × 10⁶`.

**Q_c is located at the half-height of the total-reflection plateau.** Under
resolution smearing R falls to ≈ ½ the plateau level at the true Q_c, which
makes the estimate both accurate and robust to a plateau sitting below 1.0 from
imperfect intensity normalization. The plateau level is the median of the first
`min(max(3, n/8), 12)` points of the search region; the crossing is
linearly interpolated between the bracketing samples. If no plateau was captured
(data starting above Q_c) it falls back to the steepest-descent point of
`d log₁₀R / dQ`. Search range: `0.005 ≤ Q_c ≤ 0.05` Å⁻¹, excluding any region
where fringes have already begun. At most one candidate is returned.

**Δρ is a *contrast*, not an absolute SLD** — it is `ρ_substrate − ρ_ambient`,
and the formula above assumes the incident medium is vacuum/air (ρ ≈ 0). This
matters for solvent-contrast work: a D₂O ambient shifts the whole thing by
≈ 6.36. `format_critical_edge_line` therefore renders it as "SLD contrast" and
attaches the deterministic *implied ambient SLD* hint when the contrast is
inconsistent with an H-form ambient. (The `aure extract-features` CLI and the
MCP `extract_features` tool label the same quantity simply `SLD ≈` /
`estimated_SLD`, which is looser.)

Confidence is keyed off edge sharpness `|d log₁₀R/dQ|` at Q_c: `> 50` high,
`> 20` medium, else low.

### 5.2 Kiessig fringes → total thickness

Fringe minima are spaced by `ΔQ = 2π/d`, so

```
d  =  2π / ⟨ΔQ⟩            σ_d  =  d · (σ_ΔQ / ⟨ΔQ⟩)
```

`estimate_total_thickness` takes `log₁₀R` above `q_min = 0.02` Å⁻¹, smooths with
a Savitzky–Golay filter (order 2, window `min(11, n/5)` forced odd), and finds
minima with `scipy.signal.find_peaks(-log_R, distance=5, prominence=0.05)`.
`⟨ΔQ⟩` and `σ_ΔQ` are the mean and standard deviation of the consecutive
spacings. Fewer than 2 minima returns no estimate; a single spacing gets a
default 20 % uncertainty.

Confidence: high if ≥ 5 minima and `σ_ΔQ/⟨ΔQ⟩ < 0.1`; medium if ≥ 3 minima.

### 5.3 High-Q decay → roughness

At high Q the Debye–Waller/Névot–Croce factor damps the Fresnel decay:

```
R(Q)  ≈  R_Fresnel(Q) · exp(−Q² σ²)          R_Fresnel ∝ Q⁻⁴
```

Dividing out the Q⁻⁴ envelope and taking log₁₀ linearizes it in Q²:

```
log₁₀R  +  4 log₁₀Q   =   const  −  σ² Q² / ln 10
```

`estimate_roughness` least-squares-fits that line over `Q ≥ q_min = 0.15` Å⁻¹
and recovers

```
σ  =  √( −slope · ln 10 )
```

A non-negative slope means no measurable roughness (returns `σ = 0`, low
confidence); fewer than 10 points in range or a fit failure returns the 5.0 Å
default guess with low confidence. Confidence is keyed off the standard
deviation of the fit residuals: `< 0.3` high, `< 0.6` medium, else low. Note
the single-σ, single-interface reading of a multilayer decay — this is a seed
for the first model, not a measurement.

### 5.4 Layer count

`estimate_layer_count` is an explicit **heuristic**, not a derivation: it maps
the number of distinct critical edges, the number of distinct oscillation
frequencies, and the fringe count onto 0/1/2/3+ layers. Confidence is `medium`
at best, and the LLM is free to disagree with it.

---

## 6. Residual analysis

`analyze_residual_fringes` looks for oscillations the model failed to
reproduce — the signature of a missing layer. Both methods run on the
`residual_ratio` of §1.3, cubic-polynomial-detrended to remove the
low-frequency model-mismatch envelope while preserving the oscillation.

### 6.1 FFT method

Resample onto 1024 uniform Q points, detrend, apply a Hanning window, take
`rfft`, and read peak power. A frequency `f` (conjugate to Q) maps to

```
d  =  2π f
```

Frequencies below the detectability limit are skipped: at least
`min_fringes = 5` complete oscillations must fit in the measured Q range, i.e.

```
d_min  =  5 · 2π / (Q_max − Q_min)
```

with a hard floor of bin index 3.

### 6.2 Fringe-spacing method

Detrend, smooth with Savitzky–Golay (order 2, small window `min(5, n/10)` ≥ 3,
odd) to preserve closely-spaced fringes, then find minima with a prominence
threshold scaled to the signal's own noise:

```
MAD         = median(|smooth − median(smooth)|)
prominence  = max(0.5 · MAD, 0.01)              # abort if MAD < 0.005
d           = 2π / median(ΔQ_minima)            # ≥ 3 minima required
```

The **median** spacing is used rather than the mean, for robustness against
false minima. Candidates from both methods are then de-duplicated.

---

## 7. SLD-profile artifact detection

`detect_profile_artifacts` catches a defect that is **invisible in χ²** and
visible only in the fitted profile: an error-function roughness tail reaching
across a thin layer and pulling the profile to an SLD no material present can
produce (e.g. below the substrate SLD just before the substrate).

The test is *direction-agnostic* and needs no z-alignment. Given the stack's SLD
sequence `media` (ambient..substrate or the reverse):

```
span     = max(media) − min(media)                 # abort if ≤ 0
tol      = 0.05 · span                             # unless overridden
prom     = prominence_frac · span                  # prominence_frac = 0.02
turning  = the interior turning-point values of the slab sequence
bounds   = [min(media) − tol,  max(media) + tol]
```

Every interior extremum of the profile with prominence ≥ `prom` is then an
**excursion** unless it sits within `tol` of a legitimate turning value:

- outside `bounds` → "profile SLD leaves the range reachable by the bounding media";
- inside `bounds` but at no material's value → "profile turns at an SLD no
  material provides (erf-tail excursion across a thin layer)".

Terminal media are deliberately **excluded** from `turning`: a monotone approach
to a terminal produces no interior extremum, so an interior extremum near a
terminal value is an *extra* turning point — precisely the artifact. The
extremum count (`n_found` vs `n_expected = len(turning)`) is reported as a
supporting signal only.

The σ/thickness ratio is deliberately **not** used here, and
`check_roughness_thickness_ratios` (`σ > 0.5·t`) is surfaced as an
*informational concern only* — a large roughness is legitimate when the model
parametrizes a graded profile rather than a discrete slab.

### `checked` vs `has_artifact`

`checked=False` means **unknown**, not clean. It is returned on every "cannot
check" path: fewer than 5 profile points, mismatched `z`/`rho` lengths, fewer
than 2 media, a **non-finite** sample anywhere, or a zero SLD span. The
non-finite gate matters: every comparison against NaN is False, so an all-NaN
profile from a diverged fit used to yield no extrema and report a clean bill of
health. Callers gating a decision on a clean profile must read `checked`.

An excursion **vetoes acceptance** (§2) and sets aside the fit in final
selection (§10), with a two-branch remedy suggestion: tie the roughness, or
re-label the model as a profile parametrization.

---

## 8. Boundary-hit detection

`_check_boundary_hits` reports parameters the bounds are constraining. A
parameter is pinned if **either** test fires:

```
value test:        |value − bound| ≤ tolerance · |bound|            tolerance = 0.01
uncertainty test:  value ± sigma·dx  crosses the bound              sigma = 2.0
```

The uncertainty test is invisible to a point-estimate check and is common after
a `dream` run, where the posterior is skewed towards the bound it is pressed
against. Worked example: `CuOx interface` fitted at 5.11 in `[5, 11]` is 1.8 %
from its floor — a 1 % test passes it — while its posterior runs down to 5.03.

Uncertainty hits are suppressed for parameters the data does not constrain:

```
2 · sigma · dx  >  max_width_fraction · (hi − lo)   ⟹   skip     max_width_fraction = 0.75
```

Such a parameter is not pinned against an edge, it is unconstrained, and
widening its bounds makes the next fit worse iteration after iteration. The
0.75 default is calibrated, not round: the `CuOx` case above has `dx = 0.906` on
a span of 6, so its interval covers 60 % of the range while still being a
genuine pin; a tighter guard rejects the case the check exists to catch.

With no `dx` available (optimizers that do not estimate uncertainty) this
degrades to the value test alone.

---

## 9. Resolution limit and thin layers

The smallest thickness a measurement can resolve is set by its Q range:

```
d_resolution  ≈  2π / Q_max            ≈ 30 Å for Q_max = 0.2 Å⁻¹
```

Layers thinner than that are not independently determined: their SLD contrast
and thickness enter the reflectivity essentially only through the product
`Δρ · t`, so the χ² surface has a **ridge** of near-equivalent
(Δρ, t) pairs, and a single local optimization cannot cross between basins on
it.

**Thin-layer SLD mode enumeration** (`MODE_ENUMERATION=1`, single-file only)
attacks that. For each layer with

```
t  <  THIN_LAYER_MODE_K × (2π / Q_max)             K = 1.0 by default
```

it re-seeds the layer across `THIN_LAYER_MODE_SEEDS` (default 3) discrete SLDs
spanning the layer's allowed range, cheap-polishes each with a local optimizer,
and starts the main fit from the lowest-χ² basin. Layers are visited greedily,
carrying each improvement forward. Off by default, logged, and never fatal — any
failure returns the input model unchanged.

---

## 10. Final model selection

[`finalize._select`](../src/aure/nodes/finalize.py); the full narrative is in
[finalization.md](finalization.md). Only fits with a **finite, positive** χ² are
candidates.

### 10.1 Tiers

Candidates are ranked within the best available tier, each tier falling back to
the next — reporting nothing is worse than reporting a flawed model that says so:

| Tier | Meaning |
|---|---|
| 0 | profile-clean and inside the acceptance window — the real answer |
| 1 | profile-clean but `χ² < chi2_min` — plausible, but its χ² describes the `dR` column |
| 2 | profile-vetoed (§7) — physically impossible, genuinely last |

An excursion buys χ², so ranking on χ² alone would report the very model
`evaluation` rejected.

### 10.2 Tier override

A tier that survives on physical plausibility alone, at many times the χ² of a
fit it outranks, is not the better answer:

```
tier_best  >  lowest_comparable × FINAL_TIER_CHI2_FACTOR   ⟹  fall back to χ² across all tiers
```

with `FINAL_TIER_CHI2_FACTOR = 3.0` by default. The comparison deliberately ignores
tier-1 fits: their χ² is a statement about the `dR` column, so it is not a
yardstick anything can be "far worse" than.

### 10.3 Parsimony tie-break

Within the winning tier:

```
best_χ²  = min χ² in the tier
band     = { fits : χ²  ≤  best_χ² × (1 + FINAL_SELECTION_TOL) }      tol = 0.02
winner   = the band member with the fewest free parameters,
           ties broken by earliest iteration
```

Free parameters are counted by `_free_params`, which prefers the fit's
problem-derived `_n_free_params` and falls back to `_count_free_params` (§3.3).
The model definition is resolved *before* ranking, so parsimony is counted on
exactly the model that will be promoted.

`final_fit`'s uncertainty polish is skipped for a tier-1 or tier-2 selection.

---

## 11. Constant and environment-variable reference

| Name | Default | Where | What it controls |
|---|---|---|---|
| `CHI2_MAX` / `chi2_max` | `5.0` | `evaluation` | Acceptance-window ceiling (§2) |
| `CHI2_MIN` / `chi2_min` | `0.5` | `evaluation` | Acceptance-window floor; `0` disables (§2) |
| — (hardcoded) | `1.05` | `evaluation` | χ²-regression slack (§4.1) |
| — (hardcoded) | none | `evaluation` | BIC-regression slack (§4.2) |
| `FINAL_SELECTION_TOL` | `0.02` | `finalize` | Parsimony band (§10.3) |
| `FINAL_TIER_CHI2_FACTOR` | `3.0` | `finalize` | Tier-override multiple (§10.2) |
| `FINAL_FIT_CHI2_MAX` | `= CHI2_MAX` | `final_fit` | Skip the polish above this χ² |
| `MODE_ENUMERATION` | off | `fitting` | Enable thin-layer SLD mode enumeration (§9) |
| `THIN_LAYER_MODE_K` | `1.0` | `fitting` | Thin-layer threshold, in units of `2π/Q_max` (§9) |
| `THIN_LAYER_MODE_SEEDS` | `3` | `fitting` | SLD seeds per thin layer (§9) |
| `ROUGHNESS_MAX_OUTER` | topmost layer's `roughness_max`, else `30.0` | `model_builder` | Upper bound on the outermost interface roughness |
| `FIT_METHOD` | `dream` | `fitting` | refl1d optimizer (`lm` / `de` / `dream`) |
| `FIT_STEPS` | `1000` | `fitting` | Optimizer steps |
| — | `0.01` | `evaluation` | Boundary-hit value tolerance (§8) |
| — | `2.0` | `evaluation` | Boundary-hit σ multiplier (§8) |
| — | `0.75` | `evaluation` | Boundary-hit max interval width fraction (§8) |
| — | `0.05 · span` | `feature_tools` | Profile-artifact SLD tolerance (§7) |
| — | `0.02 · span` | `feature_tools` | Profile-artifact extremum prominence (§7) |
| — | `0.5` | `feature_tools` | σ/thickness informational threshold (§7) |

See [.env.example](../.env.example) for the LLM-side variables.
