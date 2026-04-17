---
name: neutron-reflectometry
description: >
  Baseline domain knowledge for neutron reflectometry modeling and fitting with refl1d.
  Provides common material SLD values, chi-squared interpretation guidelines, model
  complexity rules (BIC), roughness constraints, refl1d API conventions, and general
  constraints that apply to ALL reflectometry analyses. Always activated.
metadata:
  author: aure
  version: "1.0"
---

## Data Structure for the REF_L INSTRUMENT
The Liquids Reflectometer (REF_L) at the Spallation Neutron Source (SNS) produces data files with a specific structure. Each file contains multiple segments corresponding to different measurement configurations, such as varying the angle of incidence or the neutron wavelength band. The segments are typically labeled with metadata that indicates the measurement conditions.

There are two common approaches to fitting REF_L data:
1. **Combined Data File Fitting**: This is the usual approach where all the segments are combined in a single data file. Since all Q points are in one file, we do not have the information about which angle was used for which Q points. When loading the data in refl1d, we can use the `load4` function, making sure to set the `FWHM` parameter according to how the Q resolution is defined in the data file.

2. **Multi-Segment Co-refinement Fitting**: Fit each segment separately, allowing for a different normalization factor, angle offset, or "sample broadening" for each segment. Sample broadening is an added component to the Q resolution that accounts for experimental factors with the sample or the instrument. When creating the probe for each segment, we have to use the incident angle (theta) corresponding to that segment, which can be extracted from the data file header. In this case we use the following code:

```python
def create_probe(data_file, theta):
    q, data, errors, dq = np.loadtxt(data_file).T
    wl = 4 * np.pi * np.sin(np.pi / 180 * theta) / q
    dT = dq / q * np.tan(np.pi / 180 * theta) * 180 / np.pi
    # Placeholder for wavelength resolution for future AuRE version
    dL = 0 * q

    # The following is how refl1d computes dQ
    # dQ = (4 * np.pi / wl) * np.sqrt((np.sin(np.pi/180*theta) * dL / wl) ** 2 + (np.cos(np.pi/180*theta) * dT * np.pi/180) ** 2)

    # dT and dL are FWHM
    probe = make_probe(
        T=theta,
        dT=dT,
        L=wl,
        dL=dL,
        data=(data, errors),
        radiation="neutron",
        resolution="uniform",
    )
    return probe 
```

During data intake, AuRE needs to recognize when multiple files represent different segments of the same measurement (usually identified by the first run number, which is sometimes called sequence_id) and set up the co-refinement with shared structural parameters but independent normalization, angle offset parameters (if needed - not by default), 
and sample broadening (if needed - not by default).

The following is an example meta data header for a combined REF_L data file:

```
# Experiment IPTS-36897 Run 226613
# Reduction 2.9.0rc3
# Run title: sample2_full_Q_OCV_end-226613-1.
# Run start time: 2026-03-29T04:50:44.076174667
# Reduction time: Sun Mar 29 00:55:34 2026
# Q summing: False
# TOF weighted: False
# Bck in Q: False
# Theta offset: 0.0
# Stitching type: None
# DataRun   NormRun   TwoTheta(deg)  LambdaMin(A)   LambdaMax(A) Qmin(1/A)    Qmax(1/A)    SF_A         SF_B          SF
# 226613    226559    0.739463       2.74975        9.4987       0.0085       0.0294       4.0          0            1.0         
# 226614    226560    2.39957        2.74978        9.49867      0.0277       0.0956       25.0         0            1.0         
# 226615    226561    7.00027        2.74977        9.49868      0.0807       0.2790       196.0        0            1.0         
# Q [1/Angstrom]        R                     dR                    dQ [FWHM] 
```

The experiment and data run information is at the top of the header. The table at the bottom lists the runs (segements) included in the data file. For each run/segment, TwoTheta represents 2 times the incident angle. 

For data files representing an individal run/segment, the table in the header will only have one row, and the TwoTheta value will correspond to that single segment. In this case, we can directly apply any necessary angle offset to the entire model when fitting that file.

**Co-refinement of multiple combined data files**: If we have multiple combined data files, we can in principle co-refine them together. However, this mode is not currently supported by AuRE since would require the implementation of flexible constraints between the model parameters for each file. Data intake should recognize this case and either reject it or only load one of the files with a warning that co-refinement of multiple combined files is not currently supported. The user can then choose to fit the files separately or combine them manually using external tools before loading into AuRE.

## Common SLD Values (×10⁻⁶ Å⁻²)

| Material | SLD |
|----------|-----|
| Silicon | 2.07 |
| SiO₂ | 3.47 |
| Air | 0.0 |
| Gold | 4.5 |
| Copper | 6.55 |
| Titanium | -1.95 |

## SLD Range Guidelines

- Set `sld_min` and `sld_max` to at least ±2.0 around the nominal SLD value for each layer.
  For example, for copper (SLD 6.55): sld_min = 4.5, sld_max = 8.5.
  For titanium (SLD -1.95): sld_min = -4.0, sld_max = 0.1.
- This allows the fitter enough freedom to find the correct values even when the
  material is not perfectly stoichiometric, has intermixing, or partial isotopic substitution.
- Never use ranges narrower than ±1.0.
- For adhesion layers like titanium that can intermix with adjacent layers, use ranges
  of ±3.0 or wider (e.g., -5.0 to 1.0 for Ti).

## Chi-Squared (χ²) Interpretation

- χ² ≈ 1: Ideal fit (model matches data within error bars)
- χ² < 0.5: Possible overfitting or overestimated errors
- χ² 1–2: Excellent fit
- χ² 2–5: Good fit, minor discrepancies
- χ² 5–10: Marginal fit, model may be missing features
- χ² > 10: Poor fit, significant model problems

## Model Complexity (BIC)

- BIC = n·ln(χ²) + k·ln(n), where n = number of data points, k = free parameters.
- Lower BIC is better.
- Each layer adds 3 free parameters (thickness, SLD, roughness).
- Adding a layer must produce a substantial χ² improvement to lower BIC.
- Do NOT suggest adding layers unless the BIC would clearly improve.
- Do NOT split existing layers into sublayers (e.g., CuO + Cu₂O) unless χ² > 10
  with clear evidence in residuals of unmodeled contrast steps.
- If a previous attempt to add a layer was reverted due to BIC regression,
  do NOT re-add the same layer — try a different approach.

## Roughness Constraints

- Roughness must be ≥ 5 Å (values below are physically unrealistic).
- Roughness must be less than half the thickness of either adjacent layer
  (otherwise artifacts occur).
- Typical roughness range: 5–30 Å.

## Refl1d API Rules

CRITICAL: `SLD(...)` objects do NOT have `.material`, `.thickness`, or `.interface`
attributes. Those attributes only exist on `Slab` objects inside the sample stack.
You MUST set parameter bounds using `sample[i]` indexing:

```
sample[0].material.rho.range(5.5, 7.0)   # ambient SLD
sample[1].thickness.range(10.0, 30.0)     # first layer thickness
sample[1].material.rho.range(2.0, 4.0)    # first layer SLD
sample[1].interface.range(0.0, 5.0)       # first layer roughness
```

NEVER write `copper.material.rho.range(...)` — this crashes with
"'SLD' object has no attribute 'material'".

## General Constraints

- NEVER suggest changing the fitting engine/method. The fitting method is chosen
  by the workflow and is not a model issue.
- NEVER suggest reversing the layer order or changing the back-reflection geometry.
  The measurement geometry is set by the user and must not be changed.
- NEVER suggest changing error bars, resolution, or Q-range — these are experimental
  parameters that cannot be modified.
- Unless specifically requested by the user, never allow the substrate SLD to vary.
- Minimum layer thickness is 5 Å — thinner layers cannot be resolved by the fitter.

## Native SiO₂ on Silicon

By default, avoid adding an SiO₂ layer on the silicon substrate. Native SiO₂ is
typically only 10–20 Å and in reflectometry it adds 3 parameters that can absorb
signal from more important layers. If an SiO₂ layer is already in the model,
consider removing it or fixing its thickness to < 20 Å to free up fitting capacity
for unknown layers. **However**, if the user explicitly requests an SiO₂ layer,
you MUST add it.

## Refinement Strategy — General

When χ² is above the acceptance threshold, follow this priority order:

1. **Constrain unphysical parameters first.** If a fitted value is far from its
   nominal/expected value (e.g., Ti thickness 5× nominal), tighten that parameter's
   bounds to a physically realistic range before trying other changes.
2. **Widen bounds on parameters hitting limits.** If a parameter is pinned at its
   bound, widen that bound — but only in the physically plausible direction.
3. **Adjust starting values.** Set starting values to the best-fit values from the
   previous iteration where they are physically reasonable.
4. **Check the ambient SLD.** If the fitted ambient SLD deviates significantly from
   the expected value for the stated solvent, flag this and constrain it. This is a
   common source of high χ² that does not require structural model changes.
5. **Enable sample_broadening for multi-segment data when indicated** (see below).
6. **Structural changes are a last resort.** Only add or remove layers when:
   - χ² remains > 10 after parameter adjustments, AND
   - residual fringes clearly indicate an unmodeled layer, AND
   - BIC analysis supports the added complexity.
7. **Never make multiple structural changes at once.** Add or remove one layer at
   a time so the effect can be evaluated.

## Refinement Strategy — Multi-Segment Co-refinement

When fitting multiple segments/files together (multi-segment co-refinement with
angle-based probes), two additional probe-level parameters become available:

### sample_broadening

`sample_broadening` adds an extra angular divergence component (in degrees) to the
Q resolution for each probe segment.  It accounts for sample curvature, waviness,
or alignment issues that broaden reflectivity features beyond the instrumental
resolution.

**When to enable sample_broadening** — enable it (`"enabled": true`) when:
- Per-segment χ² values are **uneven**, particularly when the **low-Q segment is
  significantly worse** (e.g., χ² > 2× the best segment).
- The critical edge region appears **rounder or more smeared** in the data than in
  the model prediction.
- Structural parameter adjustments (thickness, SLD, roughness) and intensity
  normalization changes have **not resolved the per-segment χ² imbalance** after
  1–2 iterations.
- Structural parameters are **drifting to unphysical values** (e.g., adhesion layer
  thickness inflating 5×, SLD far from nominal) — this often indicates the fitter
  is using structural params as proxies for missing resolution broadening.

**Do NOT enable** when:
- Single-file fitting (no angle info available; probes are Q-based).
- All segments fit equally well (uniform χ² across segments).
- χ² is already below the acceptance threshold.

**Typical ranges:** `"min": 0.0, "max": 0.5` (degrees).  Start with the default
range; widen only if the fitted value hits the upper bound.

### theta_offset

`theta_offset` allows for a small correction to the incident angle of each probe
segment (in degrees).  It accounts for sample misalignment or goniometer
calibration errors.

**When to enable theta_offset** — enable it (`"enabled": true`) when:
- The fit is poor specifically in the **overlap region** between adjacent segments
  (discontinuity in the stitched data).
- There is a systematic shift between segments that intensity normalization alone
  cannot explain.

**Do NOT enable** unless there is clear evidence of angular misalignment.

**Typical ranges:** `"min": -0.02, "max": 0.02` (degrees).  This is a small
correction; values larger than ±0.1° suggest a more serious calibration issue.

### Priority order for multi-segment issues

When per-segment χ² values are uneven (one segment much worse than others):

1. **First:** Check intensity normalization — widen intensity bounds if a segment
   is hitting its limit.
2. **Second:** Enable `sample_broadening` — this is the most common cause of
   uneven segment fits, especially when the low-Q segment is worst.
3. **Third:** Enable `theta_offset` — only if overlap regions show misalignment.
4. **Last:** Consider structural changes — only if broadening and offset do not
   resolve the issue and residual fringes indicate a missing layer.