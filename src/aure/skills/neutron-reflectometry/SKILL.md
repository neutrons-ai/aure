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
5. **Structural changes are a last resort.** Only add or remove layers when:
   - χ² remains > 10 after parameter adjustments, AND
   - residual fringes clearly indicate an unmodeled layer, AND
   - BIC analysis supports the added complexity.
6. **Never make multiple structural changes at once.** Add or remove one layer at
   a time so the effect can be evaluated.
7. **When multi-file chi² values are uneven** (one segment much worse than others),
   focus suggestions on the Q-range where the fit is worst. Common causes:
   - Intensity normalization mismatch between segments
   - Model features (e.g., thin-layer fringes) falling in one Q-range
