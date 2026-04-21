---
name: metal-oxide-interfaces
description: >
  Domain knowledge for metal oxide interface analysis in reflectometry. Covers
  native oxide formation on metals (Cu→CuO, Ti→TiO₂), oxide thickness ranges,
  and rules for when to add or avoid oxide layers. Always use when the sample involves
  metals, copper, titanium, native oxides, or SiO₂. The user may not have specified
  the oxide layer, but if the metal is exposed to ambient conditions, it is likely
  present and should be considered in the model.
metadata:
  author: aure
  version: "1.0"
---

## Native Metal Oxide Formation

When a metal layer is **directly** in contact with the ambient medium (air,
solvent, etc.) and no oxide, SEI, or other surface layer is already present,
a thin native oxide layer typically forms.

### When to Add an Oxide Layer

- Metal is the outermost layer (in contact with ambient)
- No existing oxide, SEI, or surface layer between the metal and ambient
- The fit shows systematic residuals suggesting a missing interface layer

### When NOT to Add an Oxide Layer

- An oxide or surface layer already exists between the metal and ambient
- The metal is a buried layer (e.g., Ti adhesion layer beneath Cu)
- The metal is already covered by an SEI or other surface layer
- Do NOT split an existing oxide into sublayers (e.g., CuO + Cu₂O) — keep it simple

## Common Metal Oxide Properties

| Oxide | SLD (×10⁻⁶ Å⁻²) | Typical Thickness |
|-------|-------|-------------------|
| CuO | 5.0 | 10–50 Å |
| Cu₂O | 4.0 | 10–30 Å |
| TiO₂ | 2.6 | 10–50 Å |
| Fe₂O₃ | 7.2 | 10–30 Å |
| Al₂O₃ | 5.7 | 20–50 Å |
| NiO | 6.9 | 10–30 Å |
| SiO₂ | 3.47 | 10–20 Å (native) |

## Oxide Layer Fitting Guidelines

- **Thickness bounds**: 5–200 Å (for initial oxide layers)
- **SLD bounds**: use ±2.0 around nominal oxide SLD
- Oxide roughness is typically 3–15 Å
- When a metal in contact with a reactive ambient has no oxide in the model,
  adding one is normally the top-ranked structural hypothesis. Do NOT discard
  it on the grounds of model simplicity — let BIC decide after it has been
  tried once.

## Adhesion Layers

Adhesion layers (e.g., Ti or Cr beneath Au or Cu) are internal layers and
should NOT have oxide layers added on them. Their SLD bounds should be wider
(±3.0) to account for intermixing with adjacent layers:
- Titanium adhesion: SLD range -5.0 to 1.0
- Chromium adhesion: SLD range 1.0 to 5.0

## Refinement Strategy — Metal Oxide Interfaces

When refining a model with metal layers, apply these rules in order:

### Adhesion Layer Parameter Drift

Thin adhesion layers (Ti, Cr) are prone to parameter trade-offs during fitting:
- **Thickness inflation**: If fitted Ti thickness is >2× nominal (e.g., 100 Å when
  50 Å is expected), tighten thickness bounds close to the nominal value
  (e.g., 30–80 Å for a nominal 50 Å layer). The fitter uses the adhesion layer
  as a contrast sink when given too much freedom.
- **SLD drift**: If fitted Ti SLD deviates >50% from nominal (-1.95), check whether
  the SLD bounds are too wide. Narrowing to ±2.0 around nominal may help, but keep
  bounds wide enough for realistic intermixing (minimum ±1.5).
- **Roughness pinning**: If adhesion layer roughness is pinned at its lower bound,
  the bound may be too restrictive. Ensure roughness_min ≥ 5 Å and roughness_max
  is at least 20 Å.

### Metal Surface Oxide

If an outermost metal layer is in contact with air, D₂O/H₂O, or another
reactive ambient and there is no oxide in the model, a native oxide is the
**single most likely structural gap** — add it as a high-ranked hypothesis
in the `structural-hypothesis-ranking` list at intake. For Cu in aqueous
ambient: CuO, 10–50 Å, SLD 4.5–5.5. For Ti in aqueous ambient: TiO₂,
10–50 Å, SLD 2.0–3.2. Roughness typically 3–15 Å.

During refinement, if parameter-only tweaks have not reached the
acceptance threshold and this hypothesis is `pending`, try it before
further bound fiddling. Do not gate oxide addition on arbitrary χ²
thresholds such as "only if χ² > 10" — a structural gap produces
systematic residuals even at moderate χ², and the BIC guardrail will
automatically revert the change if the added complexity is not justified.

Do NOT add an oxide layer to a metal that is buried under another layer
or a solvent unless the sample description explicitly asks for it.

### Multi-Layer Metal Stacks (e.g., Cu on Ti on Si)

- The Cu/Ti interface roughness should typically be 5–15 Å. If it grows larger
  than 20 Å, consider whether intermixing is being used to compensate for a
  missing interface layer.
- Keep the Cu SLD bounds within ±2.0 of the nominal 6.55 value.
- For Ti under Cu, the Ti SLD can deviate more due to intermixing — use bounds
  of -5.0 to 1.0 but flag deviations beyond -4.0 or above 0.0 as potential
  unphysical behavior.
