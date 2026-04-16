---
name: metal-oxide-interfaces
description: >
  Domain knowledge for metal oxide interface analysis in reflectometry. Covers
  native oxide formation on metals (Cu→CuO, Ti→TiO₂), oxide thickness ranges,
  and rules for when to add or avoid oxide layers. Use when the sample involves
  metal oxides, copper oxide, titanium oxide, native oxides, or SiO₂.
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
- Prefer keeping the model simple (fewer layers) over adding speculative oxide layers

## Adhesion Layers

Adhesion layers (e.g., Ti or Cr beneath Au or Cu) are internal layers and
should NOT have oxide layers added on them. Their SLD bounds should be wider
(±3.0) to account for intermixing with adjacent layers:
- Titanium adhesion: SLD range -5.0 to 1.0
- Chromium adhesion: SLD range 1.0 to 5.0
