---
name: polymer-films
description: >
  Domain knowledge for polymer thin film reflectometry analysis. Covers common
  polymer SLD values, isotope labeling conventions, typical film thickness ranges,
  and contrast matching strategies. Use when the sample involves polymers,
  polystyrene, polymer brushes, block copolymers, or spin-coated films.
metadata:
  author: aure
  version: "1.0"
---

## Common Polymer SLDs (×10⁻⁶ Å⁻²)

| Material | SLD |
|----------|-----|
| Polystyrene (h-PS) | 1.4 |
| d-Polystyrene (d-PS) | 6.4 |
| PMMA (h-PMMA) | 1.06 |
| d-PMMA | 7.1 |
| Polyethylene (h-PE) | -0.3 |
| d-Polyethylene (d-PE) | 7.0 |
| Polyisoprene (h-PI) | 0.27 |
| d-Polyisoprene (d-PI) | 6.7 |

## Isotope Labeling

- Deuterium labeling replaces H with D to increase SLD contrast.
- A "d-" prefix indicates full deuteration; partial deuteration gives
  intermediate SLD values.
- When the sample mentions both h- and d- versions, the SLD difference
  provides the contrast mechanism.
- Be careful not to confuse protonated and deuterated variants — check
  if the description specifies the isotope labeling.

## Typical Thickness Ranges

| Film Type | Typical Thickness |
|-----------|-------------------|
| Spin-coated thin film | 100–2000 Å |
| Polymer brush (grafted) | 50–500 Å |
| Block copolymer lamellae | 100–500 Å per domain |
| Adsorbed polymer layer | 10–100 Å |
| Self-assembled monolayer | 10–30 Å |

## Contrast Matching Strategies

- Matching the SLD of one component to the solvent makes that component
  "invisible," revealing the other component's structure.
- Common contrast match points:
  - h-PS in d-toluene: PS becomes nearly invisible
  - d-PS in h-toluene: solvent SLD ≈ 0.94, d-PS stands out
  - Polymer in D₂O/H₂O mixture: tune to match specific component

## Swelling and Solvent Penetration

- Polymer films may swell in solvent, changing thickness and SLD.
- The SLD of a swollen layer is a volume-weighted average of polymer and solvent SLDs.
- If fitting a swollen film, allow wider SLD and thickness bounds than the dry film values.
