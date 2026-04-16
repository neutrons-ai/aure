---
name: solvent-contrast-matching
description: >
  Domain knowledge for solvent contrast variation in neutron reflectometry. Covers
  D₂O/H₂O and other deuterated solvent SLD values, isotope confusion detection,
  and contrast variation experiment guidance. Use when the sample involves D2O,
  H2O, deuterated solvents, contrast matching, contrast variation, protonated or
  deuterated solvents, THF, or toluene.
metadata:
  author: aure
  version: "1.0"
---

## Common Solvent SLDs (×10⁻⁶ Å⁻²)

| Solvent | H-form SLD | D-form SLD | D-form Name |
|---------|------------|------------|-------------|
| Water | -0.56 (H₂O) | 6.36 (D₂O) | D₂O |
| THF | 0.18 | 6.35 | d8-THF |
| Toluene | 0.94 | 5.66 | d-Toluene |
| Cyclohexane | -0.28 | 6.7 | d12-Cyclohexane |
| Ethanol | -0.34 | 6.2 | d6-Ethanol |
| Methanol | -0.37 | 5.8 | d4-Methanol |

## Isotope Confusion Detection

- In back-reflection geometry through a substrate (e.g., Si, SLD=2.07), a critical
  edge at low Q indicates that either a film layer or the ambient has SLD > substrate SLD.
- If the ambient is stated as a protonated solvent (e.g., THF with SLD ≈ 0.18) but
  the fitted ambient SLD is much higher, the solvent is likely deuterated
  (e.g., d8-THF with SLD ≈ 6.35).
- Always cross-check the fitted ambient SLD against the expected solvent SLD.
- Flag isotope mismatches as an issue: the user may have specified the wrong
  isotope variant.

## Ambient SLD Check

- Check if the ambient (fronting) SLD is reasonable for the stated medium.
- If the fitted intensity is pinned at its lower or upper bound, this may indicate
  the intensity normalization range is too narrow and should be widened.

## Contrast Variation Experiments

- Multiple measurements of the same sample with different solvent isotope
  compositions constrain the model more tightly.
- H₂O/D₂O mixtures give intermediate SLD values:
  - 25% D₂O: SLD ≈ 1.17
  - 50% D₂O (CMSi): SLD ≈ 2.07 (matches silicon)
  - 75% D₂O: SLD ≈ 4.63
- "Contrast-matched to silicon" means the solvent SLD ≈ 2.07 (Si SLD).
- When fitting a contrast series, layer thicknesses and roughnesses should be
  constrained to be the same across contrasts; only SLDs of solvated components change.

## Mixed Solvents

- When the solvent is a mixture (e.g., 80% D₂O / 20% H₂O), calculate the SLD as
  a linear combination: SLD_mix = f_D × SLD_D + (1 - f_D) × SLD_H
- Allow the ambient SLD to vary within a range that covers the expected mixture
  composition uncertainty.
