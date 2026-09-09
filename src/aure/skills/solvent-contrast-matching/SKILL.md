---
name: solvent-contrast-matching
description: >
  Domain knowledge for any sample measured in a liquid medium — solvents,
  electrolytes, solutions, buffers, aqueous media. Covers D₂O/H₂O and other
  deuterated-solvent SLD values, isotope-confusion detection, and the common
  case where the user does NOT state whether the liquid is deuterated (an
  unspecified solvent may well be deuterated). ACTIVATE THIS SKILL WHENEVER the
  ambient/fronting medium is a liquid (water, electrolyte, solution, buffer,
  THF, toluene, etc.) — deuteration need not be mentioned for it to be relevant,
  because whether a solvent is H-form or D-form dominates the low-Q
  reflectivity and is the single most common omission in sample descriptions.
metadata:
  author: aure
  version: "1.1"
---

## Always Suspect Unspecified Deuteration

**Whenever the sample sits in a liquid and the user did NOT say whether it is
deuterated, treat "the solvent is actually deuterated" as a live hypothesis.**
Users routinely describe an electrolyte/solvent without naming the isotope
("0.1 M NaHCO₃ electrolyte", "in THF", "in buffer"); the H-form is parsed by
default with an SLD ≈ 0, but the experiment may well have used a D-form
(SLD ≈ 6) for contrast. Whether the liquid is H or D dominates the low-Q
reflectivity, so getting it wrong cannot be repaired by parameter tuning.

### The signature: a critical edge / low-Q upturn the H-form can't explain

A liquid with SLD ≈ 0 (H-form) against, say, Cu (SLD ≈ 6.5) gives strong
contrast; a D-form solvent (SLD ≈ 6.4) gives almost none. When the data shows a
critical edge or a strong low-Q feature that an SLD ≈ 0 ambient cannot
reproduce, the optimizer has two ways to fake it instead of recognizing a
deuterated ambient — both are tell-tale symptoms:

- **A thick, high-SLD layer near the substrate** (the "also valid but wrong"
  explanation): a layer's thickness driven to ≈ 2× its nominal value, or an
  extra interfacial/surface layer that has no physical justification.
- **A metal layer's SLD pinned toward the ambient**: e.g. a Cu layer's rho
  driven down to its lower bound, or an oxide's rho pinned at its upper bound —
  the model bending material SLDs to manufacture a contrast that a deuterated
  ambient would provide for free.

These two interpretations (deuterated ambient **vs.** thick/over-contrasted
layers) are **mutually exclusive** explanations of the same feature. When the
solvent really is deuterated, modeling it as such fits the low-Q data *better*
(lower χ²) **and** with fewer parameters (lower BIC) than the thick-layer
workaround — it wins on the data, not on parsimony alone. The χ²/BIC
regression guardrails keep the reinterpretation only if the refit actually
improves; a single worse refit is auto-reverted, which is expected — do not
re-propose the same reinterpretation after it has been reverted.

### What to do

1. **Enumerate it as a structural hypothesis at intake.** For any liquid
   ambient whose isotope was not specified, list a high-ranked hypothesis:
   *"Reinterpret the ambient as a deuterated solvent."* Set the ambient SLD
   from its H-form value (≈ 0) to the D-form table value below, and widen its
   range to span both (e.g. `[−0.6, 6.6]` for water, `[0.0, 6.5]` for THF).
   List this hypothesis even when the described stack looks complete — the
   missing piece is the ambient's isotope, not a layer.
2. **Try it when the baseline / a thick-layer interpretation stalls.** If χ²
   stops improving over a couple of iterations, or a fit "succeeds" only by
   inflating a layer thickness or pinning a metal SLD toward the ambient (see
   signature above), realize the deuterated-ambient hypothesis instead of more
   parameter tweaks.
3. **Rewind to the intake baseline when you realize it.** Because the
   thick-layer and deuterated-ambient explanations are mutually exclusive,
   start from the clean baseline (intake) model — discard the speculative
   thick/extra layer and any SLD inflation accumulated while chasing the
   contrast — and apply ONLY the deuteration (change the ambient SLD, let it
   vary). Do not stack a deuterated ambient on top of the thick layer; that is
   over-parameterized and double-counts the contrast.
4. **Use the critical edge to pin the ambient SLD — don't leave it wandering.**
   The critical edge, with the known substrate SLD, *deterministically* fixes
   the ambient SLD: `ρ_ambient ≈ ρ_substrate + (Qc/4)²/π` (assuming a film of
   typical thickness, which does not set the edge). The feature analysis reports
   this as an "implied ambient SLD". When it is available, **constrain the
   ambient SLD tightly around that value** (≈ ±1), e.g. `[5.5, 6.6]` — not the
   full H–D range `[−0.6, 6.6]`. A range left wide lets the optimizer settle on
   a physically-wrong compromise (e.g. an ambient SLD of ~3 that trades off
   against inflated layers) instead of the true deuterated value.

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
- H₂O/D₂O mixtures interpolate linearly in volume fraction *f* of D₂O
  between ρ(H₂O) = −0.56 and ρ(D₂O) = 6.36:
  `ρ ≈ −0.56 + f · 6.92`
  - 25% D₂O: SLD ≈ 1.17
  - 38% D₂O: SLD ≈ 2.07 — **this** is the silicon contrast match (CMSi)
  - 50% D₂O: SLD ≈ 2.91
  - 75% D₂O: SLD ≈ 4.64
- "Contrast-matched to silicon" means the solvent SLD ≈ 2.07 (Si SLD), which
  is reached at ≈ 38 % D₂O by volume — **not** at 50 %. If a description says
  "CMSi" or "contrast-matched", constrain the ambient SLD near 2.07 rather
  than inferring a composition, and do not assume a 50/50 mixture.
- When fitting a contrast series, layer thicknesses and roughnesses should be
  constrained to be the same across contrasts; only SLDs of solvated components change.

## Mixed Solvents

- When the solvent is a mixture (e.g., 80% D₂O / 20% H₂O), calculate the SLD as
  a linear combination: SLD_mix = f_D × SLD_D + (1 - f_D) × SLD_H
- Allow the ambient SLD to vary within a range that covers the expected mixture
  composition uncertainty.

## Refinement Strategy — Solvent Contrast

When refining models with solvent ambients:

- **Ambient SLD mismatch**: If the fitted ambient SLD deviates >0.5 from the
  expected solvent SLD, this is often the dominant source of high χ². Constrain
  the ambient SLD to ±1.0 around the expected value before making structural
  model changes.
- **Isotope confusion**: If the fitted SLD suggests a different isotope variant
  (e.g., fitted 6.3 for stated THF at 0.18), flag this issue. The sample
  likely uses dTHF, not THF. Correct the ambient SLD range accordingly.
- **Unspecified deuteration (the thick-layer trap)**: If the ambient is a
  liquid whose isotope the user never stated and the fit is stalling — or is
  "succeeding" only by inflating a layer thickness (≈ 2× nominal) or pinning a
  metal/oxide SLD toward its bound (see "Always Suspect Unspecified
  Deuteration" above) — the optimizer is faking, with structure, the contrast a
  deuterated ambient would supply. Realize the deuterated-ambient hypothesis:
  **rewind to the intake baseline model**, set the ambient SLD to the D-form
  value with a wide range, and refit. Discard the speculative thick/extra
  layer; do not keep both explanations.
- **Intensity and ambient coupling**: Intensity normalization and ambient SLD
  can trade off. If both are hitting bounds, widen both ranges slightly —
  intensity to [0.5, 1.3] and ambient SLD by ±0.5.
- **Multi-file contrast series**: When co-refining multiple contrasts, ensure
  each file has independent intensity normalization but shared structural
  parameters. If one contrast fits much worse, check that its ambient SLD
  is correctly set for the specific solvent used.
