---
name: polymer-films
description: >
  Domain knowledge for polymer, ionomer and brush film reflectometry. Covers
  recomputed dry SLDs with the densities they assume, isotope labelling,
  typical thickness ranges, converting a fitted SLD into a solvent volume
  fraction, swelling, contrast matching, and why a large roughness is usually a
  concentration gradient wearing the wrong parameter. Use when the sample
  involves polymers, polystyrene, ionomers, polymer brushes, block copolymers,
  or spin-coated films.
metadata:
  author: aure
  version: "2.0"
---

## Overview

A polymer film in a solvent is not the dry polymer. It swells, it takes up
solvent, and its interface with the solvent is often graded rather than sharp.
All three change what the model should look like, and all three are commonly
absorbed into a roughness that then means nothing.

The rule that does most of the work: **a film's fitted SLD in solvent tells you
its solvent content**, and that fraction is the number to report.

## Common Polymer SLDs (×10⁻⁶ Å⁻²)

Computed from formula and density. **The density is quoted with every value
because the SLD is directly proportional to it** — a polymer 10% denser than
assumed has an SLD 10% higher, and semicrystalline polymers vary more than that
between processing routes.

| Polymer | Formula | Density | SLD | Deuterated | Density | SLD |
|---|---|---|---|---|---|---|
| Polystyrene | C₈H₈ | 1.05 | **1.41** | d8-PS | 1.13 | **6.46** |
| PMMA | C₅H₈O₂ | 1.18 | **1.06** | d8-PMMA | 1.27 | **6.94** |
| Polyethylene | C₂H₄ | 0.94 | **−0.34** | d4-PE | 1.08 | **8.10** |
| Polyisoprene | C₅H₈ | 0.91 | **0.27** | d8-PI | 1.00 | **6.85** |
| PEO | C₂H₄O | 1.13 | **0.64** | d4-PEO | 1.23 | **7.05** |
| PDMS | C₂H₆OSi | 0.97 | **0.06** | — | | |
| PFSA ionomer (Nafion) | — | — | **4.1–4.3** | — | | |

Deuterated densities assume the same molar volume as the protonated polymer,
scaled by the molar-mass ratio — which is the standard assumption and accurate
to a percent or so.

**Fluoropolymers are the useful exception.** Fluorine gives a high SLD without
deuteration, so a PFSA ionomer at ~4.2 already contrasts strongly against both
H₂O and most metals.

Solvent values used below, for reference (see `solvent-contrast-matching`):
H₂O **−0.56**, D₂O **6.37**, h-toluene **0.94**, d8-toluene **5.66**,
Si **2.07**.

Recompute for any polymer not listed rather than trusting a reference table, and
state the density you assumed.

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

## Turn a Fitted SLD into a Solvent Fraction

This is the main quantitative step, and it is one line:

```
φ_solvent = (ρ_fit − ρ_dry) / (ρ_solvent − ρ_dry)
```

A PFSA ionomer (dry 4.2) in D₂O (6.37) fitting to 5.1:

```
φ = (5.1 − 4.2) / (6.37 − 4.2) = 0.41        ~41% solvent by volume
```

**Report the fraction, not the SLD.** An SLD of 5.1 is not interpretable by a
reader; "41% hydrated" is.

Two checks on the result: it must lie between 0 and 1, and it must be consistent
with the film's thickness change. A film that swelled 40% in thickness and
reports 5% solvent is inconsistent — one of the two numbers is wrong.

## Swelling and Solvent Penetration

- Polymer films swell in solvent, changing thickness *and* SLD. Both should move
  together; one moving alone is a warning.
- The SLD of a swollen layer is a volume-weighted average of polymer and solvent.
- Typical dry thicknesses are 50–1000 Å, and hydrated ionomers commonly swell
  10–50%.
- When the ambient changes, **thickness and SLD are both per-state** — the film
  genuinely is a different thing wet. The substrate and anything under the film
  stay shared.
- If fitting a swollen film, allow SLD bounds spanning the dry polymer value to
  the solvent value, and wider thickness bounds than the dry film.

## Do Not Let Roughness Stand In for a Gradient

A polymer–solvent interface is often a **concentration gradient** tens of
angstroms deep, not a sharp interface with roughness. A single slab plus a large
roughness can fit it, but then:

- the roughness is not a roughness, and quoting it as one is wrong;
- the model is straining — see `interface-swallowed-layers` for the quantitative
  form of when an interface width has swallowed the layer it bounds;
- the gradient's shape, which is the physics, is discarded.

The honest alternative is two or three sub-layers with decreasing polymer
fraction. That costs parameters, so decide with `thin-layer-degeneracy`'s rules:
add them only if the Q range resolves them, and check the extra layers do not
collapse or rail.

## Contrast Matching Strategies

Matching the SLD of one component to the solvent makes that component
"invisible," revealing the other component's structure. The point of deuteration
is to choose what stands out:

- **Label the polymer** — d8-PS at 6.46 against H₂O at −0.56 is close to the
  maximum contrast available.
- **Match the solvent to the polymer** to make the film invisible and isolate
  what is under it. h-PS in d8-toluene is the classic case.
- **Match the solvent to the substrate** — **38% D₂O** matches silicon at 2.07,
  removing the substrate's contribution.
- d-PS in h-toluene: solvent SLD ≈ 0.94, so the film stands out strongly.
- Polymer in a D₂O/H₂O mixture: tune to match whichever component you want gone.

## Refinement Strategy — Polymer Films

- **Thickness drift**: Polymer film thickness can vary from nominal due to
  processing conditions. Allow thickness bounds of ±50% around the nominal
  value for spin-coated films.
- **SLD between isotope variants**: If the fitted SLD falls between h- and d-
  values (e.g., 3.9 for PS where h-PS = 1.41 and d8-PS = 6.46), this suggests
  partial deuteration or isotope blending. Adjust bounds to the appropriate
  intermediate range rather than forcing a pure-isotope SLD. In a *wet*
  measurement, check the solvent-fraction explanation first — it is the more
  common cause.
- **Brush/grafted layers**: These often have graded density profiles. If χ²
  is high and roughness is large relative to thickness, consider whether
  the single-slab approximation is adequate before adding sublayers.
- **Swollen films**: When the sample is measured in solvent, the layer SLD
  should be allowed to vary between the dry polymer SLD and the solvent SLD.
  Set SLD bounds accordingly.

## Rationalizations

**"The film SLD came out between dry and solvent — the model is wrong."** That is
the expected result for a swollen film. Convert it to a volume fraction.

**"Roughness is 80 Å on a 200 Å film, but χ² is good."** It fits because a wide
error function resembles a gradient. It is not an interface width, and reporting
it as one overstates what was measured.

**"The film is thinner wet than dry."** Possible — collapse, dissolution, or
delamination — but check the fit is not trading thickness against SLD along the
degeneracy ridge first.

**"I'll fix the polymer SLD at the dry value since I know the material."** Only
valid in air. In solvent that forces all the swelling into thickness, which is
usually not where it is.

**"Ionomers are just polymers."** They have mobile ions and an uptake that
depends on counter-ion and humidity, so the same film measured twice under
nominally the same conditions can genuinely differ. Record the conditions.

## Red Flags

- A polymer SLD outside the range between its dry value and the ambient.
- Roughness exceeding half the layer it bounds.
- A film swelling in thickness with no change in SLD, or the reverse — both
  should move together.
- A solvent fraction outside 0 to 1.
- A dry polymer SLD used as a fixed value in a wet measurement.
- Three or more sub-layers used to describe a gradient over a Q range that
  cannot resolve them.
- A deuterated polymer SLD quoted without the density it assumes.

## Verification

1. **Convert every fitted polymer SLD to a solvent fraction** and check it lies
   in 0–1 and matches the thickness change.
2. **Compare roughness against thickness.** Over half means the model is
   describing a gradient with the wrong tool.
3. **Fit the swollen state from at least two starting thicknesses.** Thickness
   and SLD trade off along the ridge; if the two runs land in different places
   at similar χ², say so.
