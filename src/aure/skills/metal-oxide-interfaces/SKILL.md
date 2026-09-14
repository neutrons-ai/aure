---
name: metal-oxide-interfaces
description: >
  Domain knowledge for metal oxide interface analysis in reflectometry. Covers
  native oxide formation on metals (Cu, Ti, Al), recomputed reference SLDs for
  the common metals and oxides, rules for when to add or avoid an oxide layer,
  adhesion-layer behaviour, and how to read an oxide that changes across a
  series. Always use when the sample involves metals, copper, titanium, native
  oxides, or SiO2. The user may not have specified the oxide layer, but if the
  metal is exposed to ambient conditions it is likely present and should be
  considered in the model.
metadata:
  author: aure
  version: "2.0"
---

## Native Metal Oxide Formation

When a metal layer is **directly** in contact with the ambient medium (air,
solvent, etc.) and no oxide, SEI, or other surface layer is already present,
a thin native oxide layer typically forms.

### When to Add an Oxide Layer

- Metal is the outermost layer (in contact with ambient)
- No existing oxide, SEI, or surface layer between the metal and ambient
- The metal is one that oxidises readily (Cu, Ti, Al)
- The fit shows systematic residuals suggesting a missing interface layer
- An electrochemical step should have grown or reduced one

### When NOT to Add an Oxide Layer

- An oxide or surface layer already exists between the metal and ambient
- The metal is a buried layer (e.g., Ti adhesion layer beneath Cu)
- The metal is already covered by an SEI or other surface layer
- The film was made and measured without air exposure
- You are adding it only because χ² improved slightly — two extra free
  parameters will always improve χ² slightly
- Do NOT split an existing oxide into sublayers (e.g., CuO + Cu₂O) — two
  adjacent layers with SLDs 1.1 apart, both below the resolution limit, are not
  separable and will simply trade off against each other

**The middle case is the trap.** A layer that collapses to its minimum thickness
*while an adjacent parameter is railed at a bound* is a failed optimisation, not
evidence of absence. See `thin-layer-degeneracy`, which comes first.

## Common Metal Oxide Properties

Computed from CRC bulk densities and coherent scattering lengths — **not copied
from a quick-reference table.** Several published tables disagree with each other
and with this, and an earlier version of this skill carried three wrong values.

| Material | Bulk SLD (10⁻⁶ Å⁻²) | Density (g cm⁻³) | Typical oxide thickness |
|---|---|---|---|
| Cu | **6.554** | 8.96 | — |
| CuO (tenorite) | **6.459** | 6.31 | 10–50 Å |
| Cu₂O (cuprite) | **5.363** | 6.00 | 10–50 Å |
| Cu(OH)₂ | 2.464 | 3.37 | — |
| Ti | −1.91 | 4.506 | — |
| TiO₂ | 2.627 | 4.23 | 10–50 Å |
| Cr | 3.027 | 7.19 | — |
| Au | 4.662 | 19.3 | — |
| SiO₂ | 3.468 | 2.196 | 10–20 Å (native) |
| Si | 2.073 | 2.329 | — |
| Fe₂O₃ | 7.175 | 5.24 | 10–30 Å |
| Al₂O₃ | 5.671 | 3.95 | 20–50 Å |
| NiO | **8.660** | 6.67 | 10–30 Å |

**The check:** the same arithmetic reproduces Cu = 6.554 and Si = 2.073. If a
recomputation does not, the inputs are wrong before the answer is.

**Two entries matter more than the rest.**

*Stoichiometric CuO is nearly contrast-matched to copper* — 6.459 against 6.554,
a gap of 0.1. A dense, fully-oxidised CuO layer is close to invisible in a
neutron measurement of a copper electrode. If you are looking for one and see
nothing, that is a plausible reason, **not** evidence it is absent. This
technique speaks about Cu(I), not Cu(II).

*Cu₂O is the one you can see*, at 5.363 against copper's 6.554. A real native
oxide is porous and hydrated rather than bulk-dense, which lowers it further —
Cu₂O at 80% of bulk density is 4.29. So a fitted copper-oxide SLD anywhere in
**4.2–5.5** is a physically ordinary cuprous oxide, and the width of that range
is porosity, not measurement error.

A corollary worth knowing in aqueous and deuterated-solvent work: Cu₂O sits
*below* both D₂O (6.373) and d8-THF (5.919), and Cu sits above both. So no
mixture of copper and fluid in any proportion can fall below the fluid level —
a sub-fluid dip in the profile cannot be porosity, roughness or solvent ingress,
and requires a genuinely low-SLD third phase.

Recompute for any material not in this table rather than trusting a reference.

## Oxide Layer Fitting Guidelines

- **Thickness bounds**: 5–200 Å (for initial oxide layers)
- **SLD bounds**: for a copper oxide, 4.0–6.5 — porous cuprous up to just under
  bulk copper. A value at the top of that range means *either* dense cupric oxide
  *or* no oxide at all, and reflectivity alone will not distinguish them.
- For other oxides, use ±2.0 around the recomputed nominal
- Oxide roughness is typically 3–15 Å
- A fitted metal SLD well below bulk means porosity, roughness being absorbed, or
  solvent ingress — **not** a different metal
- When a metal in contact with a reactive ambient has no oxide in the model,
  adding one is normally the top-ranked structural hypothesis. Do NOT discard
  it on the grounds of model simplicity — let BIC decide after it has been
  tried once.

## Adhesion Layers

Adhesion layers (Ti or Cr beneath Au or Cu) are 20–60 Å — below the resolution
limit, buried under a much thicker metal, and contributing little. **Treat them
as a nuisance parameter, not a result.** They are still worth including, because
leaving one out pushes its contrast into the substrate roughness.

They are internal layers and should NOT have oxide layers added on them. Their
SLD bounds should be wider (±3.0) to account for intermixing:

- Titanium adhesion: SLD range −5.0 to 1.0
- Chromium adhesion: SLD range 1.0 to 5.0

**Tie them across states.** The adhesion layer is under the electrode; nothing an
experiment does to the surface can change it. If it drifts substantially between
two states of the same physical sample, something else in the model is wrong and
the adhesion layer is absorbing it. That is a diagnostic, not a discovery.

## Refinement Strategy — Metal Oxide Interfaces

### Adhesion Layer Parameter Drift

Thin adhesion layers (Ti, Cr) are prone to parameter trade-offs during fitting:

- **Thickness inflation**: If fitted Ti thickness is >2× nominal (e.g., 100 Å when
  50 Å is expected), tighten thickness bounds close to the nominal value
  (e.g., 30–80 Å for a nominal 50 Å layer). The fitter uses the adhesion layer
  as a contrast sink when given too much freedom.
- **SLD drift**: If fitted Ti SLD deviates >50% from nominal (−1.91), check whether
  the SLD bounds are too wide. Narrowing to ±2.0 around nominal may help, but keep
  bounds wide enough for realistic intermixing (minimum ±1.5). Before widening a
  bound on a layer that can take up hydrogen or deuterium, check which direction
  the cell's isotope drives it: b_H is negative and b_D is positive, so a hydride
  lowers a metal's SLD in an H₂O cell while a deuteride *raises* it in a D₂O one.
  Reusing hydride reasoning in a deuterated cell puts the bound on the wrong side
  and can exclude the true value.
- **Roughness pinning**: If adhesion layer roughness is pinned at its lower bound,
  the bound may be too restrictive. Ensure roughness_min ≥ 5 Å and roughness_max
  is at least 20 Å.

### Metal Surface Oxide

If an outermost metal layer is in contact with air, D₂O/H₂O, or another
reactive ambient and there is no oxide in the model, a native oxide is the
**single most likely structural gap** — add it as a high-ranked hypothesis
in the `structural-hypothesis-ranking` list at intake. For Cu in aqueous
ambient: a resolvable copper oxide is **cuprous**, 10–50 Å, SLD 4.2–5.5. For Ti
in aqueous ambient: TiO₂, 10–50 Å, SLD 2.0–3.2. Roughness typically 3–15 Å.

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
- Keep the Cu SLD bounds within ±2.0 of the nominal 6.554 value.
- For Ti under Cu, the Ti SLD can deviate more due to intermixing — use bounds
  of −5.0 to 1.0 but flag deviations beyond −4.0 or above 0.0 as potential
  unphysical behavior.

## Reading an Oxide That Changes Across a Series

When the same sample is measured in several states, or time-resolved, three
interpretations compete and they have distinguishable signatures:

- **The oxide thickening or thinning** gives an *oscillatory* change template in
  Q, with node spacing ΔQ implying a thickness change of order `π/ΔQ`.
- **The oxide changing composition** at constant thickness gives a *one-sign*
  template — an SLD contrast change with no nodes.
- **Solvent penetrating the oxide** moves its SLD toward the ambient. This also
  reads as a contrast change, but in a specific direction, which is what
  separates it from the previous case.

Model the first as a thickness varying between the two steady states, with the
endpoints being the steady-state parameters — that adds no free parameters.

## Rationalizations

**"χ² improved, so the oxide is real."** Two free parameters always improve χ².
Ask whether the improvement is worth the parameters, and whether the complex
model was optimised properly before you compare.

**"The oxide fitted to 2 Å, so there isn't one."** Check whether an adjacent
layer is railed at a bound first. A collapsed layer next to a railed neighbour
is a local minimum.

**"The Ti layer came out at 26 Å in one state and 45 Å in the other."** Nothing
reaches the adhesion layer through 500 Å of copper. Tie it, and find what it was
absorbing.

**"Cu fitted to 5.8, so it's partly oxidised throughout."** More often it is
roughness or porosity being absorbed into the SLD. Look at the roughness first.

**"I'll add both Cu₂O and CuO layers to be thorough."** Two adjacent layers with
SLDs 1.1 apart, both below the resolution limit, are not separable.

**"The oxide fitted at 6.4, which matches CuO."** It also matches copper, at
6.554. At the top of the range the measurement does not distinguish a dense
cupric oxide from no oxide at all.

## Red Flags

- An oxide SLD above the parent metal's.
- An adhesion layer differing between states of one physical sample.
- A metal SLD more than ~10% below bulk with unremarkable roughness.
- An oxide thickness at a bound.
- A model with more sub-30 Å layers than the Q range can resolve — count them
  against `2π/Q_max`.
- An oxide added *and* a large increase in the metal's roughness: both are
  describing the same interfacial smearing.
- A copper-oxide SLD quoted against a reference value below 5.0. That is the
  wrong table; see the recomputed values above.

## Verification

1. **The critical edge implies the topmost SLD.** For a metal in a solvent, that
   is the ambient; for a film in air, the film. If it disagrees with the model,
   the top of the stack is wrong.
2. **Total thickness from fringe spacing** against the sum of your layers. The
   estimate is often loose, but an order-of-magnitude disagreement is real.
3. **Every oxide fit at least twice**, from different starting SLDs, per
   `thin-layer-degeneracy`.
