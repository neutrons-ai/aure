---
name: thin-layer-degeneracy
description: >
  Always-active meta-skill on the reliability of thin-layer fits and model
  selection. Explains why layers thinner than the resolution limit are
  multimodal (the SLD x thickness "contrast-thickness" ridge), why a single
  optimizer run or a BIC comparison can silently pick a local minimum and
  wrongly reject a real layer, how to escape with discrete SLD mode
  enumeration, how to read a correlated (rho, t) uncertainty as one product
  plus one unconstrained direction, and why the SLD profile must be read before
  any thin layer's numbers are quoted. Consult whenever the model contains a
  thin layer, when a physically-expected layer appears "not needed", or when BIC
  is used to accept or reject a structural change.
metadata:
  author: aure
  version: "2.0"
---

## When this applies

Any time the model has a **thin layer** — thinner than roughly the real-space
resolution limit `2*pi / Q_max` (for a typical Q_max ~ 0.2 1/Å that is about
30 Å) — or any time a **BIC / chi-squared comparison is being used to accept or
reject a structural change**. Both situations are where automated fitting most
often goes quietly wrong.

Also: when a layer you have physical reason to expect fits to near-zero
thickness; when a fitted SLD and thickness have huge uncertainties but a
sensible-looking central value; and when two fits of the same data disagree
about a thin layer at similar χ².

## Why thin layers are multimodal (the contrast-thickness ridge)

Reflectivity constrains a thin layer mainly through the **product of its
contrast and its thickness** (Δρ · t), not through ρ and t separately. Below
the resolution limit, many (ρ, t) pairs on a curve of constant Δρ · t fit
almost equally well. Three consequences follow, and all of them look like
success:

- The layer's SLD and thickness are individually poorly determined even when
  their product is well determined — expect large, correlated uncertainties.
- The likelihood surface has **several distinct local minima** (e.g. a thin
  dense layer vs. a thicker dilute one), separated by barriers that local
  optimizers and even global optimizers at modest effort **will not cross**.
- Two fits with different (ρ, t) but the same Δρ · t can have essentially the
  same χ². The one the optimizer happens to land in is not necessarily the
  physically correct one.

## Read the uncertainty as a pair, not as two numbers

A thin layer with `ρ = 5.0 ± 2.0` and `t = 25 ± 12` is not two loose parameters.
It is **one well-determined product and one unconstrained direction along the
ridge**. Quote the product, and say that the split is not determined.

Sampling shows this directly: the 2D marginal for (ρ, t) displays the ridge. An
optimizer run gives a point on the ridge and no indication that a ridge exists.

**Fixing the SLD does not escape the ridge — it hides it.** With ρ pinned, the
thickness inherits a tight-looking interval that is *conditional on the value you
chose*, and nothing in the output says so. The measured quantity is still the
product:

```
Γ = t · Δρ        Δρ = ρ_layer − ρ_of_the_medium_it_displaces
```

`Γ`, in `10⁻⁶ Å⁻¹`, is the **integrated scattering-length deficit** of the
region — what the reflectivity actually constrains. It survives both the
smearing and the choice of ρ. A worked case: an oxide fitted at `t = 11.1 ± 0.3 Å`
with ρ pinned at 4.10 against `ρ_D₂O = 6.10` gives `Γ = 22.2 ± 1.0`. Move the
pinned SLD and `t` moves with it; `Γ` barely does.

Two states differ only if their `Γ` intervals separate, **after** inflating by
`√χ²_red`. Comparing thicknesses obtained under a fixed SLD compares two
conditional numbers, and the condition only cancels in the product.

**`Γ` has two estimators and they are not always interchangeable.** The slab form
above is valid while the layer is actually resolved in the profile. When the
bounding interfaces overlap, the fitted `ρ_layer` is never attained anywhere, so
`Δρ` taken from it is not the contrast the profile contains — and `Γ` must be
obtained by integrating the profile instead. The criterion that decides which
form applies is in `interface-swallowed-layers`; the profile form and its
companions are in `ridge-invariant-observables`.

## Look at the profile, because the layer may not be in it

A slab between interfaces of σ comparable to its own thickness is rendered as
overlapping error functions, and the result is not bounded by the media it
connects. From a real fit — an 11 Å oxide at ρ = 4.10 between σ = 18.9 and 8.9,
with D₂O at 6.10 and Cu at 6.55:

```
  z (Å)    -46    -18     +9     +80
  ρ       6.10   6.18   5.59    6.55
                  ^^^^   ^^^^
            above D₂O    the only dip: 5.59, not 4.10
```

Two things to take from it. The profile **rises 0.08 above the D₂O it starts
from**, over ~36 Å, before dipping — an excursion outside both bounding media,
which is a rendering artifact and not a layer. And the nominal 4.10 is **never
attained**: the closest the profile comes is 5.59, 1.5 away. So "an 11 Å layer of
SLD 4.10" describes something that is not in the model, while `Γ = 22.2` describes
something that is.

This is predictable **before** the fit rather than discovered after: the case
above has `σ_top + σ_bot = 27.8` against `t = 11.1`, and
`interface-swallowed-layers` predicts the 1.5 deviation from those numbers alone.

Always read the SLD profile before quoting a thin layer's numbers. If the profile
excurses outside the media on either side, the slab parameters are bookkeeping for
a shape, and the shape is what to report.

## A BIC verdict is only as good as the optimization behind it

Model selection compares the *best achievable* fit of each candidate. If the
optimizer settled in a local minimum for the more complex model, its χ² is too
high, its BIC looks too large, and the extra layer gets **rejected as
"not justified" when it is actually real**. So:

- Do **not** treat a BIC regression as proof a layer is absent when that layer
  is thin and physically expected. First make sure the complex model was
  actually optimized well.
- Signatures of a layer-absorbing local minimum (not a true rejection):
  - an *adjacent* layer's parameter pinned at a bound (e.g. an adhesion
    layer's SLD railed to its limit),
  - a roughness pinned at a bound,
  - only a tiny χ² change for the added parameters,
  - the added layer collapsing to its minimum thickness.
- When you see these, the right move is to **re-optimize the complex model
  from better starting points**, then re-compare BIC — not to conclude the
  layer isn't needed.

## Escaping the trap: discrete SLD mode enumeration

The reliable, cheap way to find the right basin for a thin layer is to **stop
treating SLD as continuous** and enumerate a few discrete seeds:

1. Pick ~3 SLD seed values for the thin layer spanning its physical range —
   e.g. the pure material, a plausible hydrated or oxidised value, the ambient.
2. For each seed, hold everything else at the current best values and run a
   short local fit.
3. Keep the basin with the lowest χ², then continue the normal fit/refinement
   from there.

This routinely finds a better minimum than a single global-optimizer run at
the same cost. Note what does **not** work: seeding the whole stack from a
sibling measurement's converged structure — the optimizer can still slide into
the wrong SLD mode of the thin layer. Enumerate the mode explicitly.

**If several modes fit comparably, that is the answer.** The data does not
distinguish them, and that belongs in the report rather than being resolved by
whichever one was run last. If `Γ` agrees across the modes while the thicknesses
do not, `Γ` is the result and the split is not.

Run each mode as its own recorded fit rather than overwriting one — several
results that can be compared beat one result whose starting point nobody
remembers.

## Use sibling / time-series measurements as a prior

When several measurements come from the **same sample** (a contrast series, an
electrochemical/anneal time series, repeated OCV points), treat a layer that is
firmly resolved in a *cleaner* member of the set as a strong prior for the
noisier ones:

- A layer clearly present in a high-quality run should not be dropped from a
  noisier sibling just because that noisier fit's BIC didn't call for it —
  re-check with mode enumeration first.
- Structural parameters that should be physically continuous across the series
  (a substrate oxide, an adhesion layer) are good candidates to carry over as
  seeds, or to co-refine (see the `multi-state-corefinement` skill).
- **A time series is a continuity constraint.** If a time-resolved run brackets
  a state, the layer cannot jump discontinuously between the two steady states.
  Tying the endpoints across the series rules out mode pairs that would require
  an implausible excursion in between.
- **The critical edge constrains the topmost layer** independently of any fringe
  analysis: `Qc` and the SLD it implies are a direct measurement.
- Report the series as a physically coherent story (what grew, what stayed
  fixed), not as independent fits that happen to disagree.

## Interaction with the hypothesis list

This skill sharpens how to walk the `structural_hypotheses` list: when a
pending hypothesis adds a *thin* layer, and a first attempt regresses BIC,
check for the local-minimum signatures above and retry with mode enumeration
before marking the hypothesis `rejected`. A thin-layer hypothesis rejected
without a mode-enumerated attempt is not yet a real rejection.

## Rationalizations

**"BIC says the layer isn't needed."** BIC compares best achievable fits. If the
complex model was not optimised well, BIC is comparing a good fit against a bad
one and the conclusion is about the optimizer.

**"The uncertainty is small, so the value is good."** An optimizer reports the
curvature of the local minimum it found. It cannot report the existence of
another minimum elsewhere on the ridge. Only sampling can.

**"Both fits give χ² near 1, so either is fine."** They are not interchangeable
if they imply different physics — a 20 Å dense oxide and a 60 Å hydrated one are
different claims about the sample.

**"I'll just widen the bounds and refit."** Wider bounds do not help an optimizer
cross a barrier. They usually make it worse by adding unphysical territory to
explore.

**"The layer must be there, so I'll fix its thickness at the literature value."**
Defensible, but then say so: it is an assumption, not a measurement, and the
uncertainty on everything downstream inherits it.

**"I fixed the SLD, so the thickness is now well determined."** Its interval got
narrow, which is not the same thing. It is a conditional interval — conditional
on a value you chose, which the output does not mention — and the ridge is still
there. Quote `Γ`, which does not depend on where on the ridge you pinned it.

**"The oxide came out 11 Å, so there is an 11 Å oxide."** Read the profile before
you believe that. If its interfaces are wider than it is, the nominal SLD is
attained nowhere and the 11 Å is a coordinate on a shape, not a layer thickness.

## Red Flags

- A thin layer reported with a precise SLD *and* a precise thickness from a
  single local-optimizer run. One of those numbers is not measured — and a
  single local-optimizer run is not evidence either way: re-running an optimizer
  with a parameter freed can return a *worse* fit than its own parent simply
  because the search failed from a scaffolded start. Sample it and read the
  posterior against the parameter's physical floor.
- An adhesion layer or oxide at exactly its lower thickness bound.
- Any parameter railed to a bound in the model that "lost" a BIC comparison.
- A thin-layer SLD outside the range of anything the sample could be made of.
  Check the bound was set for the isotope actually in the cell before widening
  it: b_H is negative and b_D positive, so uptake lowers a metal's SLD in an
  H₂O cell and raises it in a D₂O one.
- `σ_top + σ_bot > t/2`, with the slab's thickness and SLD still quoted as if
  they were separable. The older "roughness below about half the layer" rule is
  **twice too lax**: at that threshold the nominal SLD is still missed by ~0.43
  for a typical contrast. See `interface-swallowed-layers`.
- An SLD profile that excurses outside the media on either side of the layer.
- A thin layer's thickness compared between states without the comparison being
  made on `Γ` — or with the slab estimator `Γ = t · Δρ` used where
  `σ_top + σ_bot > t/2` makes it invalid. See `ridge-invariant-observables`.
- A layer's nominal SLD appearing nowhere in the fitted profile. Predictable in
  advance, not only afterwards — see `interface-swallowed-layers`.

## Verification

After sampling, on the parameter table and the fitted profile:

1. **Look at the (ρ, t) correlation** for every thin layer. A ridge means quote
   the product.
2. **Read the SLD profile.** Does the layer's nominal SLD appear in it at all?
   Does the profile stay between the media it connects? If either answer is no,
   report the profile and `Γ`, not the slab.
3. **Compute `Γ`** — by the criterion in `interface-swallowed-layers`, from the
   slab or from the profile — and compare states on that, with intervals inflated
   by `√χ²_red`. If the thicknesses separate but the `Γ` values do not, there is
   no measured difference.
4. **Check nothing is at a bound.** An interval clipped at a bound is not an
   interval.
5. **Run at least two SLD modes** for any layer under the resolution limit and
   compare. If they disagree structurally at similar χ², the degeneracy is real
   and unresolved — report it rather than picking one.
