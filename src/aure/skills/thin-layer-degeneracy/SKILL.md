---
name: thin-layer-degeneracy
description: >
  Always-active meta-skill on the reliability of thin-layer fits and model
  selection. Explains why layers thinner than the resolution limit are
  multimodal (the SLD x thickness "contrast-thickness" ridge), why a single
  optimizer run or a BIC comparison can silently pick a local minimum and
  wrongly reject a real layer, how to escape with discrete SLD mode
  enumeration, how to read a correlated (rho, t) uncertainty as one
  well-determined combination (the integrated SLD deficit Gamma, a descriptor of
  the fit and not an amount of material) plus one unconstrained direction, and
  why the SLD profile must be read before any thin layer's numbers are quoted.
  Consult whenever the model contains a
  thin layer, when a physically-expected layer appears "not needed", or when BIC
  is used to accept or reject a structural change.
metadata:
  author: aure
  version: "2.1"
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
the resolution limit, many (ρ, t) pairs on a curve of roughly constant Δρ · t
fit almost equally well. Three consequences follow, and all of them look like
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
It is **one well-determined combination and one unconstrained direction along
the ridge**. Quote the combination — `Γ`, below — and say that the split is not
determined.

Sampling shows this directly: the 2D marginal for (ρ, t) displays the ridge. An
optimizer run gives a point on the ridge and no indication that a ridge exists.

The sampled intervals are partly the prior, too. The data exclude only the
low-contrast end of the ridge: an SLD too close to its neighbours' cannot produce
the feature at any thickness. Nothing excludes the high-contrast end — a thinner
layer further from its neighbours — so on that side the interval on ρ stops
where the bound was put, and the interval on `t` is that window mapped through
the ridge.

**Fixing the SLD does not escape the ridge — it hides it.** With ρ pinned, the
thickness inherits a tight-looking interval that is *conditional on the value you
chose*, and nothing in the output says so. If that thickness then differs between
two states, what changed is the combination below. It is not evidence that the
layer got thicker or thinner.

## Γ: what the fit determines, and what it does not mean

The well-determined combination is the **integrated SLD deficit** of the dip the
layer makes in the profile, in `10⁻⁶ Å⁻¹`. It is written two ways, and they are
**different numbers**:

```
Γ_slab    = t · (ρ_ref − ρ_layer)      slab parameters; ρ_ref is ONE neighbour, say which
Γ_profile = ∫ (ρ_fluid − ρ(z)) dz      rendered profile, over the region below the fluid SLD
```

They agree only for sharp interfaces. A worked case: an oxide fitted at
`t = 11.1 ± 0.3 Å` with ρ pinned at 4.10 against `ρ_D₂O = 6.10` gives
`Γ_slab = 22.2 ± 1.0`, while the profile rendered from those parameters and the
interface widths quoted in the next section gives `Γ_profile ≈ 12`. In a
synthetic scan of this case, pinning ρ anywhere from 2.0 to 4.5 changed the total
χ² by about 1 and moved `t` from 5.7 to 13.5 Å. `Γ_slab` drifted by about 8 %
(23.5 → 21.6 against the fluid, and the other way against the substrate);
`Γ_profile` stayed within 2 %. So `Γ_profile` is what the data hold fixed and
`Γ_slab` follows it only approximately — and since they are different numbers,
never set one against the other.

**Γ describes the fitted profile. It is not an amount of material**, and it is
not the surface excess or adsorbed amount that the same symbol denotes in the
adsorption literature. It is a contrast-weighted sum over everything in the
region that is not bulk fluid:

```
Γ = Σ_k (ρ_fluid − ρ_k) · (volume of component k per unit area)
```

Solid, closed or unwetted pores, pore liquid that differs from the bulk fluid,
hydroxide or hydrated phases, and substrate material inside the region all
enter, each with its own weight and some with the opposite sign. So at a fixed
amount of solid Γ still moves: when pores open, close, fill or drain, when H
exchanges for D, when the composition changes, when the fluid changes. 20 Å of
dense Cu₂O gives Γ = 20 in D₂O and 11 in d8-THF; with 20 % closed voids the same
oxide gives 52 and 41.

Therefore:

- Do **not** call Γ, or any thickness derived from it, an amount, mass, coverage,
  inventory or surface excess of a material.
- Do **not** report a change in Γ as growth, consumption, dissolution,
  densification or swelling. Report that the integrated SLD deficit rose or
  fell, and that the measurement does not say why.
- Do **not** read an unchanged Γ as an unchanged layer. Changes can cancel.
- Do **not** convert Γ into a thickness of a named material on your own
  initiative. Naming a layer in the sample description ("copper oxide") is not a
  composition model. The conversion needs one, stated by the user, and is then
  reported as conditional on it (see `ridge-invariant-observables`).
- Compare Γ between states only like with like — same form, same reference
  medium, same fluid — and only on intervals formed inside each posterior draw
  and inflated by `√χ²_red`. Separated intervals mean the profiles differ, and
  that is the whole finding. From best-fit values alone there is no interval:
  give `Γ_slab` as a description of the fit, with no uncertainty and no
  significance.

Γ is still the right thing to steer a fit by. It says which (ρ, t) pairs are the
same fit and which parameters move together, and a change in it is a good reason
to propose growth or solvent ingress as a *hypothesis to test*. Reporting that
hypothesis as a finding is what it cannot support.

When the bounding interfaces overlap (the criterion is in
`interface-swallowed-layers`), the fitted `ρ_layer` is attained nowhere and
`Γ_slab` can be twice `Γ_profile`, as above: label it as the slab product, not as
the profile's deficit. `ridge-invariant-observables` has the full treatment —
what each component weighs in Γ, the one case in which it converts to an amount,
the check that rules that case out, and what a second measurement would add.

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
SLD 4.10" describes something that is not in the model — and so does the slab
product `Γ_slab = 22.2` built from those two numbers. What the model contains is
a dip about 0.5 deep whose integrated deficit below the D₂O level is about 12.

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

BIC also over-charges a thin layer even at the right optimum. Its parameters
are constrained only along the Δρ·t ridge, worth about 1 nat each, while BIC
charges ½·ln n, which is 2.5 nats at n = 160. When the fit came from DREAM, the
evaluation prompt shows the posterior evidence next to BIC:

- If it says the provisional evidence **contests** a BIC revert, the data favour
  the layer even though BIC rejected it. Treat the hypothesis as unresolved and
  worth another, better-optimized attempt, not as ruled out.
- If it lists the new layer's parameters as **barely constrained**, the data see
  the layer's Δρ·t and not its individual ρ and t. Report the combination — as a
  description of the fit, per the section on Γ — not the pair.

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
do not, the modes are one profile written several ways: `Γ` is what the fit
determined — as a description of that profile, not as an amount — and the split
is not.

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
- **The critical edge is independent of the fringe analysis**: it says the
  stack holds a medium whose SLD is above the incident medium's by roughly the
  reported contrast. A rough check on the model's highest-SLD thick medium, not
  a value to fit to.
- Report the series as one coherent account (what changed in the profile, what
  stayed fixed), not as independent fits that happen to disagree. "Changed in the
  profile" is as far as a thin layer's numbers go: "grew" is a claim about
  material, and Γ does not make it.

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
there. Quote `Γ`, which depends only weakly on where on the ridge you pinned it,
and quote it as what it is: an integrated SLD deficit.

**"Γ tells us how much oxide there is."** It tells you how much integrated SLD
contrast the dip holds. The worked case's `Γ_slab = 22.2` is 30 Å of dense Cu₂O,
12 Å of Cu₂O at 80 % density, or 6 Å of Cu(OH)₂, and the fit does not say which,
or whether it is a mixture.

**"Γ fell, so the layer was consumed."** Or solvent filled pores that had been
empty, or a hydroxide converted, or H exchanged for D, with nothing lost. A porous
layer that floods can lose most of its Γ with every atom of the oxide still in
place. The converse fails the same way: an unchanged Γ can hide oxide lost and
voids opened.

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
- A thin layer's thickness compared between states with no comparison of `Γ` —
  or a `Γ` comparison that mixes the slab and profile forms, different reference
  media, or different fluids. See `ridge-invariant-observables`.
- `Γ_slab` presented as the profile's integrated deficit where
  `σ_top + σ_bot > t/2`. There the two can differ by a factor of two.
- `Γ`, or a thickness derived from it, reported as an amount, coverage or mass of
  a named material — or a change in `Γ` reported as growth or consumption.
- A layer's nominal SLD appearing nowhere in the fitted profile. Predictable in
  advance, not only afterwards — see `interface-swallowed-layers`.

## Verification

After sampling, on the parameter table and the fitted profile:

1. **Look at the (ρ, t) correlation** for every thin layer. A ridge means quote
   the combination `Γ`, not the pair.
2. **Read the SLD profile.** Does the layer's nominal SLD appear in it at all?
   Does the profile stay between the media it connects? If either answer is no,
   report the profile's shape — its depth and integrated deficit — not the slab.
3. **Compute `Γ`** and say which form it is and which medium it is taken against.
   Compare states like with like, on per-draw intervals inflated by `√χ²_red`.
   If the thicknesses separate but the `Γ` intervals do not, no difference has
   been measured. If the `Γ` intervals do separate, the profiles differ: report
   that, and stop there.
4. **Check nothing is at a bound.** An interval clipped at a bound is not an
   interval.
5. **Run at least two SLD modes** for any layer under the resolution limit and
   compare. If they disagree structurally at similar χ², the degeneracy is real
   and unresolved — report it rather than picking one.
6. **Read the wording back.** No sentence may turn `Γ` into an amount of a
   material, or a change in `Γ` into growth or consumption.
