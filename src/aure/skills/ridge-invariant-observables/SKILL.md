---
name: ridge-invariant-observables
description: >
  What may be quoted instead of a slab thickness when a thin layer sits on the
  contrast-thickness ridge, and what those numbers do NOT mean. Across states a
  fitted thickness can move along the ridge and suggest a change the data do
  not support, even in a good fit (thin-layer-degeneracy covers the fitting
  side; this is the reporting side). Defines the ridge-stable profile
  descriptors (depth, integrated SLD deficit Gamma in its slab and profile
  forms, dip width) and their limits: Gamma is a contrast-weighted sum over
  everything that is not bulk fluid, so it is not an amount of material, and a
  change in it is not by itself growth, consumption or densification —
  porosity, solvent ingress, hydration and H/D exchange all move it at constant
  inventory. Consult whenever a conclusion rests on a layer's thickness changing
  between states, potentials or times, or before reporting growth, consumption,
  densification or an amount of material.
metadata:
  author: aure
  version: "2.0"
---

## When this applies

Any time a **conclusion rests on a fitted `thickness` changing** across states,
potentials or time — and any time you are about to say a layer grew, was
consumed, densified, or holds some amount of a material.

It applies even when the fit is good. This failure survives a healthy chi-squared,
a converged chain, and no parameter at a bound.

The skill does two jobs. It names the numbers that stay put along the ridge, so
they can be quoted in place of a thickness. And it limits what those numbers may
be said to mean, because the tempting reading — "Gamma is how much of the layer
there is" — is wrong in general.

## The reporting failure

In a single contrast a thin layer's `thickness` and `rho` are correlated at
r >= 0.9. Only one combination of them is determined; the position along the
ridge is not measured. So a fit can report

    t:    42.4  ->  55.7  ->  57.9  Angstrom
    rho:  5.060 ->  5.380 ->  5.576

and this does **not** establish a layer that grew and densified. Position along
the ridge is not measured, so the thickness column carries no such conclusion.
The same three states, same fit, same chain, give an integrated SLD deficit
**falling by half at 12.9 sigma** — the one well-determined number moved the
opposite way from the thickness.

What that supports is exactly this: the three profiles differ, and the dip's
integrated deficit fell. It does not support "the layer grew" (the thickness
reading). It does not support "the layer was consumed" either (the tempting
reading of the deficit): an SLD rising toward the fluid's while the deficit
falls is also what solvent filling the pores of a layer looks like, with
nothing lost.

## Quote these instead

| descriptor | definition | what it describes |
|---|---|---|
| depth | `rho_fluid - min(rho(z))` over the surface region | whether the profile has a dip at all |
| `Gamma_profile` | `integral (rho_fluid - rho(z)) dz` over the dip | the dip's integrated SLD deficit, in 1e-6 / Angstrom |
| width | full width of the dip at half its depth | whether the dip is narrow or spread out |

"The dip" is the contiguous region around the profile minimum where `rho(z)` lies
below the fluid SLD. These three describe the **shape of the fitted profile**.
None of them is a property of a material.

All are formed **inside each posterior draw**, from the rendered profile, so the
t/rho correlation cancels. Computing them from marginal medians reintroduces the
error you are trying to avoid. A difference between states is established only
when the per-draw intervals separate (inflated by the square root of the reduced
chi-squared, as in `thin-layer-degeneracy`) — and what is then established is
that the profiles differ.

If there are no draws to work from — only a table of best-fit values — these
intervals cannot be formed. Say so; do not estimate them, and do not integrate a
profile by hand. You may give the slab product `Gamma_slab` (next section) from
the best-fit values, named as that, with no uncertainty, no significance, and no
statement that two states differ.

## Two numbers are called Gamma

    Gamma_slab    = t * (rho_ref - rho_layer)           from the slab parameters
    Gamma_profile = integral (rho_fluid - rho(z)) dz    from the rendered profile, over the dip

**They are not the same number.** They agree only when both interfaces are sharp.

- `Gamma_slab` needs a reference medium, and the layer has two neighbours. For an
  11.1 Angstrom layer at 4.10 between a fluid at 6.10 and Cu at 6.55 it is 22.2
  against the fluid and 27.2 against the substrate. Always say which.
- `Gamma_profile` depends on the interface widths and on the fluid SLD, not on
  the layer alone. That same slab gives `Gamma_profile` = 22.2 with sharp
  interfaces, 16.5 with both widths at 8.9, 11.9 with 8.9 on the fluid side and
  18.9 on the substrate side, and 28.1 with 25 and 5. A change of 0.05 in the
  fluid SLD moves the 11.9 by about 11 %.
- Along the ridge, `Gamma_profile` is the one that stays put. In a synthetic scan
  of that case — the layer's SLD pinned at values from 2.0 to 4.5, everything
  else refitted, total chi-squared changing by about 1 — `t` ran from 5.7 to
  13.5 Angstrom, `Gamma_slab` against the fluid drifted from 23.5 to 21.6,
  `Gamma_slab` against the substrate drifted the other way from 26.1 to 27.6,
  and `Gamma_profile` stayed between 11.9 and 12.1.

Which to quote:

- With the rendered profile for each draw: `Gamma_profile`.
- With parameters only: `Gamma_slab`, named as the slab product, with its
  reference medium. Where `sigma_top + sigma_bot <= t/2` fails (see
  `interface-swallowed-layers`), add that the profile's integrated deficit is a
  different number — it can be half the slab product — and was not computed.
  Even where the criterion holds the two differ: 48.0 against 44.1 in a
  synthetic case sitting exactly on it.
- Never one against the other, and never two values taken against different
  media or in different fluids.

## What Gamma is made of

If the SLD at each depth is the volume-weighted mean of what is there, then

    Gamma = sum over components k of  (rho_fluid - rho_k) * v_k

where `v_k` is the volume of component *k* per unit area inside the dip (an
equivalent thickness, in Angstrom), and the sum runs over **everything that is
not fluid at the bulk SLD**. Bulk fluid weighs nothing. Every other component has
its own weight, and the weights are neither equal nor all of one sign. For a
copper surface (SLDs in 1e-6 / Angstrom^2, at bulk density):

| component | SLD | weight in D2O (6.373) | weight in d8-THF (5.919) |
|---|---|---|---|
| bulk fluid in open pores | = fluid | 0 | 0 |
| Cu2O | 5.363 | +1.01 | +0.56 |
| CuO | 6.459 | -0.09 | -0.54 |
| Cu metal | 6.554 | -0.18 | -0.63 |
| Cu(OH)2 | 2.464 | +3.91 | +3.45 |
| Cu(OD)2 (same molar volume) | 6.795 | -0.42 | -0.88 |
| empty or closed pore | 0 | +6.37 | +5.92 |
| H2O left in a pore | -0.558 | +6.93 | +6.48 |

One Angstrom of closed porosity counts as much as six to eleven Angstrom of Cu2O.
A hydroxide counts four to six times the same volume of Cu2O — until its hydrogen
exchanges for deuterium, after which it counts negative.

## What Gamma is not

**Gamma is not an amount of material.** It is not a mass, a coverage, a thickness
or an inventory, and it is not the surface excess (adsorbed amount) that the same
symbol denotes in the adsorption literature. A change in Gamma is not, by itself,
growth, consumption, dissolution or densification.

One copper inventory — the copper in 20 Angstrom of dense Cu2O — gives:

| what that copper is in | Gamma in D2O | Gamma in d8-THF |
|---|---|---|
| dense Cu2O | 20.2 | 11.1 |
| Cu2O with 30 % open pores holding bulk fluid | 20.2 | 11.1 |
| Cu2O with 20 % closed voids | 52.1 | 40.7 |
| Cu2O with 30 % pores holding unexchanged H2O | 79.6 | 66.6 |
| Cu(OH)2 | 189.8 | 167.7 |
| CuO | -1.8 | -11.4 |

A negative entry is a component above the fluid SLD: it makes no dip and is
invisible to these descriptors, or it cancels part of a dip made by something
else. **A small Gamma is therefore not evidence that little is there.**

Read the other way, one Gamma is many layers. `Gamma` = 22.2 against a fluid at
6.10 is 30 Angstrom of dense Cu2O, 12 Angstrom of Cu2O at 80 % density, or 6
Angstrom of Cu(OH)2 — or any mixture in between.

At constant inventory, Gamma moves when:

- pores that were closed or empty fill with solvent, or drain. A 50 Angstrom
  layer that is 70 % Cu2O, in d8-THF, goes from `Gamma` = 108 to 19.5 when its
  pores flood, with the oxide untouched;
- the pore liquid is not the bulk fluid (incomplete exchange, trapped water);
- an oxide hydrates or converts to a hydroxide, or the reverse;
- labile hydrogen exchanges with a deuterated solvent;
- the solid changes composition (Cu2O converting to CuO removes the dip
  altogether);
- the interface with the substrate roughens, bringing substrate material into
  the region;
- the fluid SLD changes, or the fluid is a different one.

One thing leaves Gamma alone: adding or removing fluid **at the bulk SLD**. A
layer that swells with bulk solvent keeps its Gamma while its thickness and SLD
both move, and that is why Gamma is a better thing to fit and compare than a
thickness. It is not a reason to read Gamma as the solid.

## The one case that converts, and the check that rules it out

Gamma is proportional to a volume of solid only if **all** of these hold:

1. the region contains exactly one component besides fluid;
2. that component's SLD is known independently for this film — not taken from the
   fit, and not assumed equal to the bulk crystal's;
3. every pore holds fluid at the bulk SLD;
4. no substrate material lies inside the region;
5. the fluid SLD is known.

Then, and only then, `Gamma / (rho_fluid - rho_solid)` is the thickness that solid
would have if it were dense. The reflectivity fit establishes none of the five.
They are assumptions about the sample, and they are the user's to make. Naming a
layer in the sample description ("copper oxide") is not making them.

One check can rule the case out; passing it proves nothing. With a solid whose SLD
is below the fluid's, a region made of that solid and fluid can never go lower
than the solid's own SLD. If the profile minimum — or, for a layer that passes the
`interface-swallowed-layers` criterion, its fitted `rho` — is clearly **below
`rho_solid`**, something of lower SLD is present (voids, hydrogen-bearing
material, a hydroxide), condition 1 or 3 has failed, and no conversion is
possible. For Cu2O the line is 5.363: a "copper oxide" layer sitting below that in
a deuterated fluid is already outside the case.

Do not make the conversion on your own initiative. If the user has asked for it
and has stated the composition, give it in this form only: name the assumed solid
and its assumed SLD, say that the data do not test the assumption, and give the
sensitivity. An error of 0.1 in the SLD assumed for Cu2O moves the result by 10 %
in D2O and 18 % in d8-THF, because the contrast it is divided by is only 1.0 and
0.56. Call it "the thickness of dense Cu2O that would give this deficit, **if**
...", never "the oxide thickness" or "the amount of oxide". It is never the
headline number.

`Gamma / (rho_fluid - rho_layer)` with the **fitted** `rho_layer` is not this
quantity. For a slab it is the fitted thickness again, ridge and all.

## The contrast precondition — what makes the integral exist

The dip is a bounded region only if the **fluid SLD lies between the layer's and
the substrate's**: the profile drops below the fluid in the layer and climbs back
above it into the substrate, which closes the integral. Bulk SLDs put Cu2O on Cu
in that arrangement in both D2O and d8-THF. CuO is above both fluids and makes no
dip at all. Check it for your case; do not assume it.

If the contrast does not bracket the layer this way there is no bounded dip and
these descriptors do not apply. If it does, the integral is *defined*. Nothing
about the bracketing makes it an amount.

## Reading the combination

Report depth, `Gamma` and width together, as a description of how the profile
changed. "The dip became shallower and wider and its integrated deficit fell" is
a complete and defensible sentence.

From the slab thickness alone, that change reads as growth. From `Gamma` alone it
is tempting to read it as consumption. Neither is established. Material leaving,
solvent flooding pores while the interface roughens, and conversion to a phase
nearer the fluid's SLD each produce a shallower, wider dip with less integrated
deficit.

## What it would take to say more

An amount needs information a single-contrast fit does not hold: the same state
measured in a second fluid contrast and co-refined (which separates the volume
that excludes fluid from its mean SLD), X-ray reflectivity, or an independent
measure such as charge passed or a microbalance. If the user has one of these,
say what it would add. If not, `Gamma` stays a description of the profile.

## How to word it

Write:

- "The integrated SLD deficit of the interfacial dip is `Gamma_slab` = 22 (1e-6 /
  Angstrom; slab product against the fluid, from best-fit values, no uncertainty
  available)."
- "Between the two states the integrated SLD deficit fell by half. The
  measurement does not determine why: loss of material, solvent entering pores,
  a change of composition and H/D exchange would each do this."
- Only with a composition the user stated: "If the region is only dense Cu2O (SLD
  5.36, assumed, not fitted) and bulk solvent — which these data do not test —
  the deficit corresponds to about 30 Angstrom of Cu2O; an error of 0.1 in that
  SLD changes this by 14 %."

Do not write, on the strength of `Gamma` or of a thickness: "the amount of oxide",
"oxide coverage", "surface excess", "adsorbed amount", "N Angstrom of oxide"
without the *if*, or "the oxide grew / thinned / was consumed / dissolved /
densified".

Proposing growth, dissolution or solvent ingress as a **hypothesis to test** is
fine, and a change in `Gamma` is a good reason to propose one. Reporting it as a
finding is what `Gamma` cannot support.

## Common rationalizations

- *"Chi-squared is good and nothing is at a bound."* Both are true on the ridge.
  Neither is evidence about position along it.
- *"The credible interval on thickness is tight."* It is tight **across** the ridge
  and unconstrained **along** it.
- *"I will quote the thickness and note the correlation."* The direction of the
  change may be wrong. A caveat does not repair a reversed conclusion.
- *"Gamma is robust to porosity and swelling."* Only when every pore holds fluid
  at the bulk SLD. A closed void or a pore holding another liquid outweighs the
  oxide around it six to twelve times over.
- *"Gamma fell, so material was lost."* Or pores flooded, or a phase converted, or
  hydrogen exchanged. The measurement does not choose.
- *"Gamma did not change, so nothing happened."* Gamma sums components of
  different weight and sign. Changes can cancel.
- *"An equivalent thickness is easier to read than Gamma."* It is also an
  assumption presented as a measurement. The same `Gamma` reads as 30, 12 or 6
  Angstrom depending on what the layer is taken to be.

## Red flags

- `t` and `rho` moving in the same direction across states.
- A growth or consumption claim resting on slab thickness — or on `Gamma`.
- `Gamma`, or a thickness derived from it, reported as an amount, coverage or
  mass of a named material without a composition the user supplied.
- An equivalent thickness computed with the fitted `rho_layer`, or with a
  handbook SLD nobody stated as an assumption.
- A two-component conversion alongside a profile minimum below the solid's SLD.
- `Gamma` values compared across different fluids or reference media, or slab
  form against profile form.
- A significance quoted for a `Gamma` difference built from best-fit values.
- Ridge-invariant quantities computed from posterior medians rather than per-draw.
