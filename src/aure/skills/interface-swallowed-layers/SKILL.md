---
name: interface-swallowed-layers
description: >
  A quantitative test of whether a slab exists in the SLD profile at all. When a
  layer's two bounding interfaces are wide relative to its thickness, their error
  functions overlap and the layer's nominal SLD is never reached anywhere in the
  profile — yet the fitted rho is still reported with a credible interval and
  still improves chi-squared. Gives the criterion sigma_top + sigma_bot <= t/2
  (equivalently t >= 4 sigma_bar), which is TWICE as strict as the conventional
  "roughness below half the layer thickness" rule, plus the deviation formula to
  compute per case. Consult before trusting any fitted SLD for a thin layer, and
  whenever an interfacial width was left at a scaffolded default. It governs which
  numbers may be quoted, not whether a fit is acceptable.
metadata:
  author: aure
  version: "1.1"
---

## When this applies

Before quoting a **fitted SLD** for any thin layer — and urgently whenever an
interfacial width was never declared as a parameter and sat at a scaffolded
default.

The failure is silent: it survives a good chi-squared, a converged chain, and no
parameter at a bound. In the worked case below it invalidated several rounds of
an analysis, because one interface width had never been fitted. When the widths
are fitted parameters the data determine, failing the criterion is a limit on
what can be reported, not a fault in the fit (see the last section).

## The failure

In a slab model each interface is an error function of width sigma. When the two
bounding interfaces are wide relative to the layer thickness, their profiles
overlap and the layer's nominal SLD is **never attained anywhere in the profile**.
The fitted `rho` is then not a material SLD — but nothing in the fit says so.

A worked case, a two-state co-refinement where the fluid/oxide interface width had
never been declared and sat at a default of 20 Angstrom:

| state | CuOx t | sigma_top | sigma_bot | sum vs t | nominal rho attained? |
|---|---|---|---|---|---|
| 218386 | 21.3 | 20.0 (unfitted) | 13.0 | 33.0 — 1.55x | **No** (min 4.997 vs nominal 4.005) |
| 218393 | 48.3 | 20.0 (unfitted) | 14.1 | 34.1 — ok | Yes |

That single unexamined number destroyed 218386's oxide. It is also why that
layer's SLD pinned at its lower bound: **a layer with no distinct region in the
profile has nothing holding its SLD anywhere.** A parameter on a bound is a
symptom here, not the disease.

## The criterion

For a slab of thickness *t* between media of SLD `rho_amb` and `rho_sub`, with erf
interfaces of mean width `sigma_bar = (sigma_top + sigma_bot)/2`, the profile's
extremum misses the layer's nominal `rho_lay` by

    dev = Phi( -t / (2 sigma_bar) ) * [ (rho_amb - rho_lay) + (rho_sub - rho_lay) ]

where `Phi` is the standard normal CDF. For a layer sitting 1–1.5 below both
neighbours (bracket ~ 2.5):

| t / (sigma_top + sigma_bot) | dev |
|---|---|
| 0.50 | 0.84 |
| 1.00 | 0.43 |
| 1.50 | 0.18 |
| **2.00** | **0.06** |
| 2.50 | 0.02 |

The usable rule is therefore

    sigma_top + sigma_bot <= t / 2          equivalently   t >= 4 * sigma_bar

**This is twice as strict as the conventional "roughness below half the layer
thickness".** That convention is the ratio 1.00 row, which still leaves a
deviation of 0.43 — it does not guarantee the layer exists in the profile.

The tolerable deviation scales with the SLD contrast, so **compute `dev` for your
case** rather than reusing the table.

Validated against two fitted states from the experiment it was derived on:
predicted 0.229 against 0.256 observed, and 0.045 against 0.048.

**Validated again on an independent case from a different project.** A worked
example in the reflectometry `thin-layer-degeneracy` material — an 11.1 Angstrom
oxide at nominal rho 4.10 between sigma 18.9 and 8.9, with D2O at 6.10 and Cu at
6.55 — reports the profile reaching only 5.59. The formula predicts a deviation of
**1.53**; the observed deviation is **1.49**, agreeing to 3%. That case has
`t/(sigma_top + sigma_bot) = 0.40`, below the 0.50 the criterion requires, so it is
correctly flagged in advance rather than after reading the profile.

## The consequence worth remembering

With a 5 Angstrom physical floor on interfacial width, **a layer thinner than 20
Angstrom cannot be a resolved slab at all**, since two 5 Angstrom interfaces
already require t >= 2*(5+5).

## What to do instead

If the criterion fails, do not quote the layer's `rho`. Either report the profile
and the ridge-invariant quantities (see `ridge-invariant-observables`), or
reparameterise so the quantity you report is one the data constrain.

## Red flags

- Any interfacial width still at a scaffolded default in a published fit.
- A layer SLD pinned at a bound — check the criterion before widening the bound.
- A layer thinner than ~20 Angstrom described as a slab with its own SLD.

## What the criterion does not decide

It decides one thing: whether the layer's fitted `rho` may be quoted. It does not
decide whether the fit is acceptable, in either direction.

- **Failing it is not, on its own, a reason to reject the fit**, nor to cap, tie
  or shrink widths the data have fitted, nor to remove, merge or replace the layer
  with a roughness. Report the profile's extremum or the layer's integrated
  contrast in place of `rho`. Most correct fits of thin native oxides fail it: in
  the expert reference fits of a 51-curve copper-in-THF series, the layer directly
  on the copper fails it in 40 of 50. On one pristine film, capping the oxide's
  two interfaces until it passed took χ² from 1.08 to 1.56 and moved the oxide
  from 63 Å at 4.57 to 77 Å at 5.12.
- **Passing it is not a reason to accept the fit.** It says the slab is resolved
  in the profile, not that the slab is the material it is named. Judge that
  separately, from the layer's thickness and SLD against the material it is meant
  to be: a 272 Å "copper oxide" at SLD 2.0 passes easily and is not a copper
  oxide.
