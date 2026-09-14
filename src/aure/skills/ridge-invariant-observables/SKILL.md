---
name: ridge-invariant-observables
description: >
  What to quote instead of a slab thickness when a thin layer sits on the
  contrast-thickness ridge. thin-layer-degeneracy explains why the ridge makes a
  layer multimodal and how to escape it during fitting; this skill covers the
  reporting problem that survives a good fit — across states, a fitted thickness
  can move along the ridge and report the WRONG SIGN for a change. Defines the
  ridge-invariant quantities (depth, Gamma, t_eq, sub-fluid width), why Gamma is
  robust to porosity and swelling, and the contrast precondition they require.
  Consult whenever a conclusion rests on a layer's thickness changing between
  states, potentials or times, or before reporting growth, consumption or
  densification.
metadata:
  author: aure
  version: "1.0"
---

## When this applies

Any time a **conclusion rests on a fitted `thickness` changing** across states,
potentials or time — and any time you are about to say a layer grew, was
consumed, or densified.

It applies even when the fit is good. This failure survives a healthy chi-squared,
a converged chain, and no parameter at a bound.

## The reporting failure

In a single contrast a thin layer's `thickness` and `rho` are correlated at
r >= 0.9. Only their **product** is determined; the position along the ridge is
not measured. So a fit can report

    t:    42.4  ->  55.7  ->  57.9  Angstrom
    rho:  5.060 ->  5.380 ->  5.576

and this is **not** a layer that grew and densified. It is movement along the
ridge. The same three states, same fit, same chain, give an areal deficit
**falling by half at 12.9 sigma**.

Reading the thickness alone got the sign of the change backwards.

## Quote these instead

All four are formed **inside each posterior draw**, so the t/rho correlation
cancels. Computing them from marginal medians reintroduces the error you are
trying to avoid.

| quantity | evaluated over | answers |
|---|---|---|
| `depth = rho_fluid - min(rho(z))` | the surface region | is the layer there at all? |
| `Gamma = integral (rho_fluid - rho(z)) dz` | the dip | how much of it? |
| `t_eq = Gamma / (rho_fluid - rho_layer)` | | the same, in thickness-of-dense-material units |
| width of the sub-fluid region | | is it spreading rather than growing? |

## Gamma is one quantity with two estimators

`Gamma` is the **integrated scattering-length deficit** of the region. It already
appears in the reflectometry skills as the slab form

    Gamma = t * delta_rho          delta_rho = rho_layer - rho_of_the_medium_displaced

and this skill uses the profile form

    Gamma = integral (rho_fluid - rho(z)) dz

**These are the same quantity.** The slab form is the profile form evaluated on an
ideal slab, and the two agree whenever the layer is actually resolved in the
profile. They diverge exactly when it is not — when the bounding interfaces
overlap, the fitted `rho_layer` is never attained anywhere, so `delta_rho` taken
from it is not the contrast the profile contains.

**Which estimator to use is decided by the criterion in
`interface-swallowed-layers`.** If `sigma_top + sigma_bot <= t/2` holds, use the
slab form; it is cheaper and its uncertainty is easier to propagate. If it fails,
the slab form is reporting a contrast that does not exist and you must integrate
the profile.

## Why Gamma is the robust one

For a layer that is a mixture of solid and fluid with solid fraction `f(z)`,

    rho_fluid - rho(z) = f(z) * (rho_fluid - rho_solid)

so `Gamma` is proportional to the **areal amount of solid** and is independent of
how porous or swollen the region is. That is the property a slab thickness lacks.

## The precondition — check it, do not assume it

These quantities require the **fluid SLD to lie between the layer's and the
substrate's**. Verified to hold for Cu/Cu2O in both D2O and d8-THF. If the
contrast does not bracket the layer this way, the integral is not interpretable
as an areal amount and the approach does not apply.

## Reading the combination

Falling `t_eq` together with rising sub-fluid width is a layer **being consumed
and spreading**. From the slab thickness alone that same behaviour reads as
growth. Always report the pair.

## Common rationalizations

- *"Chi-squared is good and nothing is at a bound."* Both are true on the ridge.
  Neither is evidence about position along it.
- *"The credible interval on thickness is tight."* It is tight **across** the ridge
  and unconstrained **along** it.
- *"I will quote the thickness and note the correlation."* The sign may be wrong.
  A caveat does not repair a reversed conclusion.

## Red flags

- `t` and `rho` moving in the same direction across states.
- A growth or consumption claim resting on slab thickness with no areal statistic.
- Ridge-invariant quantities computed from posterior medians rather than per-draw.
