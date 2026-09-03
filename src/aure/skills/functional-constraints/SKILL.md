---
name: functional-constraints
description: >
  Opt-in meta-skill on reparametrization — fitting a functional
  COMBINATION of parameters (a surface excess, a solvated layer's volume
  fraction) instead of the raw parameters, via a model's `derived_parameters`
  block. Explains when a combination is the quantity actually determined,
  the two canonical forms and how to write one, why a tie between layer
  attributes cannot express contrast invariance, and how to behave when a
  model already carries such a block. Activated only when reparametrization is
  enabled for the run (`allow_derived_parameters` / ALLOW_DERIVED_PARAMETERS,
  off by default). Consult whenever a thin layer's SLD and thickness trade off
  against each other, when a per-state SLD is pinned at a bound in a contrast
  series, or when an independent measurement gives a combination rather than a
  coordinate.
metadata:
  author: aure
  version: "1.0"
---

## The idea

Reflectivity does not determine the parameters a model is written in. It
determines certain **combinations** of them. For a layer thinner than the
resolution limit, `(rho_layer - rho_ambient) * thickness` is pinned tightly
while the SLD and the thickness separately are not — the likelihood is a long
narrow ridge (this is the contrast-thickness degeneracy described in the
`thin-layer-degeneracy` skill).

Independent knowledge has the same shape. QCM-D gives an adsorbed amount, not
an SLD. A known bulk density plus a swelling measurement gives a volume
fraction, not a thickness. Bounding `SEI.rho` when what is actually known is
the excess is both too weak — every SLD inside the box is equally acceptable —
and too strong, since just outside it is impossible.

A **reparametrization** fits the combination and derives the raw parameter from
it. The ridge disappears from the geometry the optimizer explores, and an
ordinary range on the combination means what it says.

## When it is warranted

Reach for one when:

- A **thin layer's** SLD and thickness trade off against each other — the fit
  moves along the ridge between iterations, or the mode-enumeration step
  reports several basins of similar quality.
- An **independent measurement** constrains a combination: an adsorbed amount /
  surface excess, a coverage, a swelling ratio, a known dry density.
- In a **contrast series**, a layer SLD is pinned at a bound, or untying that
  SLD across contrasts leaves the states effectively uncoupled (see below —
  this is the case where a reparametrization is not merely better but is the
  only correct statement of the physics).

Do **not** reach for one:

- To force a parameter to a value you prefer. A reparametrization changes the
  coordinates, not the evidence; a range on the combination has to be justified
  by something outside this dataset.
- When the raw parameters are individually well determined. It buys nothing and
  makes the model harder to read.
- As a substitute for a layer that is genuinely missing. Fix the structure
  first; reparametrize the layer afterwards if the ridge is still there.

## The two canonical forms

**Surface excess** — an adsorbed layer whose own material does not change:

```yaml
derived_parameters:
  - name: Gamma_SEI
    free: {init: -420, min: -600, max: -250}
    assign:
      SEI.rho: "dTHF.rho + Gamma_SEI / SEI.thickness"
    keep_physical: ["SEI.rho > -1.0", "SEI.rho < 6.4"]
    source: "QCM-D, 1.2 mg/m^2"
```

**Solvation** — a layer the solvent penetrates:

```yaml
derived_parameters:
  - name: phi                                # volume fraction
    free: {init: 0.30, min: 0.05, max: 1.0}
    assign:
      film.rho: "phi * rho_dry + (1 - phi) * solvent.rho"
    keep_physical: ["film.rho > -1.0", "film.rho < 6.5"]
  - name: rho_dry                            # auxiliary: free, used above
    free: {init: 2.0, min: 0.5, max: 4.0}
```

Rules for writing one:

- **Write the inverse yourself.** Nothing solves the relation for you. Say
  which raw parameter becomes dependent and what it equals.
- `free.min` / `free.max` are required. A derived parameter has no bounds of
  its own to fall back on.
- `keep_physical` guards the DERIVED value (`SEI.rho > -1.0`), because nothing
  else does. Single comparisons only; write two entries rather than chaining.
- Expressions take numbers, `<layer>.<attr>` names (`thickness`, `interface`,
  `rho`; `substrate` and `ambient` are aliases), unary minus and `+ - * / **`.
  Nothing else — no function calls.
- Record where the number came from in `source`.

## Contrast variation: the case a tie cannot express

**`(rho - rho_solvent) * t` is contrast-invariant only for a non-exchanging
adsorbate whose own SLD does not change with the solvent.** For a layer the
solvent penetrates it is not: with `rho_layer = phi*rho_dry + (1-phi)*rho_solvent`,
the difference is `phi * (rho_dry - rho_solvent)`, which depends on the
contrast. Getting this backwards yields a model that fits one contrast and
fights the other.

For a solvated film the invariants are `phi` and `rho_dry` — and **neither is a
layer attribute**, so `shared_parameters` cannot name them. Tying
`film.material.rho` across contrasts asserts something false (the SLD differs
in every contrast); untying it discards the coupling that made the contrast
series worth measuring. The solvation form above is the correct statement: it
is tied across states by default, one `phi` and one `rho_dry` for the whole
problem, with each state's `film.rho` computed from ITS own solvent.

## If the model already has a `derived_parameters` block

- The layer attributes named in `assign` are **not fit parameters**. Their
  values in `layers` are computed. A derived SLD "not moving" is correct
  behaviour, not a stuck fit — do not widen its bounds to free it.
- **Do not remove or rename a layer that an entry references.** If a structural
  change requires it, the workflow drops the affected declaration and records
  that it did — you lose the constraint. Prefer a different edit, and explain.
- The free parameter you see in the fit results (`Gamma_SEI`, `phi`) IS the
  quantity of interest. Report and reason about it directly; it is more
  meaningful than the SLD derived from it.

## Proposing one

Declarations come from the scientist's setup file — the workflow does not
install one on its own, and a block emitted by a model is discarded. When the
conditions above are met, **say so in your issues / suggestions** so the
recommendation reaches the person running the analysis. A useful recommendation
names three things: the combination to fit, which raw parameter should become
dependent, and the evidence that the raw parameters are trading off. For
example:

> The SEI thickness and SLD are anti-correlated across iterations (t 22→41 Å
> while rho 3.1→1.6, chi2 unchanged) — the fit is moving along the
> contrast-thickness ridge. Consider reparametrizing on the surface excess:
> make `(SEI.rho - dTHF.rho) * SEI.thickness` the free parameter and derive
> `SEI.rho` from it, which would also let the QCM-D coverage be used as a
> range.

Full reference, including how to declare one in YAML:
`docs/derived-parameters.md`.
