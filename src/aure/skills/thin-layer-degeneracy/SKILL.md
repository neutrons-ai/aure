---
name: thin-layer-degeneracy
description: >
  Always-active meta-skill on the reliability of thin-layer fits and model
  selection. Explains why layers thinner than the resolution limit are
  multimodal (the SLD x thickness "contrast-thickness" ridge), why a single
  optimizer run or a BIC comparison can silently pick a local minimum and
  wrongly reject a real layer, how to escape with discrete SLD mode
  enumeration, and how to use a cleaner sibling or earlier measurement as a
  prior. Consult whenever the model contains a thin layer, when a
  physically-expected layer appears "not needed", or when BIC is used to accept
  or reject a structural change.
metadata:
  author: aure
  version: "1.0"
---

## When this applies

Any time the model has a **thin layer** — thinner than roughly the real-space
resolution limit `2*pi / Q_max` (for a typical Q_max ~ 0.2 1/Å that is about
30 Å) — or any time a **BIC / chi-squared comparison is being used to accept or
reject a structural change**. Both situations are where automated fitting most
often goes quietly wrong.

## Why thin layers are multimodal (the contrast-thickness ridge)

Reflectivity constrains a thin layer mainly through the **product of its
contrast and its thickness** (Δρ · t), not through ρ and t separately. Below
the resolution limit, many (ρ, t) pairs on a curve of constant Δρ · t fit
almost equally well. Consequences:

- The layer's SLD and thickness are individually poorly determined even when
  their product is well determined — expect large, correlated uncertainties.
- The likelihood surface has **several distinct local minima** (e.g. a thin
  dense layer vs. a thicker dilute one), separated by barriers that local
  optimizers and even global optimizers at modest effort **will not cross**.
- Two fits with different (ρ, t) but the same Δρ · t can have essentially the
  same χ². The one the optimizer happens to land in is not necessarily the
  physically correct one.

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

The reliable, cheap way to find the right basin for a thin layer is to
**enumerate a few discrete SLD seeds and polish each**:

1. Pick ~3 SLD seed values for the thin layer spanning its physical range —
   e.g. a dark/low value, a mid value, and a bright/high value.
2. For each seed, hold everything else at the current best values and run a
   short local fit.
3. Keep the basin with the lowest χ², then continue the normal fit/refinement
   from there.

This routinely finds a better minimum than a single global-optimizer run at
the same cost. Note what does **not** work: seeding the whole stack from a
sibling measurement's converged structure — the optimizer can still slide into
the wrong SLD mode of the thin layer. Enumerate the mode explicitly.

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
- Report the series as a physically coherent story (what grew, what stayed
  fixed), not as independent fits that happen to disagree.

## Interaction with the hypothesis list

This skill sharpens how to walk the `structural_hypotheses` list: when a
pending hypothesis adds a *thin* layer, and a first attempt regresses BIC,
check for the local-minimum signatures above and retry with mode enumeration
before marking the hypothesis `rejected`. A thin-layer hypothesis rejected
without a mode-enumerated attempt is not yet a real rejection.
