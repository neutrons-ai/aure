---
name: structural-hypothesis-ranking
description: >
  Meta-skill for generating and acting on a ranked list of candidate structural
  changes to a reflectivity model. Always active. Teaches the agent to enumerate
  plausible structural modifications (adding, removing, splitting, or reshaping
  layers) at intake time, drawing on the other active domain skills, and to walk
  that list in rank order when parameter-only refinement stops making progress.
metadata:
  author: aure
  version: "1.0"
---

## Purpose

Reflectivity fits can stall for two very different reasons:

1. **Parameter reasons** — bounds too tight, starting point bad, a fitted value
   has drifted to an unphysical region, a segment needs `sample_broadening`.
2. **Structural reasons** — the model is missing a layer, has an extra one, or
   a layer has the wrong position (e.g. a native oxide was placed on the
   substrate instead of on top of a metal exposed to solvent).

Parameter reasons can almost always be repaired by changing bounds or starting
values. Structural reasons cannot — no amount of bound widening will add a
copper oxide to a model that doesn't have one.

The classic failure mode is: "the fit is poor, try wider bounds; the fit is
still poor, try a different parameter; try enabling `sample_broadening`;
try a different initial value; …" — spending every iteration on parameter
tweaks while the actual gap is structural. This skill exists to prevent that.

## Workflow Role

This skill is used in two places:

- **At intake**, when the sample is first parsed and the other domain skills
  are active, produce a ranked list of *structural hypotheses* — plausible
  structural changes that would be worth trying if parameter tuning fails to
  reach the acceptance threshold. Each hypothesis is motivated by one of the
  other active skills. The user's stated hypothesis is folded in as one or
  more high-priority entries (`origin: "user"`) ranked at the top.
- **During evaluation and refinement**, consult that list. If the χ²/BIC
  trajectory shows that parameter tweaks are not making meaningful progress,
  propose the top unused hypothesis instead of another parameter change.
  Evaluation may also propose *new* hypotheses and re-rank the list when fit
  evidence (residual fringes, pinned parameters, unexpected contrast) warrants
  it — re-selecting domain skills from that evidence first.

## Producing the Ranked List (at intake)

Given the parsed sample (substrate, layers, ambient) and the bodies of the
active domain skills, enumerate structural changes that are **plausible for
this specific sample** and rank them by expected value.

For each hypothesis, provide:

- **title** — one short line, e.g. "Add native CuO on top of Cu".
- **rationale** — a sentence grounded in an active skill. Cite the skill by
  name (e.g. *"metal-oxide-interfaces says a Cu layer exposed to D₂O forms a
  10–50 Å native oxide unless otherwise stated"*).
- **change** — the concrete structural edit in neutral terms: insertion
  point, typical thickness, typical SLD, typical roughness. E.g. *"insert a
  10–50 Å CuO layer (SLD 4.5–5.5) between Cu and D₂O, σ 3–15 Å"*.
- **skill_source** — the name of the skill motivating this hypothesis.

### Ranking criteria

Rank hypotheses by the combination of:

1. **Prior probability given the sample** — native oxides on exposed metals
   in aqueous ambients are almost certain; speculative extra interfacial
   layers are not.
2. **Size of effect on the fit** — a 20 Å CuO with SLD 5.0 in D₂O (SLD 6.3)
   produces a large low-Q contrast step and will move χ² substantially.
3. **Cost in parameters (BIC)** — prefer hypotheses that add fewer free
   parameters for the same effect. Splitting one layer into two sublayers is
   lower-yield than adding a genuinely missing outer layer.
4. **Irreversibility** — prefer hypotheses that are easy to evaluate and
   easy to revert (adding an outer oxide, changing a layer's nominal SLD).
   Avoid re-ordering the whole stack as an early hypothesis.

### Baseline structural hypotheses that are nearly always worth listing

When the active skills apply, include at minimum:

- **Exposed metal + aqueous/air ambient, no surface layer in model** → add
  the native oxide (skill: `metal-oxide-interfaces`). High rank.
- **Mixed probe types across segments in co-refinement** → this is not a
  structural hypothesis but a data-loading constraint; flag it as a warning
  rather than a hypothesis.
- **User-stated "hypothesized" / "expected" / "likely to form" layers that
  are not yet in the model** → add them. High rank.
- **Additional layer missing when residuals show unmodeled fringes of a
  plausible thickness** → flag as a deferred hypothesis that depends on
  residual analysis after the first fit.
- **Liquid ambient with unspecified isotope** → reinterpret the ambient as a
  deuterated solvent (skill: `solvent-contrast-matching`). High rank whenever a
  critical edge / strong low-Q feature appears, or the fit only matches it by
  inflating a layer thickness or pinning a metal SLD toward the ambient. This
  is a *reinterpretation* hypothesis (see below), not an added layer.

### Hypotheses to avoid unless justified

- Native SiO₂ on the Si substrate just because the substrate is Si. It is
  typically only 10–20 Å and adds three parameters that often absorb signal
  from more important layers. Only include when the sample description
  explicitly asks for it.
- Splitting an existing single oxide into two sublayers (CuO + Cu₂O).
- Any change that reverses the back-reflection geometry or the layer
  stacking order decided at intake.

### Reinterpretation hypotheses (and rewinding the model)

Not every hypothesis adds or removes a layer. A **reinterpretation** changes
what an *existing* material is — most commonly "the ambient solvent is actually
deuterated" (its SLD jumps from ≈ 0 to ≈ 6) — or re-labels a layer's nominal
SLD. Phrase the `change` field as the concrete SLD edit (new value + a wide
range), with `skill_source` naming the domain skill (e.g.
`solvent-contrast-matching`).

A reinterpretation is often **mutually exclusive** with an additive hypothesis
that was already tried to explain the *same* feature. For example, a critical
edge can be explained EITHER by a deuterated ambient OR by a thick, high-SLD
layer near the substrate — not both. So when you realize a reinterpretation
hypothesis:

- **Rewind to the intake baseline model.** Start from the clean structure first
  built at intake, discarding the speculative layers and inflated thicknesses/
  SLDs that earlier (now mutually-exclusive) hypotheses accumulated. Then apply
  ONLY the reinterpretation. Stacking both explanations over-parameterizes the
  fit and double-counts the same contrast.
- **The right reinterpretation wins on the data, not just parsimony.** When
  correct, it fits better (lower χ²) *and* with fewer parameters (lower BIC)
  than the structural workaround it replaces. The χ²/BIC guardrails keep it
  only if the refit actually improves; a single worse refit is auto-reverted —
  that is expected, not a reason to re-propose the same reinterpretation.

## Consuming the Ranked List (during refinement)

Each hypothesis has a `status`: `pending`, `tried`, `confirmed`, or
`rejected`. The evaluator and the refiner see the full list along with the
χ² trajectory and the BIC trajectory.

### When to propose a structural change vs a parameter change

Look at the fit-history trajectory, not any single number:

- If χ² **just improved meaningfully** on the previous step, the current
  direction is working — keep doing parameter refinement.
- If χ² **has not improved meaningfully for two or more iterations**, or
  has oscillated (better → worse → better → worse), parameter tweaks are
  exhausted. Propose the top `pending` hypothesis.
- If BIC regressed after a structural change in the previous iteration,
  that change has been auto-reverted. Mark that hypothesis `rejected` and
  move to the next `pending` one. Do **not** re-propose the rejected
  hypothesis with different parameters.
- If a parameter is still pinned at a bound *and* the bounds have already
  been auto-expanded once without improvement, this is usually evidence of
  a structural gap, not a bounds problem. Propose a hypothesis.

"Meaningful improvement" is a judgement — typical values to consider:

- Improvement in χ² < 5% over two consecutive iterations is not meaningful.
- χ² oscillating within ±10% of a baseline is stagnation.
- A single iteration that gets worse is not itself stagnation — but two
  worse iterations in a row are.

These values are guidelines, not thresholds: when in doubt, prefer trying a
structural hypothesis over another parameter tweak, because structural
hypotheses were ranked by expected value in the first place.

### Updating the list

When proposing a hypothesis, mark it `tried` with the current iteration
number. If after the next fit χ² improves and BIC does not regress, mark
`confirmed`. If BIC regressed and the guardrail reverted the change,
mark `rejected` with a short note about why.

Membership and identity are guarded by the workflow, so the rules differ by
node:

- **Modeling (status-only).** When realizing a hypothesis, return the list
  with that hypothesis marked `tried`; you may change only `status`,
  `tried_in_iteration`, and `notes` of existing entries. Any entry you add,
  drop, or rename is discarded by the merge guard — modeling cannot grow the
  backlog.
- **Evaluation (revision).** Only the evaluation-time revision step may add
  new hypotheses and re-rank the list (see below). `id`s are stable —
  re-ranking changes list order, never ids — and `rejected` hypotheses are
  never resurrected. Provenance is recorded in `origin`: `"user"` (seeded
  from the user's hypothesis, top rank), `"skill"` (enumerated at intake),
  `"evaluation"` (proposed mid-run from fit evidence).

### When to add a new hypothesis mid-run (evaluation only)

New hypotheses are proposed by the **evaluation** node once data reveals
something not visible at intake — residual fringes of a characteristic
thickness, a parameter pinned at a bound, or an artifact pointing to a
phenomenon whose skill was not obvious from the static description. At that
point the evaluator re-selects skills from the observed evidence (so a skill
like `sei-layer-analysis` can activate mid-run) and proposes new
`status: "pending"` hypotheses describing the missing structure — thickness
from the residual analysis, SLD and roughness from the now-relevant domain
skill — then re-ranks the whole list by current expected value. Propose only
genuinely new ideas; do not duplicate an existing entry.
