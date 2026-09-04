# TODO

Known issues with a decided remedy, recorded rather than fixed. Each entry says
what is wrong, what it costs, and what the change would be.

---

## Stop hardcoding a 5 Å roughness floor in `_build_layers`

**Where:** [`src/aure/nodes/modeling.py`](src/aure/nodes/modeling.py) —
`_build_layers` writes `"roughness_min": 5.0` into every layer it constructs
(both the described-layers branch and the feature-estimate branch).

**What is wrong.** The floor is asserted regardless of what the sample
description says and regardless of the roughness the intake parse chose. A
description that says, in plain English,

> This buried oxide interface is chemically sharp — its roughness is often well
> under 5 Å, so do not impose a roughness floor on it.

produces a layer with `roughness: 3.0` **and** `roughness_min: 5.0`. Two
consequences follow:

1. The parameter is built at 3.0 with bounds (5, 30) — outside its own range.
   `_ranged` (in `model_builder`) now clamps it to 5.0 and logs, so the fit no
   longer starts infeasible, but the value the description asked for is
   discarded.
2. The floor binds for the whole fit. No untied layer interface can go below
   5 Å, so an expert value below that is outside the search space before the
   optimizer starts.

**What it cost.** Measured across the validation sweeps in
`aure-validation/results` (12 sweeps with `comparisons.csv`, 165 runs with
retained artifacts):

| | |
|---|---|
| runs whose first model started a layer below its own floor | 156 / 165 |
| roughness comparisons whose expert target is below 5 Å | 128 (SiO₂ 112, Ti 16) |
| of those, comparisons where the fit actually reached below 5 Å | 22 — **all** of them only because the SLD-profile remedy had applied `roughness_tie` to that layer, which bypasses the floor |
| median \|fitted − reference\| on blocked targets | 2.87 Å |
| median error forced by the floor alone | 1.80 Å |

More than half the typical error on those interfaces was arithmetically
unavoidable. It is invisible to the run's own verdict: χ² is unaffected, and 14
of the 17 near-floor cases in `20260819-103916` still scored `good`.

### Aside: how `roughness_tie` escapes the floor

Worth recording because it explains the 22 exceptions above, not because it is
a remedy.

The floor lives in the `else` branch of the interface handling in
`_build_sample`. A layer carrying `roughness_tie` never reaches it: its
interface is replaced by the expression `fraction × thickness`, and the free
parameter becomes `fraction`, ranged (0.05, 0.5) by default. `.range(r_min,
r_max)` is never called on that interface, so nothing floors it — the
achievable σ is whatever those fractions of the fitted thickness happen to be.

In `cu_film/Cu_0/201179` (`20260819-123006`) the SiO₂ layer fitted to 17.5 Å
thick with `roughness_tie: {fraction_max: 0.5}`, which puts σ anywhere in
0.88–8.76 Å. It landed at 4.15 Å (fraction 0.24) against a reference of 1.9 Å —
still not right, but inside the range at all, which the untied layers were not.

So a guardrail aimed at something else entirely — erf-tail profile artifacts on
thin layers — is the only thing in the system that ever let a sharp buried
interface be fitted as sharp. It reached the right region for the wrong reason,
and only in the 74 of 165 runs where the profile detector happened to fire on
that layer.

It is **not** a fix, for three reasons: it applies only when the artifact
detector fires, so it cannot be relied on; it couples σ to the layer thickness,
which is physically wrong here (how sharp a buried oxide interface is has
nothing to do with how thick the oxide is); and it caps σ at half the
thickness, trading one arbitrary bound for another. The interesting part is
that the floor was invisible for so long partly *because* this accident kept
producing plausible numbers on the cases where it fired.

**The change.** Omit `roughness_min` from `_build_layers` entirely and let the
builder default apply. The default in `_build_sample` now yields to a declared
`roughness` (a default that overrides what the model states is not a default),
so dropping the hardcode restores the description-driven value while keeping
the 5 Å floor everywhere the roughness is not explicitly small. Writing
`min(5.0, roughness)` instead would also work but leaves the policy in two
places.

**Two related gaps**, both of which this change makes moot for ordinary runs
but which remain if a floor is ever wanted deliberately:

- `roughness_min` is absent from the model-JSON schema in the refinement
  prompt, so no refinement iteration can lower a floor either.
- It is reachable from a setup file only inside `states[].layers[]`. There is
  no top-level `layers:` key, so a description-driven single-file run —
  `aure analyze DATA "description"`, which is every case in the sweeps — has no
  config surface on which to set it. The codebase has an env override for the
  outer roughness *ceiling* (`ROUGHNESS_MAX_OUTER`) and nothing for the floor.

**Verifying a fix.** Re-run any case from `20260819-103916`; the SiO₂ interface
should be free to move below 5 Å and land near the reference (median 3.4 Å)
instead of pinning at the bound.

---

## Tie a shared layer to the first state that *has* it, not always to state 0

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py) —
the cross-state aliasing loop in `build_states_problem`, which reads
`ref_def = effective_defs[0]` / `ref_sample = samples[0]` unconditionally and
`continue`s when `ref_idx is None`.

**What is wrong.** Aliasing is always to state 0. When a layer is absent from
state 0 but present in two or more later states, `ref_idx` is `None`, the tie
is skipped, and it is skipped for *every* state — so layers that should be
tied to each other are fit independently. The docstring a few lines above
(`_resolve_tied_set`, "A layer present in >=2 states is tied across them")
asserts the opposite of what the code does.

A second defect rides along. The renaming pass keys off the tie *set*, not off
what was actually aliased, so both untied copies keep the tied spelling
(`"<layer> <attr>"`, no state prefix) and collide by name. This is the same
hazard `untied_by_derivation` was introduced to prevent for reparametrized
slots; the layer-absent branch never records into it.

Reproduced with three states, an oxide absent from state 0 (which inherits the
template) and present in S1 and S2:

```
oxide rho tied S1/S2: False
n param objects: 13   n unique names: 10
duplicates: [('Cu oxide interface', 2), ('Cu oxide rho', 2), ('Cu oxide thickness', 2)]
```

Three parameters are silently double-named. Anything keyed by parameter name —
the fitted-parameter dict, `apply_parameters`, the Results-page editor, the
per-state lookups in `evaluation` — sees one of the two objects and cannot
tell which. Reachable whenever the refiner scopes a new layer to a subset of
states that excludes the first one, which prompt rule 12 explicitly invites.

**The change.** Two parts, both inside the aliasing loop:

- Resolve the reference per `(layer_name, attr_path)` pair rather than once per
  problem: the first state whose `_layer_index` is not `None` *and* whose slot
  is not derived. States after it alias to that state; states before it (which
  do not have the layer) are untouched, as now.
- Record every pair that was not aliased for a state — layer-absent as well as
  derived — into the set the renaming pass consults, so an untied copy gets its
  `"<state> "` prefix. `untied_by_derivation` becomes `untied_in_fact` (or
  similar) and the layer-absent `continue` writes into it.

Note the cheaper-looking fix (keep state 0 as reference, just prefix the
untied copies) fixes the collision and leaves the missing tie in place. The
missing tie is the substantive half: it silently costs free parameters and BIC.

**Verifying a fix.** The reproduction above: `oxide rho tied S1/S2` should be
`True`, and the free-parameter count should equal the unique-name count for
every multi-state build. Worth a test in
[`tests/test_model_builder_states.py`](tests/test_model_builder_states.py)
alongside `test_per_state_structure_oxide_absent_in_one_state`, which covers
only the absent-from-a-*later*-state direction.

---

## A mid-run layer rename silently voids a config-pinned cross-state tie

**Where:** [`src/aure/nodes/modeling.py`](src/aure/nodes/modeling.py) — the
config-wins / `prune_tie_specs` sequence in `_refine_model`; and
[`src/aure/nodes/prompts.py`](src/aure/nodes/prompts.py) — the refinement
rules, where no rule protects a tie spec's layer name.

**What is wrong.** Tie specs match layer names by exact, case-sensitive string
comparison (`_layer_index`, `_valid_layer_names`); `canonical_name`
canonicalizes only the *attribute* (`rho`/`sld` -> `material.rho`), never the
layer. If a refinement iteration renames a referenced layer — `copper` ->
`Cu metal` while realizing a hypothesis — the user's pinned spec no longer
matches, `prune_tie_specs` drops it, and the constraint is gone. Config-wins
re-adds it from `user_config` on the next iteration and prune drops it again,
so it stays inert for the remainder of the run while looking, in the config,
like it is still in force.

Nothing forbids the rename. The one prompt rule that does — rule 14's "DO NOT
remove or rename any layer an entry references" — is emitted by
`_format_derived_parameters_rule` only for a model that carries a
`derived_parameters` block, so a run using `shared_parameters` alone never
sees it.

The drop does reach the run transcript, but worded "Dropped tie spec(s) for
**removed layer(s)**", which misattributes a rename as a removal and sends a
reader looking for a structural edit that never happened.

**The change.** Three parts, in increasing cost:

- Reword the transcript line and the `logger.info` to say which specs were
  dropped and that the cause is a layer name no longer present, without
  claiming the layer was removed.
- Extend the rename prohibition to tie specs: emit it (from its own small
  formatter, keyed off `shared_parameters`/`unshared_parameters` being
  non-empty) whenever the model carries a user tie set, so it ships for the
  runs that need it without adding prompt weight to the ones that do not.
- Optionally, distinguish a rename from a removal before pruning: if exactly
  one layer was renamed and the spec's old name matches nothing, follow it. This
  is guesswork and should probably not be built — the loud version of the first
  two is more honest.

**Verifying a fix.** A two-state refinement whose LLM response renames a layer
named in `shared_parameters`: the transcript must say the tie was dropped
because the name is gone, and the refinement prompt for a model with a user
tie set must contain the rename prohibition.

---

## Tie specs are validated against the union of states but applied per state

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py)
— `_valid_layer_names` (validation) versus `_layer_index` called on each
state's *effective* definition (application).

**What is wrong.** `_valid_layer_names` unions the top-level template with
every state's own stack, so a name present only in the template validates
clean and then ties nothing, because no state's effective stack contains it.
The union is deliberate and correct for the case it was written for — a layer
present in some states but not others must stay a valid tie target — but it
also admits names that are live nowhere.

This bites exactly the documented remedy for the naming hazard. The advice in
[`docs/derived-parameters.md`](docs/derived-parameters.md) ("declare the stack
explicitly in the state instead of leaving it to the description") is sound,
and there is no top-level `layers:` key in the setup schema, so following it
means giving *every* state its own stack. The LLM-parsed template still exists
underneath and its names — `copper`, say — remain valid tie targets while the
states all use `Cu`. A stale or mis-sourced spec then passes validation
silently and applies to nothing.

Related, smaller: for a single-state run `_attach_state_metadata` returns
before the tie block, so a `shared_parameters` entry in a single-state setup
is neither validated nor used, and nothing says so.

**The change.** After the build, warn on any resolved tie pair that was applied
to fewer than two states — that catches both the template-only name and a spec
naming a layer only one state has, without weakening the union. The warning
belongs where it will be read: the run transcript, not just the log. For the
single-state case, warn when `shared_parameters`/`unshared_parameters` is
non-empty and there is only one state, rather than ignoring it silently.

**Verifying a fix.** Two states that both declare `layers: [Cu, ...]` while the
template says `copper`, with `shared_parameters: [copper.material.rho]`: the
run must report that the tie applied to no state. Today it reports nothing.
