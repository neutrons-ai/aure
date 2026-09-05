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

---

## A single-state setup's `layers` — and every bound in them — is silently discarded

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py) —
`needs_states_problem`, which returns True only for `len(states) > 1` or for a
state carrying `theta_offset` / `sample_broadening` / `background`.

**What is wrong.** A setup file with exactly one state, no nuisance parameters,
and a fully declared `states[0].layers` does not take the states route. It falls
through to the description-driven path, which builds the model from
`parsed_sample` and never reads `states[0].layers`. The declared stack, and with
it every per-layer bound, is dropped without a warning.

Measured on a real run. The setup declared, for a single state:

```yaml
states:
  - name: OCV3_206931
    layers:
      - {name: copper, sld: 6.48, sld_min: 6.48, sld_max: 6.48, ...}
```

The **first** modeling checkpoint (`003_modeling.json`) already reported

```
copper sld=6.55 window=(6.3, 8.5)
```

so the value, the pin and the window were all gone before any refinement
iteration ran. Adding a second state to the same setup honours the identical
layer block, which is what localises the fault to the routing rather than to the
layer parsing.

**What it costs.** The setup file is the only surface on which a layer bound can
be declared at all — there is no top-level `layers:` key (see the roughness-floor
entry above, which records the same gap from the other side). So for a
single-curve analysis there is currently **no way to state a known constant**:
not "this copper is bulk copper at 6.48", not "the solvent SLD is known, fit its
roughness only" — which is what the expert reference fits themselves do. The
request is accepted, written into the config, echoed in the run's own setup, and
then ignored. A user has no way to tell that it did not take effect except by
reading the exported problem.

It also silently changes what an experiment measures. An arm intended to test
"does a stated materials constant improve the fit?" instead tests nothing: the
constant never reaches the model.

**The change.** Take the states route whenever a state declares its own
`layers` or `substrate`, not only when there are two or more states or a
nuisance parameter is present — i.e. add that condition to
`needs_states_problem`. A single-state states problem is already supported (the
builder handles it; only the routing predicate excludes it), and the tie
machinery is inert with one state.

Failing that, the honest fallback is to refuse: if a setup declares per-state
`layers` on a path that cannot honour them, error at load rather than proceed.
Silently discarding a declared bound is the worst of the three options.

**Aside: the workaround, and why it is not a fix.** Any nuisance parameter flips
the predicate, so a zero-width `theta_offset: {init: 0, min: 0, max: 0}` forces
the states route while being numerically inert. It works, and it is what one
would reach for under time pressure, but it couples an unrelated field to the
model-building route and would break the moment the predicate changes.

**Verifying a fix.** The setup above, run as-is: the exported `problem.json`
must show `copper rho` with bounds `(6.48, 6.48)`, and the first modeling
checkpoint must carry the declared window rather than `(6.3, 8.5)`.

---

## No layer parameter can be held fixed, and the widener would undo it anyway

**Where:** [`src/aure/nodes/prompts.py`](src/aure/nodes/prompts.py) — the model
JSON schema in the modeling and refinement prompts, and refinement rule 3.

**What is wrong.** A layer offers `sld_min` / `sld_max` and nothing else. There
is no `fixed` flag, though one exists for the probe intensity in the same
schema:

```
"intensity": { "value": ..., "min": ..., "max": ..., "fixed": <true/false> }
```

So "this copper is bulk copper at 6.48e-6 Å^-2, do not fit it" has no
representation. The nearest expression is `sld_min == sld_max`, which the prompt
never suggests — the schema glosses these as *"minimum SLD if user specifies a
range, otherwise omit"* — and which refinement rule 3 would then undo:

> 3. If parameters are hitting their bounds, widen those bounds (sld_min/sld_max,
>    thickness_min/thickness_max).

A fixed parameter sits at its bounds by construction, so the one instruction
that could hold it is also the one the widener is guaranteed to target.

**What it costs.** The instruction is understood and then lost. Measured on
`cu_film/Cu_0/201144`, whose description said *"Fix copper metal SLD at
6.48e-6 Å^-2; do not fit it"* and *"Fix dTHF solvent SLD at 6.2e-6 Å^-2; do not
fit it"*:

| stage | what happened |
|---|---|
| intake | **understood** — both recorded verbatim among 13 constraints, copper's start value set to 6.48 |
| modeling | value kept, but `SLD ∈ [4.0, 9.0]` proposed for it regardless, with no reasoning offered |
| builder | window stored as `(3.98, 8.98)`, i.e. `6.48 ± 2.5`, the code default |
| a later iteration | narrowed to `(6.3, 8.5)` by an unrelated floor in the sample description |
| fit | landed at **6.581** |

The solvent went the same way: pinned at 6.2 in the description, parsed as
`Ambient: dTHF (SLD = 6.20)`, fitted to 5.958.

This is not a comprehension failure and it should not be read as one. The
modeling node behaved reasonably given a schema with no way to say "fixed"; the
checkpoint is a bare proposal, so there is not even a place for it to record
that it could not honour a constraint it had just been handed.

**Why it matters beyond one run.** Fixing a known scattering-length density is
ordinary practice — the calibrated expert reference fits in the validation
corpus hold the ambient SLD fixed and fit only its roughness. A workflow whose
premise is that the user knows their system needs a way to accept "I know this
constant", and today there is none: not in the description (no representation),
not in a setup file for a single curve (see the single-state entry above), and
not through the refinement loop (rule 3 widens it back).

**The change.** Three parts, and the third is the one that makes the other two
stick:

- Add `"fixed": <true/false>` to the layer schema in both prompts, alongside the
  existing per-parameter bounds, and mention it in the guidance the way the
  intensity `fixed` flag is mentioned.
- Honour it in the builder: a fixed parameter is set and not ranged, so it never
  enters `problem.getp()`.
- Exempt fixed parameters from rule 3 and from the deterministic bound-widener.
  A parameter that is fixed is *at* its bounds permanently, so anything keyed on
  "hitting its bounds" must skip it explicitly.

A narrower alternative — treat `sld_min == sld_max` as a pin and exempt it from
widening — needs only the last two parts, but leaves the capability undiscoverable
from the schema, which is how it came to be missed here.

**Verifying a fix.** Run any curve with *"fix the copper SLD at 6.48, do not fit
it"* in the description: the exported `problem.json` must contain no free
parameter for that layer's SLD, and the value must still be 6.48 after five
refinement iterations.

---

## Cross-state ties only work when the setup declares the structure — say so, or fix it

**Where:** [`aure_config.example.yaml`](aure_config.example.yaml) (the
`shared_parameters` / `unshared_parameters` block), the co-refinement section of
[`README.md`](README.md), and the tie resolution in
[`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py).

**What is wrong.** Nothing in the documentation says that a tie is only
meaningful when the states declare their own `layers`, but that is the case, for
two compounding reasons.

1. **A tie must name a layer.** Specs are `<layer>.<attr>`; there are no
   wildcards and no way to say "every layer's SLD" or "the metal film, whatever
   it is called". The only name-independent targets are the literal aliases
   `ambient` and `substrate`. So a tie can only be written if the author already
   knows the layer names — which, on a description-driven run, they do not: the
   names come from the intake parse and vary run to run (the same electrode
   yielded `silicon native oxide (SiO2)` in one state and `SiO2 (native oxide)`
   in another).
2. **Per-state structure cannot be inferred.** A state with no declared
   `layers` inherits the model-level stack, so states that genuinely differ
   collapse to one structure. Measured: two states of one electrode, pristine
   and post-plating, run with no declared layers, both came out as
   `silicon native oxide / titanium / copper`; neither surface layer was ever
   proposed and no refinement iteration differentiated them (χ²_red 3.8 and 1.6,
   *worse* than either curve fitted alone).

Together these mean co-refinement cannot be used without declaring the
structure — and declaring the structure supplies the shape of the answer. The
information co-refinement is meant to add is not separable, through this
interface, from the information it is meant to help you find. `example.yaml`
comes close to saying the first half ("give the deviating state its own COMPLETE
stack") but frames it as an option for a state that differs, not as a
precondition for ties to mean anything.

**What it costs.** An experiment we ran on this is the clean illustration: a
co-refined arm scored better than single-curve fitting on every fit-quality row,
and the result was uninterpretable, because the arm had been handed a declared
per-state stack that the single-curve arm had to infer. Removing the declaration
made co-refinement *worse* than not co-refining at all. A user reading the
current docs would not anticipate either outcome.

**The decision, which is what this entry is really for.** Two honest options,
and the wrong move is to leave it undocumented and half-working:

- **Bound it loudly.** State in `example.yaml`, the README co-refinement
  section and the web UI tie panel that cross-state ties require each state to
  declare its own `layers`, that names must match across states for a tie to
  apply, and that co-refinement is therefore for the case where the user already
  knows both structures — contrast variation on a known stack, not structure
  discovery. Then make the code say so too: refuse, or warn in the transcript,
  when `shared_`/`unshared_parameters` is non-empty and no state declares
  `layers` (this pairs with the union/per-state entry above, which already asks
  for a warning when a tie applies to fewer than two states).
- **Or make it work without a declared structure.** Two pieces would be needed:
  ties addressable by something stable — role (`ambient`, `substrate`), position
  index, or material identity rather than the parsed label; and a modeling node
  that can propose *different* stacks for different states, which today it
  cannot do from a standing start. That is a real feature, not a patch, and it
  should only be built if the use-case survives the review in the next entry.

**Verifying either.** For the bounding option: a setup with
`shared_parameters` and no declared per-state `layers` must produce a loud
warning naming every tie that could not be applied. For the fixing option: two
states of one sample, no declared layers, must end with different stacks where
the data require it, and a tie expressed by role must hold across them.

---

## Take a step back: enumerate AuRE's use-cases and decide which it should serve

**Not an issue with a decided remedy** — the other entries in this file are, and
this one deliberately is not. It is a scoping decision that should be made
deliberately and written down, because several of the entries above are only
worth fixing if the use-case behind them is one AuRE is meant to serve.

**Why now.** The evidence from the validation work is that AuRE is genuinely
useful on the case it was built for, and that the further a use-case sits from
that centre the more the machinery has to be bent to reach it — each bend adding
a surface that can fail quietly rather than loudly. Three of the entries above
are of exactly that shape: a bound that cannot be declared, a tie that cannot be
expressed, a structure that cannot be inferred per state. None of them is hard
to patch individually. Together they are a signal that capability is being added
faster than the boundaries are being drawn, and the cost lands on robustness in
the centre.

**The exercise.** List every use-case the system currently admits — from the
README, the config schema, the CLI, the web UI and the MCP surface, not from
memory — and for each one record:

- what it claims to do, and where that claim is made;
- whether it has ever been run end to end on real data, and where the evidence
  is;
- what it depends on that the user must supply, and whether the interface can
  actually accept it (three of the entries above are failures of exactly this);
- how it fails when a precondition is missing: loudly, or silently;
- what it costs to keep — code paths, prompt surface, schema fields,
  documentation, and the failure modes it introduces into unrelated paths.

Then sort into: **core** (supported, tested, documented, defended);
**bounded** (works within stated limits, and the limits are enforced in code,
not just written down); **retired** (removed, with the reason recorded).

A starting inventory, to be checked against the code rather than trusted:
single-curve steady-state fitting; multi-file fitting of one sample;
multi-state co-refinement with cross-state ties; per-state structure overrides;
reparametrization via `derived_parameters`; thin-layer mode enumeration;
contrast variation; time-resolved series; the batch manifest and plan/job
surface; the web UI; the MCP tool surface; the skill library.

**The point of the exercise** is to be able to say no. "AuRE does not do this,
and here is what to use instead" is a stronger position than a feature that
works when the user already knows the answer. The single-curve case is the one
with 51 curves of evidence behind it; anything that makes that case less robust
for the sake of breadth is a bad trade.
