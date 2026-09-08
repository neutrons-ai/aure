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

Nothing forbids the rename. No refinement rule tells the LLM to leave alone a
layer name a tie spec references. (One did, for the reparametrization block
retired on 2026-09-08, and it was emitted only for models carrying that block —
so a run using `shared_parameters` alone never saw it either.)

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

This bites exactly the remedy for the naming hazard. The advice to declare the
stack explicitly in the state rather than leaving it to the description is
sound,
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

## "Fix this SLD" is honoured about half the time, and nothing says how to ask

**Where:** [`src/aure/nodes/prompts.py`](src/aure/nodes/prompts.py) — the model
JSON schema in the modeling and refinement prompts, and refinement rule 3.

**What is wrong.** A layer offers `sld_min` / `sld_max` and no `fixed` flag,
though one exists for the probe intensity in the same schema:

```
"intensity": { "value": ..., "min": ..., "max": ..., "fixed": <true/false> }
```

A pin *is* expressible — `sld_min == sld_max` reaches the builder as a
zero-width range and bumps accepts it — but nothing in the prompt says so. The
schema glosses the two keys as *"minimum SLD if user specifies a range,
otherwise omit"*, which describes a range, not a pin. So whether an explicit
instruction takes effect is left to the model inferring an undocumented
convention, and it does so inconsistently.

**What it costs.** Measured over the first 9 curves of a sweep whose
description said, in these words, *"the copper metal is 6.48e-6 Å^-2 (bulk
copper), and the dTHF solvent is 6.2e-6 Å^-2. Fix both."*

| | pinned (`sld_min == sld_max` reached the fit) |
|---|---|
| solvent SLD | **5 / 9** |
| copper SLD | **2 / 9** |
| copper within 0.02 of 6.48 without being pinned | 4 / 9 |

Identical wording, same model, same corpus, and the instruction is silently
dropped on roughly half the runs. That is worse than a missing feature: a user
cannot tell from the output whether their constraint was applied, and a sweep
that relies on it gets a mixture. On `cu_film/Cu_0/201144` the trace shows where
it goes when it goes wrong —

| stage | what happened |
|---|---|
| intake | **understood** — both constraints recorded verbatim among 13, copper's start value set to 6.48 |
| modeling | value kept, but `SLD ∈ [4.0, 9.0]` proposed for it regardless, with no reasoning offered |
| builder | stored as `(3.98, 8.98)`, i.e. `6.48 ± 2.5`, the code default |
| a later iteration | narrowed to `(6.3, 8.5)` by an unrelated floor in the sample description |
| fit | landed at **6.581**, and the solvent at 5.958 against a stated 6.2 |

— while on `201236` and `201290` the same instruction produced
`copper rho bounds=(6.48, 6.48)` and held. This is not a comprehension failure:
intake parses it correctly every time. It is a missing schema affordance, and
the checkpoint is a bare proposal with nowhere to record that a constraint could
not be honoured.

**A second hazard, latent rather than observed.** Refinement rule 3 says

> 3. If parameters are hitting their bounds, widen those bounds (sld_min/sld_max,
>    thickness_min/thickness_max).

A pinned parameter is at its bounds by construction, so anything keyed on
"hitting its bounds" should target it. The five pins above survived to the final
problem, so this did not fire in practice — but nothing prevents it, and a
deterministic widener that does not exempt zero-width ranges would silently
unpin them.

**Why it matters beyond one sweep.** Fixing a known scattering-length density is
ordinary practice — the calibrated expert reference fits in the validation
corpus hold the ambient SLD fixed and fit only its roughness. A workflow whose
premise is that the user knows their system needs to accept "I know this
constant" *reliably*, and today it is a coin flip: undocumented in the
description path, dropped entirely in a single-state setup file (see the entry
above), and unprotected against the widener.

**The change.** Three parts:

- Add `"fixed": <true/false>` to the layer schema in both prompts, alongside the
  existing per-parameter bounds, and mention it in the guidance the way the
  intensity `fixed` flag is mentioned. This is the part that converts a
  half-honoured convention into a stated one.
- Honour it in the builder: a fixed parameter is set and not ranged, so it never
  enters `problem.getp()`. Keep treating `sld_min == sld_max` the same way, for
  the models that already write it.
- Exempt fixed and zero-width parameters from rule 3 and from the deterministic
  bound-widener, explicitly.

**Verifying a fix.** Run the same curve five times with *"fix the copper SLD at
6.48, do not fit it"* in the description: the exported `problem.json` must
contain no free parameter for that layer's SLD in **all five**, and the value
must still be 6.48 after the refinement iterations. Today the same test gives a
mixture.

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
thin-layer mode enumeration;
contrast variation; time-resolved series; the batch manifest and plan/job
surface; the web UI; the MCP tool surface; the skill library.

**The point of the exercise** is to be able to say no. "AuRE does not do this,
and here is what to use instead" is a stronger position than a feature that
works when the user already knows the answer. The single-curve case is the one
with 51 curves of evidence behind it; anything that makes that case less robust
for the sake of breadth is a bad trade.

---

## Resolve a data file from its content first; fall back to the filename only when the metadata is incomplete

**Where:** [`src/aure/instruments/registry.py`](src/aure/instruments/registry.py) —
`resolve_by_name` / `resolve`, and the `file_role` / `group_key` /
`header_metadata` wrappers that pick between them.

**What is wrong.** Resolution is filename-first by design: `resolve` tries the
name and only opens the file if nothing claimed it, and the `file_role` /
`group_key` wrappers never read the file at all. The stated reason is sound in
itself — a setup file is parsed before the data need exist, and
`refl1d_import` classifies probes whose files it is still writing — but it
makes the filename authoritative over the file's own declared metadata, which
is backwards. An extension is a fine hint. A name pattern is not evidence about
what is inside.

Three consequences, all reproduced on hand-written files, all silent:

| file (identical ORSO content: θ=0.6 declared, `sQz` = 1σ) | `resolve_by_name` | `resolve` | role | group | θ | dq_is_fwhm |
|---|---|---|---|---|---|---|
| `REFL_201282_combined_data_auto.ort` | ORSO | ORSO | partial | **None** | 0.6 | False |
| `REFL_2222_1_2223_partial.txt` | REF_L | REF_L | partial | 2222 | **0.0** | **True** |
| `cu_film.txt` | **generic** | **ORSO** | unknown | None | 0.6 | False |

1. **The filename wins over the header, and the fit pays.** Row 2 is claimed by
   REF_L's name pattern, so the ORSO YAML header is never read: the declared
   incident angle is lost (θ=0.0 makes `model_builder` build a Q-probe and
   forfeit `theta_offset` / `sample_broadening`), and `dq_is_fwhm` stays `True`,
   so a 1σ `sQz` column is read as a FWHM and refl1d divides it by 2.355 — the
   resolution comes out 2.35× too narrow. `authoritative_fields` cannot help:
   it asks `instruments.resolve()`, which has already answered REF_L.

2. **Two instruments answer for one file.** Row 3 is `generic` to
   `resolve_by_name` and ORSO to `resolve`. Because `file_role` / `group_key`
   use the first and `header_metadata` uses the second, the file gets ORSO's dQ
   convention *and* generic's `role=unknown` — which also makes
   `role_supports_nuisance` refuse nuisance parameters for a file whose
   incident angle is known and declared.

3. **A set id present in the name is discarded, and a guardrail dies with it.**
   REF_L's patterns are anchored to `\.txt$`, so ORSO claims row 1 by extension
   and answers `group_key = None` — though `201282` is in the name. Both set-id
   consistency checks filter `None` before counting
   ([`config.py:393`](src/aure/config.py#L393),
   [`intake.py:653`](src/aure/nodes/intake.py#L653)), so they stop firing rather
   than erroring:

   ```
   two set_ids, .txt   -> ConfigError: partial files must share one set_id (found: ['2222', '3333'])
   two set_ids, .ort   -> kind='partials'          # accepted silently
   ```

**How academic this is.** Entirely, for now: there is no real REF_L ORSO file
to test against, and a properly written one would carry the metadata that is
missing here — its `data_source.measurement.data_files` would name the original
`REFL_<set>_..._.nxs` runs, so the set id would be recoverable from the
*content* rather than the name, and the declared angle and 1σ resolution would
be in the header where the ORSO instrument already looks. The failure modes
above are what happens to a file that is mislabelled or incompletely written,
not to a correct one. That is the reason to fix the ordering rather than to
special-case REF_L-in-ORSO: correct files stop depending on their names, and
incorrect ones degrade in a stated order instead of silently.

**The change (minimal).** Invert the precedence and make the fallback explicit:

- `resolve` reads the header first and prefers an instrument that claims the
  file by *content*; the filename decides only when no instrument claims the
  content, or when the content is unreadable / absent.
- Keep `resolve_by_name` for the two call sites that genuinely cannot read a
  file (`config` parsing a setup before the run, `refl1d_import` writing
  probes), but treat it as the desperation path it is: name it accordingly, and
  have the wrappers that *can* read the file (`file_role`, `group_key`) use
  `resolve` so one file is answered by one instrument.
- Let an instrument answer `group_key` from the header, so a set id declared in
  ORSO metadata is honoured and one omitted from a mislabelled file is
  `None` — as now.
- Restore the guardrail's teeth: `partials` whose group keys are *all* `None`
  should say so rather than pass, since "no instrument encodes grouping" and
  "these files disagree" are different situations and only the second is fine
  to ignore.

`AURE_INSTRUMENT` already exists as the override for data whose provenance
neither the name nor the header reveals, so nothing needs a new escape hatch.

**A related gap, same theme.** Run-title extraction never moved behind the
seam: it is still a module-level regex in intake
(`^#\s*(?:run\s+)?title\s*:`, [`intake.py:52`](src/aure/nodes/intake.py#L52))
applied to every file regardless of instrument. It happens to work on ORSO —
it returned `'Cu film in dTHF'` from the nested YAML `experiment.title` — but
by coincidence, and it will take the first `title:` at any nesting depth.

**Verifying a fix.** The three rows above, as a table test alongside the golden
table in [`tests/test_instruments.py`](tests/test_instruments.py): identical
ORSO content must resolve to ORSO under every name, and one instrument must
answer all four questions about a given file.

---

## An ORSO file cannot be loaded into a probe at all

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py)
— `load_probe`, which hands `.ort` to refl1d's `load4`.

**What is wrong.** It crashes. Not on a malformed file — on one written by
`orsopy` itself:

```
load_probe(valid.ort) -> AttributeError: 'NoneType' object has no attribute 'error_value'
```

The bug is upstream, in refl1d 1.0.1,
`refl1d/probe/data_loaders/load4.py:109-110`:

```python
if hasattr(v, "error") and resolution_index is None:
    header_out[refl1d_resolution_name] = v.error.error_value
```

There are **two** failure modes on that one line, and between them they cover
both shapes a real file takes:

- **No errors written** — orsopy's default. `instrument_settings.incident_angle`
  is a `Value`, whose dataclass *declares* `error` with a default of `None`. So
  `hasattr(v, "error")` is True, it guards nothing, and the next line
  dereferences `None`.
- **Errors written.** The angle is then fine, but `wavelength` is normally a
  `ValueRange`, which declares no `error` field at all — orsopy attaches it as
  an untyped attribute holding a plain **`dict`**, and the same line raises
  `'dict' object has no attribute 'error_value'`. (`Value.error` deserializes
  to an `ErrorValue` while `ValueRange.error` stays a dict; that asymmetry is
  an orsopy issue in its own right and worth raising there separately.)

So there is no way to write the header that gets past it. `resolution_index` —
the thing that would skip the branch — is only set when the file carries the
angle as a data *column* with a matching `physical_quantity` and `error_of`
pair, which a reduced R(Q) curve does not.

**Reproduction.** A standalone script covering both modes was written for the
refl1d team: it builds its files with `orsopy`'s own writer (so they are
spec-valid by construction), shows orsopy reading them back, shows `load4`
failing, prints the offending object state, and carries the patch below in its
docstring. The core of it is four lines:

```python
from orsopy import fileio
from refl1d.names import load4
fileio.save_orso([fileio.OrsoDataset(info, data)], "reduced.ort")
load4("reduced.ort")   # AttributeError: 'NoneType' object has no attribute 'error_value'
```

This is not academic. ORSO is a supported instrument on this branch: the
registry claims `.ort`, reads its metadata correctly, and declares its dQ
convention — and then the file cannot be turned into a probe, so it cannot be
fitted, which is the only thing a user wants from it.

**Why nothing caught it.** Nothing in the suite loads a `.ort` into a probe.
`load_probe` appears in **zero** tests, and `.ort` appears only in
`tests/test_instruments.py`, which exercises classification and header metadata
without ever opening a probe. The 528 lines of instrument tests are all on the
metadata side of the seam.

**The change.** Guard the dereference and tolerate both shapes — read
`getattr(v, "error", None)`, skip when it is `None`, and take `error_value` /
`value_is` from either an `ErrorValue` or a `dict` — then send it upstream to
refl1d, since every refl1d user hits this. AuRE
should not wait on the release: `load_probe` is the single entry point for data
loading, so a local workaround belongs there, either as a targeted patch or by
reading the ORSO file through `orsopy` directly and constructing the probe from
the columns (which is what `tools.data_tools.parse_ort_file` already does for
feature extraction, on a separate and more forgiving code path).

Note the two ORSO readers as a consequence worth removing later:
`parse_ort_file` is a lenient 4-column reader that succeeds on files `load4`
rejects, so a `.ort` file can pass feature extraction and then fail to fit.

**Verifying a fix.** Generate a file with `orsopy.fileio.save_orso` in a test
fixture and assert `load_probe` returns a probe whose `dQ` equals the `sQz`
column (not `sQz / 2.355`, which is the separate `dq_is_fwhm` question). Cover
both header shapes — with and without an explicit angle/wavelength error —
since they fail for different reasons. That test is the coverage gap,
independent of the crash.

---

## `instruments/orso.py` states the dQ error backwards

**Where:** [`src/aure/instruments/orso.py`](src/aure/instruments/orso.py) — the
module docstring, lines 14-17.

**What is wrong.** It says that taking an ORSO `sQz` column as a FWHM
"over-broadens an ORSO resolution by a factor of 2.35". It under-broadens it.
`dq_is_fwhm=True` means "this column is a FWHM", so refl1d converts it to a
sigma by *dividing* by 2.355. Measured on a plain 4-column file whose dQ column
is exactly `2.0e-4`:

```
dq_is_fwhm=True  -> probe dQ[0] = 8.493e-05      (= 2.0e-4 / 2.3548)
dq_is_fwhm=False -> probe dQ[0] = 2.000e-04
```

So a 1σ column read as a FWHM yields a resolution 2.35× too *narrow* — the
model is under-smeared and will chase fringe structure the measurement cannot
resolve.

**What it costs.** Nothing today: the code is right and the fix
(`meta["dq_is_fwhm"] = False`) is correct for the right reason. But the
docstring is the thing a reader consults before touching resolution handling,
and it points the wrong way — the same class of defect as the
`_resolve_tied_set` docstring that claims a tie the code does not make.

**The change.** Two words: "over-broadens" becomes "under-broadens", and the
sentence should say the conversion divides rather than multiplies.

---

## Wish: functional constraints between fit parameters, as an add-on

**Not a defect.** `derived_parameters` — declaring one parameter as a function
of others, so the fit explores a *combination* rather than the coordinates it is
written in — was **removed on 2026-09-08**. This records what it must do if it
comes back, so a future design starts from the constraints rather than
rediscovering them.

**What it was for.** Reflectivity does not determine the parameters a model is
written in; it determines combinations of them. A thin layer's
`(ρ_layer − ρ_ambient)·t` is pinned tightly while the SLD and the thickness
separately are not. Independent measurements have the same shape: QCM-D gives an
adsorbed amount, not an SLD; a density plus a swelling measurement gives a
volume fraction, not a thickness. Fitting the combination removes the degeneracy
ridge from the geometry the optimizer explores.

**Why it was removed, in order of weight:**

1. **It has to survive AuRE's own iteration, and it did not.** A declaration is
   written against layers; the refinement loop adds and removes them. The
   response was `prune_derived_parameters` — drop the declaration and log it.
   That is a workaround for the hard problem, not an answer to it, and it is
   exactly the part an add-on has to design first. A constraint that evaporates
   when the loop edits the stack is worse than no constraint, because the run
   continues and reports a χ² for a model nobody declared.
2. **No prose route, and the fallback was unsafe.** The mechanism was
   config-only. Asked in a description for a relation it could have expressed
   ("the volume fraction is the same in both contrasts, so the SLD must differ
   as the solvent does"), the tie extractor returned the nearest expressible
   thing — an untie — silently substituting two free parameters for one shared
   invariant. Whatever replaces this must **refuse** and say so in `issues`.
3. **It was never used.** Added to support benchmarking and not used for it.
   Off by default was the tell.

**What worked and is worth carrying forward:**

- **Expressions round-trip.** bumps 1.0.x preserves expression parameters *and*
  `Constraint` objects through `problem.json` — verified single-state and
  multi-state, with χ² and constraint count unchanged. The old
  `save_problem_json` refusal was over-conservative; an add-on need not inherit
  it. (Re-verify against the pinned bumps version.)
- **Cross-state relations were expressible**, if awkwardly: an auxiliary *tied*
  declaration holding one shared free parameter, plus one state-scoped
  assignment per state, each with its own expression. Verified: a 2× ratio held
  as the shared handle moved. Two of the three declarations existed only to
  carry an expression and their mandatory `free` blocks were dead — a first-class
  surface should express this directly.
- **The expression namespace was per state**, built from that state's own
  sample, with no handle on another state's parameters. The shared auxiliary was
  the only way to link states; there was no `other_state.thickness` to write.
  An add-on needs to decide whether cross-state references are first-class.
- **A whitelisted-AST evaluator, never `eval`.** These strings come from config
  files and from LLMs. The retired `expressions.py` admitted numbers, dotted
  names, unary minus, five arithmetic operators and four comparisons, and
  rejected everything else by name. Reuse that shape.
- **BIC accounting.** A derived raw parameter leaves the free set, so the count
  has to move with it. Multi-state runs read the count off the problem
  (`len(problem.getp())`), which is correct without special-casing; only the
  single-state and checkpoint-replay paths needed a delta.
- **χ² must stay the data term.** `data_chisq` exists because a `Constraint`
  makes `FitProblem.chisq()` a number about the penalty rather than the data — a
  violated one measured ~10¹⁰. It was kept on removal, so an add-on inherits a
  χ² that already means what it should.

**Where the code was**, for anyone reconstructing it: `nodes/expressions.py`
(the evaluator), a ~400-line block in `nodes/model_builder.py` between
`_build_sample` and `data_chisq`, the gate and shape-check in `config.py`, the
carry-over in `nodes/modeling.py`, refinement rule 14 in `nodes/prompts.py`, the
`functional-constraints` skill, `docs/derived-parameters.md`, and
`tests/test_derived_parameters.py`. All present up to the commit that removed
them.
