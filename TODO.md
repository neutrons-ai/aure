# TODO

Known issues with a decided remedy, recorded rather than fixed. Each entry says
what is wrong, what it costs, and what the change would be.

---

## The outer interface can only be bounded from outside the model

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py)
— `_outer_roughness_max`, and the `roughness_tie` branch of `_build_sample`.

**What this entry used to say, and why it was wrong.** It proposed making
`ROUGHNESS_MAX_OUTER` a floor on the ceiling — `max(override, declared)` —
"probably right", on the grounds that its purpose is to stop a low default
capping a diffuse interface. The corpus refutes that:

| | |
|---|---|
| outer-interface fits within 2 % of the 250 Å cap | **108 of 835**, across **32 runs**, several at exactly 250.0 |
| expert reference roughness in those runs | **46–200 Å, median 79** |
| model declarations exceeding the cap | 16, up to 400 Å |

Where the cap binds, the fit is already running 2–4× past the expert value and
the cap is the only thing stopping it. Widening it lets those 32 runs travel
*further* from the reference. The cap is a regularizer the corpus leans on, not
a safety margin — so the precedence stays as it is, and
`outer_ceiling_displacement` + a log line now make it visible instead
(`7e15c10`, this entry's phases 1 and 2).

**The real defect is that no model-side tool can bound that interface.**

`roughness_tie` is the principled bound — σ = fraction × thickness, so an
interface can never outgrow its layer — and it **cannot reach the outer
surface**. Applied to the outermost layer it rewrites that layer's *own*
interface and leaves the outer one free:

```
stack: ['dTHF', 'SEI', 'Cu', 'Si']          # back reflection, tie on SEI
  [0] dTHF   interface=Parameter   bounds=(0.0, 250.0)   <- the outer surface, untouched
  [1] SEI    interface=Expression  (fraction x thickness) <- the tie landed here
```

The outer surface lives on the *ambient* slab, and `roughness_tie` only
rewrites layer slabs. So the only instrument that can bound it is a global
constant supplied from the environment — which is exactly why
`ROUGHNESS_MAX_OUTER` exists, and why it has to be blunt.

**And neither guard catches the runaway reliably.** Of the 108 pinned fits, 90
are vetoed by the SLD-profile artifact check — but **12 were verified clean**
(`profile_checked=True, profile_artifact=False`) at 246–250 Å with χ² of
1.3–2.9, against references of 66–124 Å. A good χ², a clean profile, and an
outer roughness 2–4× the expert value. So χ² actively rewards the runaway in
those cases, presumably because a very wide erf tail mimics a diffuse layer or
density gradient the model is missing — and nothing in the loop objects.

**The change.** Give the outer surface the same principled bound the layers
have: extend `roughness_tie` (or an equivalent) to it, so in back reflection
`sample[0].interface` becomes `fraction × <outermost layer>.thickness` rather
than a free parameter under a global cap. A 250 Å tail on a 200 Å layer is
unphysical whatever any env var says, and this is the bound that says so from
inside the model.

That would make `ROUGHNESS_MAX_OUTER` redundant for its stated purpose, and it
would remove the cases where the profile check passes a runaway. It changes
what the 32 pinned runs fit, so it costs a re-sweep.

**Available today, and cheaper:** an explicit `interfaces` entry names the
outer boundary and sets both its bounds, beating the env override and the
hardcoded floor of 0 in one step — verified: `(0, 250)` becomes `(5, 400)`. It
sets fixed numbers rather than σ ≤ k·t, so it is a workaround rather than the
fix, and nothing emits it yet (see `InterfaceInfo`).

## The bottom medium's interface is the one interface with no floor

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py)
— the two `sample[0].interface.range(...)` calls at the end of `_build_sample`.

**The semantics first**, because the naming misleads. Substrate and ambient are
both slabs in the refl1d stack, and a slab's `interface` is its boundary with
whatever sits *on top of it*. So the medium at the **bottom** of the stack owns
a real fitted roughness, and the one at the **top** has an unused `0` — nothing
is above it. Which medium is which depends on the beam direction:

```
normal:           [silicon, SiO2, Cu, dTHF]
  [0] silicon   interface=4.00  FREE bounds=(0.0, 15.0)   substrate/SiO2 boundary
  [3] dTHF      interface=0.00  unused

back reflection:  [dTHF, Cu, SiO2, silicon]
  [0] dTHF      interface=7.00  FREE bounds=(0.0, 25.0)   the OUTER surface
  [3] silicon   interface=0.00  unused
```

**What is wrong.** Every interface above the bottom resolves its lower bound
through `_ranged`, which honours a declared `roughness_min` and otherwise
applies a 5 Å default that yields to a smaller declared `roughness`.
`sample[0].interface` does neither — its lower bound is the literal `0` in both
branches:

```python
sample[0].interface.range(0, _outer_roughness_max(layers_info))  # back reflection
sample[0].interface.range(0, sub_rough_max)                      # normal
```

So the one interface that cannot state a floor is whichever the beam makes the
bottom of the stack: the substrate boundary in a normal run, and the **outer
surface** in back reflection — the interface most likely to be genuinely rough,
and the one whose *ceiling* is already configurable via `ROUGHNESS_MAX_OUTER`.
`SubstrateInfo` declares no `roughness_min` and nothing reads one.

**A second observation, same code.** In back reflection the ambient's interface
is seeded from the outermost layer's `roughness` (`layers_info[-1]["roughness"]`),
so `[0] dTHF` and `[1] Cu` above both start at 7.00 — two independent free
parameters from one declared value, with different bounds ((0, 25) and (5, 25)).
That may be intended as a sensible seed; it is worth knowing that editing the
outermost layer's roughness moves two parameters.

**What it costs.** Not measured. A roughness of 0 is unphysical but it is one
parameter and the layers above carry the structure; the fit is free to leave it.
Worth knowing before someone reads the layer semantics and assumes the bottom
medium shares them.

**The change.** Route `sample[0].interface` through `_ranged` like every other
interface, so it gets the same default-yields-to-declared behaviour and can
carry a declared floor — which means `SubstrateInfo` needs a `roughness_min`,
and the back-reflection branch needs to decide whose floor it reads (the
ambient's, or the outermost layer's, whose roughness it already borrows).
Alternatively state in `SubstrateInfo` that the bottom interface is
deliberately unbounded below. The first is consistent; the second is honest
about what is implemented.

**A related gap in the declaration surface.** `roughness_min` reaches a *layer*
from a refinement (prompt schema, rule 4a) and from a setup file inside
`states[].layers[]`. It does not reach a description-driven single-file run —
`aure analyze DATA "description"` — because there is no top-level `layers:`
key. That is the same question U4 raises in [docs/scope.md](docs/scope.md);
decide that first.

---

## `intensity: true` loads, then kills the fit with a TypeError

**Where:** [`src/aure/config.py`](src/aure/config.py) — `_TRIPLET_DEFAULTS` /
`_normalise_nuisance`, against
[`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py) —
`build_states_problem`.

**What is wrong.** Three of the four per-state nuisance keys accept `true` as
"fit this over the default range": `_normalise_nuisance` expands it from
`_TRIPLET_DEFAULTS`. `intensity` is not in that table, so `true` is passed
through verbatim as a Python bool, the setup loads clean, and the run dies in
the builder several nodes later:

```
$ aure analyze -c setup.yaml        # states[0].intensity: true
TypeError: 'bool' object is not a mapping
```

from `{**intensity, **ds_intensity}` — `intensity` is `True`. The state loads,
intake runs, modeling runs, the first fit builds, and only then does the run
end, on a message that names neither the key nor the file.

`intensity` is the one nuisance key with a good implicit default already
(`init` 1.0 over `[0.7, 1.1]`, applied when the key is absent), so a user
writing `intensity: true` is asking for exactly that. It reads as the same
shorthand the neighbouring keys accept, and nothing at load time says it is
not.

**The change.** Give `intensity` a `_TRIPLET_DEFAULTS` row —
`{"init": 1.0, "min": 0.7, "max": 1.1}`, the values the builder already
falls back to — so `true` expands like the others and `false` disables it.
The builder reads `init` as well as `value`, so the expanded mapping needs no
change there. Either way `_normalise_nuisance` should reject a bare non-mapping
for any nuisance key rather than passing it through: the current failure mode
is a load-time typo surfacing as a fit-time `TypeError`.

Documented meanwhile in
[docs/launching.md](docs/launching.md#nuisance-parameters) — the mapping form
is spelled out and the `true` shorthand is called out as not applying here.

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

---

## Wish: let the refinement loop escalate the optimizer once, and read the result

**Where:** [`src/aure/nodes/evaluation.py`](src/aure/nodes/evaluation.py) —
the stall test inside `_should_revise_hypotheses`;
[`src/aure/nodes/routing.py`](src/aure/nodes/routing.py) —
`route_after_evaluation`'s existing `fitting` edge;
[`src/aure/nodes/fitting.py`](src/aure/nodes/fitting.py) — the `FIT_METHOD`
lookup.

**The observation.** A stalled fit has two quite different causes and the loop
cannot currently tell them apart. Either the search is stuck in a local
minimum, or the model is wrong. `evaluation` treats every stall as the second
and proposes a structural change, because a structural change is the only
instrument it has.

On the ionomer beamtime both cases occurred in one sample. Sample2's D2O state:
a 23-parameter model started at χ² 1074, amoeba stopped at 88.5 with the
ionomer and hydrated-layer SLDs the wrong way round, and `de` on the identical
problem — same spec, same data, same bounds — reached 17.7 with three seeds
agreeing to 0.06. No model change was involved, so no model change could have
been the fix. Sample2's air state, same beamtime: χ² ~62 under a 240k-evaluation
`de` search with every bound reopened, multiple seeds reaching the *same*
minimum. There the stack was wrong, and no optimizer was going to help. The
project's
`samples/sample2/reports/why-sample2-will-not-fit-the-sample1-stack.md`
records both.

**What it costs.** Facing case one, the loop dismantles a model that was
correct. Facing case two, it keeps proposing structure without ever
establishing that the search had been given a fair chance — which is the work
a person had to do by hand, twice, on that beamtime.

**The change.** When the fit is unacceptable, χ² has stalled (the ≥3-fit, <5%
rule already in `_should_revise_hypotheses`, extracted so both callers share
one definition), and the method in use is local (`lm` / `amoeba`), set
`state["fit_method"] = "de"` and take the `fitting` edge that already exists
for bounds-only re-fits — same model, fit it again, no LLM round. Latch rather
than probe: if `de` improves χ², stay on it, because a search that was the
limit at iteration 4 is still the limit at iteration 5; if it does not, latch
back rather than pay ~70× again.

**The point is the diagnostic, not the better fit.** Record
`optimizer_escalation = {from, to, iteration, chi2_before, chi2_after,
verdict}` and put one line in the evaluator's prompt. `model-limited` — a
global search over the whole box reached the same χ² — says the next change
*must* be structural. `search-limited` says the structure was fine and
protects a model the loop was about to take apart. A forced global default
gets the better fit and throws the diagnostic away, because nothing is left to
compare against.

**Guard rails.** Never escalate from `de` or `dream`. Once per run — allowing
repeats is precisely how "try another optimizer" becomes the loop, which is
the failure nr-workbench's `fitters.py` exists to prevent and which an
automated loop can reproduce far faster than a person. Make the cost visible,
since one iteration goes from seconds to a minute or more.

**Three things to settle first.**

1. **It is inert under the shipped default.** `FIT_METHOD` defaults to
   `dream`, which is never escalated from, so this only fires for runs set to
   `lm` or `amoeba` for speed. Whether AuRE's *exploration* default should
   move to `de` — with `dream` reserved for `FIT_METHOD_FINAL`, which is what
   `.env.example` already advises — is a separate and larger decision, with
   its own implications for the `validation/` corpus.
2. **Default off**, behind `OPTIMIZER_ESCALATION` / a setup key, for the
   reason `USE_RUN_TITLE` and `MODE_ENUMERATION` are: turning it on changes
   results for every run that would trigger it, and the reference fits in
   `validation/` are the baseline that comparability depends on.
3. **The regression guardrail needs reading first.** `evaluation` restores the
   previous model when χ² gets worse after a refinement. An escalation that
   lands worse must not be mistaken for a failed refinement — the model did
   not change, so there is nothing to restore.

**Cost:** ~120 lines plus tests. The trigger, latch and verdict are pure
functions over `fit_history`, so they test without running a fit (see
`tests/test_evaluation_chi2_clamp.py` for the shape); only an end-to-end check
on run 234277 needs a real one.

**Related measurement**, already recorded in
[`docs/plan-new-reduction-format.md`](docs/plan-new-reduction-format.md) phase
5: on run 234277, `de` reached the same χ² from every SLD seed (2.204, to
three decimals) where amoeba's varied from 2.364 to 7.045, at 71× the cost.
That is why the thin-layer mode-enumeration sweep keeps amoeba — its
seed-dependence is the mechanism the sweep exploits — and why `de` belongs
where the answer, not the landscape, is what is wanted.

## Six LLM calls bypass `invoke_with_timeout`, so the call ledger never sees them

The convention is that every call goes through `invoke_with_timeout(get_llm(), …)`,
which bounds it and writes its row in `llm_calls.jsonl`. Six call `llm.invoke`
directly instead: `analyze_fit_quality_with_llm` (the verdict on every pass) and
`propose_hypothesis_revision_with_llm` in `nodes/evaluation.py`, and
`_refine_model`, `_refine_model_legacy`, `_extract_cross_state_unshared` and
`_extract_per_state_structure` in `nodes/modeling.py`. They get no per-call
timeout and no ledger row, and they carry the long prompts: on the 51-curve
gpt-5.4 validation sweep the ledger held about 10 of a run's 21 calls but only
18% of its prompt tokens, so token and cost totals read from it are low by about
a factor of five. The validation harness recovers the
rest by replaying checkpoints (`scripts/replay_llm_calls.py`). Found while adding
the `temperature` field to the ledger; routing the six through the wrapper is
the fix.

---

## The profile veto throws away fits an analyst would repair

**Where:** [`src/aure/tools/feature_tools.py`](src/aure/tools/feature_tools.py)
— the excursion loop in `detect_profile_artifacts`;
[`src/aure/nodes/finalize.py`](src/aure/nodes/finalize.py) — `_tier` and the
tier-χ² guard in `_select` (`_DEFAULT_TIER_CHI2_FACTOR = 3.0`);
[`src/aure/nodes/evaluation.py`](src/aure/nodes/evaluation.py) — the
bounds-only refinement shortcut.

**What is wrong.** Two things. A real overshoot is often one large roughness,
wider than the layers it crosses can carry; an analyst finds it, fixes it and
re-fits. AuRE instead ranks the fit last at finalize, and nothing in the loop
asks what caused it. And the test that finds overshoots also finds things that
are not one. `detect_profile_artifacts` flags any extremum of the profile that
is not within 5 % of the SLD span of one of the slab sequence's own turning
values. Two different things fail that test:

- a thin layer whose SLD the profile never reaches, while it stays between its
  neighbours: a 19 Å native oxide fitted at 3.63 peaks at 2.86, because the
  titanium interface beside it is 11.8 Å wide. Any thin, rough layer looks like
  this;
- a profile that leaves the range of its neighbours, because a wide interface's
  erf tail reaches through thinner layers: a 243 Å solvent interface spanning
  ~150 Å of oxide, plated layer and SEI lifts the top of the copper from 6.53 to
  7.33.

Both go to tier 2, "physically impossible, so genuinely last". Only the second
is a real artifact, and it is the repairable kind.

**What it costs.** From a gpt-6-luna `de` sweep of the 51-curve copper series
(2c05e66 plus the node limit, `-m 10`):

- 201236: all five fits with the reference structure (four layers, χ² 1.61,
  lowest BIC at 311) were vetoed for the first kind. The reported fit adds two
  layers totalling ~400 Å at 6.26–6.32 against a 6.37 solvent. Every clean fit
  had five or six layers, so no ranking rule could have recovered the
  four-layer one.
- 207027: nine of ten fits were vetoed, among them pass 10 (χ² 1.47), which has
  the reference's shape and the second kind. The only clean fit, pass 8 at
  χ² 4.09, was reported and judged unusable: a 220 Å layer near SLD 0 where the
  reference has a 59 Å plated layer at 1.90. The guard did not fire: 4.09 is
  2.8 × 1.47, and the factor is 3. Pass 10 was judged a good step in the right
  direction, to be improved on, not vetoed; it was the run's last pass, so the
  loop never got the chance.

**The change.** Identify the artifact, fix it, re-fit; veto only what that
cannot clean:

1. Name the cause. `detect_profile_artifacts` already reports where the
   profile turns; it can also report which interface's tail reaches there —
   one whose roughness is wider than the layers between it and the excursion.
2. When the cause is a roughness and the fit otherwise looks good (its χ² at or
   near the best so far), fix that roughness at a value the layers can carry,
   no wider than the thickness it crosses, and re-fit from that fit. In the
   loop that is a bounds change, so it can take the shortcut an auto-expanded
   bound already takes, straight back to `fitting` without a modeling round.
   At finalize, do it once for a vetoed fit whose χ² beats every clean one,
   rank the result with the rest, and let `final_fit` polish the winner.
3. Keep the veto for an artifact with another cause, or one the re-fit does
   not clean; and veto an extremum only when it lies outside the SLD range of
   the layer it sits in and that layer's two neighbours. A thin layer that
   falls short is a note on the fit, not an artifact.

---

## The solvent's SLD window follows the last fit, so one bad pass can shut out the right value

**Where:** [`src/aure/nodes/model_builder.py`](src/aure/nodes/model_builder.py)
— the ambient range in `_build_sample`;
[`src/aure/nodes/modeling.py`](src/aure/nodes/modeling.py) — the same defaults
in `build_refl1d_script`; and the refinement step, where the model sets the
ambient's `sld_min` / `sld_max` itself.

**What is wrong.** Without explicit bounds the ambient's window is ±20 % of its
current SLD, and the current SLD is whatever the last model carried, so the
window moves with the fit. When the model sets bounds itself, it sets them
around the last fitted value too. Either way, one pass that puts the solvent in
the wrong place decides where every later pass may look.

**What it costs.** On the copper series the solvent is d-THF or THF, never a
mixture, so the right value is near 6.2 or near 0.2 and nothing in between.
Four runs from the gpt-6-luna sweeps:

- 207020, `amoeba`: the first fit put the d-THF at 5.48. A later pass left the
  bounds to the default, which centred the window on 5.48 (4.39–6.58), and the
  run ended at 4.74, χ² 4.94 against the reference's 1.28.
- 207027, `de`: the first fit put it at 6.78. From the second pass the model
  bounded it at 6.5–8.0 (at times 7.0–8.0), which shuts out both the reference
  6.10 and the textbook d-THF value, 6.35. Four of the five arms run on this
  curve failed it, all with the solvent at 6.57–6.90.
- 201290, `de`: the first fit had it right, at 6.13 against the reference's
  6.20. The model then bounded it at 6.40–8.40, later 6.80–8.00, and the
  reported fit sits on that 6.80 floor, with the top SEI's SLD on its own
  floor (0.50).
- 201334, `de`: pass 5 had the reference's structure, an SEI of 331 Å at 2.51
  with the solvent at 6.20, but at χ² 2.90; pass 1, with no SEI and the solvent
  on the 4.80 floor of the default window, won at 1.81. The final DREAM polish
  then rebuilt the default window around 4.80 (3.84–5.76), and the solvent sank
  to 4.22.

Across the 51 curves, luna with `amoeba` on this code left the solvent between
1.0 and 5.5 on 14; gpt-5.4 on v0.9.0, on 4.

Why it sinks: a thick SEI whose far edge the data barely see can be absorbed
into the ambient. On 207072 (`de`) the best pass, at χ² 0.99 against the
reference's 1.38, has no SEI and the solvent at 4.50 — the reference SEI's own
SLD is 4.47, over 473 Å. Only knowing the solvent rules that reading out, which
is what an anchored window encodes.

**The change.** Centre the window on the solvent's starting value — the one the
setup or the description gives — and keep it there for the run; let the model
narrow the window, but not past that value. Centring alone is not enough:
±20 % around 6.0 is 4.8–7.2, and 201334's first pass sat on that 4.80 floor. A
pure solvent's SLD moves by a few per cent at most (density, isotopic purity),
so its window should be about that wide. The same two lines carry the
inverted bounds for a negative ambient SLD noted in the code's `TODO(bug)`, so
one change can fix both.

---

## The report's "Fit Results" ignore an adopted final DREAM polish

**Where:** [`src/aure/cli.py`](src/aure/cli.py): `_selected_fit`, used by
`_print_analysis_results` and by `--json`'s `fit_result`. Also
[`src/aure/web/data.py`](src/aure/web/data.py): the default iteration of the
Results tab.

**What is wrong.** All three resolve the reported fit as
`fit_results[final_selection["index"]]`, which is the exploration fit.
`final_fit` appends its DREAM `FitResult` after that and, when it adopts it,
writes the DREAM values into `current_model` and `current_chi2`. With a `de`
exploration and an adopted polish, the terminal report shows:

- under *Final Model*, the DREAM values;
- under *Fit Results*, the exploration χ², `Method: de` and the exploration
  values, with **no uncertainties**.

The ± that the polish exists to produce never reach the parameter table or
`--json`. Found while adding the posterior check, which resolves this correctly
through `posterior_tools.posterior_for_report`.

**The change.** Resolve the displayed fit as the final DREAM `FitResult` when
`final_fit["adopted"]` is set, in one helper the CLI report, `--json`, `aure
batch` and the web default all use. `posterior_for_report` is the pattern.

---

## bumps' posterior-entropy estimator has four defects (upstream)

**Where:** bumps 1.0.4 `bumps/dream/entropy.py` (`entropy`,
`scipy_stats_density`) behind `MCMCDraw.entropy(method="llf")`.

**What is wrong.**

1. The normalization subset is sliced with `N_entropy` instead of `N_norm`.
2. The normalization and evaluation subsets are drawn independently, so they
   overlap. The kernel density at a point that is also a kernel centre is
   biased high.
3. `scipy_stats_density` passes the sample to `gaussian_kde` untransposed, so it
   raises "dimensions greater than samples" whenever there are more draws than
   parameters. That is always the case, so without scikit-learn the method
   fails outright.
4. The method computes $\hat N=\langle e^{\ell}/\hat p\rangle=\int e^\ell$, which
   is the evidence times the prior volume, and discards it.

**What it costs aure.** Nothing: [docs/evidence.md](docs/evidence.md) §3.4
explains why aure uses bridge sampling instead. It costs bumps users who call
`state.entropy()`.

**The change.** Report upstream with the four fixes, and propose returning
$\hat N$ alongside the entropy.

---

The next four entries come from one reading of the 51-curve copper series, made
for the reflbench question set (its `docs/sweep-habits.md` has the full tables).
Sweeps: gpt-5.4, gpt-6.1-sol and gpt-6-luna on v0.9.0; gpt-6-luna on 2c05e66
with `amoeba` and with `de` at `-m 10`; gpt-5.4 on 1097fdb (30 curves scored).
Counts of what an evaluation "cites" are keyword matches on its issues and
suggestions.

## The critical-edge hint tells the model to put the solvent near 7.4

**Where:** [`src/aure/tools/feature_tools.py`](src/aure/tools/feature_tools.py)
— `format_critical_edge_line` and `extract_critical_edges`;
[`src/aure/nodes/analysis.py`](src/aure/nodes/analysis.py) —
`implied_ambient_sld_from_edge` and the hint built from it.

**What is wrong.** The evaluation and modeling prompts carry this line whenever
the edge-derived ambient SLD is more than 1.0 from the stated one:

> … this implies an ambient SLD ≈ 7.4, far above the H-form (~0): the ambient is
> very likely DEUTERATED. Constrain the ambient SLD near 7.4, not across the
> full H–D range.

Three things fail on the copper series:

- The premise. The docstring says a film "does not set the edge; the edge is
  fixed by the substrate/ambient contrast". These samples have 500–565 Å of
  copper at 6.55, above the solvent, and the edge is the copper's. Three of the
  four protonated-THF curves (reference solvent 0.19) show it: the hint reads
  7.1–7.6 and calls the ambient deuterated.
- The number. Copper against silicon gives Qc = 0.0150; the estimator returns
  0.0156–0.0166 on the five curves checked, so even as a copper value it is
  high.
- The trigger. The brief already says d-THF, about 6.0. The hint was written to
  catch a solvent stated as protonated, but `matches_stated` compares the stated
  6.0 with the biased estimate and fails by 1.0–1.5.

**What it costs.**

| | |
|---|---|
| curves carrying the hint | 35 of 51 |
| value on the 32 deuterated ones | 7.11–7.66, against references of 6.0–6.3 (1.04–1.53 too high, median 1.21) |
| evaluations citing the critical edge | 66–85 % in every gpt-5.4 and gpt-6-luna sweep (gpt-oss-120b: 25–31 %) |
| runs where the model raised the solvent's upper bound to 6.9 or more | 35 of 51 (gpt-5.4, v0.9.0); 38 and 40 of 51 (luna, v0.9.0 and 2c05e66) |
| luna on 2c05e66: passes spent on a solvent hypothesis | 15, of which 4 improved χ² |

On 206922 (luna, 2c05e66) the first refinement set the solvent to 6.5–8.5 and
χ² went from 4.53 to 9.28. This is the likely source of the windows listed
under "The solvent's SLD window follows the last fit" (207027 at 6.5–8.0,
201290 at 6.40–8.40), and it adds to the Cu_G failures, where the description
already says "likely deuterated".

**The change.**

1. Do not emit the instruction when the stated ambient is already a deuterated
   form (within 1.0 of the solvent's tabulated `d_sld`). The question the hint
   answers is then settled.
2. Never name a target above the named solvent's `d_sld`. Say "consistent with
   the deuterated form" and leave the number out.
3. Before attributing the edge to the ambient, check the parsed stack: a layer
   a few hundred Å thick with an SLD above the stated ambient sets the edge
   itself. Skip the hint, or attribute it to that layer.
4. Check the estimator on a simulated curve with this resolution; 0.0156–0.0166
   against 0.0150 suggests the half-height rule reads high here.

**Verifying a fix.** Across the 51 curves no prompt line names an ambient SLD
above 6.4, and the four Cu_G curves carry no deuteration hint.

---

## "If parameter tweaks have stopped making meaningful progress" is read as a precondition

**Where:** [`src/aure/nodes/prompts.py`](src/aure/nodes/prompts.py) — the task
paragraph of `FIT_EVALUATION_PROMPT`;
[`src/aure/skills/structural-hypothesis-ranking/SKILL.md`](src/aure/skills/structural-hypothesis-ranking/SKILL.md)
— "walk that list in rank order when parameter-only refinement stops making
progress".

**What is wrong.** The sentence allows a structural change once tweaks have
stalled. At the first evaluation no tweak has been tried, so a literal reader
tweaks first, whatever the fit shows. The prompt also gives the trajectory but
not the pass number or the budget, so the model cannot see what a tweak costs.

**What it costs.** v0.9.0, five passes, same description:

| | gpt-5.4 | gpt-6.1-sol | gpt-6-luna |
|---|---|---|---|
| first evaluation of a failing baseline | structural 47/47 | tweak 47/47 | structural 25, tweak 22 |
| all evaluations, structural / tweak | 181 / 1 | 50 / 145 | 90 / 102 |
| runs with no structural change | 4 | 17 | 7 |
| good | 40 | 29 | 36 |

Sol states the reason: tweaks have "not yet demonstrated stagnation". Of its 34
runs that changed the structure, 33 did so at the third evaluation or later,
which leaves one or two tries. Its structural changes improved χ² 37 times in
46, its tweaks 89 in 141. gpt-5.4 ignores the sentence the other way and never
consolidates a structure that is nearly right, so the sentence steers neither
model as intended. It is unchanged on main.

A controlled test in reflbench (`questions/aure-01`, `aure-04`): v0.9.0's own
first-evaluation prompt for 206976, with a planted baseline fit at χ² 6.01, 30
repeats per model. gpt-6.1-sol tweaks in 30 of 30, citing the precondition
("Parameter-refinement stagnation has not been established, so a targeted
parameter refit should precede adding another layer"); gpt-5.4 changes the
structure in 30 of 30; gpt-6-luna tweaks in 11 of 30, all 11 with the same
reason. Swapping only the BIC paragraph for 2c05e66's changes little (Luna 8 of
30, Sol and gpt-5.4 unchanged), so the precondition, not the BIC wording, is
what holds the change back.

Partly addressed by the non-interactive note in the evaluation prompt: on AuRE
main's own prompt for the same fit, gpt-6.1-sol went from 0 to 15 of 15
structural changes with the note ("stagnation is not established.
Nevertheless, the user's explicit instruction ... applies now"), and gpt-6-luna
from 13 to 15. That covers a run whose user hypothesis says what to do when the
fit fails; a run without one still meets the precondition as written.

**The change.** Say what justifies a structural change without a stall:
residual fringes at a thickness the model lacks, a layer on a thickness or SLD
bound in the direction a pending hypothesis explains, or a user instruction
conditional on the fit failing. Reserve a tweak for a named parameter at a
bound the structure does not explain. Print "evaluation k of N". This changes
results for every run, so it needs the same care as other prompt changes
against the `validation/` baseline.

**Verifying a fix.** Sol on the cycled Cu_1 curves makes its first structural
change at the first evaluation.

---

## A passing fit is rarely accepted once the prompt carries warnings

**Where:** [`src/aure/nodes/prompts.py`](src/aure/nodes/prompts.py) — the
acceptance block of `FIT_EVALUATION_PROMPT`;
[`src/aure/nodes/evaluation.py`](src/aure/nodes/evaluation.py) — the issues
appended to a fit (`_check_boundary_hits`, the profile-excursion line);
[`src/aure/skills/interface-swallowed-layers/SKILL.md`](src/aure/skills/interface-swallowed-layers/SKILL.md).

**What is wrong.** The prompt says a fit within the threshold "should be marked
acceptable: true", and in the same prompt lists warnings without saying which
of them block acceptance. `interface-swallowed-layers` is a rule about quoting
a layer's SLD; in the evaluation context it reads as a defect of the fit.

**What it costs.** Fits evaluated at χ² ≤ 1.5, and how often they were accepted:

| | accepted |
|---|---|
| gpt-5.4, v0.9.0 / 1097fdb | 38 of 39 / 20 of 22 |
| gpt-6-luna, v0.9.0 | 22 of 42 |
| gpt-6-luna, 2c05e66, `amoeba` | 7 of 26 |
| gpt-6-luna, 2c05e66, `de`, ten passes | 11 of 143 |
| gpt-oss-120b, v0.9.0 | 0 of 27 |

In luna's 132 refusals with ten passes, the issues name a parameter at a bound
(132), a profile artifact (114), the solvent (107), the swallowed-layer
criterion (101) and the critical edge (87). It then made 40 structural changes
on fits that already passed; 201236 under "The profile veto" is one result.

The swallowed-layer criterion is cited in 125 of 210 luna evaluations on
2c05e66 (1 of 214 on v0.9.0) and 67 of 132 for gpt-5.4 (0 of 220). Luna asks
for "profile-based observables" in 42 evaluations, which the loop cannot act
on. gpt-5.4 drew 36 hypotheses from the skill, such as "Remove the discrete
CuOx slab and represent the Cu surface only by roughness". Luna on the same
optimiser went from 36 good to 26; not all of that is this.

**The change.** Mark each warning in the prompt as blocking or as a note.
Blocking: a profile excursion outside the range of the layer and its
neighbours (item 3 of the veto entry), and a violated user constraint. Notes: a
bound that was hit and already auto-expanded, and the swallowed-layer
criterion. For the skill itself, see the next entry.

**Verifying a fix.** Luna on the pristine curves accepts the first fit that
passes without a blocking warning.

---

## `interface-swallowed-layers` reads as an acceptance test, and most expert fits fail it

**Where:** [`src/aure/skills/interface-swallowed-layers/SKILL.md`](src/aure/skills/interface-swallowed-layers/SKILL.md);
[`src/aure/nodes/evaluation.py`](src/aure/nodes/evaluation.py) —
`load_skill_context`, which puts every active skill in the evaluation prompt.

**What is wrong.** The criterion's arithmetic is right: when
σ_top + σ_bot > t/2, the profile never reaches the layer's nominal SLD. What
the skill concludes from it is a reporting rule ("If the criterion fails, do
not quote the layer's `rho`"). But it never says what failing means for the
fit, and its framing reads as a verdict on the fit: "This failure is silent. It
survives a good chi-squared", "That single unexamined number destroyed
218386's oxide", "reparameterise". Its worked case is an interface width that
was never a parameter and sat at a 20 Å default. The skill does not say that
the urgency belongs to that case, not to widths the data have fitted.

**What it costs.** A controlled test in reflbench (`questions/assess-06`): the
pristine 201136 baseline fit (χ² 1.08, flat residuals; the expert accepts it)
asked with and without this skill in the prompt, three repeats each.

| | without the skill (assess-04) | with it (assess-06) |
|---|---|---|
| gpt-6.1-sol | accepts 1 of 1 | 0 of 3: refits with the oxide's interfaces capped to pass |
| gpt-6-luna | accepts 1 of 1 | 0 of 3: the same |
| gpt-oss-120b | accepts 1 of 1 | 0 of 3: replaces the oxide with a plated layer |
| gpt-5.4 | accepts 1 of 1 | 3 of 3, though it calls the oxide "well resolved" while computing that it fails |

Luna's reason, in its own words: "the fit is not acceptable for interpreting
the oxide as a resolved slab, despite its good χ²".

The skill's mirror case also goes wrong (`assess-07`): the merged 272 Å
"copper oxide" at SLD 1.96 that every model rejects without the skill passes
the criterion. With the skill, gpt-5.4 accepts it in 2 of 3 and gpt-oss-120b
in 3 of 3, citing the pass.

The criterion fails on most correct fits of this series. In the expert
reference fits, the layer directly on the copper fails it in 40 of 50, the SEI
in 18 of 36 and the native SiO₂ in 13 of 36. The reference for 201136 fails it
like the fit above: a 55.7 Å oxide between 19.0 and 42.0 Å interfaces, with a
profile minimum of 4.98 against a nominal 4.35.

Doing what Sol and Luna ask costs the fit. Refitting 201136 with the oxide's
interfaces capped at 10 and 20 Å, so that the criterion passes, gives χ² 1.56,
outside the threshold, and moves the oxide from 63 Å at 4.57 to 77 Å at 5.12.
With caps of 15 and 25 Å, χ² is 1.21 and the criterion still fails. The data
want the 41 Å oxide/solvent interface.

**The change.**

1. Add a section to the skill, "What a failed criterion does not mean". It is
   a caveat on the quoted SLD, not a defect of the fit. It is not a reason to
   withhold acceptance, to cap or tie widths the data have fitted, or to change
   the structure. Quote the profile's extremum or an integrated contrast
   instead. Passing the criterion says the slab is resolved, not that it is the
   material it is named.
2. Put the urgency where the worked case puts it: an interfacial width that is
   an unfitted default.
3. Load the skill where SLDs are reported (finalize, the report) rather than
   into the evaluation prompt, or load it there only with item 1 in place.

**Verifying a fix.** In reflbench, `assess-06` scores like `assess-04` and
`assess-07` like `assess-03` for every model.
