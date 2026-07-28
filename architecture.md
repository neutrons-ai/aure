# Architecture: AuRE

This document records the design decisions behind AuRE so they are not accidentally undone.
AuRE is an LLM-driven autonomous workflow for fitting neutron-reflectometry data with refl1d,
and for exporting the results as AI-ready records.

## 1. What AuRE does

Given reduced reflectivity data + a natural-language sample description (and optional
hypothesis), AuRE runs an agentic pipeline — **intake → analysis → modeling → fitting →
evaluation → refinement** (nodes in `src/aure/nodes/`, orchestrated by the hand-written
state machine in `workflow/runner.py`) — to produce a fitted refl1d model with
uncertainties. It is driven
three ways: a **web UI** (Flask, `web/`), a **CLI** (`analyze` / `batch`, `cli.py`), and an
**MCP server** (`mcp_server.py`).

## 2. Domain concepts (orthogonal axes)

- **State** — one physical condition of the sample (e.g. "in D2O at OCV"). A state is measured
  at one or more incident angles; each angle is a **partial** (its own run/file). Partials of a
  state share a set id (the `REFL_<setid>_<seq>_<run>_partial.txt` convention) but are distinct
  runs. A state may instead be a single **combined** file.
- **Co-refinement** — a *fitting strategy* that ties parameters across multiple files/states.
  - *Single-state co-refinement*: several Q-segments/angles of one state, structurally tied.
  - *Multi-state co-refinement*: several states (e.g. D2O / H2O) sharing structural parameters
    while differing in others.
- **Sample identity** (`ModelDefinition.distinct_sample`, default `False`) — whether the
  co-refined states are *one* physical sample under several conditions (shared sample identity
  downstream) or *distinct* physical samples that merely share a fitting strategy. Identity only,
  no physics; threaded into `run_info.json` for the data-assembler to assign one shared
  `sample_id` (default) or one per state.
- **Per-state structure ("sample ≠ structure")** — a single sample can change its layer stack
  between states (e.g. a surface oxide present in air but gone in electrolyte). `StateDefinition`
  may carry its own complete `layers`/`substrate`; absent ⇒ inherit the model-level template.

These are independent: **state** = data grouping; **co-refinement** = how the fit ties
parameters; **sample identity** = how many physical samples; **structure** = each state's stack.
The canonical config shape is `states: [{name, data_files, extra_description, layers?, …}]` plus
`shared_parameters` / `unshared_parameters` / `distinct_sample` (see `config.py`, `setup.py`,
`state.py`).

## 3. Cross-state parameter ties (shared vs unshared)

How structural parameters are tied across states is a first-class, **user-controlled** input —
never guessed from a default alone:

- **Auto** (no `shared_parameters`/`unshared_parameters`): the default tied set
  (`model_builder._DEFAULT_TIED_LAYER_ATTRS` = thickness / SLD / interface for every layer,
  plus `substrate.interface`) is tied across states.
- **Shared (whitelist)**: tie only the named parameters.
- **Unshared (blacklist)**: tie the default set *minus* the named parameters.

Flow: the UI "Cross-state parameter ties" panel (or a setup-YAML `shared_parameters` /
`unshared_parameters` block) → `user_config` → `modeling._attach_state_metadata` →
`model_def` → `model_builder._resolve_tied_set` → `build_states_problem` aliases the tied
Parameter objects across experiments. **Invariant:** a parameter the user marks unshared must
NOT be aliased across states.

The default tied set and name-validation run over the **union** of every state's layer names, so
a layer present in only some states is fine; a tie referencing a layer absent from a given state
simply does not apply there (logged, never a hard fail). When a refine removes a layer,
`modeling.prune_tie_specs` drops the now-dangling tie before the fit (regression: this used to
abort with "unshared_parameters references unknown layer").

When the user supplies no structured ties, `modeling._extract_cross_state_unshared` derives
unshared parameters from the free-text `sample_description` (e.g. "the oxide differs between
states"), and `modeling._extract_per_state_structure` derives per-state structural differences
(e.g. "the H2O state has no oxide" → that state's `layers` = template minus the oxide). Both are
LLM-driven, validated against the layer names, and no-op without an LLM. The structured ties
panel / setup block still wins when present.

### Per-state structure on refine
A refine ("remove the oxide on state 2") edits *that state's* `layers` inside `states[]`, not the
top-level template (which would change every state). `modeling._refine_model` merges only the
structural keys (`layers`/`substrate`) the LLM emits per state by name, preserving the runtime
metadata intake attached. `refl1d_import` round-trips heterogeneous co-refinements: it emits a
state's own `layers` when its refl1d stack diverges from the base, and recovers cross-state ties
by **layer name** (not stack position, which shifts when a state drops a layer).

Decision (bugfix): cross-state ties are collected in `setup.js:_buildAnalysisBody` **outside**
the file-grouping branches, so they survive both **Start Analysis** and **Save Setup**
(`/api/setup/export`) regardless of grouping-UI state (previously they were dropped after a
setup load). The backend round-trips them via `setup._setup_from_dict` / `dump_setup`.

## 4. The run directory (AI-ready provenance)

Each run writes a self-describing directory (`workflow/checkpoints.py`):

```
<output_root>/<run_name>/
  run_info.json        # run_id, sample_description, hypothesis, data_files, states[]
  checkpoints/NNN_<node>.json
  refl1d_output/        # problem.json (the fitted model), *-err.json (σ), *-refl.dat
  final_state.json      # χ², per-state conditions
```

**`run_info.json` `states[]` is the contract for the downstream store.** It carries the
user-given state names + each state's `data_files` + `extra_description` (conditions), so the
data-assembler can group runs per state without parsing file names. Single-state runs may omit
`states[]` (the flat `data_files` is one state).

### Output folder naming (`web/routes.py:_derive_run_dir_name`)

- Single run → its run number; if unnumbered, the **state name** (not a bare filename stem).
- Co-refinement → the **per-state primary run numbers joined** (e.g. `230539_230543`), so a
  multi-state fit's folder includes every state's run and **never overwrites an existing
  single-state fit** named by one run.

## 5. ISAAC export (the AI-ready handoff)

AuRE does not own the ISAAC schema; it delegates (`exporters/isaac.py`):

```
data-assembler ingest-workflow <run_dir> -o <ingest> --json   # run dir → run-level store
nr-isaac-format convert-ingest <ingest> -o <out>              # store → ISAAC record(s)
```

The data-assembler turns the run dir into a **run-level store** (one reflectivity record per
run + one fit + per-state environments, all linked by foreign keys). `convert-ingest` groups
runs by `(sample_id, environment_id)` and emits **one ISAAC record per state** (each state's
partials become that record's `measurement.series`). The exporter targets a *directory* and
validates every JSON, so it handles N records transparently. See the data-assembler and
nr-isaac-format `architecture.md` for the store/representation contract.

## 6. Invariants not to break

1. **User-controlled ties.** `shared/unshared_parameters` from the UI/setup must reach
   `_resolve_tied_set` and must survive Save Setup. Never silently default when the user
   specified ties.
2. **`run_info.states[]` is authoritative** for multi-state grouping downstream — populate it
   (names + per-state `data_files` + conditions). Never make the data-assembler infer states
   from file names.
3. **Co-refinement output folders include all states' runs** — never reuse a single-run name
   for a multi-state fit.
4. **ISAAC is delegated**, not reimplemented: export goes through `ingest-workflow` →
   `convert-ingest`. Keep AuRE's coupling to those CLIs (the `export` extra) thin.
5. **The reported model is chosen in the runner's terminal block**, not the last
   loop iteration and not a graph edge: `finalize` selects (lowest χ² +
   parsimony tie-break) and reports the untried-hypothesis backlog, the optional
   `final_fit` polishes for uncertainties, then `save_final_state` writes
   `final_state.json` + `problem.json`. `best_model`/`best_chi2` stay the loop's
   regression baseline. See [docs/finalization.md](docs/finalization.md).

   Selection **sets aside the fits `evaluation` would not have accepted** before
   ranking on χ², in tiers: profile-vetoed fits last (physically impossible), and
   sub-floor ones above them (plausible, but the χ² describes the `dR` column
   rather than the structure). Ranking on χ² alone reported exactly the model
   `evaluation` had rejected — the excursion is often what *buys* the low χ², and
   an overfitted iteration is likewise the kind that scores lowest. It reads the
   `profile_artifact` flag `evaluation` stamps on each judged `FitResult` and the
   floor from `state["chi2_min"]`. Both verdicts travel to every surface that
   renders the answer — report, `--json`, `aure batch`, the web tabs, `final_fit`'s
   gates and the ISAAC export — so keep it that way when adding another.
6. **In `evaluation`, the SLD-profile artifact check runs BEFORE the χ² acceptance
   clamp.** A finite χ² inside the acceptance window `CHI2_MIN ≤ χ² ≤ CHI2_MAX`
   deterministically forces `acceptable=True` (`_clamp_acceptance_to_chi2`) so
   the loop stops reproducibly rather than at the LLM's discretion. The clamp
   raises a verdict (`False → True`) and never lowers one, so above `CHI2_MAX`
   the LLM decides and the profile veto is the only thing that can lower an
   accept.

   The clamp decides by reading the two markers the artifact check leaves behind
   (`_profile_artifact`, `_profile_checked`) and **stands down** — declining to
   force acceptance, so the LLM decides, as before the clamp — on a vetoed fit,
   on an unverified one (no exported SLD profile, the detector declining with
   `checked=False`, or a co-refinement where any one state reported no profile —
   every state is checked against its own effective media, and partial coverage
   leaves the whole fit unverified), on one whose per-file/per-state χ² fails on its
   own, and below `CHI2_MIN` (setup key `chi2_min`, default `0.5`, `0` disables),
   where a reduced χ² far under 1 is evidence about the error model — an
   overestimated `dR` column, or free parameters absorbing the noise — rather
   than about the structure. All four are stand-downs, not vetoes.

   Ordering the clamp above the check type-checks and passes a smoke test, but
   both markers are then unset at clamp time and "not checked" means stand down:
   the clamp becomes dead code and the χ² stop silently disappears. It flips to
   the opposite failure — accepting impossible profiles — the moment anything
   sets `_profile_checked` earlier than the check itself, which is why that
   marker must stay the check's own positive statement. See
   [docs/approach.md](docs/approach.md) §4.5.

## 7. Code map

- `nodes/` — pipeline nodes (intake, analysis, modeling, fitting, evaluation, refinement).
  The terminal `finalize` (select the reported model; also emit the
  untried-improvements report) and optional `final_fit` (MCMC uncertainty polish)
  run in the runner's terminal block —
  see **[docs/finalization.md](docs/finalization.md)**.
- `workflow/` — `runner.py` (the state-machine orchestrator + terminal
  finalize/final_fit/save; the single execution engine for CLI, web UI, and MCP),
  `checkpoints.py` (run dir + `run_info.json` + `final_state.json`/`problem.json`).
- `config.py` / `setup.py` / `state.py` — states, ties, and setup-YAML (de)serialization.
- `refl1d_import.py` / `nodes/model_builder.py` — build the refl1d problem; cross-state tying.
- `web/` — Flask routes (`routes.py`), `static/setup.js`, templates.
- `exporters/isaac.py` — the ISAAC export pipeline.
