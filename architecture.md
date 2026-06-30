# Architecture: AuRE

This document records the design decisions behind AuRE so they are not accidentally undone.
AuRE is an LLM-driven autonomous workflow for fitting neutron-reflectometry data with refl1d,
and for exporting the results as AI-ready records.

## 1. What AuRE does

Given reduced reflectivity data + a natural-language sample description (and optional
hypothesis), AuRE runs an agentic pipeline — **intake → analysis → modeling → fitting →
evaluation → refinement** (LangGraph nodes in `src/aure/nodes/`, orchestrated by
`workflow/runner.py`) — to produce a fitted refl1d model with uncertainties. It is driven
three ways: a **web UI** (Flask, `web/`), a **CLI** (`analyze` / `batch`, `cli.py`), and an
**MCP server** (`mcp_server.py`).

## 2. Domain concepts (two orthogonal axes)

- **State** — one physical condition of the sample (e.g. "in D2O at OCV"). A state is measured
  at one or more incident angles; each angle is a **partial** (its own run/file). Partials of a
  state share a set id (the `REFL_<setid>_<seq>_<run>_partial.txt` convention) but are distinct
  runs. A state may instead be a single **combined** file.
- **Co-refinement** — a *fitting strategy* that ties parameters across multiple files/states.
  - *Single-state co-refinement*: several Q-segments/angles of one state, structurally tied.
  - *Multi-state co-refinement*: several states (e.g. D2O / H2O) sharing structural parameters
    while differing in others.

These are independent: state = data grouping; co-refinement = how the fit ties parameters.
The canonical config shape is `states: [{name, data_files, extra_description, …}]` plus
`shared_parameters` / `unshared_parameters` (see `config.py`, `setup.py`, `state.py`).

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

> Known gap: a sharing instruction written only in the free-text `sample_description` is **not**
> auto-extracted into `shared/unshared_parameters` (no NL→ties parser). Use the structured ties
> panel / setup block; the description alone will fall back to the Auto default.

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

## 7. Code map

- `nodes/` — pipeline nodes (intake, analysis, modeling, fitting, evaluation, refinement).
- `workflow/` — `runner.py` (orchestration), `checkpoints.py` (run dir + `run_info.json`).
- `config.py` / `setup.py` / `state.py` — states, ties, and setup-YAML (de)serialization.
- `refl1d_import.py` / `nodes/model_builder.py` — build the refl1d problem; cross-state tying.
- `web/` — Flask routes (`routes.py`), `static/setup.js`, templates.
- `exporters/isaac.py` — the ISAAC export pipeline.
