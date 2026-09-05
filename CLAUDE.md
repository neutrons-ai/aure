# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AuRE (Automated Reflectivity Evaluator) is an LLM-driven agent that fits neutron / X-ray reflectivity data with [Refl1D](https://refl1d.readthedocs.io). It accepts a raw data file plus a plain-English sample description and produces a fitted layer model. The orchestration layer is a hand-written state machine ([src/aure/workflow/runner.py](src/aure/workflow/runner.py)) — no external graph framework; LangChain is used only for the LLM calls. The science (probe loading, model building, optimization, χ²/BIC) is delegated to refl1d/bumps.

The longest-form design rationale lives in [docs/approach.md](docs/approach.md) — read it before making architectural changes to the workflow. When a change turns up an unrelated defect, record it (an issue, a `TODO` at the site) rather than growing the change to cover it: a fix that also repairs everything it touched on the way is not reviewable.

## Environment & install

- Python ≥ 3.10 (CI runs 3.10 and 3.12). Source layout (`src/aure`), installed editable.
- Standard dev setup:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e ".[dev]"           # add ,export as needed
  ```
- LLM config is read from environment / `.env` (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_TIMEOUT`, `FIT_METHOD`, …). See [.env.example](.env.example) for the full list. Provider dispatch is in [src/aure/llm/providers/](src/aure/llm/providers/); `aure check-llm` validates the live config.

## Commands

- Tests: `pytest tests/` (config in `pyproject.toml` adds `--cov=aure` and an html coverage report). Single test: `pytest tests/test_workflow.py::test_name -v`.
- Lint/format: ruff is wired via pre-commit (`ruff-check --fix --ignore=E741,E402` then `ruff-format`). Install hooks with `pre-commit install`; run on demand with `pre-commit run --all-files`. Other hooks: yamllint, taplo (TOML), gitleaks.
- CLI entry point: `aure …` (defined in `[project.scripts]` → `aure.cli:main`). Top-level commands: `analyze`, `prepare`, `batch`, `resume`, `checkpoints`, `inspect-checkpoint`, `evaluate`, `import-refl1d`, `plot-results`, `extract-features`, `lookup-sld`, `list-materials`, `mcp-server`, `serve` / `interactive`, `check-llm`. Full reference is in [README.md](README.md).
- Validation harness (separate package): `python -m validation.cli compare|chi2|diagnose` — compares fitted runs against reference models. See [validation/](validation/).

## Architecture

### Workflow state machine ([src/aure/workflow/runner.py](src/aure/workflow/runner.py))

A hand-written state machine over `ReflectivityState` (a `TypedDict` in [src/aure/state.py](src/aure/state.py)). `run_workflow_with_checkpoints` iterates `NODE_ORDER` and follows the `ROUTING_FUNCTIONS` to pick the next node:

```
START → intake → analysis → modeling → fitting → evaluation ─┐
                                ▲              │              │
                                │              └─ (bounds) ──┘
                                └─ (refine model) ───────────┘
                                                              │
                                                            COMPLETE / END
```

Each node lives in [src/aure/nodes/](src/aure/nodes/) and returns a state-delta dict:

- `intake` — load probe, LLM-parse the sample description into `ParsedSample` (substrate / layers / ambient / hypothesis); a second LLM call from the `structural-hypothesis-ranking` skill produces a ranked list of candidate structural modifications the refinement loop may try. The user's `-h` hypothesis is folded into that list as top-ranked `origin="user"` entries (and is **not** baked into the baseline structure); skill-enumerated entries are `origin="skill"`. The data file's free-form run title (`# Run title: …`) is **always** extracted deterministically (regex, no LLM) onto each `DatasetInfo.run_title` and checkpointed, but only *interpreted* when `USE_RUN_TITLE` is enabled — then it seeds `origin="header"` entries ranked between `user` and `skill`, never a correction to the parsed baseline, so a wrong title is rejected by the regression guardrail rather than poisoning every iteration. Conflicting titles among one state's files are reported and discarded, not merged.
- `analysis` — deterministic feature extraction (critical edge → substrate SLD, Kiessig fringe spacing → total thickness, etc.). No LLM.
- `modeling` — LLM generates / refines a `ModelDefinition` JSON. When entered from `evaluation`, the LLM is told whether to do a parameter tweak or realize a specific structural hypothesis.
- `fitting` — builds a `bumps` `FitProblem` via `nodes.model_builder.build_problem()` and runs refl1d's optimizer (`lm` / `de` / `dream`). Optional **thin-layer SLD mode enumeration** (env `MODE_ENUMERATION=1`, single-file only): before the main fit, re-seed each layer thinner than `THIN_LAYER_MODE_K × 2π/Q_max` across discrete SLDs, cheap-polish each, and start from the best basin (escapes the Δρ·t-ridge local minima a single run can't cross). Off by default, logged, never fatal.
- `evaluation` — LLM judges fit quality (χ², BIC, residual structure, parameter sanity) and chooses next route. Has **deterministic regression guardrails**: if χ² or BIC got worse after a refinement, the previous model is restored and the attempted hypothesis is marked rejected. A deterministic **SLD-profile artifact check** (`tools.feature_tools.detect_profile_artifacts`) flags non-physical erf-tail excursions (profile leaving the range its bounding media can produce — a χ²-invisible defect) as an issue with a two-branch remedy suggestion (tie the roughness, or re-label as a profile parametrization) and vetoes acceptance; the σ/thickness ratio is surfaced as an informational concern only.
  It then applies the **deterministic χ² stop** (`_clamp_acceptance_to_chi2`): a finite χ² inside `chi2_min ≤ χ² ≤ chi2_max` forces `acceptable = True`, so the loop stops instead of re-litigating a passing fit. One-directional — only `False → True`, so *above* the ceiling the LLM's verdict still decides and none of the guards below are consulted. It **stands down** (declining to force acceptance; the LLM's verdict decides, as before the clamp) on a vetoed profile (`_profile_artifact`), on a profile that was not *verified* (`_profile_checked` unset: no exported profile as in library/MCP runs, detector returned `checked=False`, or a co-refinement where any one state reported no profile — every state is checked, each against its own effective media, and partial coverage leaves the whole fit unverified), on a per-file/per-state χ² over the ceiling or carrying the `+inf` fit-failed sentinel, and below `chi2_min` (default 0.5, `0` disables — a reduced χ² far under 1 is evidence about the `dR` column, not about the structure, so it must not read as a pass; the node also records it as an issue and `prompts._format_chi2_floor` tells the evaluator what it implies). `_simple_evaluation` never asserts acceptance, so the clamp is the single acceptance point. **Ordering invariant:** the clamp must run *after* `_detect_profile_artifacts_into`, whose two markers it reads — hoisted above, both are unset, "not checked" means stand down, and the χ² stop silently becomes dead code. `CHI2_MAX` / `CHI2_MIN` (setup keys `chi2_max` / `chi2_min`) are pinned into `state` by the runner on the first pass, so a resume keeps the window it was launched with.
  When the fit isn't acceptable **and** there's a signal worth it (residual fringes, χ² stalled ≥2 iters, or no `pending` hypotheses left), it runs a **gated revision step**: re-selects skills from the observed artifacts (`select_skills(..., extra_context=…)`), proposes new `origin="evaluation"` hypotheses, and re-ranks the list.
- `routing.*` — pure functions returning the edge name; no state mutation.

Routing lives in `nodes/routing.py` (`route_after_*`, pure functions). After the loop breaks, a terminal block in the runner runs `finalize` (select the reported model — profile-vetoed fits set aside, then **lowest χ² with a parsimony tie-break** among fits within `FINAL_SELECTION_TOL`; plus a second, reporting-only message listing the still-`pending` hypotheses and what was attempted, which `cli` re-renders for the `analyze` report and `aure batch` through the shared `pending_hypotheses` / `hypothesis_label` / `format_attempted_counts` helpers) and the optional `final_fit` (uncertainty polish) — see [docs/finalization.md](docs/finalization.md). Selection **sets profile-vetoed fits aside** first (reading the `profile_artifact` flag `evaluation` stamps on each judged `FitResult`), reporting one only if every fit was vetoed — the excursion buys χ², so ranking on χ² alone reported the model `evaluation` rejected. Sub-floor fits are set aside the same way, one tier above vetoed ones (physically plausible but its χ² describes the `dR` column, versus physically impossible), so the ladder is clean-and-in-window → clean-but-sub-floor → vetoed; `final_fit` skips the polish for either. The veto reaches every surface that renders the answer — the report, `--json`, `aure batch`, both web tabs, `final_fit`'s gate and the ISAAC export's context/warnings — via the `profile_artifact` flag and `final_selection`; a new such surface is expected to disclose it too. `run_prepare` (and the `prepare` command / `aure batch` `prepare` mode) stops after `modeling`, writing a `problem.json` consumable directly by refl1d.

### Models are JSON, not scripts

The historical design generated Python refl1d scripts; the current design stores models as `ModelDefinition` dicts (see `state.py`) and builds refl1d `Experiment`/`FitProblem` objects on-the-fly in [nodes/model_builder.py](src/aure/nodes/model_builder.py). When extending model semantics, change the dict schema + builder together — don't reintroduce script-string round-tripping.

### Reparametrization: `derived_parameters` ([nodes/expressions.py](src/aure/nodes/expressions.py), `model_builder.apply_derived_parameters`)

User-facing guide: [docs/derived-parameters.md](docs/derived-parameters.md). **Off by default** — gated by `allow_derived_parameters` / `ALLOW_DERIVED_PARAMETERS` (`config.derived_parameters_enabled`), because a reparametrized model asks more of the LLM than a plain stack. Declaring the block without the gate is a hard error; with the gate off the refinement rule and the `functional-constraints` skill never reach a prompt.

A `ModelDefinition` may declare `derived_parameters`: each entry adds one **free** parameter and derives raw ones from it through `assign` (`"<layer>.<attr>" -> expression`), so the fit explores a *combination* — a surface excess `(ρ−ρ_ambient)·t`, a volume fraction — instead of coordinates the data does not resolve. `keep_physical` entries become bumps `Constraint` objects guarding the derived value, which has no bounds of its own. Expressions are evaluated by a whitelisted-AST evaluator (`expressions.evaluate`), **never** `eval` — these strings can come from a config file or an LLM. Nothing does algebra: the inverse is written out in `assign`.

Consequences that are easy to get wrong:

- A raw parameter named in `assign` leaves the free set (bumps discovers parameters by traversal, and an expression is not one), so **`_count_free_params` adjusts** via `_derived_param_delta`: a one-for-one swap is BIC-neutral, solvation (two free, one derived) costs one.
- **Reported χ² is the data term only** (`model_builder.data_chisq`). `FitProblem.chisq()` scales `pmodel + pparameter + pconstraints`, and bumps returns `pmodel = 0.0` when a constraint fails — so a violated guard would report χ² = 0, reading as a perfect fit *and* landing under the acceptance floor. `data_chisq` returns `inf` for infeasible instead. Every surface that reports χ² uses it (fitting, cli, web, refl1d import).
- **Multi-state**: a declaration is *tied* by default — one parameter object across states, `assign` re-evaluated per state — which is what solvent contrast actually needs (the invariant is composition, not SLD, and it is not a layer attribute, so `shared_parameters` cannot express it). `tied: false` gives one copy per state; `states: [...]` scopes it. A slot owned by a reparametrization is excluded from tie aliasing **at both ends**, and the renaming pass treats it as untied so two states falling back to a free parameter cannot collide on one name.
- `save_problem_json` **refuses** a model with `derived_parameters`: bumps serialization does not round-trip expression parameters (the same limitation `roughness_tie` re-applies around), so the export would silently be a different model.
- The refine LLM does not know about the block, so `modeling` carries it over explicitly (config wins) — otherwise the first refinement would drop it and quietly revert to raw coordinates.
- A structural edit that removes a referenced layer **prunes** the declaration (`prune_derived_parameters`, mirroring `prune_tie_specs`) instead of failing the build — with the drop recorded in the transcript. Auxiliary declarations left unreferenced are pruned with it.

Declared in the setup/user config as a top-level `derived_parameters:` list; shape-checked in `config._parse_derived_parameters`, resolved against the built model by `model_builder.validate_derived_parameters`. Fully optional — absent means today's behaviour exactly.

### Co-refinement (multi-file and multi-state)

Two distinct co-refinement modes share the same workflow:

- **Multi-file (single physical sample)**: several data files of *the same physical sample* (e.g. spliced Q-segments) share one refl1d `Sample` so every layer parameter is tied automatically. Driven via `data_files=[…]` / `-d` flag / `data_files:` in a manifest. Built by `build_multi_problem`.
- **Multi-state (different physical states of one sample)**: several physical states — e.g. solvent contrast, anneal step, applied potential — each with their own data file(s) and `ambient`/`intensity`/`back_reflection`. Structural parameters are tied **across states** based on `shared_parameters` (whitelist) or `unshared_parameters` (blacklist); the default tied set is `thickness`, `material.rho`, `interface` per layer plus substrate `interface`. Driven via a `states:` block in the YAML config (`-c config.yaml`) or per-job in a manifest. Built by `build_states_problem` in [model_builder.py](src/aure/nodes/model_builder.py), which aliases shared `bumps.Parameter` objects across the per-state samples and renames untied params with a `"<state> "` prefix.

`state.iter_states(definition)` and `state.flatten_data_files(states)` are the canonical accessors — they synthesize a one-state view from the legacy single-file shape, so consumers don't need to branch. Intake refuses ambiguous flat multi-file invocations where files come from different REF_L set_ids and asks the user to migrate to `states:`. The `multi-state-corefinement` Agent Skill auto-activates whenever `len(states) > 1`.

### Importing refl1d fits ([src/aure/refl1d_import.py](src/aure/refl1d_import.py))

`aure import-refl1d REFL1D_DIR -o OUTPUT_DIR` ingests a hand-run refl1d `problem.json` into an AuRE workspace so it can be opened with `aure serve`, evaluated, or extended via `aure resume`. The module:

- `definition_from_problem(problem)` walks `problem.models`, groups experiments by sample identity (`id(sample)`), and reconstructs a `ModelDefinition` with explicit `states`. Tied-parameter detection uses Python identity on `bumps.Parameter` instances (which is the protocol `build_states_problem` follows). Material-name heuristics guess `back_reflection` orientation; `--back-reflection` overrides.
- `extract_fit_result_from_problem(problem, …)` — moved from `cli.py`; reused by `aure evaluate` and the importer.
- Probes are dumped to `<output_dir>/data/*.txt` so the imported run is self-contained.

### Checkpoints ([src/aure/workflow/checkpoints.py](src/aure/workflow/checkpoints.py))

State is serialized to JSON after every node into `output_dir/checkpoints/NNN_<node>.json`, plus `run_info.json` and `final_state.json`. `aure resume` reloads from any checkpoint; `aure inspect-checkpoint` dumps one; `aure checkpoints` lists them. When adding state fields, make sure they're JSON-serializable (numpy arrays are handled, but watch for refl1d/bumps objects — those are built on demand from the JSON model, never persisted directly).

### Agent Skills ([src/aure/skills/](src/aure/skills/))

Each skill is a directory containing a `SKILL.md` (Agent Skills spec format). `selector.select_skills(...)` chooses which to inject into LLM prompts based on the parsed sample (e.g. `polymer-films`, `metal-oxide-interfaces`, `solvent-contrast-matching`, `sei-layer-analysis`, `neutron-reflectometry`). The `structural-hypothesis-ranking` skill is special — it drives the initial hypothesis list used by the refinement loop, not the modeling prompt directly. That list is then mutated through a single guarded merge in [nodes/hypotheses.py](src/aure/nodes/hypotheses.py): `modeling` may only change entry *status* (membership-frozen), while `evaluation` may append new entries and re-rank. Skills are loaded via `SkillRegistry` and rendered into prompts in `nodes/prompts.py`.

### LLM layer ([src/aure/llm/](src/aure/llm/))

- `config.py` — reads env vars, returns a normalized config dict; supports `openai`, `gemini`, `local` (OpenAI-compatible).
- `providers/` — one module per backend; `get_llm()` dispatches.
- `timeout.py` — signal-based wrapper (`invoke_with_timeout`, raises `LLMTimeoutError`); the per-call timeout comes from `LLM_TIMEOUT`.
- Any OpenAI-compatible endpoint (a self-hosted server, or a remote facility inference API) is reached through the `local` provider with `LLM_BASE_URL` + `LLM_API_KEY`. AuRE deliberately carries **no** provider-specific credential code: obtaining and refreshing a facility token is the facility's tooling's job, not AuRE's.

### Web UI & MCP server

- `aure serve` ([src/aure/web/](src/aure/web/)) — Flask app, dual-mode: interactive setup (no arg) or viewer (`aure serve OUTPUT_DIR`). Three tabs: Setup / History (χ² progression) / Results (R(Q), SLD, live parameter editor with Refl1D-recomputed dashed "User" curve, ISAAC export).
- `aure mcp-server` ([src/aure/mcp_server.py](src/aure/mcp_server.py)) — FastMCP server (stdio or SSE) exposing the workflow to AI assistants.

### Setup file format ([src/aure/setup.py](src/aure/setup.py), [aure_config.example.yaml](aure_config.example.yaml))

A "setup" YAML describes ONE analysis run. It is the canonical format shared by:

- `aure analyze -c setup.yaml` (positional DATA_FILE / SAMPLE_DESCRIPTION become optional when the setup carries them)
- `aure batch setup.yaml` — flat single-job manifest (no `jobs:` wrapper required)
- `aure batch manifest.yaml` — each job entry is a setup, merged with the top-level `defaults:` block
- The web UI's Setup tab **Load / Save** buttons (`POST /api/setup/load`, `POST /api/setup/export`)

The schema is **states-only**. Top-level `data_file:` / `data_files:` are no longer accepted — every analysis declares its files inside a `states:` block, even a single-file analysis (`states: [{name: state0, data_files: [{file: ...}]}]`). The `aure analyze DATA_FILE` CLI positional remains for ad-hoc one-off runs and internally wraps the file in a synthetic `state0`.

Analyzer-compat synonyms: `describe:` / `description:` for `sample_description:`, and `data:` for `data_files:` inside a state — so output from analyzer's `plan-data` command loads straight into AuRE. A `metadata:` block is preserved verbatim on round-trip but otherwise ignored.

`aure.setup.load_setup` / `dump_setup` / `load_manifest` are the entry points; they reuse `_parse_states` from [config.py](src/aure/config.py) so all multi-state validation lives in one place.

### Optional `export` extra

The ISAAC AI-Ready Data exporter ([src/aure/exporters/isaac.py](src/aure/exporters/isaac.py)) depends on `nr-isaac-format`, fetched from GitHub. Guard imports so the core install (`pip install -e .`) doesn't require it.

## Conventions worth knowing

- All LLM calls go through `invoke_with_timeout(get_llm(), …)` — don't bypass the timeout wrapper, and don't call `langchain_*` clients directly from nodes.
- Workflow nodes mutate state **only** by returning a dict; never call `state.update(...)` in-place. The list fields that accumulate (`messages`, `fit_results`, `model_history`, `llm_calls`) are appended by `runner._merge_state_updates`, so returning `{"messages": [...]}` appends; every other key is overwritten.
- `route_*` functions in `nodes/routing.py` must be pure — the runner calls them to pick the next node and they must not have side effects.
- Pre-commit hooks are authoritative for formatting (ruff + taplo + yamllint). Don't hand-format files differently.
- Docker image (`ghcr.io/neutrons-ai/aure`) installs `[export]` and uses `aure` as ENTRYPOINT — changes to the CLI surface area are user-visible there.
