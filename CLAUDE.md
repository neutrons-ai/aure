# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AuRE (Automated Reflectivity Evaluator) is an LLM-driven agent that fits neutron / X-ray reflectivity data with [Refl1D](https://refl1d.readthedocs.io). It accepts a raw data file plus a plain-English sample description and produces a fitted layer model. The orchestration layer is [LangGraph](https://github.com/langchain-ai/langgraph); the science (probe loading, model building, optimization, χ²/BIC) is delegated to refl1d/bumps.

The longest-form design rationale lives in [docs/approach.md](docs/approach.md) — read it before making architectural changes to the workflow.

## Environment & install

- Python ≥ 3.9 (CI runs 3.12). Source layout (`src/aure`), installed editable.
- Standard dev setup:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e ".[dev,agent]"     # add ,alcf and/or ,export as needed
  ```
- LLM config is read from environment / `.env` (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_TIMEOUT`, `FIT_METHOD`, …). See [.env.example](.env.example) for the full list. Provider dispatch is in [src/aure/llm/providers/](src/aure/llm/providers/); `aure check-llm` validates the live config.

## Commands

- Tests: `pytest tests/` (config in `pyproject.toml` adds `--cov=aure` and an html coverage report). Single test: `pytest tests/test_workflow.py::test_name -v`.
- Lint/format: ruff is wired via pre-commit (`ruff-check --fix --ignore=E741,E402` then `ruff-format`). Install hooks with `pre-commit install`; run on demand with `pre-commit run --all-files`. Other hooks: yamllint, taplo (TOML), gitleaks.
- CLI entry point: `aure …` (defined in `[project.scripts]` → `aure.cli:main`). Top-level commands: `analyze`, `batch`, `resume`, `checkpoints`, `inspect-checkpoint`, `evaluate`, `import-refl1d`, `plot-results`, `extract-features`, `lookup-sld`, `list-materials`, `mcp-server`, `serve` / `interactive`, `check-llm`. Full reference is in [README.md](README.md).
- Validation harness (separate package): `python -m validation.cli compare|chi2|diagnose` — compares fitted runs against reference models. See [validation/](validation/).

## Architecture

### Workflow state machine ([src/aure/workflow/graph.py](src/aure/workflow/graph.py))

LangGraph `StateGraph` over `ReflectivityState` (a `TypedDict` in [src/aure/state.py](src/aure/state.py)):

```
START → intake → analysis → modeling → fitting → evaluation ─┐
                                ▲              │              │
                                │              └─ (bounds) ──┘
                                └─ (refine model) ───────────┘
                                                              │
                                                            COMPLETE / END
```

Each node lives in [src/aure/nodes/](src/aure/nodes/) and returns a state-delta dict:

- `intake` — load probe, LLM-parse the sample description into `ParsedSample` (substrate / layers / ambient / hypothesis); a second LLM call from the `structural-hypothesis-ranking` skill produces a ranked list of candidate structural modifications the refinement loop may try.
- `analysis` — deterministic feature extraction (critical edge → substrate SLD, Kiessig fringe spacing → total thickness, etc.). No LLM.
- `modeling` — LLM generates / refines a `ModelDefinition` JSON. When entered from `evaluation`, the LLM is told whether to do a parameter tweak or realize a specific structural hypothesis.
- `fitting` — builds a `bumps` `FitProblem` via `nodes.model_builder.build_problem()` and runs refl1d's optimizer (`lm` / `de` / `dream`).
- `evaluation` — LLM judges fit quality (χ², BIC, residual structure, parameter sanity) and chooses next route. Has **deterministic regression guardrails**: if χ² or BIC got worse after a refinement, the previous model is restored and the attempted hypothesis is marked rejected.
- `routing.*` — pure functions returning the edge name; no state mutation.

Conditional edges are wired in `create_workflow()`. `include_fitting=False` truncates the graph after `modeling` (used by the `prepare` command and the `aure batch` `prepare` mode, which writes a `problem.json` consumable directly by refl1d).

### Models are JSON, not scripts

The historical design generated Python refl1d scripts; the current design stores models as `ModelDefinition` dicts (see `state.py`) and builds refl1d `Experiment`/`FitProblem` objects on-the-fly in [nodes/model_builder.py](src/aure/nodes/model_builder.py). When extending model semantics, change the dict schema + builder together — don't reintroduce script-string round-tripping.

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

Each skill is a directory containing a `SKILL.md` (Agent Skills spec format). `selector.select_skills(...)` chooses which to inject into LLM prompts based on the parsed sample (e.g. `polymer-films`, `metal-oxide-interfaces`, `solvent-contrast-matching`, `sei-layer-analysis`, `neutron-reflectometry`). The `structural-hypothesis-ranking` skill is special — it drives the initial hypothesis list used by the refinement loop, not the modeling prompt directly. Skills are loaded via `SkillRegistry` and rendered into prompts in `nodes/prompts.py`.

### LLM layer ([src/aure/llm/](src/aure/llm/))

- `config.py` — reads env vars, returns a normalized config dict; supports `openai`, `gemini`, `alcf`, `local` (OpenAI-compatible).
- `providers/` — one module per backend; `get_llm()` dispatches.
- `timeout.py` — signal-based wrapper (`invoke_with_timeout`, raises `LLMTimeoutError`); the per-call timeout comes from `LLM_TIMEOUT`.
- ALCF auth: `LLM_PROVIDER=alcf` uses `ALCF_ACCESS_TOKEN`, falling back to `globus_sdk` (if `aure[alcf]` is installed) and then to a subprocess call to ALCF's `inference_auth_token.py`. `aure check-llm --fix` will download and run that helper.

### Web UI & MCP server

- `aure serve` ([src/aure/web/](src/aure/web/)) — Flask app, dual-mode: interactive setup (no arg) or viewer (`aure serve OUTPUT_DIR`). Three tabs: Setup / History (χ² progression) / Results (R(Q), SLD, live parameter editor with Refl1D-recomputed dashed "User" curve, ISAAC export).
- `aure mcp-server` ([src/aure/mcp_server.py](src/aure/mcp_server.py)) — FastMCP server (stdio or SSE) exposing the workflow to AI assistants.

### User config ([aure_config.example.yaml](aure_config.example.yaml))

`-c config.yaml` passes plain-English `evaluation_criteria` and `model_constraints` strings that are interpolated into the LLM prompts (`config.format_user_constraints`). They supplement, not replace, built-in checks.

### Optional `export` extra

The ISAAC AI-Ready Data exporter ([src/aure/exporters/isaac.py](src/aure/exporters/isaac.py)) depends on `nr-isaac-format`, fetched from GitHub. Guard imports so the core install (`pip install -e .`) doesn't require it.

## Conventions worth knowing

- All LLM calls go through `invoke_with_timeout(get_llm(), …)` — don't bypass the timeout wrapper, and don't call `langchain_*` clients directly from nodes.
- Workflow nodes mutate state **only** by returning a dict; never call `state.update(...)` in-place. `Message` history uses an `Annotated[..., operator.add]` reducer so returning `{"messages": [...]}` appends.
- `route_*` functions in `nodes/routing.py` must be pure — they're called by LangGraph to pick edges and must not have side effects.
- Pre-commit hooks are authoritative for formatting (ruff + taplo + yamllint). Don't hand-format files differently.
- Docker image (`ghcr.io/neutrons-ai/aure`) installs `[agent,export]` and uses `aure` as ENTRYPOINT — changes to the CLI surface area are user-visible there.
