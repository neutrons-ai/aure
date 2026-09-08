# Scope: what AuRE actually does

An inventory of every capability in this repository, assembled from the code
rather than from memory, so that a keep / bound / retire decision can be made
against facts. This answers the last entry in [TODO.md](../TODO.md).

It records **what exists, where it is exposed, what it weighs, and when it
arrived**. It deliberately does *not* record whether each capability works, is
tested, or has been validated — that is a different question, and mixing the two
is what makes a scoping list unusable.

Measurements are from `next` at the time of writing: 185 commits, 2026-02-09 to
2026-09-07, 27,912 lines of Python under `src/aure`, plus 3,836 lines of web
assets and 1,466 lines of skill markdown.

---

## What the repository was on day one

The first commit (2026-02-09) already contained: the CLI, the node pipeline
(`intake`, `analysis`, `modeling`, `fitting`, `evaluation`), the workflow runner
and checkpoints, the feature-extraction tools, the materials database, the web
UI, and the MCP server.

That matters for reading the "arrived" column. The web UI was **not** a later
accretion, and neither was MCP — being in the first commit is not by itself an
argument for keeping something, as the MCP removal below shows. What accreted
afterwards is a different set of things, and the dates separate them.

Retirements to date, which establish that pruning is normal here:

| Date | Removed | Why |
|---|---|---|
| 2026-03-13 | `llm/providers/{alcf,local,openai}.py` | consolidated into `openai_compat.py` |
| 2026-07-27 | `workflow/graph.py` | LangGraph replaced by the hand-written runner |
| 2026-09-04 | `llm/providers/alcf_auth.py` | an OpenAI-compatible endpoint needs no bespoke provider |
| 2026-09-07 | `mcp_server.py`, `nodes/refinement.py` | see **Decisions taken** |
| 2026-09-08 | `cli.py` `evaluate` command | see **Decisions taken** |
| 2026-09-08 | `cli.py` `lookup-sld`/`list-materials`, then `database/` | see **Decisions taken** |

---

## Decisions taken

**MCP server — retired (2026-09-07, `c22b994`).** Never needed in practice. It
exposed 11 tools: three thin wrappers over the materials database and feature
extraction, two one-shot wrappers over `run_analysis`, and a five-tool
hand-driven session loop. Everything but one detail was already reachable from
the CLI (`aure lookup-sld`, `aure extract-features`, `aure analyze`,
`aure analyze -c`); the exception is that `analyze_reflectivity_features`
accepted in-memory Q/R arrays where `aure extract-features` requires a file.
`fastmcp` was a **core** runtime dependency, so this also shrank the default
install.

**`nodes/refinement.py` — retired with it (`265ef0a`).** 114 lines that edited
refl1d model *scripts* by regex, against a representation abandoned on
2026-03-17 when `model_builder.py` arrived. MCP's `modify_model` was its only
importer, and every one of that tool's five operations had been raising into an
`except Exception` and returning `{"error": …}`: the two bound-wideners ran
`re.sub` against a dict, and `_add_layer` was called with two arguments against
a one-argument definition and was a self-declared stub returning `False`
regardless. Dormant since 2026-03-06. The `result["evaluation"]` branch in
`_print_analysis_results`, which only MCP ever populated, collapsed with it.

Net effect of both: **−1,079 lines**, one core dependency dropped, 783 tests
still green.

**Facility coupling — bounded behind a seam (2026-09-07, `57a562d`,
`92a9c5b`, `ca2b7a0`).** The ORNL REF_L filename conventions were hard-coded
in three modules; adding another instrument meant editing all of them. They
now sit in `aure/instruments/` (698 lines) behind a four-question protocol,
with REF_L's regexes and theta parser moved verbatim and a golden parity
table pinning their classifications. Consequences worth recording:

- The previously silent branch is loud. A file no instrument recognises used
  to be counted as a combined curve without comment; it now warns, records
  `_instrument` on the state, and refuses a setup that declares
  `theta_offset` / `sample_broadening` on files AuRE cannot classify.
- ORSO `.ort` is the second instrument, which is what proves the seam. It
  parses the YAML header `data_tools.parse_ort_file` was reading and
  discarding, and declares `dq_is_fwhm = False` — an ORSO `sQz` column is one
  sigma by specification, so the global REF_L-derived default had been
  over-broadening every ORSO resolution by 2.35.
- Third parties register through an `aure.instruments` entry point or
  `AURE_INSTRUMENT`; see [instruments.md](instruments.md).

**`aure evaluate` — retired (2026-09-08).** Scope creep: an LLM-judgement CLI
wrapper whose inputs were a strict *subset* of what the library call it wrapped
accepts. Removing it takes 310 lines out of `cli.py`, plus a 23-line helper and
a 34-line test that existed only for it.

The case against keeping it:

- **It starved the judge it called.** `analyze_fit_quality_with_llm` takes
  `bic` / `n_params` / `n_layers` / `skill_context` / `features` /
  `residual_analysis`; the command passed none of them. With `bic=None` the
  whole complexity block renders `(not computed)`
  ([`prompts.py:627`](../src/aure/nodes/prompts.py#L627)), and with no
  `skill_context` no domain skill reaches the prompt. The parsimony argument
  and the physics grounding are what distinguish that judge from a χ²
  threshold, and the command asked for a verdict without either.
- **`aure import-refl1d` already does the richer version of the same ingest.**
  It reads the same `problem.json` and computes BIC from `bic_inputs(problem)`,
  extracts features, and sets `active_skills`
  ([`refl1d_import.py:1562`](../src/aure/refl1d_import.py#L1562)). A verdict on
  an external fit is `aure import-refl1d` then `aure serve` / `aure resume` —
  the same capability through the door that feeds the evaluator properly.
- **Its `--json` was an unwritten wire format, and its only consumer was
  reading it wrong.** nr-analyzer shelled out to `aure evaluate <dir> --json`
  (pipeline step 6, default on) and rendered `verdict` / `quality` / `status`,
  `chi2`, `physical_plausibility`, `summary` — none of which AuRE emits. Its
  actual keys are `quality_assessment`, `chi_squared`, `physical_concerns`.
  Every fallback chain missed, so the rendered report carried the issues and
  suggestions lists and nothing else: no verdict, no χ², no physical judgement,
  no sign the verdict was advisory. nr-analyzer's pipeline papered over it with
  a hard-coded "Verdict: (none reported by aure evaluate)". The fault was the
  consumer's, but the guess was available to make because AuRE published the
  format only by emitting it — and no test pinned the key names.
- **The command also dropped most of what the judge returned.** The verdict
  dict carries eleven keys
  ([`evaluation.py:1446`](../src/aure/nodes/evaluation.py#L1446)); `--json`
  forwarded five. `hypothesis_addressed` was omitted although the command took
  `-h`, and `_used_fallback` was omitted, so a consumer could not tell an LLM
  verdict from `_simple_evaluation`'s three χ² bands.
- **It had no test coverage.** Only `analyze` and `batch` were exercised
  through `CliRunner`.

What replaces it for the two out-of-tree callers is what one of them was
already doing: import the function. nr-workbench reaches
`analyze_fit_quality_with_llm` directly through a pinned contract table, and
nr-analyzer already imports `aure.llm` in four places — the subprocess was the
odd path in its own codebase. The migration PR is tracked in
[TODO.md](../TODO.md).

**What this leaves unresolved.** `analyze_fit_quality_with_llm` is now reached
by the `evaluation` node and by out-of-tree importers, with no CLI in between
and still no `__all__` entry (see row 28). The verdict stays advisory outside
the node: the deterministic guardrails — profile veto, χ² clamp — are the
node's alone, and any external caller gets the model's opinion on one exported
fit. That distinction was worth a JSON key; it is now worth a docstring.

**Materials / SLD database — retired entirely (2026-09-08).** The two CLI
commands (`lookup-sld`, `list-materials`, 126 lines of `cli.py`) and then the
module behind them (`database/`, 448 lines).

The question that prompted this was whether `database/` belongs in AuRE at all,
given that refl1d ships `periodictable`. The first answer looked like "yes":
**nothing in `database/` reimplements periodictable** — `compute_sld` is a
four-line delegation to `neutron_sld()` — and what periodictable lacks is real:

| | periodictable | `aure.database` |
|---|---|---|
| element densities | yes (`Si` 2.33, `Cu` 8.96) | — |
| **compound** densities | no (`D2O`, `H2O`, `SiO2`, `Al2O3` all `None`) | 18 entries |
| common-name resolution | no (`quartz`, `sapphire`, `silicon` all raise) | 66 aliases |
| contrast matching | no | `get_contrast_match_ratio`, `get_mixture_sld` |

What overturned that is where the SLD in a fitted model actually comes from.
**Not from this module.** The intake prompt asks the LLM for `sld` directly on
the substrate, every layer and the ambient
([`prompts.py:37`](../src/aure/nodes/prompts.py#L37)), and `_build_layers` uses
that number as-is. The database was never consulted on that path. Its only
internal caller was one line — `get_sld("silicon")`, resolving a constant of
nature — now inlined as `_SILICON_SLD`.

And an LLM estimate is good enough, because the SLD is a *fitted* parameter and
the estimate only has to seed it. With the formula known and only the density
guessed, ±1 in SLD needs the density to ~15% for deuterated species and Al₂O₃,
~30% for SiO₂/TiO₂, and is essentially unconstrained for anything protiated.
What actually matters is protiation, not density: `b_c(D) − b_c(H) = 10.409 fm`,
so deuteration adds the hydrogen number density times 1.041 — 5 to 8 for any
organic — which is why "deuterated organics sit near 5.5, protiated near 0.4"
holds across the 18 species measured for this decision. A 15 % density slip
costs 1; a missed H/D swap costs 5. AuRE already treats the latter as a
first-class structural hypothesis (refine rule 13's rewind), which is the right
place for it.

So the tables encode what the LLM already knows, for a number the fit refines
anyway. Retiring them costs nothing AuRE was using.

**The one thing that was load-bearing** was the fallback for when the parse
supplies *no* SLD: `2.0` with bounds seeded at ±2.5, i.e. `(-0.5, 4.5)`. Real
neutron SLDs are bimodal, so that window sits in the empty middle and excludes
the entire deuterated half of the distribution — and a layer fenced into the
wrong half cannot be fitted out of it, because the bound rather than the data
is holding it. That is now `(-0.6, 7.5)`, spanning both clusters, with the
reasoning recorded at the constant and pinned by a test. This was the real
defect the scoping exercise turned up, and it had nothing to do with the
database.

**Downstream.** nr-workbench was the only external consumer, and barely: its
`sld_for` wrapper is referenced nowhere in that repo, `lookup_material` was
pinned in its contract table but never called, and `contrast_match_ratio` is
live only as an *agent-facing* API documented in two of its SKILL.md files. It
is six lines of interpolation over two solvent SLDs, which that repo now
carries itself — consistent with its own stated policy of reimplementing
AuRE's short arithmetic rather than paying for the import.

`periodictable` remains a declared dependency though nothing in `src/aure`
imports it any more; refl1d requires it regardless.

---

## The inventory

Footprint is what would be deleted, not what would be touched.

| # | Capability | Exposed via | Footprint | Arrived |
|---|---|---|---|---|
| 1 | **Fit one curve from a plain-English description** | `aure analyze DATA "desc"` | the reason for the rest | 02-09 |
| 2 | Agent refinement loop (intake→…→evaluation, routing) | implicit | `workflow/` 1,575 + `nodes/` core | 02-09 |
| 3 | Structural hypothesis ranking & refinement targeting | implicit; `-h` flag | `nodes/hypotheses.py` 231 + skill 206 | 04-20, 06-03 |
| 4 | Deterministic guardrails (χ² clamp, profile-artifact veto, regression guard) | implicit; `CHI2_MIN/MAX` | inside `evaluation.py` 2,139 | grown |
| 5 | Final model selection + parsimony tie-break | implicit | `nodes/finalize.py` 874 | 07-26 |
| 6 | Final uncertainty polish (`dream` re-fit) | `fit_method_final`, `FIT_*_FINAL` | `nodes/final_fit.py` 420 | 07-27 |
| 7 | Checkpointing, `resume`, checkpoint inspection | `resume`, `checkpoints`, `inspect-checkpoint` | `workflow/checkpoints.py` 672 | 02-09 |
| 8 | Setup-YAML declaration surface | `-c setup.yaml` | `setup.py` 665 + `config.py` 608 | 05-22 |
| 9 | **Multi-file co-refinement** (one sample, spliced Q) | `-d`, `data_files:` | `build_multi_problem` ~150 | early |
| 10 | **Multi-state co-refinement + cross-state ties** | `states:`, `shared_/unshared_parameters` | ~320 of `model_builder.py` + skill 234 | 05-17 |
| 11 | Per-state structure overrides | `states[].layers/substrate` | `config.py:352`, `_state_overrides` | 05-17 |
| 12 | Nuisance/resolution parameters | `theta_offset`, `sample_broadening`, `background`, `intensity` | scattered, partials-only | grown |
| 13 | `derived_parameters` reparametrization | `derived_parameters:` + `allow_derived_parameters` | `nodes/expressions.py` 205 + ~200 in builder + own doc | **09-02** |
| 14 | Thin-layer SLD mode enumeration | `MODE_ENUMERATION=1` env only | inside `fitting.py` + skill 101 | 07-24 |
| 15 | `roughness_tie` profile reparametrization | model JSON only | `model_builder.py:383-396` | grown |
| 16 | Batch manifest runner (2 modes) | `aure batch` | `cli.py:1318-1750` (433) | grown |
| 17 | `prepare` → `problem.json` handoff to bare refl1d | `aure prepare`, batch mode | `cli.py:976-1314` (339) | grown |
| 18 | **Web UI** (setup / history / results, live param editor, file browser) | `aure serve`, `aure interactive` | 3,054 py + 3,836 assets = **6,890** | 02-09 |
| 19 | **Import a hand-run refl1d fit** | `aure import-refl1d` | `refl1d_import.py` **1,740** | 05-22 |
| 20 | ISAAC AI-ready export | `EXPORT_FORMAT`, web button | `exporters/` 523 + optional dep | 03-07 |
| 21 | Standalone feature extraction | `extract-features` | `tools/feature_tools.py` 1,142 (shared with node 2) | 02-09 |
| 22 | Load reflectivity data (`.txt`, `.dat`, `.csv`, `.asc`, `.refl`, `.ort`) | implicit; every command that takes a data file | `tools/data_tools.py` 389 | 02-09 |
| 23 | **Pluggable instrument / file-format support** | `aure.instruments` entry point, `AURE_INSTRUMENT`, `register()` | `instruments/` 698 + own doc | **09-07** |
| 24 | Result plotting | `aure plot-results` | `cli.py:1998-2294` (297) | grown |
| 25 | Domain skill library (9 skills) | LLM-selected into prompts | 1,466 md + `selector.py` 401 + `loader.py` 172 | 04-16 → **09-03** |
| 26 | LLM provider layer (3 providers, timeout, retries, ledger) | `LLM_*` env | `llm/` ~600 | 02-14 |
| 27 | Docker image | `ghcr.io/neutrons-ai/aure` | Dockerfile + CI | grown |
| 28 | **AuRE as an importable library** | `import aure` — no dedicated code | `__all__` names 3 of them | 02-09 |

12 CLI commands remain.

Row 28 is the one surface this inventory previously missed, and it has a
consumer. `__all__` declares `ReflectivityState`, `create_initial_state` and
`run_analysis` ([`__init__.py:33`](../src/aure/__init__.py#L33)). nr-workbench
pins twelve callables across four modules in its own contract table — so a
rename here breaks its CI rather than a scientist's fit — and reaches three more
in `aure.llm` besides. None of the fifteen appears in any `__all__`. Whatever
is decided about the CLI, that list is a de-facto API, and the scoping decision
should say whether it becomes a declared one.

---

## Things the inventory turned up that bear on the decision

### Physics knobs reachable only through environment variables

15 setup keys are mapped to env vars for batch
([`cli.py:1685-1717`](../src/aure/cli.py#L1685-L1717)). These are **not** among
them, and have no CLI flag or config key:

`MODE_ENUMERATION`, `THIN_LAYER_MODE_K`, `THIN_LAYER_MODE_SEEDS`,
`ROUGHNESS_MAX_OUTER`, `FINAL_SELECTION_TOL`, `FINAL_TIER_CHI2_FACTOR`,
`USE_RUN_TITLE`, `EXPORT_FORMAT`.

The first six change what model comes out. They are invisible in the setup file
that otherwise documents a run, so a run's own config does not record the
physics policy it ran under.

### One boundary the code sets that the documentation does not

- **No absorption.** `_build_sample` constructs `SLD(name, rho)` only
  ([`model_builder.py:309-314`](../src/aure/nodes/model_builder.py#L309-L314));
  nothing anywhere sets `irho` (it appears once, as a defensive key in the
  bound-widener's lookup table). An absorbing layer cannot be represented.
  [README:9](../README.md#L9) says "neutron and X-ray reflectivity"; without
  `irho` the X-ray half is not generally reachable. No magnetism either.

Smaller, same species: `data_tools` advertises ".refl: NIST reflectivity
format" ([`data_tools.py:6`](../src/aure/tools/data_tools.py#L6)) and prints
it among the supported formats, but `load_reflectivity_data` routes `refl`
straight to `parse_ascii_columns` — there is no NIST parser, only the column
reader. A `.refl` file whose layout differs from bare columns is not read
specially, it is read wrongly.

(The second entry here was the REF_L filename coupling. It is addressed —
see **Decisions taken** — and is kept out of this list because the
conventions are now declarable rather than hard-coded, and an unrecognised
file says so instead of taking the combined branch in silence.)

### Documentation weight is inverted

Of 1,039 README lines: the setup-YAML / co-refinement block is **187**, `aure
batch` **97**, the manifest **22** — 306 lines, 29% of the README, for
extensions. `aure analyze`, the core, gets **47**.

### Minor duplication

`aure interactive` is an explicit alias for `aure serve` in setup mode — its own
docstring says so ([`cli.py:3133`](../src/aure/cli.py#L3133)).

The `-v` verbose block is copy-pasted across six commands. One copy had drifted
to logger names that do not exist (`agent.nodes.*` rather than `aure.nodes.*`)
and was fixed by deletion in `cc60731` — `basicConfig` already covers every
`aure.*` logger, so the per-module loop was redundant as well as misnamed. No
test covers `-v` on any command, which is why it survived; closing that properly
means extracting the six blocks into one testable helper.

---

## The shape of the accretion

Sorting the inventory by arrival, the pattern is not "features added randomly".
It is a steady outward drift from one curve:

- **Feb** — the core, plus its two alternative front-ends.
- **Mar–Apr** — model representation replaced (scripts → JSON); ISAAC export;
  the skill library opens.
- **May** — the declaration surface (`setup.py`, `config.py` states) and
  `import-refl1d`, i.e. two large capabilities in one month, both about
  *getting other people's structure in*.
- **Jun–Jul** — hypothesis machinery, finalize, final_fit: the loop learning to
  stop well.
- **Aug–Sep** — `expressions.py` (derived parameters) and the
  `functional-constraints` skill, then `instruments/` — the first addition
  that *removed* coupling rather than adding capability, and the newest code
  in the repo.

The three largest single blocks that are not the core loop are still the **web
UI (6,890)**, **`refl1d_import.py` (1,740)** and the **skill library (1,867
incl. selector)**. `instruments/` (698) is the newest and the only one that
exists to *hold coupling in one place* rather than to add reach.

---

## What is deliberately not decided here

Which of the remaining capabilities to keep. That is the owner's call, and the
inventory exists to make it on evidence rather than recollection. Where this
document takes a position, the position is factual rather than editorial: the
two evaluation paths disagree by construction, six physics knobs are
env-var-only, and the undocumented absorption boundary above is real.
