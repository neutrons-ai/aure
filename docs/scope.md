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
argument for keeping something, as the MCP removal in the table below shows.
What accreted afterwards is a different set of things, and the dates separate
them.

Retirements to date, which establish that pruning is normal here:

| Date | Removed | Why |
|---|---|---|
| 2026-03-13 | `llm/providers/{alcf,local,openai}.py` | consolidated into `openai_compat.py` |
| 2026-07-27 | `workflow/graph.py` | LangGraph replaced by the hand-written runner |
| 2026-09-04 | `llm/providers/alcf_auth.py` | an OpenAI-compatible endpoint needs no bespoke provider |
| 2026-09-07 | `mcp_server.py`, `nodes/refinement.py` | `c22b994`, `265ef0a` |
| 2026-09-08 | `cli.py` `evaluate` command | `89d2156` |
| 2026-09-08 | `cli.py` `lookup-sld`/`list-materials`, then `database/` | `2d5e1f1` |
| 2026-09-08 | `cli.py` `plot-results` | never worked; see the use-case pass |
| 2026-09-08 | `cli.py` `extract-features` | never worked; see the use-case pass |

---

## Use-cases

The inventory below says what exists. It cannot say what should — that depends
on who AuRE is for and what they are trying to do, which is a judgement about
the science, not about the code. This section holds that judgement, so the
keep / bound / retire question has something to be answered *against*.

Each entry should say what someone is trying to accomplish, not which command
they type: "fit a single curve from a plain-English description" is a use-case,
"`aure analyze`" is a surface that serves one. A capability in the inventory
that serves no entry here is a candidate for retirement; an entry here that no
capability serves is a gap.

### The list

| # | Surface | Use-case |
|---|---|---|
| **U1** | CLI | Fit a **single** reflectivity curve (1 state, 1 file) from a textual description. |
| **U2** | CLI | Co-refine several curves that are **one state** (1 state, N files — spliced Q segments) from a textual description. Every structural parameter is tied; nuisance parameters are *not* tied unless the description says so. |
| **U3** | CLI | Co-refine **several states** (N states, N files) from a textual description, with some parameters tied as the description directs. |
| **U4** | CLI | **Generate the setup YAML** for U1–U3, for the user to then edit by hand. |
| **U5** | CLI | Run a fit **from a setup YAML**. |
| **U6** | UI | U1 and U2 through the web interface. |
| **U7** | UI | Import and export the setup YAML. |
| **U8** | CLI → UI | Import a refl1d fit **run by hand**, then visualise it and resume fitting in the UI. |
| **U9** | CLI | **Hand a model to bare refl1d**: build the stack from a description or a setup YAML and write a `problem.json`, to fit or inspect in refl1d directly rather than through AuRE's loop. |
| **U10** | CLI | **Run a corpus unattended**: fit many samples in one invocation from a manifest, each job a setup merged with a shared `defaults:` block, with per-job selection and a dry run. Inherits U1–U3 unchanged — both modes pass the job's `states` into the same `run_analysis` / `run_prepare` entry points that `analyze -c` uses, so it is a loop over U5 rather than a second implementation. Also available in `prepare` mode, giving U9 for every job at once. |
| **U11** | UI | **Deliver an AI-ready record**: export a finished fit in the ISAAC format for the data portal, so the result leaves AuRE as a citable record rather than a directory of files. |

**U3 is the one that needs assessing, not just implementing.** Expressing a
cross-state tie in prose is qualitatively harder than expressing a stack: the
description has to say which parameters are shared, and "shared" is a claim
about physics that the text may not pin down. The open question is whether
**parametric constraints** — a parameter written as a function of another
rather than tied to it — are viable inside AuRE at all, or whether they demand
a declaration surface that prose cannot carry. Row 13 (`derived_parameters`) is
the existing partial answer and is off by default for exactly this reason.

Two things bear on that assessment, both established by round-tripping real
problems through `bumps.serialize` (see U9):

- **A reparametrization is exportable, so it need not be a dead end.**
  `save_problem_json` refuses any model carrying `derived_parameters`, on the
  stated grounds that bumps serialization does not preserve expression
  parameters. In bumps 1.0.x it does. A single-state declaration round-trips
  with its free parameter, both `keep_physical` constraints and the derived
  `Expression` intact and χ² unchanged; so does a multi-state **tied**
  declaration (`assign: {SEI.rho: "ambient.rho + Gamma / SEI.thickness"}`) —
  one shared free parameter, an `Expression` in each state, structural ties
  preserved by object identity. `roughness_tie`, which that refusal cites as
  its precedent, survives too. The guard looks obsolete rather than wrong in
  principle, and should be re-tested against the pinned bumps version before
  it is relaxed. If parametric constraints are the answer for U3, this is one
  fewer thing standing in the way.
- **The failure path is not safe yet.** When an `assign` expression cannot be
  resolved, the declaration is pruned and the slot falls back to a free
  parameter — but the slot stays marked as reparametrized, so the renaming pass
  skips its `"<state> "` prefix and two distinct parameters end up with the
  same name (`SEI rho` twice, in a two-state problem). Anything keyed by
  parameter name then sees one of the two and cannot tell which. This is
  reachable without a typo: `prune_derived_parameters` drops a declaration
  whenever a structural edit removes a layer it references. It is the same
  class of defect as the state-0 tie reference recorded in
  [TODO.md](../TODO.md), and it should be fixed before `derived_parameters` is
  offered as U3's mechanism.
- **A tie spec needs names the user does not control.** This is the practical
  obstacle, and it is not hypothetical: a four-job manifest exercising U1, U2,
  U3 and U9 ran clean except that the U3 job died at the initial build with

  ```
  shared_parameters references unknown layer 'Cu'; known:
  ['D2O', 'D2O/H2O', 'H2O', 'ambient', 'copper', 'silicon', 'substrate']
  ```

  The manifest said `Cu.thickness`; the intake LLM had named the layer
  `copper`. Tie specs match layer names exactly, and for a description-driven
  run those names do not exist until intake has already run — so U3 asks the
  user to write a reference to something they cannot see yet. The failure is at
  least loud and lists the candidates; the quiet variants (a mid-run rename, a
  name present only in the template) are recorded in
  [TODO.md](../TODO.md). The documented remedy is to declare the stack
  explicitly in the `states` block rather than leaving the names to the parse,
  which for U10 means a manifest that pins layer names. Whether U3 can be
  *description-driven at all*, as stated, turns on this more than on the tie
  machinery, which works.

### What serves what

| Use-case | Inventory rows |
|---|---|
| U1 | 1, 2, 3, 4, 5, 6, 25 |
| U2 | 9, 12 (+ the U1 set) |
| U3 | 10, 11, 12, 13? (+ the U1 set) |
| U4 | **nothing** — see below |
| U5 | 8 |
| U6 | 18 (+ the U1/U2 sets) |
| U7 | 8, 18 |
| U8 | 19, then 18, 7 |
| U9 | 17, and whichever of 9 / 10 the shape needs |
| U10 | 16, then whatever U1–U3 or U9 the jobs are |
| U11 | 20, 18, and 5 (it exports the *selected* fit) |
| all fitting | 22, 23, 26 |

### Three findings the mapping produces

**U4 has no implementation.** `setup.dump_setup` exists and round-trips, but its
only caller is the web UI's `/api/setup/export`
([`web/routes.py:951`](../src/aure/web/routes.py#L951)). No CLI command writes a
setup file, so "generate the YAML, then edit it" is reachable only by starting
the web server — which makes U4 as stated a gap, and U7 the workaround people
would find. `aure prepare` is the nearest thing and serves U9 instead: it emits
a refl1d `problem.json`, not a setup YAML — a different artifact for a
different purpose.

**U11 is UI-only, which its inventory row obscures.** Row 20 lists the export
as "Exposed via `EXPORT_FORMAT`, web button", which reads as though the env var
were a CLI trigger. It is not: `EXPORT_FORMAT` only *selects* which exporter
the web button uses, and `exporters.get_exporter()` is called from exactly two
places, both web routes
([`web/routes.py:1797`](../src/aure/web/routes.py#L1797),
[`:1817`](../src/aure/web/routes.py#L1817)). Neither the CLI nor the workflow
ever exports. So a corpus run through U10 produces no records without opening
each result in the browser, which is the combination most likely to be wanted
and the one that does not exist. Whether U11 should also be a CLI use-case is a
real decision, not a formality.

**The UI already exceeds U6.** `/api/start-analysis` accepts `states` and
`data_files`, not just a single `data_file`, so the web path can launch U3 as
well — its own docstring documents only the single-file body and is out of
date. Worth deciding whether U6 should claim U3 rather than leaving working
code undocumented.

### Capabilities no listed use-case asks for

Four inventory rows serve nothing in the list above, and **none of them is a
CLI command any more**. That is the result of asking the question, not the
state it started in:

- Of the five candidates this list first produced, **three became use-cases** —
  batch, `prepare` and the ISAAC export are U10, U9 and U11.
- **Two were retired, both because they had never worked.** `plot-results`
  globbed `refl1d_output/fit_iter*_*/problem.json`, but bumps names that export
  `<model_name>.json` — the same fact `CheckpointManager._find_problem_json`
  exists to handle and which `plot-results` never consulted; 297 lines, no
  test, five months. `extract-features` unpacked `load_reflectivity_data` as a
  3-tuple when it returns a dict, so a 4-column file raised "Error loading
  data" (blaming a file that had loaded) and a 3-column file bound the strings
  `'Q'`, `'R'`, `'dR'` and crashed later; 108 lines, no test, and broken since
  the **first commit**.

  Neither capability was redundant in principle — one showed every iteration on
  one axis, the other reported what the data says before a model exists. Both
  were dead code advertising a feature. The underlying feature extraction is
  untouched and heavily used: the `analysis` node calls
  `extract_all_features`, which the retired command did not.

What remains unclaimed:

- **14 (thin-layer mode enumeration), 15 (`roughness_tie`)** — fit strategy
  rather than user-facing capability; they serve U1–U3 indirectly and are
  reachable only by env var or hand-edited model JSON.
- **25 (Docker), 26 (importable library)** — delivery and integration, not
  use-cases. Row 26 has out-of-tree consumers regardless of what this list
  says.

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
| 21 | Load reflectivity data (`.txt`, `.dat`, `.csv`, `.asc`, `.refl`, `.ort`) | implicit; every command that takes a data file | `tools/data_tools.py` 389 | 02-09 |
| 22 | **Pluggable instrument / file-format support** | `aure.instruments` entry point, `AURE_INSTRUMENT`, `register()` | `instruments/` 698 + own doc | **09-07** |
| 23 | Domain skill library (9 skills) | LLM-selected into prompts | 1,466 md + `selector.py` 401 + `loader.py` 172 | 04-16 → **09-03** |
| 24 | LLM provider layer (3 providers, timeout, retries, ledger) | `LLM_*` env | `llm/` ~600 | 02-14 |
| 25 | Docker image | `ghcr.io/neutrons-ai/aure` | Dockerfile + CI | grown |
| 26 | **AuRE as an importable library** | `import aure` — no dedicated code | `__all__` names 3 of them | 02-09 |

10 CLI commands remain.

Row 26 is the one surface this inventory previously missed, and it has a
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

(The second entry here was the REF_L filename coupling. It is addressed by
the `instruments/` seam — `57a562d`, `92a9c5b`, `ca2b7a0` — and is kept out of
this list because the conventions are now declarable rather than hard-coded,
and an unrecognised file says so instead of taking the combined branch in
silence.)

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
document takes a position, the position is factual rather than editorial: six
physics knobs are env-var-only, and the undocumented absorption boundary above
is real.
