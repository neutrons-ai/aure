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
| **U3** | CLI | Co-refine **several states** (N states, N files) from a textual description, with some parameters tied as the description directs. Equality ties only — see the assessment below. |
| **U4** | CLI | **Generate the setup YAML** for U1–U3, for the user to then edit by hand. |
| **U5** | CLI | Run a fit **from a setup YAML**. |
| **U6** | UI | U1 and U2 through the web interface. |
| **U7** | UI | Import and export the setup YAML. |
| **U8** | CLI → UI | Import a refl1d fit **run by hand**, then visualise it and resume fitting in the UI. |
| **U9** | CLI | **Hand a model to bare refl1d**: build the stack from a description or a setup YAML and write a `problem.json`, to fit or inspect in refl1d directly rather than through AuRE's loop. |
| **U10** | CLI | **Run a corpus unattended**: fit many samples in one invocation from a manifest, each job a setup merged with a shared `defaults:` block, with per-job selection and a dry run. Inherits U1–U3 unchanged — both modes pass the job's `states` into the same `run_analysis` / `run_prepare` entry points that `analyze -c` uses, so it is a loop over U5 rather than a second implementation. Also available in `prepare` mode, giving U9 for every job at once. |
| **U11** | UI | **Deliver an AI-ready record**: export a finished fit in the ISAAC format for the data portal, so the result leaves AuRE as a citable record rather than a directory of files. |

### U3 assessed

U3 was flagged as needing assessment rather than implementation. It has now had
one, against the code and against a live model. **The verdict splits on a line
the use-case as stated does not draw: whether the relation between states is an
equality or a function.**

- **Equality ties from prose — viable, and working.** Keep U3 as stated.
- **Functional relations from prose — not viable, and currently unsafe**, because
  the prose path silently substitutes an untie. Either narrow U3 to exclude them
  or make the substitution refuse.
- **Functional relations from a config file — viable today**, with an awkward
  idiom and no prose route to it.

#### What works

*The tie machinery.* Parameters are aliased across states as shared
`bumps.Parameter` objects, resolved by layer *name* so an inserted or removed
layer above the tied one does not break it. The aliasing survives a
`bumps.serialize` round-trip by object identity — a `problem.json` written from
a multi-state model reloads with `sample_a[1].thickness is sample_b[1].thickness`
and no duplicated parameter names. Per-state structure works in both directions
(a layer present in some states and absent in others), and a tie naming a layer
a state does not have is skipped rather than failing.

*The prose-to-tie-name mapping.* This was the expected weak link and is not.
Given a three-layer stack in two contrasts, `_extract_cross_state_unshared`
produced:

| description | result |
|---|---|
| nothing said about differences | `None` → the default tied set |
| "the copper oxide thickness may differ … everything else is the same sample" | `['CuOx.thickness']` |
| "only the copper thickness and SLD should be common; let everything else float" | the full complement — including `Cu.interface`, correctly reading "thickness and SLD" as excluding roughness |
| "the buried interfaces are identical, but the outer surface roughens" | `['CuOx.interface']` — inferring both that `CuOx` is outermost and that roughness is `interface` |

The third case is the interesting one: it inverted a whitelist statement into
the blacklist the schema wants. The vocabulary problem — mapping physics prose
onto `<layer>.<attr>` — is solved well enough to build on.

*End to end.* A two-state job run through `aure batch` completed with exactly
the intended structure: one shared `copper thickness/rho/interface`, per-state
`D2O rho` / `H2O rho`, `intensity` and `silicon interface`.

#### Where prose fails, and how

A cross-state relation that is not an equality has no representation in
`shared_parameters`, which can only tie or not tie. Asked to express one, the
extractor does not decline — it returns the nearest expressible thing, which is
an untie:

| description | returned | what it means |
|---|---|---|
| "the oxide in the second measurement is twice as thick, it was left in air overnight" | `['SiO2.thickness']` | *independent*, not 2× — and it picked the wrong oxide |
| "the polymer volume fraction is the same in both, so the layer SLD must differ exactly as the solvent SLD does" | `['Cu.material.rho']` | two free SLDs instead of one shared volume fraction |

Both are silent. The stated constraint is discarded, replaced by a free
parameter, and nothing in the run says so — the fit then reports a χ² for a
model the description did not ask for. That is worse than refusing, and it is
the substantive finding of this assessment.

The second row is the case a reparametrization would have expressed, which
makes the failure precise: prose states a relation the schema cannot hold, and
the fallback quietly contradicts it.

#### Functional constraints: retired, deferred to an add-on

`derived_parameters` — declaring one parameter as a function of others — was
**removed on 2026-09-08**, and U3 is narrowed to equality ties as a result.

It worked, and more than the earlier note credited: a cross-state ratio was
expressible with an auxiliary tied handle plus one scoped assignment per state,
verified end to end, and `bumps` 1.0.x round-trips the resulting expressions
and constraints through `problem.json` intact — so the export refusal it
carried was over-conservative. The reasons to retire it were not that it failed
to work:

- **It has to survive AuRE's own iteration and does not.** A declaration is
  written against layers; the refinement loop adds and removes them. The
  response was to prune the declaration and log it — a workaround for the hard
  problem, not an answer to it, and the mechanism that most needed designing.
- **No prose route, and the fallback is unsafe.** The two rows above are the
  evidence: the one mechanism that could express those relations was reachable
  only from a config file, while the description path silently substituted an
  untie.
- **It was never used.** Added to support benchmarking and not used for it.
  Off by default was the tell.

The design constraints established here are recorded as a wish in
[TODO.md](../TODO.md), so a future add-on starts from them rather than
rediscovering them.

#### What gates it

Five recorded defects sit on this path, all in [TODO.md](../TODO.md) except
where noted:

1. **Tie names do not exist when the user has to write them.** A tie spec
   matches layer names exactly, and for a description-driven run those names
   are chosen by the intake parse. Observed live: a manifest saying
   `Cu.thickness` failed against a parse that named the layer `copper`. Loud,
   with the candidates listed — but it means U3 as stated asks the user to
   reference something they cannot see yet. The remedy is to declare the stack
   in the `states` block, which trades the prose interface away.
2. **State 0 is the tie reference.** A layer absent from state 0 but present in
   two or more others is tied nowhere, and both copies keep the tied name, so
   three parameters end up double-named.
3. **A mid-run rename voids a pinned tie**, silently, reported as a removed
   layer.
4. **Union validation, per-state application.** A name present only in the
   model-level template validates and then ties nothing.
Only (1) is a design question. (2)–(4) are bounded fixes.

#### Recommendation

Keep U3 for equality ties; it works and the prose mapping is good. Before
functional constraints return as an add-on:

- make the untie substitution **refuse** rather than approximate — if a
  description states a relation the schema cannot express, that belongs in
  `issues` and in front of the user, not in a silently different model;
- fix (2)–(4);
- design the add-on around surviving structural iteration, which is the
  constraint the retired mechanism did not meet, and decide whether it has a
  prose surface at all — "from a textual description" could not reach the old
  one.

### What serves what

| Use-case | Inventory rows |
|---|---|
| U1 | 1, 2, 3, 4, 5, 6, 22 |
| U2 | 9, 12 (+ the U1 set) |
| U3 | 10, 11, 12 (+ the U1 set) |
| U4 | **nothing** — see below |
| U5 | 8 |
| U6 | 17 (+ the U1/U2 sets) |
| U7 | 8, 17 |
| U8 | 18, then 17, 7 |
| U9 | 16, and whichever of 9 / 10 the shape needs |
| U10 | 15, then whatever U1–U3 or U9 the jobs are |
| U11 | 19, 17, and 5 (it exports the *selected* fit) |
| all fitting | 20, 21, 23 |

### Three findings the mapping produces

**U4 has no implementation.** `setup.dump_setup` exists and round-trips, but its
only caller is the web UI's `/api/setup/export`
([`web/routes.py:951`](../src/aure/web/routes.py#L951)). No CLI command writes a
setup file, so "generate the YAML, then edit it" is reachable only by starting
the web server — which makes U4 as stated a gap, and U7 the workaround people
would find. `aure prepare` is the nearest thing and serves U9 instead: it emits
a refl1d `problem.json`, not a setup YAML — a different artifact for a
different purpose.

**U11 is UI-only by design — decided, not a gap.** Worth stating explicitly
because the absence of a CLI export otherwise reads as an oversight, and
because row 20 used to invite that reading: it listed the export as "Exposed
via `EXPORT_FORMAT`, web button", as though the env var were a CLI trigger.
It is not, and the row now says so. `EXPORT_FORMAT` only *selects* which
exporter the web button uses, and
`exporters.get_exporter()` is called from exactly two places, both web routes
([`web/routes.py:1797`](../src/aure/web/routes.py#L1797),
[`:1817`](../src/aure/web/routes.py#L1817)). Neither the CLI nor the workflow
ever exports.

The consequence is accepted: a corpus run through U10 produces no records
until someone opens each result. Publishing a record is a deliberate act, and
it stays behind the surface where a person has actually looked at the fit. So
U11 is tagged UI and there is no CLI counterpart to add.

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

- **13 (thin-layer mode enumeration), 14 (`roughness_tie`)** — fit strategy
  rather than user-facing capability; they serve U1–U3 indirectly and are
  reachable only by env var or hand-edited model JSON.
- **24 (Docker), 25 (importable library)** — delivery and integration, not
  use-cases. Row 25 has out-of-tree consumers regardless of what this list
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
| 13 | Thin-layer SLD mode enumeration | `MODE_ENUMERATION=1` env only | inside `fitting.py` + skill 101 | 07-24 |
| 14 | `roughness_tie` profile reparametrization | model JSON only | `model_builder.py:383-396` | grown |
| 15 | Batch manifest runner (2 modes) | `aure batch` | `cli.py:1318-1750` (433) | grown |
| 16 | `prepare` → `problem.json` handoff to bare refl1d | `aure prepare`, batch mode | `cli.py:976-1314` (339) | grown |
| 17 | **Web UI** (setup / history / results, live param editor, file browser) | `aure serve`, `aure interactive` | 3,054 py + 3,836 assets = **6,890** | 02-09 |
| 18 | **Import a hand-run refl1d fit** | `aure import-refl1d` | `refl1d_import.py` **1,740** | 05-22 |
| 19 | ISAAC AI-ready export | web button only; `EXPORT_FORMAT` selects the format | `exporters/` 523 + optional dep | 03-07 |
| 20 | Load reflectivity data (`.txt`, `.dat`, `.csv`, `.asc`, `.refl`, `.ort`) | implicit; every command that takes a data file | `tools/data_tools.py` 389 | 02-09 |
| 21 | **Pluggable instrument / file-format support** | `aure.instruments` entry point, `AURE_INSTRUMENT`, `register()` | `instruments/` 698 + own doc | **09-07** |
| 22 | Domain skill library (8 skills) | LLM-selected into prompts | 1,466 md + `selector.py` 401 + `loader.py` 172 | 04-16 → **09-03** |
| 23 | LLM provider layer (3 providers, timeout, retries, ledger) | `LLM_*` env | `llm/` ~600 | 02-14 |
| 24 | Docker image | `ghcr.io/neutrons-ai/aure` | Dockerfile + CI | grown |
| 25 | **AuRE as an importable library** | `import aure` — no dedicated code | `__all__` names 3 of them | 02-09 |

10 CLI commands remain.

Row 25 is the one surface this inventory previously missed, and it has a
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
- **Aug–Sep** — `instruments/`, the first addition that *removed* coupling
  rather than adding capability, and the newest code in the repo. (A
  reparametrization mechanism arrived here too and was retired in the same
  month; see **Use-cases**.)

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
