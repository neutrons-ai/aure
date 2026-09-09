# Launching a run: the CLI routes to U1–U3

`aure analyze` is the surface that serves use-cases **U1**, **U2** and **U3**
of [scope.md](scope.md):

| # | Use-case |
|---|---|
| **U1** | Fit a **single** curve — 1 state, 1 file. |
| **U2** | Co-refine several files that are **one state** — spliced Q segments of one physical sample. |
| **U3** | Co-refine **several states** — solvent contrast, anneal step, applied potential — with parameters tied as the description or the setup directs. |

There are two ways to declare any of them, and they differ in *what they can
express*, not in what they run: both paths end in the same `run_analysis`
entry point.

- **Ad-hoc (positional)** — the data file and the sample description are
  command-line arguments. Reaches U1 and U2.
- **Setup YAML (`-c`)** — the run is declared in a file whose `states:` block
  carries the data. Reaches U1, U2 and U3, and is the only route to U3.

This document covers the CLI only. The web UI is a separate surface (U6/U7);
see [README](../README.md#aure-serve).

| Route | U1 | U2 | U3 |
|---|---|---|---|
| `aure analyze DATA "description"` | ✅ | ✅ with `-d` | ❌ |
| `aure analyze -c setup.yaml` | ✅ | ✅ | ✅ |

---

## U1 — one curve, one file

### Ad-hoc

```bash
aure analyze data.txt "100 nm polystyrene on silicon" -o ./output -v
```

The two positionals are `DATA_FILE` and `SAMPLE_DESCRIPTION`. AuRE wraps the
file in a synthetic single state named `state0` internally, so this is the
same shape as the YAML below — there is no separate single-file code path.

### Setup YAML

```yaml
sample_description: |
  100 nm polystyrene on silicon.

states:
  - name: state0
    data_files:
      - file: data.txt
```

```bash
aure analyze -c setup.yaml -o ./output -v
```

The schema is **states-only**: even a one-file run declares a `states:` block.
Top-level `data_file:` / `data_files:` are not accepted. A state needs a
non-empty `name` and at least one file; `data_files:` entries may be bare
paths (`- data.txt`) or mappings (`- {file: data.txt, label: lowQ}`), and
`data:` is accepted as a synonym of `data_files:` for analyzer interop.

Reach for the YAML at U1 when you want anything the command line has no flag
for — evaluation criteria, model constraints, fit budgets, the χ² window, a
per-state ambient — or when the run has to be repeatable. Every key is
documented in [aure_config.example.yaml](../aure_config.example.yaml).

---

## U2 — one state, several files

Several files of *the same physical sample* (typically spliced Q segments)
share one refl1d `Sample`, so every layer parameter is tied automatically.

### Ad-hoc

```bash
aure analyze low-Q.dat "Cu/Ti on Si in dTHF" -d mid-Q.dat -d high-Q.dat -o ./output
```

`-d / --extra-data` is repeatable. The first file is the positional; the
extras join it in the same synthetic `state0`. Each dataset's label defaults
to its filename stem.

### Setup YAML

One state, N files:

```yaml
sample_description: |
  2 nm CuOx / 50 nm Cu / 3 nm Ti on Si.

states:
  - name: state0
    data_files:
      - file: low-Q.dat
      - file: mid-Q.dat
      - file: high-Q.dat
```

The YAML route additionally validates the state's files against the resolved
instrument — combined and partial files cannot be mixed in one state, and
partials must share one set id (a REF_L `set_id`, or whatever the instrument
uses as its group key). The ad-hoc `-d` route builds its state directly and
skips that check, so it will accept a mix that the YAML route refuses.

Nuisance parameters are declarable only in the YAML, and U2 is where their
scoping matters: `theta_offset:`, `sample_broadening:` and `background:` become
*one* parameter tied across the state's files, while `intensity:` stays
per-file. Full semantics and defaults are under
[Nuisance parameters](#nuisance-parameters) — they apply to a one-state run
exactly as written there.

---

## U3 — several states

Several *physical states* of one sample — solvent contrast, anneal step,
applied potential — each with its own file(s), its own ambient, and its own
instrument nuisance parameters. Structural parameters are tied across states.

**There is no ad-hoc route.** `states` has no CLI flag, so U3 requires
`-c setup.yaml`. `-d` is a single-state mechanism and combining it with a
`states:` block is a hard error (exit 2) rather than a silent merge.

### The minimal shape: one file per state, ties from the prose

```yaml
sample_description: |
  2 nm CuOx / 50 nm Cu / 3 nm Ti on Si.
  The copper oxide thickness may differ between the two contrasts;
  everything else is the same sample.

states:
  - name: D2O
    extra_description: ambient is D2O (SLD ~6.4)
    data_files:
      - file: Rawdata/REFL_226642_combined_data_auto.txt
  - name: H2O
    extra_description: ambient is H2O (SLD ~-0.56)
    data_files:
      - file: Rawdata/REFL_226660_combined_data_auto.txt
```

```bash
aure analyze -c setup.yaml -o ./output -v
```

Nothing is tied explicitly here: the default tied set applies, minus whatever
the prose says varies. This is U3 "from a textual description".

### The full shape: several files per state, explicit ties, nuisance parameters

Each state may hold as many files as the measurement produced — the three
single-angle partials of one REF_L set, say. Within a state those files share
one `Sample` exactly as in U2, so they are tied to each other automatically;
the `shared_parameters` / `unshared_parameters` spec governs only what is tied
*across* states.

```yaml
sample_description: |
  A copper electrode on a titanium adhesion layer on silicon, measured in
  D2O and then in H2O. Neutrons enter through the silicon substrate.

states:
  - name: D2O
    extra_description: ambient is D2O (SLD ~6.36)
    ambient: {rho: 6.36}
    back_reflection: true
    data_files:
      - {file: REFL_226642_1_226643_partial.txt, label: D2O_0p3deg}
      - {file: REFL_226642_2_226644_partial.txt, label: D2O_0p8deg}
      - {file: REFL_226642_3_226645_partial.txt, label: D2O_2p5deg}
    theta_offset: true
    background: {init: 5.0e-7, min: 0.0, max: 5.0e-6}
    intensity: {init: 1.0, min: 0.9, max: 1.1}

  - name: H2O
    extra_description: ambient is H2O (SLD ~-0.56)
    ambient: {rho: -0.56}
    back_reflection: true
    data_files:
      - {file: REFL_226660_1_226661_partial.txt, label: H2O_0p3deg}
      - {file: REFL_226660_2_226662_partial.txt, label: H2O_0p8deg}
    theta_offset: {init: 0.0, min: -0.01, max: 0.01}
    background: true
    intensity: {init: 1.0, min: 0.9, max: 1.1}

shared_parameters:
  - Cu.thickness
  - Cu.material.rho
  - Cu.interface
  - Ti.thickness
  - Ti.material.rho
  - Ti.interface
  - substrate.interface
```

Reading that as the fit sees it:

- **Tied across the two states:** copper and titanium thickness / SLD /
  interface, and the silicon interface. Seven parameters, each fitted once
  against both contrasts.
- **Free per state:** everything not listed — the ambient SLDs (never tied by
  default, and here each state's is declared anyway), plus any layer the
  refinement loop adds later. A whitelist is exhaustive: `substrate.interface`
  is in the default tied set but has to be re-listed here, or it comes out
  untied.
- **Per state, and never tied:** `theta_offset`, `background`, `intensity` —
  see [Nuisance parameters](#nuisance-parameters) below.
- **Each state's own partials must come from one set** (one REF_L `set_id`):
  226642 for D2O, 226660 for H2O. Mixing sets inside one state is rejected —
  that is the shape that means "these are different measurements", which is
  what states are for. Mixing combined and partial files inside one state is
  rejected too.

One caveat about that `shared_parameters` block, before you copy it: it names
`Cu` and `Ti`, and nothing in the setup above declares those names. They come
from the intake parse of `sample_description`, and a parse that named the layer
`copper` — or `titanium adhesion layer` — makes every tie above fail. The
failure is loud (the run stops and lists the names it does know), but the
reliable fix is to declare each state's `layers:` explicitly, which means
supplying part of the answer you are fitting for. See
[Known limits](#known-limits-before-you-write-ties-by-hand) below: writing ties
by hand and driving the run from prose pull against each other today. The
minimal shape above, which names no layers at all, does not have this problem.

### Which parameters are tied across states

With neither key present, the default tied set is every layer's `thickness` /
`material.rho` / `interface`, plus the substrate's `interface`. The ambient is
never in it. Override the default with *one* of:

```yaml
shared_parameters:      # whitelist — only these are tied
  - Cu.thickness
  - Cu.material.rho
  - Cu.interface
```

```yaml
unshared_parameters:    # blacklist — the default set, minus these
  - CuOx.thickness
```

The two are mutually exclusive; declaring both is an error. Each entry is
`<layer>.<attr>`, where:

| Part | Accepted |
|---|---|
| `<layer>` | a layer name from the parsed stack (or from any state's own `layers:`), a substrate/ambient material name, or the literal aliases `substrate` / `ambient` |
| `<attr>` | `thickness`, `material.rho` (`rho` and `sld` are accepted synonyms), or `interface` — **nothing else** |

Naming any other attribute is refused at build time with the valid list, so
`Cu.roughness` and `intensity` are not tie targets. Roughness *is* tieable, as
`Cu.interface` — an `interface` is the roughness of that layer's boundary with
whatever sits above it.

A tie naming a layer a given state does not have is skipped, not an error, so
per-state structure needs no matching edit to the tie spec.

If you declare **neither** key, the ties are read from the prose: AuRE derives
`unshared_parameters` from `sample_description` at the modeling step, which is
what makes "the copper oxide thickness may differ … everything else is the same
sample" a working declaration. Declaring either key turns that extraction off —
the config wins.

**The names you will read back.** A tied parameter keeps refl1d's plain
`"<layer> <attr>"` spelling and appears once in the fit; an untied one is
prefixed with its state, `"<state> <layer> <attr>"`. So `Cu thickness` in a
report means tied, `H2O CuOx thickness` means that state's own. Per-file
intensities are named `intensity <label>`, which is where the `label:` on each
data file shows up.

### Which structure each state has

By default every state fits the same stack. When a state genuinely differs — an
oxide reduced away under potential, a swollen layer with no dry counterpart —
give that state its own **complete** `layers:` (and `substrate:`, if it
differs); states without `layers:` keep inheriting the shared stack. AuRE can
also infer the difference from the description, or propose it mid-run as a
hypothesis. See
[README](../README.md#when-the-states-differ-in-structure).

All states must share the same orientation: mixing normal and back-reflecting
states in one run is refused, because refl1d puts the substrate roughness on
`sample[0]` in a normal stack and on `sample[n+1]` in a back-reflected one, so
aliasing `substrate.interface` across the two would silently drop the range on
one side.

### Nuisance parameters

Four instrument / measurement parameters are declarable per state, alongside
`extra_description` — they describe the *measurement*, not the structure, and
none of them can appear in a tie spec:

| Key | What it fits | Where it applies | Scope |
|---|---|---|---|
| `theta_offset` | a constant error in the incident angle, degrees | **partials only** | one parameter, tied across that state's files |
| `sample_broadening` | added angular width beyond the declared resolution | **partials only** | one parameter, tied across that state's files |
| `background` | one flat additive background | any data kind | one parameter, tied across that state's files |
| `intensity` | beam-intensity scaling | any data kind | **one per file** |

Two scopes are at work, and the distinction is the point of declaring them per
state:

- *Within* a state, `theta_offset` / `sample_broadening` / `background` are one
  parameter shared by all that state's files, because they describe one
  measurement of one sample. `intensity` is the exception: each file gets its
  own, since separate angles are separately normalised.
- *Across* states they are always independent. There is no way to tie them, and
  no reason to want one — a second contrast is a second measurement, with its
  own alignment and its own incoherent background.

Each accepts either `true` (use the default range) or an explicit
`{init, min, max}` mapping; `false` or omission leaves the parameter fixed and
unfitted. `background` also accepts `fixed: true` to set a constant value
without fitting it. The defaults behind `true`:

| Key | `init` | `min` | `max` |
|---|---|---|---|
| `theta_offset` | 0.0 | −0.02 | 0.02 |
| `sample_broadening` | 0.0 | 0.0 | 0.05 |
| `background` | 1e-6 | 0.0 | 1e-5 |

`intensity` has no `true` shorthand — write the mapping. (`intensity: true`
loads without complaint and then fails in the builder with `'bool' object is
not a mapping`; recorded in [TODO.md](../TODO.md).) Its own defaults, when the
key is absent altogether, are `init` 1.0 over `[0.7, 1.1]`.

**`theta_offset` and `sample_broadening` need an incident angle.** They exist
only on refl1d's angle-based probe, which AuRE builds when a file's header
yields a single theta — i.e. for single-angle partials. Declaring either on a
state whose files are combined curves is refused when the setup loads:

```
State 'D2O': `theta_offset` is only valid for partials (single-angle) files:
REFL_226642_combined_data_auto.txt → REF_L reports 'combined'.
```

If no registered instrument recognises the filenames at all, the same error
says so and points at [docs/instruments.md](instruments.md) — AuRE will not
guess whether an unknown file is a single angle. Note the asymmetry: the
*filename* gates the declaration at load time, but the *header* decides whether
an angle-based probe can actually be built, and a partial-looking file whose
header carries no theta yields a Q-only probe on which the parameter silently
does nothing.

All four keys work for a single-state run too — a `states:` block of length one
carrying any of them is routed through the multi-state builder precisely so
they get wired. They are not, however, reachable from the ad-hoc positional
route: the synthetic `state0` it builds has no place to put them.

### Known limits before you write ties by hand

U3 is scoped to **equality ties**. Three things worth knowing before you hand
one a setup file:

- **A relation that is not an equality has no representation.** The schema can
  tie a parameter or leave it free, and nothing else — there is no route to a
  functional constraint. Asked for one ("the oxide in the second measurement is
  twice as thick"), the prose extractor returns the nearest expressible thing —
  an *untie* — silently, and the fit then reports a χ² for a model the
  description did not ask for.
- **Tie names must match the parsed layer names, which you cannot see yet.**
  Specs are `<layer>.<attr>` with no wildcards — there is no way to say "every
  layer's SLD" — and on a description-driven run the layer names come from the
  intake parse and vary between runs (one real pair of states yielded
  `silicon native oxide (SiO2)` and `SiO2 (native oxide)` for the same layer).
  A hand-written `Cu.thickness` against a parse that said `copper` stops the run
  with the known names listed. The remedy is to declare each state's stack in
  the `states:` block — which trades away the prose interface and supplies the
  shape of the answer. Treat explicit ties and a purely prose-driven run as
  alternatives, not a combination.
- **A tie that was valid at the start can go inert later.** Names are matched
  exactly, and nothing stops a refinement iteration from renaming a layer the
  spec references (`copper` → `Cu metal`). When that happens the spec no longer
  matches and is dropped, logged as "Dropped tie spec(s) for removed layer(s)"
  — which reads as a structural edit rather than the rename it was. The setup
  still shows the tie; the fit no longer has it. Check the run log if a tie
  matters.

The first two are recorded, with the rest of the assessment, in
[scope.md](scope.md#u3-assessed); the tie-name problem and the mid-run rename
each have their own entry in [TODO.md](../TODO.md).

---

## Mixing the two routes

`-c` and the positionals can be combined, with these rules:

| Situation | Behaviour |
|---|---|
| Positional `SAMPLE_DESCRIPTION` and the setup both set it | The positional wins; a note is printed. |
| Positional `DATA_FILE` given, setup has `states:` | The positional is **ignored**; a note is printed. The setup's files are used. |
| `-d` given, setup has `states:` | **Error**, exit 2. Move the files into the states block. |
| `--data-dir` given without `-c` | Ignored, with a warning. |
| No data file and no `states:` | Error, exit 2. |
| No description in either place | Error, exit 2. |

Because a setup file must declare at least one state to load at all, the
`-c` route always brings its own data. And because the first positional is
validated as an existing path, you cannot pass *only* a description alongside
`-c` (`aure analyze -c setup.yaml "some description"` reads the description as
a filename and fails) — put `sample_description:` in the YAML.

`-h/--hypothesis`, `-m/--max-refinements` and `-n/--model-name` are folded into
the setup as `hypothesis`, `max_refinements` and `model_name`, so the flag
always wins over the file.

### Locating the data

Relative `data_files:` paths are resolved against the first directory that
contains them, in this order:

1. `--data-dir` on the command line (highest priority; requires `-c`),
2. the top-level `data_dir:` key in the YAML,
3. the directory holding the YAML,
4. the current working directory.

A file found in none of them is a hard error listing the directories searched.
Absolute paths are used as given.

---

## Options that apply to every route

| Option | Effect |
|---|---|
| `-o, --output-dir PATH` | Write checkpoints, the refl1d/bumps exports, `final_state.json` and the LLM ledger here. Without it the run leaves nothing on disk, cannot be resumed, and — see below — is judged differently. |
| `-m, --max-refinements N` | Ceiling on refinement iterations (default 5). Not a target: the loop stops as soon as χ² lands in the acceptance window. |
| `-h, --hypothesis TEXT` | Seeded at intake as top-ranked candidate structural changes (`origin="user"`). Not baked into the baseline stack. |
| `-n, --model-name NAME` | Basename for the exported refl1d/bumps files (default: the output-dir name, then the data-file stem). |
| `-v, --verbose` | Stream node-by-node progress to stderr. |
| `--json` | Machine-readable summary on stdout — selected fit, χ², parameters, and the still-`pending` hypotheses. |

`-o` is worth passing even for a throwaway run. refl1d exports the SLD profile
only when the run has an output directory, and the deterministic SLD-profile
artifact check needs that profile: with no export it cannot verify, so the
deterministic χ² stop stands down and the evaluator LLM's verdict decides
acceptance instead. An `-o`-less run is therefore not just unrecorded, it is
judged by a different rule.

## Where the run controls come from

Three layers, innermost wins:

1. `.env` / the ambient environment — `LLM_*`, `FIT_METHOD`, `CHI2_MAX`, …
2. Setup keys, which are applied as environment overrides for the duration of
   the run and restored afterwards: `chi2_max`, `chi2_min`, `fit_method`,
   `fit_steps`, `fit_burn`, `fit_method_final`, `fit_steps_final`,
   `fit_burn_final`, `final_fit_chi2_max`, and the `llm_*` keys.
3. The command-line flags above.

The acceptance window in force is echoed in the run banner and pinned into the
run's checkpoints, so `aure resume` keeps the window the run was launched with
rather than the resuming shell's.

Some knobs that change what model comes out have **no setup key and no flag** —
`MODE_ENUMERATION`, `THIN_LAYER_MODE_K`, `THIN_LAYER_MODE_SEEDS`,
`ROUGHNESS_MAX_OUTER`, `FINAL_SELECTION_TOL`, `FINAL_TIER_CHI2_FACTOR`,
`USE_RUN_TITLE`. They are environment-only, and a setup file therefore does not
fully record the physics policy its run used.

---

## Adjacent CLI routes

Same declarations, different destination — all four take the setup YAML in the
same form:

| Command | What it does instead |
|---|---|
| `aure prepare` | Stops after `modeling` and writes a `problem.json` for bare refl1d (U9). |
| `aure batch` | Runs many setups in one invocation, each merged with a `defaults:` block (U10). A flat single-job manifest *is* a setup file, so the same YAML works with both `analyze -c` and `batch`. |
| `aure resume` | Restarts an interrupted run from any checkpoint. |
| `aure serve` | Opens a finished run's output directory in the web UI. |

Full option reference for each: [README](../README.md#cli-reference).
